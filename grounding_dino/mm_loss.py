#!/usr/bin/env python3
"""多模态指代理解的匹配器与损失。

仓库里原本**没有任何 criterion**(上游 GroundingDINO 的损失跑在 MMDetection 那套框架里,
不在这个精简仓库内), 所以这里从零写一套 DETR 风格的 `SetCriterion`:

    总损失 = 1.0 * focal(分类, token 级) + 5.0 * L1(框) + 2.0 * GIoU(框)

三处值得单独说明的设计:

1. **匹配退化成 argmin。** 本数据集每张图只有 1 个 GT 框, 所以「900 个 query 里挑一个」
   就是取代价最小者 —— Hungarian 算法没有必要。这也顺带让 `scipy` 变成非必需依赖
   (它现在只是 `supervision` 的传递依赖, 不该被直接 import)。多目标图会显式报错,
   而不是静默算错。

2. **分类是 token 级的。** `pred_logits` 形状是 `[B, nq, T]`(T = 文本 token 槽位数),
   正样本不是「某个 query 属于第几类」, 而是「某个 query 在第几个 token 上应激活」。
   所以 `target_cls` 是在命中的 (样本, query) 上、把**所有真实 token 槽位**置 1。

3. **`-inf` 必须先 clamp。** `ContrastiveEmbed` 把 padding token 槽位掩成 `-inf`
   (实测 900x248 个, 是设计行为)。直接送进 `binary_cross_entropy_with_logits` 在 fp16/AMP
   下会出 NaN。clamp 到 -50 不改变 loss(sigmoid(-50) 已经是 0), 但把风险消掉了。
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GroundingDINO")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss  # noqa: E402
from groundingdino.util.box_ops import (  # noqa: E402
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)

# `pred_logits` 里 padding 槽位是 -inf; 这个下限保证 fp16 下也不出 NaN
LOGIT_CLAMP = 50.0


class HungarianMatcher(nn.Module):
    """单 GT 框时退化为 argmin 的匹配器。

    代价(全部在归一化坐标上算):
      cost_class = -(1-p)^gamma * log(p+eps)   p = 该框在真实 token 上的最大分数
      cost_bbox  = L1(cxcywh)
      cost_giou  = -GIoU
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets, token_mask):
        """返回 length-B 的 list, 每项是 (pred_idx[Tensor], tgt_idx[Tensor])。

        `token_mask` [B,T] bool, True = 真实文本 token(用于取短语分数)。
        """
        # .float(): AMP 下 pred_logits 是 fp16, 代价里含 cdist / GIoU, 用 fp32 算更稳
        logits = outputs["pred_logits"].clamp(-LOGIT_CLAMP, LOGIT_CLAMP).float()
        boxes = outputs["pred_boxes"].float()

        n_tgt = [len(t["boxes"]) for t in targets]
        if any(n != 1 for n in n_tgt):
            raise NotImplementedError(
                f"当前只实现了「每张图恰好 1 个 GT 框」的匹配(实测数据集即如此), "
                f"实际每个样本的框数为 {n_tgt}。多目标需要真正的 Hungarian 算法: "
                f"把 scipy 显式加进 pyproject.toml 后改用 scipy.optimize.linear_sum_assignment。"
            )

        indices = []
        for i, tgt in enumerate(targets):
            prob = logits[i].sigmoid()                      # [nq, T]
            # 短语分数: 与 run_grounding.py 的 top-1 打分口径一致
            p = prob[:, token_mask[i]].max(dim=-1).values    # [nq]
            cost_class = -((1 - p) ** self.focal_gamma) * torch.log(p + 1e-8)

            # .to(boxes.dtype): AMP 下 pred_boxes 已是 fp32, 而 target 可能还是 fp16
            # (torch.cdist 不接受 fp16)
            gt_box = tgt["boxes"].to(boxes.dtype)             # [1,4] cxcywh
            cost_bbox = torch.cdist(boxes[i], gt_box, p=1).squeeze(-1)   # [nq]

            giou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(boxes[i]), box_cxcywh_to_xyxy(gt_box)
                )
            )
            cost_giou = -giou

            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            q = int(cost.argmin().item())
            indices.append(
                (
                    torch.tensor([q], dtype=torch.long),
                    torch.tensor([0], dtype=torch.long),
                )
            )
        return indices


class SetCriterion(nn.Module):
    """DETR 风格的集合预测损失(分类用 token 级 focal loss)。

    `loss_dict` 里的键: `loss_ce` / `loss_bbox` / `loss_giou` 是最后一层 decoder 的,
    `loss_*_aux{i}` 是各中间层(i 从 0 开始)。总损失 = 所有值之和(aux 权重与主层相同,
    这是 DETR 的标准做法)。
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_dict=None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict or {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _one_layer(self, pred_logits, pred_boxes, targets, token_mask, num_boxes, indices, sfx=""):
        # 见文件头的第 3 点; .float() 让整个 criterion 在 fp32 下算(AMP 下输入是 fp16,
        # 直接在 fp16 上做 BCE / GIoU 容易出 NaN)
        logits = pred_logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP).float()
        pred_boxes = pred_boxes.float()

        # ---- 分类: 在命中的 (样本, query) 上, 把所有真实 token 槽位置 1 ----
        target_cls = torch.zeros_like(logits)
        for i, (pred_idx, _) in enumerate(indices):
            target_cls[i, pred_idx] = token_mask[i].to(target_cls.dtype)

        loss_ce = sigmoid_focal_loss(
            logits, target_cls, num_boxes,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
        )

        # ---- 框: 只算命中 query 上的 L1 + GIoU ----
        src_boxes = torch.cat([pred_boxes[i][pi] for i, (pi, _) in enumerate(indices)])
        tgt_boxes = torch.cat([t["boxes"] for t in targets]).to(src_boxes.dtype)

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes
        loss_giou = (
            1.0
            - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)
                )
            )
        ).sum() / num_boxes

        return {
            f"loss_ce{sfx}": loss_ce,
            f"loss_bbox{sfx}": loss_bbox,
            f"loss_giou{sfx}": loss_giou,
        }

    def forward(self, outputs, targets):
        if "text_token_mask" not in outputs:
            raise KeyError(
                "outputs 里没有 text_token_mask —— 请用 `model(samples, targets, ...)` "
                "调用(targets 非 None 时 forward 才会附加该键), 不要在 criterion 里重新 tokenize。"
            )
        token_mask = outputs["text_token_mask"].bool()
        dev = outputs["pred_logits"].device
        token_mask = token_mask.to(dev)

        # 匹配只在最后一层做一次, 索引复用到各 aux 层(DETR 标准做法)
        indices = self.matcher(outputs, targets, token_mask)
        num_boxes = max(1, sum(len(t["boxes"]) for t in targets))

        losses = self._one_layer(
            outputs["pred_logits"], outputs["pred_boxes"],
            targets, token_mask, num_boxes, indices,
        )
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            losses.update(
                self._one_layer(
                    aux["pred_logits"], aux["pred_boxes"],
                    targets, token_mask, num_boxes, indices, sfx=f"_aux{i}",
                )
            )
        return losses


def build_criterion(focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> SetCriterion:
    matcher = HungarianMatcher(focal_alpha=focal_alpha, focal_gamma=focal_gamma)
    return SetCriterion(matcher, focal_alpha=focal_alpha, focal_gamma=focal_gamma)
