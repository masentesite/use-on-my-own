#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练链路(`mm_data.py` + `mm_loss.py`)的验证测试 —— 纯 CPU, 不需要完整模型。

用法(仓库根目录):
    .venv/bin/python test_training.py
    .venv/bin/python test_training.py --data-dir ../TrainSet. --samples 4

覆盖两类东西:
  A. 数据管道: 三模态对齐 / padding 语义 / 增强的同步性与不变量
  B. 损失: 匹配、归一化、-inf 防护、aux 层

⚠️ 两条「假通过」陷阱, 测试里已按此设计断言(与 test_multimodal.py 的教训同源):
  1. `DepthPreprocessor` 的百分位归一化对单调仿射变换**不变**, 所以不能用「depth 加个常数
     看结果变不变」来验证 depth 分支 —— 这里的做法是直接断言**像素值本身**:
     depth 补 0 后必须真的是 0, 且最近邻 resize 不能凭空造出新的中间值。
  2. 翻转会让 query 的置信度排序整体重排, 按下标配对会拿 A 物体的坐标比 B 物体。
     这里只验证**框变换公式**(x1'=1-x2)这个纯几何量, 不碰模型输出。
"""

import argparse
import os
import sys
import traceback

# ---- 代理环境变量必须在 import torch / transformers 之前设好(与 run_grounding.py 一致) ----
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for _v in ("no_proxy", "NO_PROXY"):
    if "hf-mirror.com" not in os.environ.get(_v, ""):
        os.environ[_v] = "hf-mirror.com," + os.environ.get(_v, "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from mm_data import (  # noqa: E402
    IR_MEAN,
    IR_STD,
    MultiModalReferDataset,
    collate_fn,
    load_items,
    split_image_ids,
)
from mm_loss import HungarianMatcher, SetCriterion, build_criterion  # noqa: E402

DEFAULT_DATA = os.path.join(REPO, "../TrainSet.")


# ======================================================================
#  基础设施
# ======================================================================
class Failure(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Failure(msg)


def fmt(x):
    return f"{x:.3e}" if isinstance(x, float) and (abs(x) < 1e-3 or abs(x) > 1e5) else f"{x:.6f}"


class Ctx(dict):
    __getattr__ = dict.get


# ======================================================================
#  测试
# ======================================================================
def test_dataset_basic(ctx):
    """每个样本的形状/取值范围/框语义都对。"""
    ds = ctx["ds"]
    for i in range(min(3, len(ds))):
        s = ds[i]
        img, ir, depth = s["image"], s["ir"], s["depth"]
        check(img.shape[0] == 3, f"可见光应为 3 通道, 实际 {tuple(img.shape)}")
        check(ir.shape[0] == 1, f"IR 应为 1 通道(通道已取均值), 实际 {tuple(ir.shape)}")
        check(depth.shape[0] == 1, f"depth 应为 1 通道, 实际 {tuple(depth.shape)}")
        check(img.shape[-2:] == ir.shape[-2:] == depth.shape[-2:],
              f"三模态空间尺寸不一致: {tuple(img.shape)} {tuple(ir.shape)} {tuple(depth.shape)}")

        # 归一化后的可见光: 大致落在 [-2.2, 2.7] 附近, 且不是常数
        check(img.max() > 0.5 and img.min() < 0, "可见光看起来没有做 ImageNet 归一化")
        # IR 用标量 mean/std
        check(ir.max() > 0.5 and ir.min() < 0, "IR 看起来没有做归一化")

        # depth 保持原始 uint16 数值(未归一化): 0 是 invalid 哨兵, 必须原样保留
        check(depth.max() > 100.0,
              f"depth 的最大值只有 {depth.max().item():.1f} —— 像是被归一化或截断了, "
              f"0 哨兵语义会被破坏")

        # 框: 归一化 cxcywh, 0<cx<1, 0<w<=1
        b = s["boxes"]
        check(b.shape == (1, 4), f"框形状应为 (1,4), 实际 {tuple(b.shape)}")
        check(0 <= b[0, 0] <= 1 and 0 <= b[0, 1] <= 1 and 0 < b[0, 2] <= 1 and 0 < b[0, 3] <= 1,
              f"框不像归一化 cxcywh: {b.tolist()}")

        # caption 规范化和 run_grounding.py 一致
        check(s["caption"] == s["caption"].lower(), f"caption 没有小写: {s['caption']!r}")
        check(s["caption"].endswith("."), f"caption 没有补句点: {s['caption']!r}")


def test_split_by_image(ctx):
    """划分必须按图, 同一张图的 5 条 query 不能跨集。"""
    items = ctx["items"]
    tr, va = split_image_ids(items, 0.1, seed=42)
    check(len(tr) + len(va) == 400, f"图总数应为 400, 实际 {len(tr)}+{len(va)}")
    check(len(va) == 40, f"val 应为 40 张图, 实际 {len(va)}")
    check(not (set(tr) & set(va)), "train / val 的图 id 有重叠")
    check(tr == sorted(tr) and va == sorted(va), "划分结果没有排序(会影响可复现性)")

    # 同 seed 必须完全一致, 换 seed 必须不同
    tr2, va2 = split_image_ids(items, 0.1, seed=42)
    check(tr == tr2 and va == va2, "同一个 seed 两次划分结果不同")
    tr3, _ = split_image_ids(items, 0.1, seed=7)
    check(tr != tr3, "换了 seed 划分结果却没变")

    # 每张图的 query 都落在同一侧
    side = {}
    for it in items:
        s = "train" if it["img_id"] in set(tr) else "val"
        check(side.setdefault(it["img_id"], s) == s, f"{it['img_id']} 的 query 跨集了")


def test_resize_invariance(ctx):
    """等比 resize 不改变归一化框 —— 所以数据集里不需要动框。"""
    items = ctx["items"]
    ds_a = MultiModalReferDataset(ctx["data_dir"], image_ids=[items[0]["img_id"]],
                                  train=False, resize=800, max_size=1333, items=items)
    ds_b = MultiModalReferDataset(ctx["data_dir"], image_ids=[items[0]["img_id"]],
                                  train=False, resize=400, max_size=666, items=items)
    a, b = ds_a[0], ds_b[0]
    ha, wa = a["image"].shape[-2:]
    hb, wb = b["image"].shape[-2:]
    check((ha, wa) != (hb, wb), f"两组 resize 尺寸相同({ha}x{wa}), 测不出不变性")
    check(float((a["boxes"] - b["boxes"]).abs().max()) < 1e-6,
          f"resize 改变了归一化框: {a['boxes'].tolist()} vs {b['boxes'].tolist()}")

    # 尺寸数学与 run_grounding.py 的 RandomResize([800], 1333) 对 1920x1080 的结果一致
    check((ha, wa) == (750, 1333),
          f"1920x1080 经 resize=800/max_size=1333 应为 750x1333, 实际 {ha}x{wa}")


def test_hflip_geometry(ctx):
    """水平翻转: cx' = 1 - cx, 其余不变; 且三模态一起翻。"""
    items = ctx["items"]
    img_id = items[0]["img_id"]

    ds_no = MultiModalReferDataset(ctx["data_dir"], image_ids=[img_id], train=True,
                                   resize=400, max_size=666, hflip_p=0.0, items=items)
    ds_yes = MultiModalReferDataset(ctx["data_dir"], image_ids=[img_id], train=True,
                                    resize=400, max_size=666, hflip_p=1.0, items=items)
    a, b = ds_no[0], ds_yes[0]

    ba, bb = a["boxes"][0], b["boxes"][0]
    check(abs(float(bb[0]) - (1.0 - float(ba[0]))) < 1e-6,
          f"翻转后 cx 应为 1-cx: {float(ba[0]):.6f} -> {float(bb[0]):.6f}")
    check(abs(float(bb[1]) - float(ba[1])) < 1e-6, "水平翻转不应该动 cy")
    check(abs(float(bb[2]) - float(ba[2])) < 1e-6 and abs(float(bb[3]) - float(ba[3])) < 1e-6,
          "水平翻转不应该动 w/h")

    # 三个模态必须一起翻: 拿图像张量自己验证
    for k in ("image", "ir", "depth"):
        flipped = torch.flip(b[k], dims=[-1])
        d = float((flipped - a[k]).abs().max())
        check(d < 1e-5, f"{k} 没有跟着一起翻转(与手工镜像的最大偏差 {fmt(d)})")


def test_depth_padding_and_nearest(ctx):
    """depth 补 0 = invalid; 最近邻 resize 不产生新的中间值。"""
    items = ctx["items"]
    ids = [items[0]["img_id"]]
    ds = MultiModalReferDataset(ctx["data_dir"], image_ids=ids, train=False,
                                resize=400, max_size=666, items=items)
    s = ds[0]
    depth = s["depth"]

    # 原始值的集合(去重)应该很小 —— 16 位深度图的实际取值远少于像素数;
    # 如果用了双线性插值, 值域会爆炸成几乎每个像素一个新值。
    uniq = torch.unique(depth)
    ratio = uniq.numel() / depth.numel()
    check(ratio < 0.2,
          f"depth 去重后占 {ratio * 100:.1f}% 的像素 —— 数量过多, 像是用了双线性插值"
          f"(最近邻不会产生新数值)")

    # 0 必须还是 0(补 0 的 padding 与 invalid 哨兵共用同一个值, 语义一致)
    check(float(depth.min()) == 0.0, f"depth 最小值应为 0, 实际 {float(depth.min())}")

    # 合成一个 batch, 检查 collate 的 padding
    ds2 = MultiModalReferDataset(ctx["data_dir"], image_ids=ids[:1], train=False,
                                 resize=400, max_size=666, items=items)
    batch = [ds2[0], ds2[0]]
    samples, targets, ir, depth_b = collate_fn(batch)
    B, C, H, W = samples.tensors.shape
    check((B, C) == (2, 3), f"RGB batch 形状异常: {tuple(samples.tensors.shape)}")
    check(ir.shape == (B, 1, H, W), f"IR batch 应为 {(B, 1, H, W)}, 实际 {tuple(ir.shape)}")
    check(depth_b.shape == (B, 1, H, W),
          f"depth batch 应为 {(B, 1, H, W)}, 实际 {tuple(depth_b.shape)}")
    check(samples.mask.shape == (B, H, W), f"mask 形状异常: {tuple(samples.mask.shape)}")
    check(len(targets) == B, "targets 数量与 batch 不符")
    check(set(targets[0]) == {"caption", "boxes"}, f"target 键集异常: {set(targets[0])}")


def _find_colorized_ir(ctx, limit=60):
    """找一张「真彩色」的 IR 图 —— 只有这种图能把「通道均值」和「亮度公式」区分开。

    实测 400 张里 382 张近灰度(通道差 <=1), 18 张通道差最大到 253; 在灰度图上
    两个公式的差别小到测不出来, 所以必须挑一张彩色的。
    """
    from PIL import Image

    for it in ctx["items"][:limit]:
        p = os.path.join(ctx["data_dir"], it["paths"]["infrared"])
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        spread = a.max(axis=2) - a.min(axis=2)
        # 用 **最大** 通道差, 不用均值: 实测 400 张 IR 的平均通道差全都在 0.5 以下
        # (整体上都是灰的), 只有 18 张存在一小片真正饱和的彩色区域(最大差 253,
        # 占比约 0.04%) —— 用均值判据会一张都找不到。
        if float(spread.max()) > 100.0:
            return it
    return None


def test_ir_channel_mean(ctx):
    """IR 用的是**通道均值**, 不是 PIL convert("L") 的亮度公式 0.299R+0.587G+0.114B。"""
    from PIL import Image
    from torchvision.transforms import functional as TF

    check(abs(IR_MEAN - 0.449) < 1e-3 and abs(IR_STD - 0.226) < 1e-3,
          f"IR 的标量统计应约等于 ImageNet 三通道统计的均值, 实际 {IR_MEAN}/{IR_STD}")

    it = _find_colorized_ir(ctx)
    check(it is not None, "没找到彩色 IR 图, 这个测试就失去意义了(先用 --samples 调大)")

    ds = MultiModalReferDataset(ctx["data_dir"], image_ids=[it["img_id"]], train=False,
                                resize=400, max_size=666, items=ctx["items"])
    idx = [j for j, x in enumerate(ds.items) if x["qid"] == it["qid"]][0]
    got = ds[idx]["ir"]

    # 按数据集的同一条变换路径重算两个候选公式
    pil = Image.open(os.path.join(ctx["data_dir"], it["paths"]["infrared"])).convert("RGB")
    t = TF.to_tensor(TF.resize(pil, (got.shape[-2], got.shape[-1])))
    mean_path = ((t.mean(dim=0, keepdim=True)) - IR_MEAN) / IR_STD
    luma_path = (0.299 * t[0:1] + 0.587 * t[1:2] + 0.114 * t[2:3] - IR_MEAN) / IR_STD

    d_mean = float((mean_path - got).abs().max())
    d_luma = float((luma_path - got).abs().max())
    check(d_mean < 1e-4, f"IR 不是通道均值(最大偏差 {fmt(d_mean)})")
    check(d_luma > 1e-3,
          f"这张彩色 IR 上「均值」和「亮度公式」几乎一样({fmt(d_luma)}) —— 区分不出来, "
          f"换一张通道差更大的图")


# ---------------------------------------------------------------- 损失

def test_matcher_picks_argmax(ctx):
    """匹配最小化的是**总代价**, 不是单纯取分数最高的 query —— 两个分量分别验。"""
    B, NQ, T = 2, 20, 8
    torch.manual_seed(0)
    token_mask = torch.zeros(B, T, dtype=torch.bool)
    token_mask[:, 1:5] = True
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    targets = [{"boxes": gt.clone()} for _ in range(B)]

    # --- (a) 框全部相同 -> 只剩 cost_class 起作用, 必须选中分数最高的 query ---
    logits = torch.randn(B, NQ, T) - 2.0
    logits[0, 3, :] = 5.0
    logits[1, 11, :] = 5.0
    boxes_same = gt.repeat(B, NQ, 1).clone()
    idx = HungarianMatcher()({"pred_logits": logits, "pred_boxes": boxes_same},
                             targets, token_mask)
    check(int(idx[0][0][0]) == 3,
          f"框全相同时应选中分数最高的 query 3, 实际 {int(idx[0][0][0])}")
    check(int(idx[1][0][0]) == 11,
          f"框全相同时应选中分数最高的 query 11, 实际 {int(idx[1][0][0])}")

    # --- (b) 分数全部相同 -> 只剩框代价起作用, 必须选中框最准的 query ---
    logits_flat = torch.zeros(B, NQ, T)
    boxes = torch.rand(B, NQ, 4) * 0.3 + 0.35
    # 把精确命中 GT 的框放在指定的 query 上(其余 query 随机 -> 几乎不可能一样准)
    boxes[0, 7] = gt[0]
    boxes[1, 15] = gt[0]
    idx = HungarianMatcher()({"pred_logits": logits_flat, "pred_boxes": boxes},
                             targets, token_mask)
    check(int(idx[0][0][0]) == 7,
          f"分数全相同时应选中框最准的 query 7, 实际 {int(idx[0][0][0])}")
    check(int(idx[1][0][0]) == 15,
          f"分数全相同时应选中框最准的 query 15, 实际 {int(idx[1][0][0])}")

    # 多目标图必须显式报错, 不能静默算错
    bad = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]])}]
    try:
        HungarianMatcher()({"pred_logits": logits[:1], "pred_boxes": boxes[:1]}, bad,
                           token_mask[:1])
        raise Failure("多 GT 框时 matcher 没有报错 —— 会静默算错")
    except NotImplementedError:
        pass


def test_criterion_perfect_prediction(ctx):
    """预测 == GT 时分类/框损失都应该接近 0。"""
    B, NQ, T = 2, 16, 6
    token_mask = torch.zeros(B, T, dtype=torch.bool)
    token_mask[:, 1:4] = True                       # 3 个真实 token

    gt = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    # 分类: 只有 query 0 在真实 token 槽位上给大 logit, 其余一律 -12。
    # ⚠️ 不能让**所有** query 都在真实槽位上给大 logit —— 那样 899 个没被匹配到的
    # query 会变成「高置信但被标为负样本」, focal loss 会正确地给出巨大惩罚(实测 25.3),
    # 而那是模型该被罚的情形, 不是「完美预测」。
    logits = torch.full((B, NQ, T), -12.0)
    logits[:, 0, 1:4] = 12.0
    boxes = gt.repeat(B, NQ, 1).clone()             # 每个 query 都精确命中

    crit = build_criterion()
    losses = crit({"pred_logits": logits, "pred_boxes": boxes,
                   "text_token_mask": token_mask},
                  [{"boxes": gt.clone()} for _ in range(B)])
    check(losses["loss_ce"] < 0.05, f"完美分类下 loss_ce 应≈0, 实际 {float(losses['loss_ce']):.4f}")
    check(losses["loss_bbox"] < 1e-4, f"完美框下 loss_bbox 应≈0, 实际 {fmt(float(losses['loss_bbox']))}")
    check(losses["loss_giou"] < 1e-4, f"完美框下 loss_giou 应≈0, 实际 {fmt(float(losses['loss_giou']))}")


def test_criterion_num_boxes_normalization(ctx):
    """num_boxes = 命中对数 = B, 所以 loss_bbox 是「每样本平均」而不是「每张量平均」。"""
    B, NQ, T = 3, 8, 5
    token_mask = torch.zeros(B, T, dtype=torch.bool)
    token_mask[:, 1:3] = True
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    logits = torch.zeros(B, NQ, T)
    # 让每个 query 的框都差 0.1(在 cx 上) -> 单个命中对的 L1 = 0.1
    boxes = gt.repeat(B, NQ, 1).clone()
    boxes[:, :, 0] += 0.1

    crit = build_criterion()
    losses = crit({"pred_logits": logits, "pred_boxes": boxes, "text_token_mask": token_mask},
                  [{"boxes": gt.clone()} for _ in range(B)])
    # 每个命中对 L1 = 0.1; 总和 = B*0.1; /num_boxes = B -> 0.1
    check(abs(float(losses["loss_bbox"]) - 0.1) < 1e-5,
          f"loss_bbox 应为 0.1(每样本归一化), 实际 {fmt(float(losses['loss_bbox']))}")


def test_criterion_neg_inf_safe(ctx):
    """-inf 的 padding 槽位不能产生 NaN —— 这是 AMP 下最容易炸的地方。"""
    B, NQ, T = 2, 8, 6
    token_mask = torch.zeros(B, T, dtype=torch.bool)
    token_mask[:, 1:4] = True
    logits = torch.randn(B, NQ, T)
    logits[:, :, 4:] = float("-inf")                # padding 槽位(设计行为)
    boxes = torch.rand(B, NQ, 4) * 0.5 + 0.25
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])

    crit = build_criterion()
    losses = crit({"pred_logits": logits, "pred_boxes": boxes, "text_token_mask": token_mask},
                  [{"boxes": gt.clone()} for _ in range(B)])
    for k, v in losses.items():
        check(not torch.isnan(v).any(), f"{k} 出现了 NaN(-inf 槽位没被 clamp 住)")
        check(torch.isfinite(v).all(), f"{k} 不是有限值: {float(v)}")

    # fp16 下同样不能炸(AMP 路径)
    losses16 = crit({"pred_logits": logits.half(), "pred_boxes": boxes.half(),
                     "text_token_mask": token_mask},
                    [{"boxes": gt.clone().half()} for _ in range(B)])
    for k, v in losses16.items():
        check(torch.isfinite(v).all(), f"fp16 下 {k} 不是有限值: {v}")


def test_criterion_aux_layers(ctx):
    """aux 层的 loss 必须真的被算出来, 且键名与主层区分开。"""
    B, NQ, T = 2, 8, 5
    token_mask = torch.zeros(B, T, dtype=torch.bool)
    token_mask[:, 1:3] = True
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    base = {"pred_logits": torch.randn(B, NQ, T), "pred_boxes": torch.rand(B, NQ, 4),
            "text_token_mask": token_mask}
    aux = [{"pred_logits": torch.randn(B, NQ, T), "pred_boxes": torch.rand(B, NQ, 4)}
           for _ in range(2)]
    crit = build_criterion()
    losses = crit({**base, "aux_outputs": aux}, [{"boxes": gt.clone()} for _ in range(B)])

    for k in ("loss_ce", "loss_bbox", "loss_giou", "loss_ce_aux0", "loss_ce_aux1"):
        check(k in losses, f"缺少 {k}, 实际有 {sorted(losses)}")
    check(all(torch.isfinite(v).all() for v in losses.values()), "aux loss 里有非有限值")
    check("loss_ce_aux0" in losses and "loss_ce_aux1" in losses,
          "aux 层的键没有编号, 会互相覆盖")

    # 缺 text_token_mask 时必须明确报错, 而不是默默用错的口径
    no_mask = {k: v for k, v in base.items() if k != "text_token_mask"}
    check("text_token_mask" not in no_mask, "测试自身写错了: 去掉 mask 的字典里还有 mask")
    try:
        crit(no_mask, [{"boxes": gt.clone()} for _ in range(B)])
        raise Failure("缺少 text_token_mask 时 criterion 没有报错")
    except KeyError:
        pass


def test_missing_gt_guard(ctx):
    """没有 GT 的条目必须被过滤掉(否则会造出「有图无框」的坏监督)。"""
    img_id = ctx["items"][0]["img_id"]
    same_img = [dict(it) for it in ctx["items"] if it["img_id"] == img_id]
    check(len(same_img) >= 2, "这张图的 query 太少, 测不出过滤行为")

    dropped = same_img[0]["qid"]
    same_img[0]["bbox"] = None
    ds = MultiModalReferDataset(ctx["data_dir"], image_ids=[img_id], train=False,
                                resize=400, max_size=666, items=same_img)

    kept = {x["qid"] for x in ds.items}
    check(dropped not in kept, f"无 GT 的条目 {dropped} 没有被过滤掉")
    check(len(ds) == len(same_img) - 1,
          f"应保留 {len(same_img) - 1} 条, 实际 {len(ds)} 条")
    for i in range(len(ds)):
        check(ds[i]["boxes"].shape == (1, 4), "过滤后仍有形状异常的框")


# ======================================================================
#  驱动
# ======================================================================
TESTS = [
    ("1. 数据管道: 样本形状与语义", test_dataset_basic),
    ("2. 数据管道: 按图划分", test_split_by_image),
    ("3. 数据管道: resize 不改变归一化框", test_resize_invariance),
    ("4. 数据管道: 水平翻转的同步性与几何", test_hflip_geometry),
    ("5. 数据管道: depth padding 与最近邻插值", test_depth_padding_and_nearest),
    ("6. 数据管道: IR 通道均值", test_ir_channel_mean),
    ("7. 损失: 匹配退化为 argmin", test_matcher_picks_argmax),
    ("8. 损失: 完美预测 ≈ 0", test_criterion_perfect_prediction),
    ("9. 损失: num_boxes 归一化", test_criterion_num_boxes_normalization),
    ("10. 损失: -inf 防护(fp32/fp16)", test_criterion_neg_inf_safe),
    ("11. 损失: aux 层与接口守卫", test_criterion_aux_layers),
    ("12. 数据管道: 无 GT 条目被过滤", test_missing_gt_guard),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--samples", type=int, default=4, help="数据集测试用几张图")
    args = ap.parse_args()

    print("=" * 78)
    print(f"data_dir={os.path.abspath(args.data_dir)}")
    print("=" * 78)

    items = load_items(args.data_dir)
    ids = sorted({it["img_id"] for it in items})[: args.samples]
    ctx = Ctx(
        data_dir=args.data_dir,
        items=items,
        ids=ids,
        ds=MultiModalReferDataset(args.data_dir, image_ids=ids, train=False,
                                  resize=400, max_size=666, items=items),
    )

    passed, failed = 0, []
    for name, fn in TESTS:
        print(f"\n[ RUN  ] {name}")
        try:
            fn(ctx)
            print(f"[  OK  ] {name}")
            passed += 1
        except Failure as e:
            print(f"[ FAIL ] {name}\n         {e}")
            failed.append(name)
        except Exception:
            print(f"[ FAIL ] {name}\n{traceback.format_exc()}")
            failed.append(name)

    print("\n" + "=" * 78)
    print(f"通过 {passed} / {len(TESTS)}")
    if failed:
        for n in failed:
            print(f"  ✗ {n}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
