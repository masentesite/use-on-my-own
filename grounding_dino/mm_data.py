#!/usr/bin/env python3
"""多模态(RGB + IR + Depth)指代理解数据集 —— 训练用。

与 `run_grounding.py` 的关系:那个脚本是**评估协议**的单一事实来源, 本文件只负责训练侧,
但凡涉及「预处理口径」的地方(图像 resize、归一化、caption 规范化)都刻意复刻它, 免得
训练与评估的输入分布不一致。

数据事实(实测):
  - `TrainSet./queries/queries.json`: 2000 条 query, 400 张图 × 5 条, key 形如 "001_001"
  - 每条 query 自带 `bbox`(归一化 xyxy, 全部 2000 条都有)
  - 三模态尺寸全部 1920x1080; visible/infrared 是 3 通道 uint8,
    depth 是 **1 通道 uint16**(0~19999), **0 = invalid**, 约 27.7% 的像素是 0

两条硬规则(踩过坑, 见 `test_training.py` 的断言):
  1. **depth 只能用最近邻插值**。双线性会在有效像素和 0(invalid)之间插出虚假的近零值,
     污染 `DepthPreprocessor` 依赖的 valid/invalid 边界。
  2. **不能对 depth 做仿射增强**(改亮度/对比度/整体缩放)。`DepthPreprocessor` 的百分位
     归一化对单调仿射变换**不变**(quantile(aD+b) = a*quantile(D)+b, 约掉后逐位相同),
     这类增强会被预处理静默吃掉。depth 的增强交给模型内的 `ModalityAugment`(挖洞 +
     整路 dropout), 同理 IR 的噪声/模糊/对比度也在那里, 不要在这里重复实现。
"""
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GroundingDINO")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from groundingdino.datasets.transforms import RandomResize  # noqa: E402
from groundingdino.util.box_ops import box_xyxy_to_cxcywh  # noqa: E402
from groundingdino.util.misc import nested_tensor_from_tensor_list  # noqa: E402

MODALITIES = ("visible", "infrared", "depth")

# visible 与 run_grounding.py / demo 的评估口径逐字一致
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# IR 是**单通道**热图, 经 IREncoder 的 Conv2d(1,3,1) stem 复制成 3 个**相同**的通道。
# 所以这里用标量统计(ImageNet 三通道均值/标准差的平均), 而不是 per-channel:
# 三个不同的 mean/std 作用在三个相同的值上会得到三个不同的值, 与 stem 的语义自相矛盾;
# 而 warm-start 来的 patch_embed 期望的是 ImageNet 量级的输入, 标量统计正是单通道的对应物。
IR_MEAN = float(np.mean(IMAGENET_MEAN))  # 0.449
IR_STD = float(np.mean(IMAGENET_STD))    # 0.226


def normalize_caption(query: str) -> str:
    """指代句的规范化:小写 + 去首尾空白 + 补结尾句号。

    **单一事实来源** —— dataset 与推理脚本(run_grounding_mm.py)都走这里。
    尾句号不能省:它决定 tokenizer 的 [SEP] 落在哪一位, 而打分取的正是
    `arange(1, n_tok-1)` 这段真实词 token 上的最大值, 少一个句号就整体错位一格。
    """
    s = query.lower().strip()
    return s if s.endswith(".") else s + "."


# ================================================================ 划分

def load_items(data_dir: str):
    """读 queries.json, 返回按 (img_id, qid) 排序的原始条目列表。"""
    qfile = os.path.join(data_dir, "queries", "queries.json")
    if not os.path.exists(qfile):
        raise FileNotFoundError(f"找不到 {qfile}")
    queries = json.load(open(qfile, encoding="utf-8"))

    items = []
    for qid, v in queries.items():
        rel = v["visible"]  # e.g. "Images/visible/001.png"
        items.append(
            {
                "qid": qid,
                "query": v["query"],
                "img_id": os.path.splitext(os.path.basename(rel))[0],
                "paths": {m: os.path.join(data_dir, v[m]) for m in MODALITIES},
                "bbox": v.get("bbox"),
            }
        )
    items.sort(key=lambda it: (it["img_id"], it["qid"]))
    return items


def split_image_ids(items, val_ratio: float = 0.1, seed: int = 42):
    """按**图**划分 train/val, 同一张图的 5 条 query 不跨集。

    返回 (train_ids, val_ids), 均为排序后的 list。划分在运行期确定、不落盘,
    但会写进 checkpoint, 保证续训和事后评估用的是同一套划分。
    """
    img_ids = sorted({it["img_id"] for it in items})
    rng = random.Random(seed)
    rng.shuffle(img_ids)
    n_val = int(round(len(img_ids) * val_ratio))
    if val_ratio > 0:
        n_val = max(1, min(n_val, len(img_ids) - 1))
    return sorted(img_ids[n_val:]), sorted(img_ids[:n_val])


# ================================================================ Dataset

class MultiModalReferDataset(Dataset):
    """每个样本 = 一条 (图, 指代句, GT 框)。

    `train=True` 时做同步几何增强(resize + 水平翻转), 三个模态共用同一套随机决策;
    `train=False` 时只做确定性的 resize(与评估同口径)。
    """

    def __init__(
        self,
        data_dir: str = "../TrainSet.",
        image_ids=None,
        train: bool = True,
        resize: int = 800,
        max_size: int = 1333,
        hflip_p: float = 0.5,
        items=None,
    ):
        self.data_dir = data_dir
        self.train = train
        self.hflip_p = hflip_p if train else 0.0

        items = items if items is not None else load_items(data_dir)
        if image_ids is not None:
            keep = set(image_ids)
            items = [it for it in items if it["img_id"] in keep]
        # 没有 GT 的条目不能用来训练
        self.items = [it for it in items if it["bbox"] is not None]

        # resize 的尺寸数学必须与评估侧同源。直接用上游的 RandomResize:
        # sizes 只放一个元素 -> 每次都是同一尺寸, 随机选择实际是确定的。
        # 注意它返回的是 **PIL 图**, 所以后面必须跟一个 ToTensor(评估侧的
        # `T.Compose([RandomResize, ToTensor, Normalize])` 就是这个顺序)。
        self.geo = RandomResize([resize], max_size=max_size)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.items)

    def _load(self, path: str, gray: bool = False) -> Image.Image:
        if not os.path.exists(path):
            raise FileNotFoundError(f"图片不存在: {path}")
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        it = self.items[idx]

        img = self._load(it["paths"]["visible"])
        ir = self._load(it["paths"]["infrared"])

        # depth 必须走 cv2 的 IMREAD_UNCHANGED 才能拿到 uint16 原始值(PIL 会截成 8 位);
        # 保持原始数值不做任何归一化 —— 0 是 invalid 哨兵, 归一化是 DepthPreprocessor 的职责。
        depth = cv2.imread(it["paths"]["depth"], cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"图片不存在或无法解码: {it['paths']['depth']}")
        if depth.ndim == 3:  # 某些 PNG 会解成 (H,W,1)
            depth = depth[..., 0]
        depth = depth.astype(np.float32)

        # ---- resize: 三张图用同一个目标尺寸 ----
        img_t = self.to_tensor(self.geo(img, None)[0])   # [3,h,w] float32 in [0,1]
        ir_t = self.to_tensor(self.geo(ir, None)[0])     # 同尺寸(同源图尺寸, 决策确定)
        # 通道均值 —— 不是 PIL convert("L") 的亮度公式(0.299R+0.587G+0.114B),
        # 与 IREncoder 内部对 3 通道输入的处理语义一致
        ir_t = ir_t.mean(dim=0, keepdim=True)

        h, w = img_t.shape[-2:]
        # 最近邻: 见文件头的硬规则 1
        depth_t = torch.from_numpy(
            cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        ).unsqueeze(0)  # [1,h,w]

        # ---- 同步几何增强: 三模态共用同一枚硬币 ----
        # 翻转放在 **resize 之后**, 三个模态都在同一张输出网格上做镜像。
        # 若放在 resize 之前, depth 走的是 cv2 的最近邻(取样点是 floor(dst*scale)),
        # 它的取样网格在水平镜像下并不对称 —— 「先翻后缩」与「先缩后翻」会差一个像素,
        # 于是 depth 与 RGB 的像素栅格对不齐(实测偏差正好是 19999, 即最大值被错位取到)。
        # 「先缩后翻」让镜像在张量层面是精确的, 三个模态严格同源。
        if random.random() < self.hflip_p:
            img_t = torch.flip(img_t, dims=[-1])
            ir_t = torch.flip(ir_t, dims=[-1])
            depth_t = torch.flip(depth_t, dims=[-1])
            flipped = True
        else:
            flipped = False

        img_t = (img_t - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(
            IMAGENET_STD
        ).view(3, 1, 1)
        ir_t = (ir_t - IR_MEAN) / IR_STD

        # ---- GT: 归一化 xyxy -> cxcywh (归一化框对等比缩放不变, 无需改动) ----
        boxes = torch.tensor([it["bbox"]], dtype=torch.float32)  # [1,4] xyxy
        boxes = box_xyxy_to_cxcywh(boxes)  # [1,4] cxcywh
        if flipped:
            # 水平镜像: cx' = 1 - cx; w/h 不变
            boxes[:, 0] = 1.0 - boxes[:, 0]

        caption = normalize_caption(it["query"])

        return {
            "image": img_t,      # [3,h,w]  已归一化
            "ir": ir_t,          # [1,h,w]  已归一化
            "depth": depth_t,    # [1,h,w]  原始 uint16 数值(未归一化)
            "caption": caption,
            "boxes": boxes,      # [1,4] 归一化 cxcywh
            "img_id": it["img_id"],
            "qid": it["qid"],
        }


# ================================================================ collate

def collate_fn(batch):
    """把不同尺寸的样本补齐到同一 batch 形状。

    RGB 走 `nested_tensor_from_tensor_list` 得到 NestedTensor + padding mask;
    IR / depth 手工 pad 到**同一个** (Hmax, Wmax), 这一点很关键:

      - IR 补 0 -> 会被 `_aux_modality` 返回的 `samples.mask`(即 RGB 的 padding mask)遮掉,
        Swin 的窗口注意力看不到 padding。这里依赖 `_aux_modality` 的既有契约:
        **裸 Tensor 输入时它返回 samples.mask**, 所以只要 aux 的 padding 与 RGB 完全对齐就是对的。
      - depth 补 0 -> `DepthPreprocessor` 的 `valid = depth > depth_min` 自动判为 invalid,
        其逐样本百分位也只在非 padding 像素上统计。零额外处理, 天然正确。

    返回 `(samples, targets, ir, depth)`。
    """
    images = [b["image"] for b in batch]
    samples = nested_tensor_from_tensor_list(images)
    h, w = samples.tensors.shape[-2:]

    def pad_to(x):
        _, hx, wx = x.shape
        if hx == h and wx == w:
            return x
        return torch.nn.functional.pad(x, (0, w - wx, 0, h - hx))

    ir = torch.stack([pad_to(b["ir"]) for b in batch])
    depth = torch.stack([pad_to(b["depth"]) for b in batch])
    targets = [{"caption": b["caption"], "boxes": b["boxes"]} for b in batch]
    return samples, targets, ir, depth
