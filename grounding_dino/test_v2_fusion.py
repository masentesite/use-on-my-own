#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GroundingDINO 第二版融合 (局部 Cross-Attention + 空间矩阵 Gate) 验证测试。

覆盖 V2 技术方案 §10「必须通过的测试」的全部 7 项, 另加 4 项针对 §12
「风险与排查顺序」的常驻守卫 —— 那 4 项正是 §12 表格里逐行对应的判据:

    §10-1  shape          F_new,l 与 F_rgb,l 完全同形状, 通道 256
    §10-2  gate_init      三模态 0.79/0.11/0.11; 单辅助 0.88/0.12
    §10-3  missing        只传 RGB / RGB+IR / RGB+Depth / 三模态都能 forward, 无 NaN
    §10-4  mask           ir_valid=0 ⇒ W_ir ≡ 0; depth 同理(逐样本、逐像素都测)
    §10-5  gradient       LocalCrossAttention2d / SpatialMixtureGate / IR / Depth
                          Encoder 有梯度; 冻结的 RGB Swin / BERT 无更新
    §10-6  memory         H/8 level 不出现全局 H*W x H*W 大矩阵
    §10-7  regression     不传辅助模态时输出逐位等于原 GroundingDINO
    §12    window_exact   局部窗口 attention 与逐像素暴力实现逐位等价
    §12    no_zero_init   gate bias / out_proj 都不是零初始化(§5.3 §5.4)
    §9     stats_keys     要求的 5 个指标(逐 level + 跨 level 平均)都在
    §12    stats_mask     gate 统计遵守 mask 口径, 不被 modality dropout 拉偏

用法(在仓库根目录执行):
    .venv/bin/python test_v2_fusion.py --device cpu
    .venv/bin/python test_v2_fusion.py --device cpu --weights weights/groundingdino_swint_ogc.pth
    .venv/bin/python test_v2_fusion.py --device cuda --only gradient

第一版 LanguageGuidedFusion 的套件是 test_multimodal.py(beta 语义贯穿全部断言),
它同时充当 V2 §11 消融矩阵里 V2-E4(residual fusion 对照)的回归测试。

⚠️ 两个已知的、不是 bug 的现象(下面的断言都按此处理):
  1. pred_logits 里的 -inf 是设计行为: ContrastiveEmbed 把 padding token 槽位掩成
     -inf, 使 sigmoid 后为 0。所以判定「无 NaN」要用 torch.isnan(...).sum() == 0,
     不能用 torch.isfinite(...).all()。
  2. 因此两个 pred_logits 相减会在 -inf 位置产生 nan, 相等判断必须用 torch.equal。
"""

import argparse
import math
import os
import resource
import sys
import traceback

# ---- 代理环境变量必须在 import torch / transformers 之前设好(与 run_grounding.py 一致) ----
os.environ.setdefault("ALL_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("all_proxy", "http://127.0.0.1:7892")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for _v in ("no_proxy", "NO_PROXY"):
    if "hf-mirror.com" not in os.environ.get(_v, ""):
        os.environ[_v] = "hf-mirror.com," + os.environ.get(_v, "")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
GDINO = os.path.join(REPO, "GroundingDINO")
sys.path.insert(0, GDINO)

from demo.inference_on_a_image import load_image  # noqa: E402
from groundingdino.models import build_model  # noqa: E402
from groundingdino.models.GroundingDINO.multimodal_modules import (  # noqa: E402
    CrossAttentionGatedFusion,
    LocalCrossAttention2d,
    SpatialMixtureGate,
    masked_mean_pool,
)
from groundingdino.util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402

RGB_CFG = os.path.join(GDINO, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
MM_CFG = os.path.join(GDINO, "groundingdino/config/GroundingDINO_MultiModal_SwinT.py")
SAMPLE_IMAGE = os.path.join(GDINO, ".asset/cat_dog.jpeg")

CAPTION = "cat . dog ."
V2_FUSION = "local_cross_attention_spatial_gate"

# 多模态专属的模块名前缀 —— 用于把「RGB 子树」从多模态模型里筛出来
AUX_PREFIXES = (
    "ir_encoder.",
    "depth_encoder.",
    "ir_proj.",
    "depth_proj.",
    "fusion.",
)

# 单元测试用的合成特征. 故意让三个 level 都是奇数尺寸:
#   - LocalCrossAttention2d 的 pad+切片在奇数 H/W 上最容易越界
#   - _valid_map 的 nearest 下采样在非整除比例下最容易错位
UNIT_LEVELS = ((8, 9), (4, 5), (2, 3))
UNIT_B, UNIT_C = 2, 256
UNIT_HEADS, UNIT_WINDOW = 8, 5


# ======================================================================
#  基础设施
# ======================================================================
class Failure(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Failure(msg)


def check_close(actual, expect, tol, msg):
    check(abs(actual - expect) <= tol,
          f"{msg}: 期望 {expect}±{tol}, 实际 {actual}")


def fmt(x):
    return f"{x:.4g}"


def build(cfg_path, weights=None, device="cpu", fusion_type=None):
    """按 config 建模型; weights 给了就加载 checkpoint(走 clean_state_dict + strict=False)。"""
    args = SLConfig.fromfile(cfg_path)
    args.device = device
    if fusion_type is not None and getattr(args, "use_multimodal", False):
        args.fusion_type = fusion_type
    model = build_model(args)
    info = {}
    if weights:
        try:
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        res = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        info["missing"] = list(res.missing_keys)
        info["unexpected"] = list(res.unexpected_keys)
    model.eval().to(device)
    return model, info


def off_checkpointing(model):
    """关掉 gradient checkpointing: 测试里不需要省显存, 开着只会拖慢并产生 reentrant 警告。"""
    enc = getattr(model.transformer, "encoder", None)
    if enc is not None:
        enc.use_checkpoint = False
        enc.use_transformer_ckpt = False


def load_small_image(path, device, long_side):
    """读样例图并缩到 long_side —— 第二版要在真实模型上跑反传, 原图 1333x750 CPU 太慢。"""
    _, image = load_image(path)
    if long_side and max(image.shape[-2:]) > long_side:
        scale = long_side / max(image.shape[-2:])
        new = (max(32, int(round(image.shape[-2] * scale)) // 32 * 32),
               max(32, int(round(image.shape[-1] * scale)) // 32 * 32))
        image = F.interpolate(image[None], size=new, mode="bilinear",
                              align_corners=False)[0]
    return image.to(device)


def make_aux_inputs(image_tensor, device, holes=True):
    """造出与 RGB 同尺寸的 IR / Depth 输入(测试用, 内容是合成的)。"""
    _, h, w = image_tensor.shape
    g = torch.Generator().manual_seed(0)
    ir = (torch.rand(1, 1, h, w, generator=g) * 0.5 + 0.2).to(device)
    depth = (torch.rand(1, 1, h, w, generator=g) * 18000 + 1500).to(device)
    if holes:
        depth[:, :, : h // 8, : w // 8] = 0.0  # 一块 invalid
    return ir, depth


def loss_from(out):
    """从 forward 输出构造一个可反传的标量损失。

    pred_logits 在 padding token 槽位上是 -inf(设计使然), 必须先过滤掉,
    否则 sum() 会污染成 nan。
    """
    boxes = out["pred_boxes"]
    logits = out["pred_logits"]
    finite = torch.isfinite(logits)
    return boxes.sum() + logits[finite].sum() * 1e-3


def any_grad_nonzero(module):
    """该模块是否存在至少一个非零梯度。"""
    return any(p.grad is not None and bool(p.grad.abs().max() > 0)
               for p in module.parameters())


def grad_summary(module):
    n_none = sum(1 for p in module.parameters() if p.grad is None)
    n_zero = sum(1 for p in module.parameters()
                 if p.grad is not None and not bool(p.grad.abs().max() > 0))
    return f"{n_none} 个无梯度, {n_zero} 个零梯度"


# ======================================================================
#  单元测试用的合成数据与辅助函数
# ======================================================================
def make_fusion(device, seed=0, **kw):
    """建一个独立的第二版 Fusion 模块(不经过完整模型), 单元测试全靠它保证速度与确定性。"""
    torch.manual_seed(seed)
    opts = dict(dim=UNIT_C, num_levels=len(UNIT_LEVELS), text_dim=UNIT_C,
                num_heads=UNIT_HEADS, window_size=UNIT_WINDOW)
    opts.update(kw)
    return CrossAttentionGatedFusion(**opts).to(device).eval()


def make_text_dict(device, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {
        "encoded_text": torch.randn(UNIT_B, 7, UNIT_C, generator=g, device=device),
        "text_token_mask": torch.ones(UNIT_B, 7, dtype=torch.bool, device=device),
    }


def make_feats(device, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(UNIT_B, UNIT_C, h, w, generator=g, device=device) * scale
            for (h, w) in UNIT_LEVELS]


def text_up_of(fusion, text_dict):
    t = masked_mean_pool(text_dict["encoded_text"], text_dict["text_token_mask"])
    return fusion.gate.text_proj(t)


def gate_inputs(fusion, rgb, ir, dep, lvl):
    """复现 forward 内部对 gate 的输入, 以便直接检查 B x 3 x H x W 的权重矩阵。"""
    a_ir = (fusion._candidate(rgb[lvl], ir[lvl], fusion.ir_attn[lvl],
                              fusion.ir_adapters[lvl], fusion.ir_norms[lvl])
            if ir is not None else None)
    a_dep = (fusion._candidate(rgb[lvl], dep[lvl], fusion.depth_attn[lvl],
                               fusion.depth_adapters[lvl], fusion.depth_norms[lvl])
             if dep is not None else None)
    return a_ir, a_dep


def gate_W(fusion, lvl, rgb, ir, dep, text_dict, ir_ok=None, depth_ok=None):
    a_ir, a_dep = gate_inputs(fusion, rgb, ir, dep, lvl)
    with torch.no_grad():
        return fusion.gate(lvl, rgb[lvl], a_ir, a_dep, text_up_of(fusion, text_dict),
                           ir_ok, depth_ok)


def run_fusion(fusion, rgb, ir, dep, text_dict, **kw):
    """跑一遍 forward 并返回 (输出, stats, gate 权重列表)。"""
    with torch.no_grad():
        out = fusion(rgb, ir_srcs=ir, depth_srcs=dep, text_dict=text_dict, **kw)
    return out, fusion.last_stats


# ======================================================================
#  §10-1 Shape test
# ======================================================================
def test_shape(ctx):
    """F_new,l 与 F_rgb,l 完全同形状, 通道数为 256。

    额外覆盖「辅助模态分辨率与 RGB 不一致」这条路径: IR Swin 与 Depth 金字塔的输出
    尺寸是 ceil 取整算出来的, 在奇数输入上会和 RGB 的 input_proj 差一个像素。
    """
    dev = ctx["device"]
    f = make_fusion(dev)
    rgb, ir, dep = make_feats(dev, 10), make_feats(dev, 11), make_feats(dev, 12)
    td = make_text_dict(dev)

    out, stats = run_fusion(f, rgb, ir, dep, td)
    check(len(out) == len(UNIT_LEVELS), f"输出应有 {len(UNIT_LEVELS)} 个 level, 得到 {len(out)}")
    for lvl, (o, r) in enumerate(zip(out, rgb)):
        check(o.shape == r.shape,
              f"level{lvl} 形状不一致: F_new={tuple(o.shape)} vs F_rgb={tuple(r.shape)}")
        check(o.shape[1] == 256, f"level{lvl} 通道数为 {o.shape[1]}, 应为 256")

    # 辅助模态放大 2 倍 —— 必须被 bilinear 拉回 RGB 的尺寸
    up = [F.interpolate(x, scale_factor=2, mode="nearest") for x in ir]
    out2, _ = run_fusion(f, rgb, up, dep, td)
    for lvl, (o, r) in enumerate(zip(out2, rgb)):
        check(o.shape == r.shape,
              f"level{lvl} IR 尺寸为 RGB 的 2 倍时未对齐: "
              f"{tuple(o.shape)} vs {tuple(r.shape)}")

    # 逐像素 valid 的形状与 level 分辨率不一致时, 也要能对齐
    dv = torch.ones(UNIT_B, 1, UNIT_LEVELS[0][0] * 3, UNIT_LEVELS[0][1] * 3,
                    dtype=torch.bool, device=dev)
    out3, _ = run_fusion(f, rgb, ir, dep, td, depth_valid=dv)
    for lvl, (o, r) in enumerate(zip(out3, rgb)):
        check(o.shape == r.shape, f"level{lvl} 逐像素 valid 尺寸不匹配时未对齐")

    return (f"3 个 level 输出形状全部等于 RGB 输入 "
            f"{[tuple(r.shape) for r in rgb]}, 通道 256; "
            f"辅助模态 2x 放大 / 逐像素 valid 尺寸不匹配两种情形均正确对齐")


# ======================================================================
#  §10-2 Gate init test  +  §12 no_zero_init
# ======================================================================
def test_gate_init(ctx):
    """初始化后三模态 gate 均值接近 RGB 0.79 / IR 0.11 / Depth 0.11; 单辅助 0.88 / 0.12。

    方案 §5.3 的推导: softmax([2,0,0]) = [0.786, 0.107, 0.107],
    softmax([2,-1e4,0]) = [0.881, 0.000, 0.119]。

    ⚠️ 单辅助模态的 0.88/0.12 只有在**缺失通道被 mask 成 -1e4** 时才成立。
    如果实现改成「候选填零但不 mask」, softmax([2,0,0]) 的三个通道仍然是
    0.786/0.107/0.107 —— 辅助模态拿不到 0.12, 而 RGB 也从 0.88 掉回 0.79。
    这个测试因此同时是「整路缺失必须 mask」(§13 第 7 条)的回归守卫。
    """
    dev = ctx["device"]
    rgb, ir, dep = make_feats(dev, 20), make_feats(dev, 21), make_feats(dev, 22)
    td = make_text_dict(dev)
    tol = 0.02

    cases = [
        # (说明, ir, depth, 期望 rgb, 期望 ir, 期望 depth)
        ("三模态",          ir,  dep, 0.787, 0.107, 0.107),
        ("单辅助 RGB+IR",   ir,  None, 0.881, 0.119, 0.0),
        ("单辅助 RGB+Depth", None, dep, 0.881, 0.0, 0.119),
    ]
    lines = []
    for tag, i, d, e_rgb, e_ir, e_dep in cases:
        # 每个 case 都用一个全新的模块: gate 统计依赖初始化, 不能复用被前向污染过的
        f = make_fusion(dev)
        _, stats = run_fusion(f, rgb, i, d, td)
        got = (float(stats["gate_rgb_mean"]), float(stats["gate_ir_mean"]),
               float(stats["gate_depth_mean"]))
        check_close(got[0], e_rgb, tol, f"[{tag}] gate_rgb_mean")
        check_close(got[1], e_ir, tol, f"[{tag}] gate_ir_mean")
        check_close(got[2], e_dep, tol, f"[{tag}] gate_depth_mean")
        # 三模态时三路 mask 全开, 逐像素权重和为 1 ⇒ 三个均值之和必须精确为 1
        if i is not None and d is not None:
            check_close(sum(got), 1.0, 1e-5, f"[{tag}] 三路 gate 均值之和")
        lines.append(f"{tag}={got[0]:.3f}/{got[1]:.3f}/{got[2]:.3f}")

    # ---- §5.3「不要把辅助模态初始化成 0 贡献」的行为断言 ----
    # 喂一路**恒零**的 IR 特征: 候选 A_ir 会精确等于 0(cross-attn 输出 → out_proj
    # 无 bias → adapter 零残差 → GroupNorm(0)=0)。此时 gate 仍然必须给它非零权重,
    # 否则说明 gate 被零初始化了。
    f = make_fusion(dev)
    zeros_ir = [torch.zeros_like(r) for r in rgb]
    _, stats = run_fusion(f, rgb, zeros_ir, None, td)
    w_ir = float(stats["gate_ir_mean"])
    check(w_ir > 0.05,
          f"辅助候选恒零时 gate_ir_mean={w_ir:.4f}, 说明 gate 被零初始化了(§5.3 禁止)")
    w_ir_zero_init = w_ir

    # out_proj / gate conv 的权重不能是全零
    for name, mod in (("q_proj", f.ir_attn[0].q_proj), ("out_proj", f.ir_attn[0].out_proj)):
        check(float(mod.weight.abs().max()) > 0,
              f"LocalCrossAttention2d.{name}.weight 全零 —— 违反 §5.4「候选不要零初始化」")
    for lvl, c in enumerate(f.gate.conv):
        check(float(c.weight.abs().max()) > 0,
              f"gate.conv[{lvl}].weight 全零 —— 违反 §5.3「gate 不要初始化为全 0」")
    bias = f.gate.conv[0].bias.detach().tolist()
    check(bias == [2.0, 0.0, 0.0],
          f"gate bias 初值应为 [2.0, 0.0, 0.0](§5.3 的 fusion_gate_rgb_bias/aux_bias), 实际 {bias}")

    return ("; ".join(lines) + f"; 零候选 IR 仍拿到 gate_ir_mean={w_ir_zero_init:.3f}; "
            "gate bias=[2,0,0], gate/out_proj 权重非零")


# ======================================================================
#  §10-3 Missing modality test
# ======================================================================
def test_missing(ctx):
    """只传 RGB、RGB+IR、RGB+Depth、三模态都能 forward, 无 NaN。

    走的是**完整模型**(不是孤立模块), 因为「辅助模态缺失」的短路发生在
    groundingdino._fuse_multimodal 里, 只有整模型才能覆盖到。
    """
    dev = ctx["device"]
    mm = ctx["mm"]
    img = ctx["image"]
    ir, depth = make_aux_inputs(img, dev)
    iv = torch.ones(1, dtype=torch.bool, device=dev)
    dv = torch.ones(1, dtype=torch.bool, device=dev)
    # 与 make_aux_inputs 造的那块 depth 空洞对齐的逐像素 valid
    dv_hole = torch.ones(1, 1, *img.shape[-2:], dtype=torch.bool, device=dev)
    h, w = img.shape[-2:]
    dv_hole[:, :, : h // 8, : w // 8] = False

    combos = [
        ("RGB",            {}),
        ("RGB+IR",         dict(ir_samples=ir, ir_valid=iv)),
        ("RGB+Depth",      dict(depth_samples=depth, depth_valid=dv)),
        ("RGB+IR+Depth",   dict(ir_samples=ir, ir_valid=iv,
                                depth_samples=depth, depth_valid=dv)),
        ("IR 整路有效但 valid=False",
         dict(ir_samples=ir, ir_valid=torch.zeros(1, dtype=torch.bool, device=dev),
              depth_samples=depth, depth_valid=dv)),
        ("Depth 逐像素有洞",
         dict(ir_samples=ir, ir_valid=iv, depth_samples=depth, depth_valid=dv_hole)),
    ]

    ref_boxes = ref_logits = None
    lines = []
    with torch.no_grad():
        for tag, kw in combos:
            out = mm(img[None].to(dev), captions=[CAPTION], **kw)
            logits, boxes = out["pred_logits"], out["pred_boxes"]
            n_nan = int(torch.isnan(logits).sum() + torch.isnan(boxes).sum())
            check(n_nan == 0, f"[{tag}] 输出含 {n_nan} 个 NaN")
            n_inf_boxes = int(torch.isinf(boxes).sum())
            check(n_inf_boxes == 0, f"[{tag}] pred_boxes 含 {n_inf_boxes} 个 inf")
            if ref_boxes is None:
                ref_boxes, ref_logits = boxes, logits
            check(boxes.shape == ref_boxes.shape and logits.shape == ref_logits.shape,
                  f"[{tag}] 输出形状与纯 RGB 不一致: {tuple(boxes.shape)}")
            stats = mm._fusion_stats
            if tag == "RGB":
                check(stats is None,
                      f"[{tag}] 纯 RGB 路径不该产生 gate 统计, 得到 "
                      f"{'None' if stats is None else sorted(stats)}")
                check(torch.equal(boxes, ref_boxes), "纯 RGB 自比不一致")
            else:
                check(stats is not None, f"[{tag}] 没有产生 gate 统计")
            lines.append(f"{tag}: nan=0 shape={tuple(boxes.shape)}")

    return "; ".join(lines)


# ======================================================================
#  §10-4 Mask test
# ======================================================================
def test_mask(ctx):
    """ir_valid=0 时 W_ir 接近 0; depth_valid=0 时 W_depth 接近 0。

    方案 §13 第 7 条要求的是「缺失模态把 gate logit 置 -1e4」, **不是**「把输入置零」。
    两者可以直接分辨: 置零输入时 channel 1 的 logit 只剩 conv bias(0.0), 仍会和
    RGB 通道的 logit 竞争出一个 ~0.03 的权重; 置 -1e4 则 softmax 严格下溢, 权重
    **精确等于 0.0**。所以下面断言的是 `== 0.0` 而不是 `< 1e-3`。
    """
    dev = ctx["device"]
    f = make_fusion(dev)
    rgb, ir, dep = make_feats(dev, 30), make_feats(dev, 31), make_feats(dev, 32)
    td = make_text_dict(dev)
    lvl = 0
    b, _, h, w = rgb[lvl].shape

    a_ir, a_dep = gate_inputs(f, rgb, ir, dep, lvl)
    check(float(a_ir.abs().max()) > 0,
          "IR 候选特征恒零 —— 违反 §5.4「候选不要零初始化」")

    # ---- ① 不给 valid: 三路都得有非零权重(基线) ----
    W0 = gate_W(f, lvl, rgb, ir, dep, td)
    check(W0.shape == (b, 3, h, w),
          f"gate 权重形状应为 B x 3 x H x W = {(b, 3, h, w)}, 得到 {tuple(W0.shape)}")
    check_close(float((W0.sum(dim=1) - 1).abs().max()), 0.0, 1e-5,
                "gate 权重未沿 dim=1 归一化")
    check(float(W0[:, 1].min()) > 0 and float(W0[:, 2].min()) > 0,
          "未提供 valid 时 IR / Depth 不该被 mask")

    # ---- ② IR 整批 valid=False ----
    W = gate_W(f, lvl, rgb, ir, dep, td, ir_ok=torch.zeros(b, dtype=torch.bool, device=dev))
    check(float(W[:, 1].abs().max()) == 0.0,
          f"ir_valid=0 时 W_ir 应为精确 0.0(§13 第 7 条: logit 置 -1e4 而非输入置零), "
          f"实际 max={float(W[:, 1].abs().max()):.3e}")
    check(torch.equal(W[:, 1], torch.zeros_like(W[:, 1])),
          "ir_valid=0 时 W_ir 必须逐元素精确为 0")
    check_close(float((W.sum(dim=1) - 1).abs().max()), 0.0, 1e-5,
                "mask 掉 IR 后权重未重新归一化")
    # mask 掉一路之后, 它的权重应该重新分配给 RGB 而不是凭空消失
    check(float(W[:, 0].mean()) > float(W0[:, 0].mean()),
          "mask 掉 IR 之后 W_rgb 没有变大 —— 权重被丢掉了而不是重新归一化")

    # ---- ③ Depth 整批 valid=False ----
    W_d = gate_W(f, lvl, rgb, ir, dep, td,
                 depth_ok=torch.zeros(b, dtype=torch.bool, device=dev))
    check(float(W_d[:, 2].abs().max()) == 0.0,
          f"depth_valid=0 时 W_depth 应为精确 0.0, 实际 {float(W_d[:, 2].abs().max()):.3e}")
    check_close(float((W_d.sum(dim=1) - 1).abs().max()), 0.0, 1e-5,
                "mask 掉 Depth 后权重未重新归一化")

    # ---- ③' Depth 整路不传: 走完整 Fusion, 用 last_stats 观测 ----
    # 不能直接调 fusion.gate 测这一条: 那个调用要自己传 depth_ok, 一旦忘了 mask
    # 就测不出问题(这正是原先实现里的真实 bug —— 「整路缺失」这条路径没被 mask)。
    # 现在 mask 由 gate 内部依据「候选 is None」决定, 所以从对外可观测的 last_stats
    # 断言才是有效回归: 缺失若不 mask, gate_depth_mean 会是 ~0.107 而不是 0。
    _, stats_all = run_fusion(f, rgb, ir, dep, td)
    _, stats_absent = run_fusion(f, rgb, ir, None, td)
    w_gone = float(stats_absent["gate_depth_mean"])
    check(w_gone == 0.0,
          f"depth 整路不传时 gate_depth_mean={w_gone:.3e}, 应为精确 0.0 —— "
          f"缺失却没有 mask, gate 会给零候选留下约 "
          f"{float(stats_all['gate_depth_mean']):.3f} 的权重(§5.4)")
    check(float(stats_absent["gate_ir_mean"]) > 0.05,
          "depth 缺失时 IR 的权重不该跟着消失")
    check(float(stats_absent["gate_rgb_mean"]) > float(stats_all["gate_rgb_mean"]),
          "depth 缺失后 W_rgb 没有变大 —— 权重被丢掉了而不是重新归一化")

    # ---- ④ 逐像素 mask: 左半 depth 无效 ----
    dv = torch.ones(b, 1, h, w, dtype=torch.bool, device=dev)
    dv[:, :, :, : w // 2] = False
    Wp = gate_W(f, lvl, rgb, ir, dep, td, depth_ok=dv)
    left, right = Wp[:, 2, :, : w // 2], Wp[:, 2, :, w // 2:]
    check(float(left.abs().max()) == 0.0,
          f"逐像素 depth_valid=False 的区域 W_depth 应为 0, 实际 {float(left.abs().max()):.3e}")
    check(float(right.min()) > 0,
          "逐像素 depth_valid=True 的区域 W_depth 应大于 0")
    check(float(Wp[:, 0, :, : w // 2].mean()) > float(Wp[:, 0, :, w // 2:].mean()),
          "左半 mask 掉 depth 后 W_rgb 没有相应变大")

    # ---- ⑤ 逐样本 mask: 只有第 1 个样本的 IR 有效 ----
    iv = torch.tensor([False, True], device=dev)
    Ws = gate_W(f, lvl, rgb, ir, dep, td, ir_ok=iv)
    check(float(Ws[0, 1].abs().max()) == 0.0, "样本 0 的 IR 无效, W_ir 应为 0")
    check(float(Ws[1, 1].min()) > 0, "样本 1 的 IR 有效, W_ir 应大于 0")

    return ("三路权重形状 Bx3xHxW 且逐像素和为 1; 整批 / 逐样本 / 逐像素三种粒度的 "
            "ir_valid=0 与 depth_valid=0 均使对应权重**精确**为 0; "
            "「整路不传」与「valid 全 False」的 mask 语义一致; "
            "mask 后权重重新分配给 RGB")


# ======================================================================
#  §10-5 Gradient test
# ======================================================================
def test_gradient(ctx):
    """LocalCrossAttention2d、SpatialMixtureGate、IR/Depth Encoder 有梯度; 冻结的
    RGB Swin / BERT 无更新。

    分两段:
      A) 单元: 独立 Fusion 模块前向 + 反传, 逐个参数组检查梯度。
      B) 集成: 完整模型 Stage 1 前向 + 反传, 检查辅助分支有梯度、RGB Swin / BERT 冻结。
    """
    dev = ctx["device"]

    # ---------- A) 单元 ----------
    f = make_fusion(dev).train()
    rgb, ir, dep = make_feats(dev, 40), make_feats(dev, 41), make_feats(dev, 42)
    td = make_text_dict(dev)
    out = f(rgb, ir_srcs=ir, depth_srcs=dep, text_dict=td)
    loss = sum(o.float().pow(2).sum() for o in out)
    loss.backward()

    groups = {
        "ir_attn(LocalCrossAttention2d)": f.ir_attn,
        "depth_attn(LocalCrossAttention2d)": f.depth_attn,
        "gate(SpatialMixtureGate)": f.gate,
        "ir_adapters(BottleneckAdapter2d)": f.ir_adapters,
        "depth_adapters(BottleneckAdapter2d)": f.depth_adapters,
        "ir_norms(candidate_norm)": f.ir_norms,
        "depth_norms(candidate_norm)": f.depth_norms,
    }
    unit_lines = []
    for name, mod in groups.items():
        check(any_grad_nonzero(mod),
              f"[单元] {name} 完全没有非零梯度({grad_summary(mod)})")
        unit_lines.append(f"{name}: ok")
    # 所有参数都必须参与计算图(grad 不为 None) —— 包括 adapter 的 down 层
    # (它的梯度因 up 层零初始化而恰好为 0, 但必须有 grad 张量, 否则说明没接上)
    for name, mod in groups.items():
        n_none = sum(1 for p in mod.parameters() if p.grad is None)
        check(n_none == 0, f"[单元] {name} 有 {n_none} 个参数不在计算图里")

    # ---------- B) 集成 ----------
    mm = ctx["mm"]
    off_checkpointing(mm)
    stage_info = mm.set_train_stage(1)
    mm.zero_grad(set_to_none=True)
    ir, depth = make_aux_inputs(ctx["image"], dev)
    iv = torch.ones(1, dtype=torch.bool, device=dev)
    out = mm(ctx["image"][None].to(dev), captions=[CAPTION], ir_samples=ir,
             depth_samples=depth, ir_valid=iv, depth_valid=iv)
    loss_from(out).backward()

    aux_groups = {
        "ir_encoder": mm.ir_encoder,
        "depth_encoder": mm.depth_encoder,
        "ir_proj": mm.ir_proj,
        "depth_proj": mm.depth_proj,
        "fusion": mm.fusion,
        "enc_adapter": getattr(mm.transformer.encoder, "vision_adapters", None),
        "dec_adapter": getattr(mm.transformer.decoder, "decoder_adapters", None),
    }
    int_lines = []
    for name, mod in aux_groups.items():
        if mod is None:
            continue
        params = list(mod.parameters())
        check(params, f"[集成] {name} 没有参数, 检查挂载位置")
        check(all(p.requires_grad for p in params),
              f"[集成] {name} 在 Stage 1 应全部可训练, 但有参数 requires_grad=False")
        check(any_grad_nonzero(mod),
              f"[集成] {name} 没有非零梯度({grad_summary(mod)})")
        int_lines.append(f"{name}: ok")

    frozen = {"rgb_swin": mm.backbone[0], "bert": mm.bert, "feat_map": mm.feat_map}
    froz_lines = []
    for name, mod in frozen.items():
        params = list(mod.parameters())
        n_trainable = sum(1 for p in params if p.requires_grad)
        check(n_trainable == 0,
              f"[集成] Stage 1 的 {name} 应全部冻结, 但有 {n_trainable} 个参数可训练")
        n_grad = sum(1 for p in params if p.grad is not None)
        check(n_grad == 0,
              f"[集成] 冻结的 {name} 有 {n_grad} 个参数拿到了梯度, 说明冻结只挡了优化器没挡反传")
        froz_lines.append(f"{name}: {len(params)} 参数全冻结、无梯度")

    stage_note = (f"Stage1 可训练张量 "
                  f"{stage_info['num_trainable_tensors']}/{stage_info['num_total_tensors']}"
                  if isinstance(stage_info, dict) else str(stage_info))
    mm.zero_grad(set_to_none=True)
    return (f"[单元] {len(unit_lines)} 个参数组全部有非零梯度; "
            f"[集成 Stage1] {len(int_lines)} 个辅助分支可训练且有梯度; "
            f"冻结项 {', '.join(froz_lines)}; {stage_note}")


# ======================================================================
#  §10-6 Memory test
# ======================================================================
def test_memory(ctx):
    """H/8 level 不出现全局 attention 的 H*W x H*W 大矩阵, 显存应接近第一版可控范围。

    判据(方案 §12 第一行「显存突然暴涨」的两条排查线索):
      1. 不许出现 H*W x H*W 的 softmax 输入 —— 局部窗口的 attention 矩阵必须是
         B x heads x H x W x k*k。
      2. 不许调用 F.unfold —— 即使窗口是局部的, unfold 的中间张量也是
         B x (C*k*k) x (H*W), 在 H/8 level 上同样会炸。
    另外把实际峰值与「全局方案需要多少」并排打印出来, 便于人工核对。
    """
    dev = ctx["device"]
    f = make_fusion(dev)

    # 真实分辨率: 750 x 1333 的 H/8 level
    H8, W8 = 94, 167
    rgb = [torch.randn(1, UNIT_C, H8, W8, device=dev),
           torch.randn(1, UNIT_C, 47, 84, device=dev),
           torch.randn(1, UNIT_C, 24, 42, device=dev)]
    ir = [torch.randn_like(x) for x in rgb]
    dep = [torch.randn_like(x) for x in rgb]
    td = {"encoded_text": torch.randn(1, 7, UNIT_C, device=dev),
          "text_token_mask": torch.ones(1, 7, dtype=torch.bool, device=dev)}

    n_token = H8 * W8
    global_bytes = n_token * n_token * UNIT_HEADS * 4  # 单 level 单模态的 fp32 全局矩阵

    sm_shapes, unfold_calls = [], []
    real_softmax, real_unfold = torch.softmax, F.unfold

    def spy_softmax(inp, dim=None, *a, **k):
        sm_shapes.append(tuple(inp.shape))
        return real_softmax(inp, dim, *a, **k)

    def spy_unfold(inp, *a, **k):
        res = real_unfold(inp, *a, **k)
        unfold_calls.append((tuple(inp.shape), tuple(res.shape)))
        return res

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    torch.softmax, F.unfold = spy_softmax, spy_unfold
    try:
        with torch.no_grad():
            out = f(rgb, ir_srcs=ir, depth_srcs=dep, text_dict=td)
    finally:
        torch.softmax, F.unfold = real_softmax, real_unfold
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20

    check(len(sm_shapes) >= 1, "没有观测到任何 softmax 调用, 探针没挂上")
    check(not unfold_calls,
          f"attention 路径调用了 F.unfold(方案 §12 明令排查): {unfold_calls[:2]}")

    kk = UNIT_WINDOW * UNIT_WINDOW
    for s in sm_shapes:
        check(not (len(s) >= 2 and s[-1] == n_token and s[-2] == n_token),
              f"出现了全局 {n_token}x{n_token} 的 attention 矩阵: shape={s}(§12 第一行)")
    budget = UNIT_HEADS * n_token * kk
    worst = max(sm_shapes, key=lambda s: math.prod(s) if s else 0)
    worst_numel = math.prod(worst)
    check(worst_numel <= budget,
          f"最大 attention 矩阵 {worst_numel} 元素超过窗口预算 {budget}(heads*H*W*k^2)")

    out_elems = sum(o.numel() for o in out)
    return (f"H/8 level {H8}x{W8}={n_token} token: 最大 attention 矩阵 {worst} = "
            f"{worst_numel * 4 / 2**20:.2f} MiB(fold 预算 {kk} 倍窗口), "
            f"全局方案需 {global_bytes / 2**30:.2f} GiB/level/模态 —— 相差 "
            f"{global_bytes / (worst_numel * 4):.0f} 倍; 未调用 F.unfold; "
            f"输出 {out_elems * 4 / 2**20:.2f} MiB; "
            f"进程峰值 RSS {rss_before:.2f} -> {rss_after:.2f} GiB")


# ======================================================================
#  §10-7 Regression test
# ======================================================================
def test_regression(ctx):
    """关闭辅助模态后输出接口与原 GroundingDINO 一致。

    第二版没有 beta 可以「拨到 0」, 所以这条约束靠 _fuse_multimodal 的早退实现:
    两个辅助模态都没传时直接 `return srcs`。这里同时验证
      (a) 与一个真正由 RGB config 构建的模型输出**逐位**相等;
      (b) fusion / ir_encoder / depth_encoder 一次都没被调用(没有白算);
      (c) _fusion_stats 为 None。
    """
    dev = ctx["device"]
    img = ctx["image"]
    mm = ctx["mm"]

    if ctx["weights"]:
        rgb_model, info = build(RGB_CFG, ctx["weights"], dev)
        mode = "checkpoint"
    else:
        rgb_model, info = build(RGB_CFG, None, dev)
        sd = {k: v for k, v in mm.state_dict().items()
              if not k.startswith(AUX_PREFIXES) and "adapters" not in k}
        res = rgb_model.load_state_dict(sd, strict=False)
        check(not res.unexpected_keys,
              f"拷贝 RGB 子树时有意外键: {list(res.unexpected_keys)[:5]}")
        check(not res.missing_keys,
              f"RGB 模型缺少权重: {list(res.missing_keys)[:5]}")
        mode = "state_dict 拷贝"
    off_checkpointing(rgb_model)
    off_checkpointing(mm)

    called = {}
    handles = []
    for name, mod in (("fusion", mm.fusion), ("ir_encoder", mm.ir_encoder),
                      ("depth_encoder", mm.depth_encoder), ("ir_proj", mm.ir_proj),
                      ("depth_proj", mm.depth_proj)):
        if mod is None:
            continue
        called[name] = 0

        def hook(_m, _i, _o, _n=name):
            called[_n] += 1

        handles.append(mod.register_forward_hook(hook))

    try:
        with torch.no_grad():
            ref = rgb_model(img[None].to(dev), captions=[CAPTION])
            o_none = mm(img[None].to(dev), captions=[CAPTION])
    finally:
        for h in handles:
            h.remove()

    check(torch.equal(ref["pred_boxes"], o_none["pred_boxes"]),
          f"[{mode}] 不传辅助模态时 pred_boxes 不一致, "
          f"maxdiff={(ref['pred_boxes'] - o_none['pred_boxes']).abs().max().item()}")
    check(torch.equal(ref["pred_logits"], o_none["pred_logits"]),
          f"[{mode}] 不传辅助模态时 pred_logits 不一致")
    check(set(o_none.keys()) == set(ref.keys()),
          f"输出接口不同: 多模态 {sorted(o_none)} vs 原版 {sorted(ref)}")
    check(mm._fusion_stats is None, "纯 RGB 路径不该产生 gate 统计")
    check(all(v == 0 for v in called.values()),
          f"纯 RGB 路径白跑了辅助模块: {called}")

    return (f"[{mode}] pred_logits / pred_boxes 逐位相等(torch.equal), "
            f"输出键一致 {sorted(ref.keys())}, "
            f"辅助模块调用次数 {called}, _fusion_stats=None")


# ======================================================================
#  §12 附加-1: 局部窗口 attention 数值正确性
# ======================================================================
def test_window_exact(ctx):
    """LocalCrossAttention2d 与「逐像素暴力 attention」逐位等价。

    方案 §4 把全局 attention 换成本地窗口, 但没说窗口怎么实现。这里用最笨的
    双重 for 循环当基准 —— 它没有任何切片/reshape 的排列自由度, 所以能唯一地
    钉住「窗口内的 k 顺序」与「softmax 的归一化轴」这两件最容易写错的事。

    为什么必须钉住顺序: Q 与 K 的点积只依赖配对、不依赖顺序, 但**输出**是
    Σ_k attn_k · v_k, 一旦 attn 与 v 的窗口排列不一致, 权重就会配错位置。这种
    错误在所有张量形状上都看不出来, 只能靠数值比对。
    """
    dev = ctx["device"]
    torch.manual_seed(50)
    b, heads, hd, h, w = 2, UNIT_HEADS, UNIT_C // UNIT_HEADS, 6, 7
    m = LocalCrossAttention2d(dim=UNIT_C, num_heads=UNIT_HEADS,
                              window_size=UNIT_WINDOW).to(dev).eval()
    rgb = torch.randn(b, UNIT_C, h, w, device=dev)
    aux = torch.randn(b, UNIT_C, h, w, device=dev)

    with torch.no_grad():
        got = m(rgb, aux)
        q4 = m.q_proj(m.norm_q(rgb)).view(b, heads, hd, h, w)
        kv = m.norm_kv(aux)
        k4 = m.k_proj(kv).view(b, heads, hd, h, w)
        v4 = m.v_proj(kv).view(b, heads, hd, h, w)
        p = UNIT_WINDOW // 2
        kp, vp = F.pad(k4, (p, p, p, p)), F.pad(v4, (p, p, p, p))
        buf = torch.zeros(b, heads, hd, h, w, device=dev)
        for y in range(h):
            for x in range(w):
                kb = kp[..., y:y + UNIT_WINDOW, x:x + UNIT_WINDOW].reshape(b, heads, hd, -1)
                vb = vp[..., y:y + UNIT_WINDOW, x:x + UNIT_WINDOW].reshape(b, heads, hd, -1)
                logits = torch.einsum("bhc,bhck->bhk", q4[..., y, x], kb) * (hd ** -0.5)
                buf[..., y, x] = torch.einsum("bhk,bhck->bhc",
                                              torch.softmax(logits, -1), vb)
        ref = m.out_proj(buf.reshape(b, UNIT_C, h, w))

    diff = float((got - ref).abs().max())
    scale = float(ref.abs().max())
    check(diff <= 1e-5 * max(1.0, scale),
          f"局部窗口 attention 与暴力实现不一致: maxdiff={diff:.3e} (输出量级 {scale:.3e}); "
          f"多半是窗口内 k/v 的排列与 attn 权重错位")

    # ---- 「局部」二字的直接证据: 输出的感受野必须是 (2p+1) x (2p+1) ----
    # 直接测 _window_attention 而不是整个模块: LocalCrossAttention2d.forward 里有
    # GroupNorm(1, dim), 它是**全局**归一化 —— 在 aux 上打一个脉冲会同时改变全图
    # 归一化的均值/方差, 于是远处像素的输出也会微微变化, 把感受野测量搅浑。
    # 传入裸的 q/k/v 才能干净地只测窗口这一件事。
    cy, cx = h // 2, w // 2
    with torch.no_grad():
        q = torch.randn(b, heads, hd, h, w, device=dev)
        k = torch.randn(b, heads, hd, h, w, device=dev)
        v = torch.zeros(b, heads, hd, h, w, device=dev)
        v[:, :, :, cy, cx] = 1.0
        resp = m._window_attention(q, k, v).abs().amax(dim=(1, 2))  # [b, h, w]
    eye = torch.zeros(h, w, dtype=torch.bool, device=dev)
    eye[cy - p:cy + p + 1, cx - p:cx + p + 1] = True
    outside = float(resp[:, ~eye].max())
    inside = float(resp[:, eye].min())
    check(outside == 0.0,
          f"窗口外的输出被中心脉冲影响了(max={outside:.3e}) —— attention 不是局部的; "
          f"感受野应为以 ({cy},{cx}) 为中心的 {UNIT_WINDOW}x{UNIT_WINDOW} 方框")
    check(inside > 0, "窗口内的输出对中心脉冲无响应, 探针无效")

    return (f"与逐像素暴力 attention 的 maxdiff={diff:.2e}(fp32), 窗口 "
            f"{UNIT_WINDOW}x{UNIT_WINDOW}; 感受野实测恰为 {UNIT_WINDOW}x{UNIT_WINDOW} 方框"
            f"(窗口外响应精确为 0, 全局实现会全图非零); 边界用 0 填充")


# ======================================================================
#  §9 附加-2: 统计指标齐全
# ======================================================================
def test_stats_keys(ctx):
    """§9 要求的 5 个指标逐 level 都要有, 并额外给出跨 level 平均值。"""
    dev = ctx["device"]
    f = make_fusion(dev)
    rgb, ir, dep = make_feats(dev, 60), make_feats(dev, 61), make_feats(dev, 62)
    td = make_text_dict(dev)

    _, stats = run_fusion(f, rgb, ir, dep, td)
    want = ("gate_rgb_mean", "gate_ir_mean", "gate_depth_mean",
            "aux_ir_ratio", "aux_depth_ratio")
    missing = []
    for key in want:
        for lvl in range(len(UNIT_LEVELS)):
            if f"{key}/l{lvl}" not in stats:
                missing.append(f"{key}/l{lvl}")
        if key not in stats:
            missing.append(f"{key}(跨 level 平均)")
    check(not missing, f"缺失的统计键: {missing}")
    for key, v in stats.items():
        check(torch.isfinite(v).all(), f"统计项 {key} 不是有限值: {v}")
        check(v.dim() == 0, f"统计项 {key} 应是标量, 得到 shape {tuple(v.shape)}")
    check(not any(p.requires_grad for p in [stats[k] for k in stats]),
          "统计项不应带梯度")

    # 单模态缺失时, 那一组的 aux_* 键必须整体消失(而不是留下一个 0)
    _, s_ir = run_fusion(make_fusion(dev), rgb, ir, None, td)
    check(not any(k.startswith("aux_depth_ratio") for k in s_ir),
          f"只传 IR 时不该有 aux_depth_ratio, 得到 {sorted(k for k in s_ir if 'depth' in k)}")
    check(float(s_ir["gate_depth_mean"]) == 0.0, "只传 IR 时 gate_depth_mean 应为 0")
    check(any(k.startswith("aux_ir_ratio") for k in s_ir), "只传 IR 时应保留 aux_ir_ratio")

    # log_stats=False 时必须彻底不算, 一个键都不留
    f_off = make_fusion(dev, log_stats=False)
    _, s_off = run_fusion(f_off, rgb, ir, dep, td)
    check(s_off is None, f"fusion_log_stats=False 时 last_stats 应为 None, 得到 {s_off}")

    return (f"三模态 {len(stats)} 个统计键(5 指标 x {len(UNIT_LEVELS)} level + 5 个跨 level 平均), "
            f"全为有限标量且无梯度; 单模态缺失时对应 aux_* 键整体消失; "
            f"fusion_log_stats=False 时 last_stats=None")


# ======================================================================
#  §12 附加-3: gate 统计的 mask 口径
# ======================================================================
def test_stats_mask(ctx):
    """gate 统计必须按 mask 口径算, 不能被「本该被丢掉的样本」拉偏。

    §9 的判据是「gate_ir_mean 长期接近 0 ⇒ IR 没被用上」。开了 modality dropout
    之后同一批里总有一部分样本的 IR 被丢掉, 那部分 W_ir 恒为 0; 若统计用朴素平均,
    gate_ir_mean 会随 dropout 比例系统性偏低, 判据就失真了。
    """
    dev = ctx["device"]
    f = make_fusion(dev)
    rgb, ir, dep = make_feats(dev, 70), make_feats(dev, 71), make_feats(dev, 72)
    td = make_text_dict(dev)

    _, all_on = run_fusion(f, rgb, ir, dep, td)
    _, half = run_fusion(f, rgb, ir, dep, td,
                         ir_valid=torch.tensor([False, True], device=dev))

    # 一半样本的 IR 被 mask 掉, 但统计口径应该只看有效的那一半
    check_close(float(half["gate_ir_mean"]), float(all_on["gate_ir_mean"]), 0.02,
                "一半样本 IR 无效后 gate_ir_mean 明显偏了 —— "
                "统计没有按 mask 口径算, 会被 modality dropout 系统性拉低")
    check(float(half["gate_ir_mean"]) > 0.02,
          "有效样本的 gate_ir_mean 也该是正的")
    check_close(float(half["gate_depth_mean"]), float(all_on["gate_depth_mean"]), 0.02,
                "Depth 两批都全有效, gate_depth_mean 不该变")

    # gate_rgb_mean 按方案 §9 的原式就是 W[:,0].mean() —— **纯平均, 不按 mask 口径**。
    # IR 被丢弃的样本只剩 RGB / Depth 两路竞争, 它的 W[:,0] 天然更高(§5.3: 0.881 对
    # 0.787), 所以这个读数会随「本批有哪些模态可用」移动。这是指标定义的固有性质,
    # 不是 bug: 训练时 modality dropout 的比例是固定超参, 因此该读数在 epoch 之间
    # 平稳, §9「长期接近 1 ⇒ 只走 RGB」的判据仍然成立。要对比不同可用组合, 应当看
    # 按 mask 口径算的 gate_ir_mean / gate_depth_mean(上面两条)以及 aux_*_ratio。
    # 这里断言的是重新归一化的方向: 少一路竞争, RGB 的份额必须变大。
    check(float(half["gate_rgb_mean"]) > float(all_on["gate_rgb_mean"]),
          f"IR 丢掉一半后 gate_rgb_mean 没有上升({float(half['gate_rgb_mean']):.4f} vs "
          f"{float(all_on['gate_rgb_mean']):.4f}) —— 与 §5.3 的 gate 初值不符")
    check(float(half["gate_rgb_mean"]) < 1.0, "gate_rgb_mean 不该到 1")

    # 全批 mask 掉时返回 0 而不是 NaN
    _, none_on = run_fusion(f, rgb, ir, dep, td,
                            ir_valid=torch.zeros(2, dtype=torch.bool, device=dev))
    check(float(none_on["gate_ir_mean"]) == 0.0,
          f"全批 IR 无效时 gate_ir_mean 应为 0, 得到 {float(none_on['gate_ir_mean'])}")
    check(not bool(torch.isnan(none_on["gate_ir_mean"])), "全批无效时统计出现 NaN")

    return (f"全有效 gate_ir_mean={float(all_on['gate_ir_mean']):.4f}, "
            f"半批有效={float(half['gate_ir_mean']):.4f}(口径一致), "
            f"全批无效=0.0(不是 NaN)")


# ======================================================================
#  主流程
# ======================================================================
TESTS = [
    ("shape", test_shape),
    ("gate_init", test_gate_init),
    ("missing_modality", test_missing),
    ("mask", test_mask),
    ("gradient", test_gradient),
    ("memory", test_memory),
    ("regression", test_regression),
    ("window_exact", test_window_exact),
    ("stats_keys", test_stats_keys),
    ("stats_mask", test_stats_mask),
]


def main():
    ap = argparse.ArgumentParser(description="GroundingDINO 第二版融合 (V2) 验证测试")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--weights", default=None, help="GroundingDINO Swin-T 预训练权重")
    ap.add_argument("--image", default=SAMPLE_IMAGE)
    ap.add_argument("--size", type=int, default=384,
                    help="模型级测试用的图片长边(默认 384, 只为跑得快; 与正确性无关)")
    ap.add_argument("--only", default=None, help="只跑名字里含该子串的测试")
    ap.add_argument("--fusion", default=V2_FUSION,
                    help="覆盖 config 里的 fusion_type(默认就是第二版)")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("!! CUDA 不可用, 回退到 cpu")
        args.device = "cpu"

    print("=" * 78)
    print(f"device={args.device}  weights={args.weights or '(无, 用 state_dict 拷贝)'}")
    print(f"fusion={args.fusion}  image={args.image}  long_side={args.size}")
    print("=" * 78)

    image = load_small_image(args.image, args.device, args.size)
    mm, info = build(MM_CFG, args.weights, args.device, fusion_type=args.fusion)
    off_checkpointing(mm)
    check(getattr(mm, "fusion_type", None) == args.fusion,
          f"模型实际用的是 {getattr(mm, 'fusion_type', None)!r}, 不是 {args.fusion!r}")
    if args.weights:
        print(f"IR Swin warm-start: {getattr(mm, '_ir_warm_start_info', None)}")
        non_aux_missing = [k for k in info.get("missing", [])
                           if not k.startswith(AUX_PREFIXES) and "adapters" not in k]
        print(f"checkpoint missing(非多模态键, 应为空): {non_aux_missing}")
    print(f"图片尺寸 {tuple(image.shape)}  说明: 第二版没有 beta, 纯 RGB 路径靠早退保证一致")
    print()

    ctx = {"device": args.device, "image": image, "mm": mm, "weights": args.weights}

    def reset_model():
        """每个测试开始前复位: eval 模式 + 清梯度 + 清融合统计。

        必须有这一步 —— 前面的测试可能调过 train() / set_train_stage() / 反传,
        而 BERT 自带 hidden_dropout_prob=0.1, 留在 train 模式会让逐位比较失效。
        """
        mm.eval()
        mm.zero_grad(set_to_none=True)
        mm._fusion_stats = None

    passed, failed = [], []
    for name, fn in TESTS:
        if args.only and args.only not in name:
            continue
        reset_model()
        print(f"[ RUN  ] {name}")
        try:
            msg = fn(ctx)
            print(f"[  OK  ] {name}\n         {msg}\n")
            passed.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[ FAIL ] {name}\n         {type(e).__name__}: {e}\n")
            traceback.print_exc()
            failed.append(name)

    print("=" * 78)
    print(f"通过 {len(passed)} / {len(passed) + len(failed)}")
    for n in failed:
        print(f"  失败: {n}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
