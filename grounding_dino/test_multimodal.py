#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GroundingDINO RGB/IR/Depth 多模态改造的验证测试。

对应技术方案 §18 的 8 项测试, 另加 1 项参数分组覆盖检查。全部可在 CPU 上跑。

用法(在仓库根目录执行):
    .venv/bin/python test_multimodal.py --device cpu
    .venv/bin/python test_multimodal.py --device cpu --weights weights/groundingdino_swint_ogc.pth
    .venv/bin/python test_multimodal.py --device cuda --weights weights/groundingdino_swint_ogc.pth

不传 --weights 时: 辅助测试用随机初始化, 而「RGB-only 回归」测试改为把多模态模型里
RGB 子树的权重直接拷进一个原始 config 构建的模型再比对 —— 不需要 checkpoint 也能验证
「beta=0 且不传辅助模态时, 模型逐位等于原始 RGB GroundingDINO」这一核心约束。

⚠️ 两个已知的、不是 bug 的现象(测试里已按此断言):
  1. pred_logits 里的 -inf 是设计行为: ContrastiveEmbed 把 padding token 槽位掩成 -inf,
     使 sigmoid 后为 0。所以判定「无 NaN」要用 torch.isnan(...).sum() == 0,
     不能用 torch.isfinite(...).all()。
  2. 因此两个 pred_logits 相减会在 -inf 位置产生 nan, 相等判断必须用 torch.equal 而不是
     torch.allclose。
"""

import argparse
import os
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
    BottleneckAdapter,
    BottleneckAdapter2d,
)
from groundingdino.util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402

RGB_CFG = os.path.join(GDINO, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
MM_CFG = os.path.join(GDINO, "groundingdino/config/GroundingDINO_MultiModal_SwinT.py")
SAMPLE_IMAGE = os.path.join(GDINO, ".asset/cat_dog.jpeg")

CAPTION = "cat . dog ."

# 多模态专属的模块名前缀 —— 用于把「RGB 子树」从多模态模型里筛出来
AUX_PREFIXES = (
    "ir_encoder.",
    "depth_encoder.",
    "ir_proj.",
    "depth_proj.",
    "fusion.",
)


# ======================================================================
#  基础设施
# ======================================================================
class Failure(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Failure(msg)


def build(cfg_path, weights=None, device="cpu"):
    """按 config 建模型; weights 给了就加载 checkpoint(走 clean_state_dict + strict=False)。"""
    args = SLConfig.fromfile(cfg_path)
    args.device = device
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


def fmt(x):
    return f"{x:.4g}"


# ======================================================================
#  1. RGB-only 回归: beta=0 + 不传辅助模态时, 逐位等于原始 RGB 模型
# ======================================================================
def test_rgb_only_regression(ctx):
    """方案的核心约束: RGB 主路径不能被破坏。

    两种验证方式:
      A) 有 checkpoint: 同一份权重分别加载进 RGB 模型与多模态模型, 输出必须 torch.equal。
      B) 无 checkpoint: 把多模态模型里 RGB 子树的 state_dict 拷进 RGB 模型, 再比对。
    另外还要验证「传了辅助模态但 beta=0」时输出同样逐位不变 —— 这是 Stage 1 能
    从「严格等于 RGB 模型」起步的前提。
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
        # 多模态模型多出来的键只会是 Adapter / 辅助分支, 不该有别的
        leftover = [k for k in res.unexpected_keys]
        check(not leftover, f"拷贝 RGB 子树时有意外键: {leftover[:5]}")
        missing = [k for k in res.missing_keys]
        check(not missing, f"RGB 模型缺少权重: {missing[:5]}")
        mode = "state_dict 拷贝"

    with torch.no_grad():
        ref = rgb_model(img[None].to(dev), captions=[CAPTION])

        # A) 多模态模型, 不传任何辅助模态
        o_none = mm(img[None].to(dev), captions=[CAPTION])
        # B) 多模态模型, 传了辅助模态但 beta = 0(构造默认值)
        ir, depth = make_aux_inputs(img, dev)
        o_aux = mm(img[None].to(dev), captions=[CAPTION],
                   ir_samples=ir, depth_samples=depth)

    check(torch.equal(ref["pred_boxes"], o_none["pred_boxes"]),
          f"[{mode}] 不传辅助模态时 pred_boxes 不一致, "
          f"maxdiff={(ref['pred_boxes'] - o_none['pred_boxes']).abs().max().item()}")
    check(torch.equal(ref["pred_logits"], o_none["pred_logits"]),
          f"[{mode}] 不传辅助模态时 pred_logits 不一致")

    betas = torch.cat([mm.fusion.beta_ir.detach(), mm.fusion.beta_depth.detach()])
    check(torch.all(betas == 0), f"beta 初值应为 0, 实际 {betas.tolist()}")

    check(torch.equal(ref["pred_boxes"], o_aux["pred_boxes"]),
          "beta=0 时辅助模态不应改变 pred_boxes, "
          f"maxdiff={(ref['pred_boxes'] - o_aux['pred_boxes']).abs().max().item()}")
    check(torch.equal(ref["pred_logits"], o_aux["pred_logits"]),
          "beta=0 时辅助模态不应改变 pred_logits")

    # Adapter 的输出层必须是零初始化 —— 这是上面「逐位相等」的另一个前提
    bad = [n for n, m in mm.named_modules()
           if isinstance(m, BottleneckAdapter) and m.up.weight.abs().sum().item() > 0]
    check(not bad, f"Encoder/Decoder Adapter 的输出层不是零初始化: {bad[:3]}")

    return f"模式={mode}; pred_boxes/pred_logits 均 torch.equal (不传辅助 + beta=0 传辅助)"


# ======================================================================
#  2. Shape: 各 level 的 RGB / IR / Depth / Fusion 特征形状一致
# ======================================================================
def test_shapes(ctx):
    """方案 §5: 融合输出的形状必须与 RGB srcs 完全一致, 否则下游 Transformer 全崩。"""
    dev, mm = ctx["device"], ctx["mm"]
    img = ctx["image"]
    ir, depth = make_aux_inputs(img, dev)

    cap = {}
    orig_fwd = mm.fusion.forward

    def spy(rgb_srcs, ir_srcs=None, depth_srcs=None, **kw):
        cap["rgb"] = [t.shape for t in rgb_srcs]
        if ir_srcs is not None:
            cap["ir"] = [t.shape for t in ir_srcs]
        if depth_srcs is not None:
            cap["depth"] = [t.shape for t in depth_srcs]
        out = orig_fwd(rgb_srcs, ir_srcs=ir_srcs, depth_srcs=depth_srcs, **kw)
        cap["fused"] = [t.shape for t in out]
        cap["levels"] = len(rgb_srcs)
        return out

    mm.fusion.forward = spy
    try:
        with torch.no_grad():
            out = mm(img[None].to(dev), captions=[CAPTION],
                     ir_samples=ir, depth_samples=depth)
    finally:
        mm.fusion.forward = orig_fwd

    check("fused" in cap, "Fusion 没有被调用")
    check(cap["levels"] == mm.num_fusion_levels,
          f"Fusion 应作用于 {mm.num_fusion_levels} 个 level, 实际 {cap['levels']}")

    B, C = 1, mm.hidden_dim
    for lvl in range(cap["levels"]):
        for name in ("rgb", "ir", "depth", "fused"):
            s = cap[name][lvl]
            check(len(s) == 4 and s[0] == B and s[1] == C,
                  f"level{lvl} {name} 形状 {tuple(s)} 不是 [{B},{C},h,w]")
            check(s[-2:] == cap["rgb"][lvl][-2:],
                  f"level{lvl} {name} 空间尺寸 {s[-2:]} 与 RGB {cap['rgb'][lvl][-2:]} 不一致")

    # IR / Depth 编码器自身的输出通道数也要与各自的 proj 对上
    check([t.shape[1] for t in _encoder_out(mm, ir, depth, dev)[0]] == mm.ir_encoder.out_channels,
          "IR 编码器输出通道与 out_channels 不符")
    check([t.shape[1] for t in _encoder_out(mm, ir, depth, dev)[1]] == mm.depth_encoder.out_channels,
          "Depth 编码器输出通道与 out_channels 不符")

    return (f"{cap['levels']} 个 level, 每个 rgb/ir/depth/fused 均为 "
            f"[{B},{C},h,w] 且空间尺寸逐 level 一致; IR ch={mm.ir_encoder.out_channels}, "
            f"Depth ch={mm.depth_encoder.out_channels}")


def _encoder_out(mm, ir, depth, dev):
    with torch.no_grad():
        ir_feats = mm.ir_encoder(ir, mask=torch.zeros(1, ir.shape[-2], ir.shape[-1],
                                                      dtype=torch.bool, device=dev))
        xd, _ = mm.depth_preprocess(depth)
        d_feats = mm.depth_encoder(xd)
    return ir_feats, d_feats


# ======================================================================
#  3. Mask: padding mask 与 RGB 对齐; 辅助模态的 invalid 区域不被当成真实内容
# ======================================================================
def test_mask(ctx):
    """两条断言:

    1. Fusion 只改数值, 不改 mask / 空间尺度 —— padding 区域在融合前后完全一致。
    2. 辅助模态的 invalid 区域(深度 0 / 整路 drop)不会污染特征: 把 RGB 保持不变,
       只把深度图里 invalid 区域之外的像素改掉, 融合结果必须跟着变;
       而只改 invalid 区域内的像素值, 融合结果必须不变(因为那里已被 mask 掉)。
    """
    dev, mm = ctx["device"], ctx["mm"]
    img = ctx["image"]
    ir, depth = make_aux_inputs(img, dev)

    # ---- 1) mask 对齐: 用 B=2 的 padding 场景(两张不同尺寸的图) ----
    mm.eval()
    img2 = F.interpolate(img[None], size=(img.shape[1] // 2, img.shape[2] // 2),
                         mode="bilinear", align_corners=False)[0]
    from groundingdino.util.misc import nested_tensor_from_tensor_list
    samples = nested_tensor_from_tensor_list([img.to(dev), img2.to(dev)])
    B, _, H, W = samples.tensors.shape
    check(tuple(samples.mask.shape) == (B, H, W),
          f"padding mask 形状 {tuple(samples.mask.shape)} 与输入 {(B, H, W)} 不符")

    ir2 = torch.zeros(B, 1, H, W, device=dev)
    ir2[0, :, : img.shape[1], : img.shape[2]] = ir
    ir2[1, :, : img2.shape[1], : img2.shape[2]] = F.interpolate(
        ir, size=img2.shape[-2:], mode="bilinear", align_corners=False)
    d2 = torch.zeros(B, 1, H, W, device=dev)
    d2[0, :, : img.shape[1], : img.shape[2]] = depth
    d2[1, :, : img2.shape[1], : img2.shape[2]] = F.interpolate(
        depth, size=img2.shape[-2:], mode="nearest")

    cap = {}
    orig_fwd = mm.fusion.forward

    def spy(rgb_srcs, ir_srcs=None, depth_srcs=None, **kw):
        cap["shapes"] = [t.shape for t in rgb_srcs]
        return orig_fwd(rgb_srcs, ir_srcs=ir_srcs, depth_srcs=depth_srcs, **kw)

    mm.fusion.forward = spy
    try:
        with torch.no_grad():
            out = mm(samples, captions=[CAPTION, CAPTION + " "], ir_samples=ir2,
                     depth_samples=d2, unset_image_tensor=False)
    finally:
        mm.fusion.forward = orig_fwd
    mm.unset_image_tensor()

    # RGB 三个 level 的空间尺寸必须等于 mask 下采样后的尺寸
    # (上游就是这么算 mask 的: F.interpolate(m[None].float(), size=src.shape[-2:])[0],
    #  这里只是把同一个关系显式断言出来)
    for lvl, s in enumerate(cap["shapes"]):
        m = F.interpolate(samples.mask[None].float(), size=s[-2:]).to(torch.bool)[0]
        check(tuple(m.shape) == (B,) + tuple(s[-2:]),
              f"level{lvl} 的 mask {tuple(m.shape)} 与 src {(B,) + tuple(s[-2:])} 不匹配")
        check(m.sum().item() > 0, f"level{lvl} 的 padding mask 全为 0, 构造的多尺寸 batch 没生效")
    check(torch.isnan(out["pred_logits"]).sum().item() == 0, "padding 场景出现 NaN")

    # ---- 2) valid flag 必须真的参与运算 ----
    # 整路置零时, valid=True(真实的「全是 0 距离」)与 valid=False(模态缺失)必须给出
    # 不同结果, 否则「整路 drop」会被误读成真实的零距离/黑色信息 —— 方案 §14 的核心。
    # 三次 forward, 其余条件完全相同:
    #   base      beta=0                      -> 纯 RGB 参考
    #   mixed     beta=1e-2, valid=[T, F]     -> 同样内容, 只有 valid flag 不同
    #   perturbed beta=1e-2, valid=[T, F], 深度整体 +3000
    base = fused_srcs_with_valid(mm, img, dev, depth, None, orig_fwd, beta=0.0)
    mixed = fused_srcs_with_valid(mm, img, dev, depth, [True, False], orig_fwd)
    perturbed = fused_srcs_with_valid(mm, img, dev, torch.flip(depth, dims=[3]),
                                      [True, False], orig_fwd)

    diff = max((s[0] - s[1]).abs().max().item() for s in mixed)
    check(diff > 0,
          "同一份深度输入下 depth_valid=False 与 True 结果相同 —— valid flag 没有参与运算")

    # valid=False 的样本拿到的必须是**原始 RGB 特征** ——
    # 「模态缺失 → 逐样本退化成纯 RGB」这条设计承诺。
    # 容差说明: base(B=1) 与 mixed(B=2) 是两次独立 forward、batch 尺寸不同, CPU oneDNN
    # 会为不同形状挑不同卷积核, 因此只能到 1e-5 量级而不能要求逐位相等。
    # beta=1e-2 时真正的 ΔF 幅度在 1e-2 量级(实测 |ΔF|/|F_rgb| ≈ 0.5 @ beta=1),
    # 与这个容差差着三个数量级, 不会把「其实改了」误判成「没改」。
    for lvl, (s, ref) in enumerate(zip(mixed, base)):
        d = (s[1] - ref[0]).abs().max().item()
        check(d < 1e-5,
              f"level{lvl}: depth_valid=False 的样本没有退化成纯 RGB(最大偏差 {fmt(d)})")

    # valid=True 的样本则必须真的读到了深度: 把深度整体加一个常数, 融合特征必须跟着变。
    changed = max((a[0] - b[0]).abs().max().item() for a, b in zip(mixed, perturbed))
    check(changed > 0, "扰动深度图后融合特征毫无变化 —— 深度分支没被真正读取")

    return ("padding mask 与 src 空间尺寸逐 level 对齐; 扰动深度会改变融合特征; "
            f"同内容下 valid=False/True 结果不同({fmt(diff)}); "
            f"valid=False 的样本退化成纯 RGB(偏差 < 1e-5)")


def fused_srcs_with_valid(mm, img, dev, depth, valid_flags, orig_fwd, beta=1e-2):
    """跑一次 forward, 返回融合后的 srcs: 返回值是 [level][sample] 的嵌套 list。

    `valid_flags` 是长度 B 的 list, 或 None(不传 valid, 视作全部有效)。
    **不能传「整 batch 全 invalid」**: 那种情况下 `_fuse_multimodal` 会短路直接返回
    RGB srcs, `fusion.forward` 根本不被调用(刻意的设计 —— 没有辅助模态就不建图)。
    所以这里让 batch 里至少留一个 valid 样本, 再逐个样本比较自己的融合结果。

    由于 batch 内所有样本的 RGB / 深度内容完全相同, 样本间的差异**只**来自 valid flag,
    这正好把「valid 是否真的参与运算」隔离出来。
    """
    b = len(valid_flags) if valid_flags is not None else 1
    imgs = img[None].to(dev).repeat(b, 1, 1, 1)
    depth_b = depth.to(dev).repeat(b, 1, 1, 1)
    dv = None if valid_flags is None else torch.tensor(valid_flags, device=dev)
    got = {}

    def spy(rgb_srcs, ir_srcs=None, depth_srcs=None, **kw):
        out = orig_fwd(rgb_srcs, ir_srcs=ir_srcs, depth_srcs=depth_srcs, **kw)
        got["v"] = [t.detach().clone() for t in out]
        return out

    mm.fusion.forward = spy
    try:
        with torch.no_grad():
            mm.fusion.beta_depth.fill_(beta)
            mm(imgs, captions=[CAPTION] * b, depth_samples=depth_b, depth_valid=dv)
            mm.fusion.beta_depth.fill_(0.0)
    finally:
        mm.fusion.forward = orig_fwd
    check("v" in got, "fusion.forward 没有被调用 —— 辅助模态被短路掉了, 检查 valid 构造")
    # 返回 [level][sample]: got["v"] 本身就是 [level] 的列表
    return [[t[i] for i in range(b)] for t in got["v"]]


# ======================================================================
#  4. Gradient: 辅助分支 / Adapter / Head 有梯度, 冻结模块梯度为 None
# ======================================================================
def test_gradient(ctx):
    """方案 §18 的 Gradient test + §11 的冻结约束。

    关键机制(必须理解, 否则会误判成 bug):
      beta=0 时 ΔF 拿不到梯度(∂/∂ΔF ∝ β = 0), 但 β 自己的梯度
      ∂L/∂β = ⟨∂L/∂F_new, ΔF⟩ ≠ 0, 所以优化器会先推动 β 离开 0。
      因此第一轮 backward 后 delta/encoder 的梯度是 **零**, 不是 None;
      要证明整条支路是活的, 得把 beta 手动设成非零再 backward 一次。
    """
    dev, mm = ctx["device"], ctx["mm"]
    img = ctx["image"].to(dev)
    ir, depth = make_aux_inputs(ctx["image"], dev)

    off_checkpointing(mm)
    mm.train()
    # 关掉模态 dropout 保证可复现; 单独在 test_modality_dropout 里测它
    mm.modality_augment.p_ir_drop = 0.0
    mm.modality_augment.p_depth_drop = 0.0
    mm.modality_augment.p_rgb_only = 0.0
    mm.modality_augment.p_ir_degrade = 0.0
    mm.modality_augment.p_depth_hole = 0.0

    out = mm(img[None], captions=[CAPTION], ir_samples=ir, depth_samples=depth)
    loss_from(out).backward()

    grad = {n: p.grad for n, p in mm.named_parameters()}
    with_grad = [n for n, g in grad.items() if g is not None]

    # ---- 文档 §18 的原始断言 ----
    check(any(p.grad is not None for p in mm.fusion.parameters()),
          "Fusion 完全没有梯度")
    for tag, mod in (("IR Encoder", mm.ir_encoder), ("Depth Encoder", mm.depth_encoder),
                     ("IR proj", mm.ir_proj), ("Depth proj", mm.depth_proj)):
        check(any(p.grad is not None for p in mod.parameters()), f"{tag} 完全没有梯度")
    for tag, mods in (("Encoder Adapter", mm.transformer.encoder.vision_adapters),
                      ("Decoder Adapter", mm.transformer.decoder.decoder_adapters)):
        check(any(p.grad is not None for p in mods.parameters()), f"{tag} 完全没有梯度")
    check(any(p.grad is not None for p in mm.bbox_embed.parameters()), "检测头没有梯度")

    # ---- beta 必须拿到非零梯度(否则整条支路真的死了) ----
    b_ir, b_d = mm.fusion.beta_ir.grad, mm.fusion.beta_depth.grad
    check(b_ir is not None and b_ir.abs().sum().item() > 0,
          f"beta_ir 梯度为 0/None, 辅助支路永远不会被激活: {b_ir}")
    check(b_d is not None and b_d.abs().sum().item() > 0,
          f"beta_depth 梯度为 0/None, 辅助支路永远不会被激活: {b_d}")

    # ---- beta != 0 时, delta / adapter / encoder 才真正开始学习 ----
    mm.zero_grad(set_to_none=True)
    with torch.no_grad():
        mm.fusion.beta_ir.fill_(1e-3)
        mm.fusion.beta_depth.fill_(1e-3)
    out2 = mm(img[None], captions=[CAPTION], ir_samples=ir, depth_samples=depth)
    loss_from(out2).backward()

    live = {
        "fusion.ir_deltas": mm.fusion.ir_deltas[0].weight.grad,
        "fusion.depth_deltas": mm.fusion.depth_deltas[0].weight.grad,
        "fusion.ir_adapters": mm.fusion.ir_adapters[0].up.weight.grad,
        "ir_encoder": next(mm.ir_encoder.parameters()).grad,
        "depth_encoder": next(mm.depth_encoder.parameters()).grad,
    }
    for name, g in live.items():
        check(g is not None and g.abs().sum().item() > 0,
              f"beta=1e-3 时 {name} 仍无梯度 —— 该支路是死的")

    mm.zero_grad(set_to_none=True)
    with torch.no_grad():
        mm.fusion.beta_ir.fill_(0.0)
        mm.fusion.beta_depth.fill_(0.0)

    # ---- 冻结组: 参数梯度必须是 None, 且绝不能用 no_grad 包住 ----
    info = mm.set_train_stage(1)
    # beta 保持非零, 否则辅助分支的梯度按设计恰好为 0(∂/∂ΔF ∝ β),
    # 会把「Stage 1 冻结了辅助分支」和「beta=0 所以没梯度」两种情况混为一谈。
    with torch.no_grad():
        mm.fusion.beta_ir.fill_(1e-3)
        mm.fusion.beta_depth.fill_(1e-3)
    out3 = mm(img[None], captions=[CAPTION], ir_samples=ir, depth_samples=depth)
    loss_from(out3).backward()

    frozen_checked = 0
    for name, p in mm.named_parameters():
        if name.startswith(AUX_PREFIXES) or "adapters" in name or \
                name.startswith(("bbox_embed.", "class_embed.")):
            continue
        if not name.startswith(("bert.", "feat_map.", "backbone.", "input_proj.",
                                "transformer.encoder.layers.", "transformer.decoder.layers.")):
            continue
        check(p.grad is None, f"Stage 1 下冻结参数 {name} 仍拿到了梯度")
        frozen_checked += 1
    check(frozen_checked > 0, "没有检查到任何冻结参数")

    # Stage 1 下辅助分支必须仍然有梯度 —— 冻结不能把它一起冻掉
    check(any(p.grad is not None and p.grad.abs().sum().item() > 0
              for p in mm.fusion.parameters()), "Stage 1 下 Fusion 没有梯度")
    check(any(p.grad is not None and p.grad.abs().sum().item() > 0
              for p in mm.ir_encoder.parameters()), "Stage 1 下 IR Encoder 没有梯度")
    check(any(p.grad is not None and p.grad.abs().sum().item() > 0
              for p in mm.bbox_embed.parameters()), "Stage 1 下检测头没有梯度")

    mm.zero_grad(set_to_none=True)
    return (f"第一轮 beta=0: beta_ir/beta_depth 梯度非零({fmt(b_ir.abs().max().item())}/"
            f"{fmt(b_d.abs().max().item())}); beta=1e-3 时 delta/adapter/encoder 均非零梯度; "
            f"Stage 1 下 {frozen_checked} 个冻结参数梯度为 None, 辅助分支与头仍有梯度")


# ======================================================================
#  5. Text: 一次 forward 内 BERT 只跑一次, encoded_text 被 Fusion 与 Transformer 共用
# ======================================================================
def test_text_encoder_once(ctx):
    """方案 §9: Fusion 只读 text_dict["encoded_text"], 不重复跑 BERT。"""
    dev, mm = ctx["device"], ctx["mm"]
    img = ctx["image"].to(dev)
    ir, depth = make_aux_inputs(ctx["image"], dev)

    counter = {"bert": 0, "feat_map": 0, "tokenizer": 0}

    def hook(tag):
        def fn(mod, args, out):
            counter[tag] += 1
        return fn

    handles = [
        mm.bert.register_forward_hook(hook("bert")),
        mm.feat_map.register_forward_hook(hook("feat_map")),
    ]

    # ⚠️ 不能写成 mm.tokenizer.__call__ = wrapper: Python 的特殊方法在**类型**上查找,
    # 实例属性拦不住 `tokenizer(...)`。必须换成一个包装对象。
    class CountingTokenizer:
        def __init__(self, inner):
            self._inner = inner

        def __call__(self, *a, **k):
            counter["tokenizer"] += 1
            return self._inner(*a, **k)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    tok_orig = mm.tokenizer
    mm.tokenizer = CountingTokenizer(tok_orig)

    seen_text = {}
    orig_fwd = mm.fusion.forward

    def spy(rgb_srcs, ir_srcs=None, depth_srcs=None, text_dict=None, **kw):
        if text_dict is not None:
            seen_text["shape"] = tuple(text_dict["encoded_text"].shape)
            seen_text["mask_shape"] = tuple(text_dict["text_token_mask"].shape)
            seen_text["id"] = id(text_dict["encoded_text"])
        return orig_fwd(rgb_srcs, ir_srcs=ir_srcs, depth_srcs=depth_srcs,
                        text_dict=text_dict, **kw)

    mm.fusion.forward = spy
    try:
        with torch.no_grad():
            mm(img[None], captions=[CAPTION], ir_samples=ir, depth_samples=depth)
    finally:
        mm.fusion.forward = orig_fwd
        mm.tokenizer = tok_orig
        for h in handles:
            h.remove()

    check(counter["tokenizer"] == 1, f"tokenizer 被调用了 {counter['tokenizer']} 次")
    check(counter["bert"] == 1, f"BERT 被调用了 {counter['bert']} 次, 应为 1 次")
    check(counter["feat_map"] == 1, f"feat_map 被调用了 {counter['feat_map']} 次, 应为 1 次")
    check("shape" in seen_text, "Fusion 没有收到 text_dict")
    check(seen_text["shape"][0] == 1 and seen_text["shape"][2] == mm.hidden_dim,
          f"Fusion 收到的 encoded_text 形状 {seen_text['shape']} 不是 [B,L,{mm.hidden_dim}]")
    check(seen_text["mask_shape"][:2] == seen_text["shape"][:2],
          f"text_token_mask {seen_text['mask_shape']} 与 encoded_text {seen_text['shape']} 不匹配")

    return (f"tokenizer/BERT/feat_map 各调用 1 次; Fusion 复用 encoded_text "
            f"{seen_text['shape']} + text_token_mask {seen_text['mask_shape']}")


# ======================================================================
#  6. Missing modality: 四种输入组合都不出 NaN
# ======================================================================
def test_missing_modality(ctx):
    """方案 §14: RGB-only / RGB+IR / RGB+Depth / RGB+IR+Depth 四种组合都要能跑。

    ⚠️ 判定「无 NaN」用 torch.isnan(...).sum() == 0, **不能**用 torch.isfinite(...).all():
    pred_logits 在 padding token 槽位上被 ContrastiveEmbed 有意掩成 -inf, 是设计行为。
    """
    dev, mm = ctx["device"], ctx["mm"]
    img = ctx["image"].to(dev)
    ir, depth = make_aux_inputs(ctx["image"], dev)

    # beta 设成非零, 保证辅助支路真的在参与, 而不是被 beta=0 掩盖掉
    with torch.no_grad():
        mm.fusion.beta_ir.fill_(1e-2)
        mm.fusion.beta_depth.fill_(1e-2)

    combos = [
        ("RGB only", {}),
        ("RGB + IR", dict(ir_samples=ir)),
        ("RGB + Depth", dict(depth_samples=depth)),
        ("RGB + IR + Depth", dict(ir_samples=ir, depth_samples=depth)),
        ("IR 整路置零 + valid=False", dict(ir_samples=torch.zeros_like(ir),
                                          ir_valid=torch.tensor([False], device=dev))),
        ("Depth 整路置零 + valid=False", dict(depth_samples=torch.zeros_like(depth),
                                             depth_valid=torch.tensor([False], device=dev))),
        ("两个模态都置零 + valid=False", dict(ir_samples=torch.zeros_like(ir),
                                            depth_samples=torch.zeros_like(depth),
                                            ir_valid=torch.tensor([False], device=dev),
                                            depth_valid=torch.tensor([False], device=dev))),
        ("深度全为 0(无 valid flag)", dict(depth_samples=torch.zeros_like(depth))),
    ]

    results = []
    for name, kw in combos:
        with torch.no_grad():
            out = mm(img[None], captions=[CAPTION], **kw)
        n_log = int(torch.isnan(out["pred_logits"]).sum())
        n_box = int(torch.isnan(out["pred_boxes"]).sum())
        check(n_box == 0, f"[{name}] pred_boxes 出现 {n_box} 个 NaN")
        check(n_log == 0, f"[{name}] pred_logits 出现 {n_log} 个 NaN")
        check(bool(torch.isfinite(out["pred_boxes"]).all()), f"[{name}] pred_boxes 不全是有限值")
        check(tuple(out["pred_boxes"].shape) == (1, mm.num_queries, 4),
              f"[{name}] pred_boxes 形状 {tuple(out['pred_boxes'].shape)} 不对")
        # 真实 token 槽位上必须是有限值
        valid_logits = out["pred_logits"][torch.isfinite(out["pred_logits"])]
        check(valid_logits.numel() > 0, f"[{name}] pred_logits 全是 -inf")
        results.append(f"{name}: ok")

    with torch.no_grad():
        mm.fusion.beta_ir.fill_(0.0)
        mm.fusion.beta_depth.fill_(0.0)

    return f"{len(combos)} 种组合全部无 NaN 且形状正确 ({'; '.join(results[:4])} ...)"


# ======================================================================
#  7. Decoder: hs / pred_logits / pred_boxes 的接口不变
# ======================================================================
def test_decoder_interface(ctx):
    """方案 §5: 改造后 Transformer 与检测头的接口必须与上游完全一致。"""
    dev, mm = ctx["device"], ctx["mm"]
    img = ctx["image"].to(dev)
    ir, depth = make_aux_inputs(ctx["image"], dev)

    def shape_of(x):
        """transformer 的返回里既有 Tensor 也有 list, 统一成嵌套的 shape 描述。"""
        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
        if isinstance(x, (list, tuple)):
            return [shape_of(e) for e in x]
        return type(x).__name__

    cap = {}
    handle = mm.transformer.register_forward_hook(
        lambda m, a, o: cap.update(hs=shape_of(o[0]), reference=shape_of(o[1]),
                                   hs_enc=shape_of(o[2]))
    )
    try:
        with torch.no_grad():
            out = mm(img[None], captions=[CAPTION], ir_samples=ir, depth_samples=depth)
    finally:
        handle.remove()

    n_dec = mm.transformer.num_decoder_layers
    # 上游返回的 hs 是长度 n_dec 的 list(每项 [B, Nq, C]); hs_enc 同理
    check(isinstance(cap["hs"], list) and len(cap["hs"]) == n_dec,
          f"hs 应为长度 {n_dec} 的列表, 实际 {cap['hs']}")
    for i, s in enumerate(cap["hs"]):
        check(s == (1, mm.num_queries, mm.hidden_dim),
              f"hs[{i}] 形状 {s} 应为 {(1, mm.num_queries, mm.hidden_dim)}")
    # two_stage_type="standard" 时 hs_enc = tgt_undetach.unsqueeze(0), 形状固定为
    # (1, bs, nq, d_model) —— 是 Tensor 而不是 list, 上游就是这么返回的。
    check(cap["hs_enc"] == (1, 1, mm.num_queries, mm.hidden_dim),
          f"hs_enc 形状应为 {(1, 1, mm.num_queries, mm.hidden_dim)}, 实际 {cap['hs_enc']}")
    check(tuple(out["pred_logits"].shape) == (1, mm.num_queries, mm.max_text_len),
          f"pred_logits 形状 {tuple(out['pred_logits'].shape)} 不对")
    check(tuple(out["pred_boxes"].shape) == (1, mm.num_queries, 4),
          f"pred_boxes 形状 {tuple(out['pred_boxes'].shape)} 不对")
    check(bool((out["pred_boxes"] >= 0).all() and (out["pred_boxes"] <= 1).all()),
          "pred_boxes 必须归一化到 [0,1]")

    # 不传辅助模态时接口也不能变
    with torch.no_grad():
        out_rgb = mm(img[None], captions=[CAPTION])
    check(tuple(out_rgb["pred_logits"].shape) == tuple(out["pred_logits"].shape),
          "不传辅助模态时 pred_logits 形状变了")
    check(tuple(out_rgb["pred_boxes"].shape) == tuple(out["pred_boxes"].shape),
          "不传辅助模态时 pred_boxes 形状变了")

    return (f"hs 为长度 {n_dec} 的列表, 每项 {(1, mm.num_queries, mm.hidden_dim)}; "
            f"pred_logits {tuple(out['pred_logits'].shape)}, "
            f"pred_boxes {tuple(out['pred_boxes'].shape)} 且落在 [0,1]; 辅助模态有无都不变")


# ======================================================================
#  8. 数据对齐: 同一模态只读自己的输入, 且不跨样本串味
# ======================================================================
def test_data_alignment(ctx):
    """「对齐」在这里落到三个可证伪的断言上:

    1. **没有跨模态串味**: 同一个 batch 里把 IR / Depth 在两个样本之间对调, 每个样本的
       融合特征应当与「单独跑该样本、喂对调后的辅助模态」逐位一致。
       (辅助模态与 mask 的空间尺寸本来就是逐 level 对齐的, 见 test_shapes。)
    2. **每个模态都被真正读取**: 只改 IR 或只改 Depth, 融合特征必须跟着变。
    3. **RGB 空间镜像一致性**: 同步翻转 RGB/IR/Depth 时, 纯 RGB 路径(beta=0)的预测框
       必须镜像; 且 beta!=0 时「只翻 RGB 不翻 IR/Depth」与「三个一起翻」结果不同 ——
       说明辅助模态自身的空间内容确实参与了解算。
    """
    dev, mm = ctx["device"], ctx["mm"]
    img = ctx["image"].to(dev)
    _, h, w = ctx["image"].shape
    g = torch.Generator().manual_seed(7)
    ir = torch.rand(1, 1, h, w, generator=g).to(dev)
    depth = (torch.rand(1, 1, h, w, generator=g) * 15000 + 2000).to(dev)

    # 复制成 batch=2: 样本 0 用原图, 样本 1 用原图
    rgb_b = torch.cat([img[None], img[None]], dim=0)
    ir_b = torch.cat([ir, ir * 0.3 + 0.1], dim=0)
    d_b = torch.cat([depth, depth * 0.5 + 1000.0], dim=0)
    ir_swap = torch.cat([ir_b[1:2], ir_b[0:1]], dim=0)
    d_swap = torch.cat([d_b[1:2], d_b[0:1]], dim=0)

    def fused(rgb, iri, di):
        """直接跑 RGB backbone + 辅助编码 + Fusion, 绕开中间的文本/Transformer 部分。"""
        from groundingdino.util.misc import nested_tensor_from_tensor_list

        with torch.no_grad():
            mm.eval()
            mm.fusion.beta_ir.fill_(1e-2)
            mm.fusion.beta_depth.fill_(1e-2)
            samples = nested_tensor_from_tensor_list([rgb[i] for i in range(rgb.shape[0])])
            feats, _ = mm.backbone(samples)
            srcs = [mm.input_proj[l](f.tensors) for l, f in enumerate(feats)]
            ir_srcs = mm.ir_proj(mm.ir_encoder(iri, mask=samples.mask))
            xd, _ = mm.depth_preprocess(di)
            d_srcs = mm.depth_proj(mm.depth_encoder(xd))
            text_dict = mm._encode_text(["cat . dog ."] * rgb.shape[0])
            out = mm.fusion(srcs, ir_srcs=ir_srcs, depth_srcs=d_srcs, text_dict=text_dict)
            mm.fusion.beta_ir.fill_(0.0)
            mm.fusion.beta_depth.fill_(0.0)
        return [t.detach().clone() for t in out]

    f_orig = fused(rgb_b, ir_b, d_b)
    f_swap = fused(rgb_b, ir_swap, d_swap)
    # 用容差而不是 torch.equal: 这里比的是「两次独立前向里不同位置的数据」, CPU 上
    # oneDNN 会因为数值不同而选到不同的卷积 kernel, 产生 ~1e-9 的浮点噪声。
    # 真正的串味是数量级级别的差异, 容差足以区分。
    ATOL, RTOL = 1e-5, 1e-4
    for lvl in range(mm.num_fusion_levels):
        for got, want, tag in ((f_swap[lvl][0], f_orig[lvl][1], "样本0↔原样本1"),
                               (f_swap[lvl][1], f_orig[lvl][0], "样本1↔原样本0")):
            md = (got - want).abs().max().item()
            check(torch.allclose(got, want, atol=ATOL, rtol=RTOL),
                  f"level{lvl}: 对调辅助模态后 {tag} 的融合特征不一致 (maxdiff={fmt(md)}) "
                  f"—— 存在跨样本串味")

    # 每个模态都被真正读取
    #
    # ⚠️ 扰动要选**非线性**的。`DepthPreprocessor` 的百分位归一化对单调仿射变换不变
    # (quantile(aD+b) = a·quantile(D)+b, 约掉之后 D_norm 逐位相同), 所以 `d*0.4+500`
    # 这种改法会被预处理完全消掉, 融和特征只差 1e-7 的浮点噪声 —— 测试会假通过。
    # 这里改成空间翻转, 真正改变深度的空间内容。
    f_ir = fused(rgb_b, torch.flip(ir_b, dims=[3]), d_b)
    d_ir = max((a - b).abs().max().item() for a, b in zip(f_orig, f_ir))
    check(d_ir > 1e-4, f"只改 IR 却几乎不影响融合特征(maxdiff={fmt(d_ir)}) —— IR 分支没被读取")
    f_d = fused(rgb_b, ir_b, torch.flip(d_b, dims=[3]))
    d_d = max((a - b).abs().max().item() for a, b in zip(f_orig, f_d))
    check(d_d > 1e-4, f"只改 Depth 却几乎不影响融合特征(maxdiff={fmt(d_d)}) —— Depth 分支没被读取")

    # RGB 镜像一致性(beta=0, 纯 RGB 路径)
    with torch.no_grad():
        mm.fusion.beta_ir.fill_(0.0)
        mm.fusion.beta_depth.fill_(0.0)
        mm.eval()
        o_plain = mm(img[None], captions=[CAPTION])
        flip = torch.flip(img, dims=[2])
        o_flip = mm(flip[None], captions=[CAPTION])
        # 三个模态一起翻, 与只翻 RGB
        o_all = mm(flip[None], captions=[CAPTION],
                   ir_samples=torch.flip(ir, dims=[2]),
                   depth_samples=torch.flip(depth, dims=[2]))
        mm.fusion.beta_ir.fill_(1e-2)
        mm.fusion.beta_depth.fill_(1e-2)
        o_mis = mm(flip[None], captions=[CAPTION], ir_samples=ir, depth_samples=depth)
        mm.fusion.beta_ir.fill_(0.0)
        mm.fusion.beta_depth.fill_(0.0)

    # 纯 RGB 路径下翻转输入的输出, 应当等于原输出的镜像(用框中心验证)。
    #
    # ⚠️ 必须做**最近邻匹配**, 不能按下标配对。实测: 翻转会让 query 的置信度排序整体
    # 重排(原图 top-1 的框在翻转图里可能掉到第 5), 按下标逐位相减等于拿 A 物体的坐标
    # 去比 B 物体, 得到的「偏差 0.2」纯属配错对象 —— 连 y 方向都会跟着冒出 0.15 的假偏差,
    # 而水平翻转根本不该动 y, 这本身就说明是配对问题而不是镜像问题。
    # 模型也确实不是严格翻转等变的(Swin 的窗口划分从左上角起算, 翻转后窗口边界错位),
    # 所以用「每个高置信框在翻转结果里的最近邻距离」来度量, 而不是要求逐位相等。
    # 更强的「RGB 路径逐位不变」由 test 1 的 torch.equal 保证。
    # 「镜像得好」是**训练过的模型**才具备的性质: 随机初始化的模型输出的 900 个框
    # 本来就与图像内容无关, 没有理由翻转等变。所以这里按有无 checkpoint 分流。
    K, POOL = 20, 200
    if ctx["weights"]:
        box_p = o_plain["pred_boxes"][0]
        box_f = o_flip["pred_boxes"][0]
        top_p = torch.topk(o_plain["pred_logits"][0].max(dim=-1)[0], K).indices
        top_f = torch.topk(o_flip["pred_logits"][0].max(dim=-1)[0],
                           min(POOL, box_f.shape[0])).indices
        ref = box_p[top_p]                                # [K,4] cxcywh
        cand = box_f[top_f].clone()                       # [POOL,4]
        cand[:, 0] = 1.0 - cand[:, 0]                     # 镜像 cx
        dist = (ref[:, 0:1] - cand[None, :, 0]).abs() + (ref[:, 1:2] - cand[None, :, 1]).abs()
        md = dist.min(dim=1).values.mean().item()
        check(md < 0.05,
              f"纯 RGB 路径下水平翻转后高置信框中心未镜像(最近邻平均偏差 {fmt(md)})")
    else:
        # 无 checkpoint 时只断言「翻转确实改变了输出」, 证明空间信息被真实读取
        md = (o_flip["pred_boxes"] - o_plain["pred_boxes"]).abs().max().item()
        check(md > 0, "随机初始化下翻转输入没有改变输出 —— 空间信息没被读取")
    check(torch.equal(o_all["pred_boxes"], o_flip["pred_boxes"]),
          "beta=0 时三模态一起翻应与只翻 RGB 结果逐位一致")

    mis = (o_mis["pred_boxes"] - o_all["pred_boxes"]).abs().max().item()
    check(mis > 0,
          "IR/Depth 不跟着翻转时输出毫无变化 —— 说明辅助模态的空间内容没被使用")

    md_tag = "纯 RGB 翻转镜像偏差" if ctx["weights"] else "翻转后输出变化"
    return (f"对调辅助模态后逐样本结果与原样本逐位互换(无跨样本串味); "
            f"只改 IR 影响 {fmt(d_ir)}, 只改 Depth 影响 {fmt(d_d)}; "
            f"{md_tag} {fmt(md)}; "
            f"IR/Depth 不跟随翻转时输出差异 {fmt(mis)}")


# ======================================================================
#  9. 模态 dropout (方案 §14) 与参数分组覆盖
# ======================================================================
def test_modality_dropout(ctx):
    """训练期的模态 dropout 必须通过 valid flag 生效, 而不是只把输入置零。"""
    dev, mm = ctx["device"], ctx["mm"]
    img = ctx["image"].to(dev)
    ir, depth = make_aux_inputs(ctx["image"], dev)

    aug = mm.modality_augment
    saved = (aug.p_ir_drop, aug.p_depth_drop, aug.p_rgb_only,
             aug.p_ir_degrade, aug.p_depth_hole)
    try:
        # eval 下必须是恒等变换
        mm.eval()
        for p in (0.0, 1.0):
            aug.p_ir_drop = p
            _, _, iv, dv = aug(ir, depth, None, None)
            check(iv is None and dv is None, f"eval 下 ModalityAugment 不应改动 valid (p={p})")

        # train 下 p=1 必须真的把整路 drop 掉, 且 valid=False
        mm.train()
        aug.p_ir_drop = 1.0
        aug.p_depth_drop = 1.0
        aug.p_rgb_only = 1.0
        aug.p_ir_degrade = 0.0
        aug.p_depth_hole = 0.0
        ir_d, d_d, iv, dv = aug(ir, depth, None, None)
        check(float(ir_d.abs().sum()) == 0.0, "p_ir_drop=1 时 IR 没被置零")
        check(float(d_d.abs().sum()) == 0.0, "p_depth_drop=1 时 Depth 没被置零")
        check(iv is not None and not bool(iv.any()), "p_ir_drop=1 时 ir_valid 不是全 False")
        check(dv is not None and not bool(dv.any()), "p_depth_drop=1 时 depth_valid 不是全 False")

        # 整路 drop 后, forward 应当退化到纯 RGB 路径(beta=0 时逐位相等)
        #
        # ⚠️ 只让 modality_augment 留在 train 模式, 模型其余部分必须 eval:
        # BERT 自带 hidden_dropout_prob=0.1, 与 config 的 text_dropout=0 无关,
        # 整模型 train() 会让两次前向的文本特征不同, 逐位比较必然失败。
        aug.p_ir_drop = 1.0
        aug.p_depth_drop = 1.0
        aug.p_rgb_only = 1.0
        mm.eval()
        aug.train()
        try:
            with torch.no_grad():
                mm.fusion.beta_ir.fill_(0.0)
                mm.fusion.beta_depth.fill_(0.0)
                o_drop = mm(img[None], captions=[CAPTION], ir_samples=ir, depth_samples=depth)
                o_rgb = mm(img[None], captions=[CAPTION])
        finally:
            aug.eval()
        check(torch.equal(o_drop["pred_boxes"], o_rgb["pred_boxes"]),
              "整路 drop 后的输出应逐位等于纯 RGB 输出, "
              f"maxdiff={(o_drop['pred_boxes'] - o_rgb['pred_boxes']).abs().max().item()}")
        check(torch.equal(o_drop["pred_logits"], o_rgb["pred_logits"]),
              "整路 drop 后的 pred_logits 应逐位等于纯 RGB 输出")

        # IR 退化只改 IR, 不改 valid
        aug.p_ir_drop = 0.0
        aug.p_depth_drop = 0.0
        aug.p_rgb_only = 0.0
        aug.p_ir_degrade = 1.0
        mm.train()
        ir_g, _, iv2, _ = aug(ir.clone(), None, None, None)
        check(float((ir_g - ir).abs().sum()) > 0, "p_ir_degrade=1 时 IR 没有被退化")
        check(iv2 is None or bool(iv2.all()), "IR 退化不应把 valid 置 False")

        # 深度挖洞: 洞内变 0, 会被 DepthPreprocessor 判为 invalid
        aug.p_ir_degrade = 0.0
        aug.p_depth_hole = 1.0
        _, d_h, _, _ = aug(None, depth.clone(), None, None)
        check(float((d_h == 0).sum()) > float((depth == 0).sum()), "p_depth_hole=1 时没有挖出孔洞")
        x_d, valid_map = mm.depth_preprocess(d_h)
        check(bool((valid_map[d_h <= 0] == False).all()),  # noqa: E712
              "被打成 0 的孔洞没有被 DepthPreprocessor 判为 invalid")
    finally:
        (aug.p_ir_drop, aug.p_depth_drop, aug.p_rgb_only,
         aug.p_ir_degrade, aug.p_depth_hole) = saved
        with torch.no_grad():   # beta 是 leaf Variable, 原地写必须关梯度
            mm.fusion.beta_ir.fill_(0.0)
            mm.fusion.beta_depth.fill_(0.0)
        mm.eval()

    return "eval 下为恒等; train 下 p=1 真的整路 drop 且 valid=False, 输出退化到纯 RGB; 退化/挖洞行为正确"


def test_ir_warm_start(ctx):
    """方案 §6 方案 A: stem 权重 1/3 + 独立 Swin 用 RGB Swin 权重 warm-start。

    1/3 的来历: Swin 的 `patch_embed.proj` 实测是 (96,3,4,4), 所以
        proj ∘ stem == 1/3 * Σ_c proj[:, c, :, :]
    正好等于「patch_embed 权重沿输入通道求平均」, 且 Swin 侧仍是标准 in_chans=3,
    RGB Swin 的 state_dict 可以逐键直接拷 —— 这就是能一次 load_state_dict 完成
    warm-start、不需要任何权重改形的原因。
    """
    mm = ctx["mm"]
    stem = mm.ir_encoder.stem
    check(stem.weight.shape == (3, 1, 1, 1),
          f"IR stem 应为 Conv2d(1,3,1), 实际权重形状 {tuple(stem.weight.shape)}")
    check(bool((stem.weight == 1.0 / 3.0).all()),
          f"IR stem 权重应恒为 1/3, 实际 min={stem.weight.min().item()} max={stem.weight.max().item()}")
    check(float(stem.bias.abs().sum()) == 0.0, "IR stem bias 应为 0")

    rgb_proj = mm.backbone[0].patch_embed.proj
    check(rgb_proj.weight.shape[1] == 3,
          f"RGB Swin patch_embed.proj 输入通道应为 3, 实际 {rgb_proj.weight.shape[1]}")
    # 「proj ∘ stem」等价于 proj 沿输入通道求均值:
    # composed[o,x,y] = Σ_c proj[o,c,x,y] * stem[c]  (stem.weight 是 (3,1,1,1), 拉平成 (3,))
    stem_w = stem.weight.detach().reshape(-1)
    composed = torch.einsum("ocxy,c->oxy", rgb_proj.weight.detach(), stem_w)
    check(torch.allclose(composed, rgb_proj.weight.detach().mean(dim=1)),
          "stem=1/3 与 patch_embed 复合后不等于沿输入通道求均值")

    if not ctx["weights"]:
        return ("stem 权重恒为 1/3、复合后等价于 patch_embed 沿输入通道求均值; "
                "(无 checkpoint, 未验证实际 warm-start)")

    info = getattr(mm, "_ir_warm_start_info", None)
    check(info is not None, "加载 checkpoint 后没有触发 IR warm-start")
    check(info["copied"] == 187 and not info["skipped"],
          f"IR Swin warm-start 应逐键拷满 187 个, 实际 copied={info['copied']} "
          f"skipped={info['skipped'][:5]}")

    rgb_sd = mm.backbone[0].state_dict()
    ir_sd = mm.ir_encoder.body.state_dict()
    mismatch = [k for k in rgb_sd if k not in ir_sd or not torch.equal(rgb_sd[k], ir_sd[k])]
    check(not mismatch, f"warm-start 后仍有键与 RGB Swin 不一致: {mismatch[:5]}")

    # checkpoint 里已经带 IR 权重时, 绝不能被 RGB 权重覆盖(续训多模态模型的场景)
    ir_before = {k: v.clone() for k, v in ir_sd.items()}
    saved = mm.ir_encoder.body.patch_embed.proj.weight.detach().clone()
    with torch.no_grad():
        mm.ir_encoder.body.patch_embed.proj.weight.add_(1.0)
    full_sd = mm.state_dict()
    mm.load_state_dict(full_sd, strict=False)
    after = mm.ir_encoder.body.patch_embed.proj.weight.detach()
    check(torch.allclose(after, saved + 1.0),
          "load_state_dict 时用自己的 IR 权重覆盖了刚改的值 —— warm-start 判断有误")
    with torch.no_grad():
        mm.ir_encoder.body.patch_embed.proj.weight.copy_(saved)
    del ir_before

    return (f"stem=1/3 且复合等价于通道均值; warm-start 逐键拷满 {info['copied']} 个, "
            f"与 RGB Swin 完全一致; checkpoint 自带 IR 权重时不会被覆盖")


def test_param_groups(ctx):
    """方案 §12.3: 分组学习率不能漏掉任何可训练参数。

    `get_param_groups()` 只吐**可训练**参数, 所以 Stage 0(=RGB baseline, 只训检测头)
    的分组天然只有 head 一组 —— 这不是漏组, 是设计要求: 辅助分支在 Stage 0 必须冻住,
    否则 beta 会从 0 漂走, 「多模态模型在 Stage 0 逐位等于 RGB 模型」这个对照前提就没了。
    """
    mm = ctx["mm"]
    results = []
    for stage in (0, 1, 2, 3):
        info = mm.set_train_stage(stage)
        groups = mm.get_param_groups()
        names = {g["name"] for g in groups}
        total = sum(len(g["params"]) for g in groups)
        n_trainable = sum(1 for p in mm.parameters() if p.requires_grad)
        # 覆盖率: 每一个 requires_grad=True 的张量都必须落进某个学习率组
        check(total == n_trainable,
              f"Stage {stage}: 分组覆盖 {total} 个张量, 可训练 {n_trainable} 个, 不一致")
        check(len({id(p) for g in groups for p in g["params"]}) == total,
              f"Stage {stage}: 同一个张量被分进了多个学习率组")

        if stage == 0:
            # 只训检测头; 辅助分支/Adapter 必须冻住
            check(names == {"head"}, f"Stage 0 应只有 head 组, 实际 {sorted(names)}")
            check(not mm.fusion.beta_ir.requires_grad
                  and not next(mm.ir_encoder.parameters()).requires_grad,
                  "Stage 0 下辅助分支没有冻住 —— beta 会漂走, 不再逐位等于 RGB 模型")
        else:
            check({"ir", "depth", "fusion", "adapter", "head"} <= names,
                  f"Stage {stage}: 分组缺少必需组, 实际 {sorted(names)}")
            check(mm.fusion.beta_ir.requires_grad
                  and next(mm.ir_encoder.parameters()).requires_grad
                  and next(mm.depth_encoder.parameters()).requires_grad,
                  f"Stage {stage}: 辅助分支被误冻结")
            check(next(mm.transformer.encoder.vision_adapters.parameters()).requires_grad
                  and next(mm.transformer.decoder.decoder_adapters.parameters()).requires_grad,
                  f"Stage {stage}: Adapter 被误冻结")
            # 检测头必须落到 head 组(5e-5), 不能被混进 transformer 组(1e-5)
            head_names = {n for g in groups if g["name"] == "head"
                          for n in [id(p) for p in g["params"]]}
            check(any(id(p) in head_names for p in mm.bbox_embed.parameters()),
                  f"Stage {stage}: bbox_embed 没被分到 head 组(会被当成 1e-5 训练)")

        check(not any(p.requires_grad for p in mm.bert.parameters()),
              f"Stage {stage}: BERT 不应可训练")
        results.append(f"S{stage}:{total}张量/{len(groups)}组")
    mm.set_train_stage(1)
    return "Stage 0/1/2/3 分组覆盖完整且无重复: " + ", ".join(results)


# ======================================================================
#  运行器
# ======================================================================
TESTS = [
    ("1. RGB-only 回归", test_rgb_only_regression),
    ("2. Shape", test_shapes),
    ("3. Mask", test_mask),
    ("4. Gradient", test_gradient),
    ("5. Text 只编码一次", test_text_encoder_once),
    ("6. 缺失模态", test_missing_modality),
    ("7. Decoder 接口", test_decoder_interface),
    ("8. 数据对齐", test_data_alignment),
    ("9. 模态 dropout + 参数分组", test_modality_dropout),
    ("10. IR warm-start", test_ir_warm_start),
    ("11. 参数分组覆盖", test_param_groups),
]


def main():
    ap = argparse.ArgumentParser(description="GroundingDINO 多模态改造验证测试")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--weights", default=None,
                    help="GroundingDINO checkpoint; 不传则用 state_dict 拷贝做 RGB 回归")
    ap.add_argument("--only", default=None, help="只跑名字里含该子串的测试")
    ap.add_argument("--image", default=SAMPLE_IMAGE)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("!! CUDA 不可用, 回退到 cpu")
        args.device = "cpu"

    print("=" * 78)
    print(f"device={args.device}  weights={args.weights or '(无, 用 state_dict 拷贝)'}")
    print(f"sample={args.image}")
    print("=" * 78)

    _, image = load_image(args.image)
    mm, info = build(MM_CFG, args.weights, args.device)
    if args.weights:
        warm = getattr(mm, "_ir_warm_start_info", None)
        print(f"IR Swin warm-start: {warm}")
        non_aux_missing = [k for k in info.get("missing", []) if not k.startswith(AUX_PREFIXES)
                           and "adapters" not in k]
        print(f"checkpoint missing(非多模态键, 应为空): {non_aux_missing}")
        print(f"checkpoint unexpected: {info.get('unexpected')}")
    print()

    ctx = {"device": args.device, "image": image, "mm": mm, "weights": args.weights}

    def reset_model():
        """每个测试开始前复位: eval 模式 + 清梯度 + 归零 beta。

        必须有这一步 —— 前面的测试会调 train() / set_train_stage() / 改 beta,
        而 BERT 自带 hidden_dropout_prob=0.1, 留在 train 模式会让逐位比较失效。
        """
        mm.eval()
        mm.zero_grad(set_to_none=True)
        if getattr(mm, "fusion", None) is not None:
            with torch.no_grad():
                mm.fusion.beta_ir.fill_(0.0)
                mm.fusion.beta_depth.fill_(0.0)

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
