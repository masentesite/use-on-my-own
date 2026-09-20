#!/usr/bin/env python3
"""GroundingDINO 三模态(RGB + IR + Depth)指代表达理解 —— 评估 / 推理 / §12 逐图排查。

与 `run_grounding.py` 的关系:
  - **评估协议逐字复用**:`iou` 与 `slice_by_images` 直接从那个脚本 import(它是评估口径的
    单一事实来源),分数口径也一样 —— `pred_logits.sigmoid()` 后取真实词 token
    `arange(1, n_tok-1)`(排除 [CLS]/[SEP])的最大值作为该框分数, 取 top-1 框算 IoU。
  - **预处理逐字复用**:三模态的 resize / 归一化 / depth 最近邻,全部交给
    `mm_data.MultiModalReferDataset(train=False)` —— 训练与推理的输入分布因此不可能漂移。
    这里**不重新实现**任何变换(那正是 `mm_data.py` 文件头警告过的事)。
  - 只做推理,不碰训练(训练见 `train_multimodal.py`)。

为什么要单独一个脚本(V2 §12):
  `train_multimodal.py` 的 `evaluate()` 只报**集合级**指标(mean_iou / gate 均值)。
  当 §12 出现「辅助模态没贡献」时, 集合级的读数说不清是**哪张图**、**哪个 level**、
  是「权重没给」还是「权重给了但候选是 0」。本脚本按 query 打印 gate 诊断量:
      gate r/i/d = W[:,0] / W[:,1] / W[:,2] 的逐样本空间均值
      aux_ir / aux_depth = ‖W_aux·A_aux‖ / ‖W_rgb·F_rgb‖(§9 第 4/5 行)
  并对每条样本做一次 §12 的三段判据(见 `triage()`), 汇总出 `aux_dead_rate`。

模态开关: `--ir/--no-ir` 与 `--depth/--no-depth` 覆盖 §7.4 的四种组合
(RGB / RGB+IR / RGB+Depth / RGB+IR+Depth)。缺的那一路**不传 kw**,走的是模型里
「整路缺失」的分支 —— 与训练时 modality dropout 掉它完全同一条路径。

用法示例:
  python run_grounding_mm.py --num-images 3                    # 三模态, 前 3 张图
  python run_grounding_mm.py --no-depth                        # RGB + IR
  python run_grounding_mm.py --no-ir --no-depth                # 纯 RGB(对照档)
  python run_grounding_mm.py --start 200 --num-images 200      # 按图切片续跑
  python run_grounding_mm.py --weights runs/stage2/best.pth --gate-per-level
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# run_grounding.py 必须在 _HERE 进 sys.path 之后再 import(它在模块顶层
# `sys.path.insert(0, "./GroundingDINO")` 用的是相对 cwd 的路径)。它已不再设置任何
# 代理环境变量, import 它是安全的; 本脚本刻意**只**取它的两个纯函数, 不复制协议。
from run_grounding import iou, slice_by_images  # noqa: E402

import bert_local  # noqa: E402
from mm_data import (  # noqa: E402
    MultiModalReferDataset,
    collate_fn,
    load_items,
    normalize_caption,
)

_REPO = os.path.join(_HERE, "GroundingDINO")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from groundingdino.models import build_model  # noqa: E402
from groundingdino.util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402

# ---------------- 默认路径(本机固定项,可被命令行覆盖)----------------
DEFAULT_DATA = os.path.join(_HERE, "../TrainSet.")
DEFAULT_CFG = os.path.join(_REPO, "groundingdino/config/GroundingDINO_MultiModal_SwinT.py")
# 多模态 checkpoint 只能由 train_multimodal.py 存出 —— RGB 预训练权重
# (weights/groundingdino_swint_ogc.pth)里没有任何 ir_*/depth_*/fusion.* 键, 加载会当场报错。
DEFAULT_WEIGHTS = os.path.join(_HERE, "runs/smoke/best.pth")

#: 判定「候选被压没了」的强度比阈值 —— 权重拿到了, 但辅助支路进主路径的强度近似为 0
_AUX_DEAD_RATIO = 1e-3


# ================================================================ 数据集

def load_dataset(data_dir: str):
    """读 queries.json,返回按 (img_id, qid) 排序的条目(训练侧同一份 `load_items`)。

    每个 item: qid / query / img_id / paths{visible,infrared,depth} / bbox(归一化 xyxy 或 None)
    """
    items = load_items(data_dir)
    if not items:
        raise SystemExit(f"{data_dir} 里没有 query")
    return items


def build_dataset(data_dir: str, items, resize: int, max_size: int):
    """按训练侧的口径建一个**只做确定性预处理**的 dataset(train=False: 不翻转)。

    返回的 dataset 与传入的 items **同序**(无 GT 的那些会被换成带占位框的副本), 所以主循环
    可以直接用位置索引取样本 —— main 里有一道顺序检查兜底。

    `boxes` 这一项本脚本一概不用(GT 一律取自 `load_items` 的原始条目, 已经由
    `slice_by_images` 切过片), 但 dataset 会把「没有 GT」的条目过滤掉 ——
    纯推理 dump 的场景下那会把数据全滤干净。所以无 GT 时塞一个退化框占位,
    只为绕开那条过滤, 占位框不会被读到。
    """
    placeholder = [0.0, 0.0, 0.0, 0.0]
    items = [it if it["bbox"] is not None else {**it, "bbox": placeholder} for it in items]
    return MultiModalReferDataset(
        data_dir=data_dir, items=items, train=False, resize=resize, max_size=max_size
    )


# ================================================================ 模型

def load_model(cfg_path: str, weights_path: str, device: str):
    """建多模态模型并**严格**加载 checkpoint —— 任何一个键对不上就当场停下。

    为什么不用 strict=False 凑合:
      融合分支(ir_attn / depth_attn / gate.conv / adapter)全部是随机初始化起步的。
      若某个键没加载上而脚本照跑, 输出会「看起来正常」但融合是随机的, 到 gate 诊断
      那一栏才表现为 §12 的「辅助模态没贡献」—— 排查成本极高。加载不了就说明
      checkpoint / config 选错了, 这时应该报错而不是给数。
    """
    args = SLConfig.fromfile(cfg_path)
    # 与 run_grounding.py 一致: 本地有 weights/bert-base-uncased 就读它, 建模型时不联网
    args.text_encoder_type = bert_local.resolve(args.text_encoder_type)
    args.device = device
    model = build_model(args)

    if not os.path.exists(weights_path):
        raise SystemExit(
            f"找不到 checkpoint: {weights_path}\n"
            f"  先用 train_multimodal.py 训一个(或加 --smoke 跑个冒烟), 再用 --weights 指过来。"
        )
    # weights_only=False: checkpoint 里除权重外还有 args / train_image_ids 等非张量
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # ---- 融合版本守卫: 两版 Fusion 的键完全不同(v1 是 beta_*/gate.*, v2 是 ir_attn/depth_attn)。
    # 拿 v1 的 checkpoint 去加载 v2 模型, strict 会报一堆 missing/unexpected; 更糟的是有人
    # 关掉 strict 后拿到「随机初始化的 Fusion + 正常的 RGB 主干」, 读数全不可信。
    want, got = model.fusion_type, ckpt.get("fusion_type")
    if got is not None and got != want:
        raise SystemExit(
            f"[ckpt] 融合版本不匹配: {weights_path} 是 {got!r} 训出来的, "
            f"当前 config({os.path.basename(cfg_path)})是 {want!r}。\n"
            f"  改 --config(用 fusion_type={got!r} 的那份)或换 checkpoint。"
        )

    try:
        # clean_state_dict 去掉可能存在的 "module." 前缀(DataParallel 存出来的)
        model.load_state_dict(clean_state_dict(ckpt.get("model", ckpt)), strict=True)
    except RuntimeError as e:
        n_aux = len([k for k in model.state_dict()
                     if k.startswith(("fusion.", "ir_", "depth_"))])
        raise SystemExit(
            f"[ckpt] 加载 {weights_path} 失败 —— 这不是与当前 config 配套的多模态 checkpoint。\n"
            f"  该 checkpoint 记录的 fusion_type={got!r}; 模型里多模态相关键共 {n_aux} 个。\n"
            f"  最常见的原因: 传进来的是 RGB 预训练权重 weights/groundingdino_swint_ogc.pth, "
            f"它没有任何辅助分支的键。\n{str(e)[:600]}"
        )
    model.eval().to(device)
    print(f"[ckpt] {weights_path}")
    print(f"       fusion_type={want}  stage={ckpt.get('stage')}  epoch={ckpt.get('epoch')}")
    return model


# ================================================================ 推理

@torch.no_grad()
def predict_top1_mm(model, batch, caption: str, use_ir: bool, use_depth: bool, device: str,
                    box_threshold: float = 0.0, amp: bool = False):
    """返回 (归一化 xyxy 或 None, score, gate_stats 或 None)。

    `batch` 是 `collate_fn([dataset[i]])` 的四元组 —— 与 train_multimodal.py 的
    dataloader 同一条代码路径: NestedTensor(RGB + padding mask)、IR [1,1,h,w] 已归一化、
    depth [1,1,h,w] 原始 uint16 数值(0=invalid, 归一化是 DepthPreprocessor 的职责)。
    `caption` **由调用方逐条传入**, 不从 batch 里取 —— 同一张图的 5 条 query 共用一份
    图像张量(run_grounding.py 也是这么缓存图像的), 从 batch 取会把 5 条 query 全判成
    第 1 条的那句话(症状: 同图 5 条 score 逐位相同)。

    分数口径与 `run_grounding.py::predict_top1` 逐字一致, 只多了两个 kw:
    传了 `ir_samples` / `depth_samples` 就走融合路径, 不传就是纯 RGB(上游行为)。

    `gate_stats` 取自模型的 `_fusion_stats`(= Fusion 的 last_stats)。**batch=1**,
    所以它就是**这一条样本**的 gate 均值, 正好用于 §12 的逐图排查。
    """
    samples, _, ir, depth = batch
    samples = samples.to(device)

    kw = {}
    if use_ir:
        kw["ir_samples"] = ir.to(device)
    if use_depth:
        kw["depth_samples"] = depth.to(device)

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
        outputs = model(samples, captions=[caption], **kw)

    logits = outputs["pred_logits"].float().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].float()[0]               # (nq, 4) 归一化 cxcywh

    n_tok = model.tokenizer(caption, return_tensors="pt")["input_ids"].shape[1]
    valid_pos = torch.arange(1, max(1, n_tok - 1))
    scores = logits[:, valid_pos].max(dim=1)[0]            # (nq,)

    i = int(scores.argmax().item())
    best = float(scores[i].item())
    if best < box_threshold:
        pred = None
    else:
        cx, cy, w, h = boxes[i].tolist()
        pred = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    stats = getattr(model, "_fusion_stats", None)
    gate = None if not stats else {k: float(v.detach().float()) for k, v in stats.items()}
    return pred, best, gate


# ================================================================ gate 诊断打印

def format_gate(gate, per_level: bool = False) -> str:
    """把一条样本的 gate 统计拼成一行(V2 §9)。

    `last_stats` 里既有逐 level 的明细(`gate_ir_mean/l0`), 也有跨 level 平均
    (`gate_ir_mean`)。第一版 Fusion 没有这些键(只有逐 level 的 beta_*),
    挑不到就整体退化成逐键打印 —— 两版融合共用同一段打印代码。
    """
    if not gate:
        return "gate=(无)"
    parts = []
    if "gate_rgb_mean" in gate:
        parts.append("gate r/i/d={:.3f}/{:.3f}/{:.3f}".format(
            gate["gate_rgb_mean"], gate.get("gate_ir_mean", 0.0),
            gate.get("gate_depth_mean", 0.0)))
    for key, tag in (("aux_ir_ratio", "aux_ir"), ("aux_depth_ratio", "aux_depth")):
        if key in gate:
            parts.append(f"{tag}={gate[key]:.3f}")
    if not parts:  # 第一版 Fusion 的键(beta_* 等), 原样列出
        parts = [f"{k}={gate[k]:.4f}" for k in sorted(gate)]
    else:
        parts += _per_level_parts(gate) if per_level else []
    return " ".join(parts)


def _per_level_parts(gate):
    """逐 level 的 r/i/d, 形如 `l0=0.786/0.107/0.107`。"""
    out = []
    for lvl in sorted(k.rsplit("/l", 1)[-1] for k in gate if k.startswith("gate_rgb_mean/l")):
        r = gate.get(f"gate_rgb_mean/l{lvl}")
        i = gate.get(f"gate_ir_mean/l{lvl}", float("nan"))
        d = gate.get(f"gate_depth_mean/l{lvl}", float("nan"))
        out.append(f"l{lvl}={r:.3f}/{i:.3f}/{d:.3f}")
    return out


def triage(gate, use_ir: bool, use_depth: bool):
    """§12「辅助模态没贡献」的逐图判据 —— 返回人话告警(空 = 这一路在用)。

    三段判据的区别决定了接下来该去查哪儿:
      1. gate 权重**恒为 0**        -> 候选是 None 或 valid 全 False, 问题在**输入/掩码**;
      2. 权重非零但 aux_ratio ≈ 0  -> 候选本身被压没了, 问题在 **adapter / attention**;
      3. 权重极低(但非 0)         -> 模型只是不倾向用它, 问题在**优化**(这才是 §12 那一条)。
    """
    if not gate:
        return []
    out = []
    for on, name in ((use_ir, "ir"), (use_depth, "depth")):
        if not on:
            continue
        w = gate.get(f"gate_{name}_mean")
        ratio = gate.get(f"aux_{name}_ratio")
        if w is None:
            continue
        if w <= 0.0:
            out.append(f"{name}: gate 恒 0(候选=None 或 valid 全 False, 查输入/掩码)")
        elif ratio is not None and ratio < _AUX_DEAD_RATIO:
            out.append(f"{name}: gate={w:.3f} 但 aux_ratio={ratio:.1e}≈0 ⇒ 候选被 adapter/attention 压没了")
        elif w < 0.02:
            out.append(f"{name}: gate={w:.3f} 偏低(近乎不用)")
    return out


def combo_name(use_ir: bool, use_depth: bool) -> str:
    return "_".join(["rgb"] + (["ir"] if use_ir else []) + (["depth"] if use_depth else []))


def _fmt_iou(v):
    return "  n/a" if v is None else f"{v:.3f}"


# ================================================================ 主流程

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="GroundingDINO 三模态(RGB+IR+Depth)指代表达理解评估 / §12 逐图排查",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据 / 数量控制
    p.add_argument("--data-dir", default=DEFAULT_DATA, help="数据集根目录(含 queries/queries.json)")
    p.add_argument("--start", type=int, default=0, help="从第几张图开始(按 img_id 排序,续跑用)")
    p.add_argument("--num-images", type=int, default=0, help="跑多少张图,0=全部;配合 --start 分批续跑")
    # 模态开关 —— 覆盖 §7.4 的四种组合
    p.add_argument("--ir", action=argparse.BooleanOptionalAction, default=True,
                   help="是否送入 IR(--no-ir 关掉; 缺的那路走模型里「整路缺失」的分支)")
    p.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True,
                   help="是否送入 Depth")
    # 模型 / 设备
    p.add_argument("--config", default=DEFAULT_CFG,
                   help="多模态 config .py(决定 fusion_type, 必须与 checkpoint 配套)")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="多模态 checkpoint(由 train_multimodal.py 存出的 best.pth/last.pth)")
    p.add_argument("--device", default="cuda", help="推理设备 cuda/cpu")
    p.add_argument("--amp", action="store_true",
                   help="半精度推理; 默认 fp32(评估口径更稳。训练开过 AMP 也不影响本脚本)")
    # 常规超参数
    p.add_argument("--box-threshold", type=float, default=0.0,
                   help="低于此分数的 top-1 框判为无检测;0.0=REC 模式总返回 top-1")
    p.add_argument("--resize", type=int, default=800, help="最短边,与训练 / run_grounding.py 对齐")
    p.add_argument("--max-size", type=int, default=1333)
    # gate 诊断
    p.add_argument("--gate-every", type=int, default=1,
                   help="每多少条 query 打印一次 gate 诊断(1=每条;逐样本明细一律写进 JSON)")
    p.add_argument("--gate-per-level", action="store_true",
                   help="gate 行里额外打印逐 level 的 r/i/d(默认只打跨 level 平均)")
    p.add_argument("--no-triage", action="store_true",
                   help="关掉 §12 逐图判据的告警行(只打印数值)")
    # 输出
    p.add_argument("--out", default=None,
                   help="结果 JSON 路径,默认 {data-dir}/grounding_mm_{组合}[_start{N}]_results.json")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    items = slice_by_images(load_dataset(args.data_dir), args.start, args.num_images)
    n_imgs = len({it["img_id"] for it in items})
    combo = combo_name(args.ir, args.depth)
    print(f"数据集: {args.data_dir}")
    print(f"模态  : {combo}   设备: {args.device}   预处理: train=False(resize={args.resize})")
    print(f"待跑  : {n_imgs} 张图 / {len(items)} 条 query (start={args.start}, num_images={args.num_images})")
    if not items:
        print("没有要跑的数据,退出")
        return

    dataset = build_dataset(args.data_dir, items, args.resize, args.max_size)
    # 主循环直接用位置索引取样本(dataset[idx]), 这里确认 dataset 没过滤/重排过
    if len(dataset.items) != len(items) or any(
        d["qid"] != s["qid"] for d, s in zip(dataset.items, items)
    ):
        raise SystemExit("dataset 的条目与切片后的 items 不同序 —— 位置索引会取错样本")

    model = load_model(args.config, args.weights, args.device)
    # gate 诊断只有在**真的走了融合**时才有值, 提前把「为什么没有」说清楚, 免得逐条
    # 打印一串没有信息的 gate=(无)
    log_gate = args.ir or args.depth
    if not log_gate:
        print("gate: 未走融合 —— 两种辅助模态都没传, 模型直接返回 RGB srcs(纯对照档)")
    elif not hasattr(model.fusion, "log_stats"):
        # 第一版 Fusion: 只有逐 level 的 beta_*, 没有 B x 3 x H x W 的 gate 分布,
        # §12 的三段判据(gate 权重 vs 强度比)在它身上无从谈起 —— 那正是 V2 要解决的问题
        print(f"gate: fusion_type={model.fusion_type} 只有逐 level 的 beta_*(没有 gate 分布)")
    elif not model.fusion.log_stats:
        print("⚠️ config 的 fusion_log_stats=False, 本次不会产生任何 gate 诊断量")
    every = f"每 {args.gate_every} 条" if args.gate_every > 0 else "不逐条(只写 JSON)"
    print(f"模型就绪 (gate 诊断打印: {every})\n", flush=True)

    # 同一张图的 5 条 query 共用一次解码 + resize: dataset 的 __getitem__ 每次都要读三张
    # PNG, 逐 query 重算等于把 I/O 与 resize 放大 5 倍。items 已按 (img_id, qid) 排序,
    # 所以只在 img_id 变化时重新 collate, 缓存里恒为一张图的张量。
    batch, cur_img_id = None, None

    results = []
    gate_acc = {}          # gate 键 -> [sum, count]
    n_detected = 0
    n_warned = 0
    triage_hits = {"ir": 0, "depth": 0}
    t0 = time.perf_counter()

    for idx, it in enumerate(items):
        if it["img_id"] != cur_img_id:
            # 只缓存图像张量, caption 逐条传 —— 见 predict_top1_mm 的说明
            batch = collate_fn([dataset[idx]])
            cur_img_id = it["img_id"]

        pred, score, gate = predict_top1_mm(
            model, batch, normalize_caption(it["query"]), args.ir, args.depth, args.device,
            box_threshold=args.box_threshold, amp=args.amp,
        )

        if pred is not None:
            n_detected += 1
            iou_val = iou(pred, it["bbox"]) if it["bbox"] else None
        else:
            iou_val = 0.0 if it["bbox"] else None

        if gate:
            for k, v in gate.items():
                slot = gate_acc.setdefault(k, [0.0, 0])
                slot[0] += v
                slot[1] += 1

        warns = [] if args.no_triage else triage(gate, args.ir, args.depth)
        if warns:
            n_warned += 1
            for name in triage_hits:
                if any(w.startswith(name + ":") for w in warns):
                    triage_hits[name] += 1

        results.append({
            "qid": it["qid"],
            "img_id": it["img_id"],
            "query": it["query"],
            "input_size": [int(batch[0].tensors.shape[-1]), int(batch[0].tensors.shape[-2])],
            "gt_box": it["bbox"],   # 归一化 xyxy(框对缩放不变, 与像素空间 IoU 等价)
            "pred_box": pred,       # 归一化 xyxy 或 None
            "score": score,
            "iou": iou_val,
            "gate": gate,           # None = 没走融合(last_stats 未记录)
        })

        if args.gate_every > 0 and (idx + 1) % args.gate_every == 0:
            tail = f" | {format_gate(gate, args.gate_per_level)}" if log_gate else ""
            print(f"  [{idx + 1:>5}/{len(items)}] {it['qid']} iou={_fmt_iou(iou_val)} "
                  f"score={score:.3f}{tail}", flush=True)
            for w in warns:
                print(f"        ⚠️ {w}", flush=True)
        elif (idx + 1) % 100 == 0:
            print(f"  ...{idx + 1}/{len(items)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    dt = time.perf_counter() - t0

    # ---- 汇总 ----
    gate_stats = {k: v / n for k, (v, n) in gate_acc.items() if n}
    with_gt = [r for r in results if r["gt_box"] is not None]
    summary = {
        "combo": combo,
        "use_ir": args.ir,
        "use_depth": args.depth,
        "num_images": n_imgs,
        "num_queries": len(results),
        "num_detected": n_detected,
        "detect_rate": n_detected / len(results) if results else 0.0,
        "elapsed_s": round(dt, 1),
        # §9 的五个诊断量(逐样本平均) + 逐 level 明细 + §12 的告警计数
        "gate_stats": {k: v for k, v in gate_stats.items() if "/l" not in k},
        "gate_stats_per_level": {k: v for k, v in gate_stats.items() if "/l" in k},
        "aux_dead_queries": n_warned,
        "aux_dead_rate": n_warned / len(results) if results else 0.0,
        "triage_hits": triage_hits,
    }
    print(f"\n跑完 {len(results)} 条 query / {n_imgs} 张图,耗时 {dt:.1f}s")
    if with_gt:
        ious = np.array([r["iou"] for r in with_gt if r["iou"] is not None])
        for thr in (0.25, 0.5, 0.75):
            summary[f"acc@{thr}"] = float((ious >= thr).mean())
            print(f"  Acc@IoU>={thr}: {(ious >= thr).mean() * 100:.2f}%")
        summary["mean_iou"] = float(ious.mean())
        summary["median_iou"] = float(np.median(ious))
        print(f"  mean IoU  : {ious.mean():.4f}    median IoU: {np.median(ious):.4f}")
    else:
        print("  无 GT bbox,仅推理 dump")

    # v1 Fusion 没有 gate_*_mean 这些跨 level 键(只有逐 level 的 beta_*), 退化成逐键打印
    shown_gate = summary["gate_stats"] or summary["gate_stats_per_level"]
    if shown_gate:
        print(f"  gate(均值): {format_gate(shown_gate)}")
        if n_warned:
            print(f"  ⚠️ {n_warned}/{len(results)} 条命中 §12 辅助模态告警 "
                  f"(ir={triage_hits['ir']}, depth={triage_hits['depth']}); "
                  f"逐条明细见 JSON 的 results[].gate")

    # 默认文件名带上 --start: 切片续跑时每一批各写各的, 不会把上一批的结果覆盖掉
    tail = f"_start{args.start}" if args.start else ""
    out = args.out or os.path.join(args.data_dir, f"grounding_mm_{combo}{tail}_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "data_dir": args.data_dir,
                "combo": combo,
                "use_ir": args.ir,
                "use_depth": args.depth,
                "start": args.start,
                "num_images": args.num_images,
                "box_threshold": args.box_threshold,
                "config": args.config,
                "weights": args.weights,
                "resize": args.resize,
                "max_size": args.max_size,
                "amp": args.amp,
                "device": args.device,
                "fusion_type": getattr(model, "fusion_type", None),
            },
            "summary": summary,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已存: {out}")


if __name__ == "__main__":
    main()
