#!/usr/bin/env python3
"""多模态 GroundingDINO 微调入口 —— 同一份代码既跑本地冒烟测试, 也跑云端 4090 训练。

设计要点:

* **一条命令一个 stage。** `--stage N` 决定冻结策略, `--resume` 从上一段的 checkpoint 续。
  脚本**不在运行中切换 stage**(那会让 AdamW 动量在 cosine 中途断档); 每个 stage 独立启动,
  模型/优化器/调度器状态整体恢复。

* **不 import `run_grounding.py`。** 那个脚本在**模块顶层**设置代理环境变量, import 它就等于
  把死代理注进本进程。这里只复刻它的评估协议(`predict_top1` 的打分方式与 `iou`), 约 15 行。

* **`--smoke`** 是 CPU 上 1~2 分钟跑完的端到端闸门: 真的走完 forward → backward →
  optimizer.step → 一次验证打分。它考的是「管道通不通」, 不是「模型准不准」。

用法参见文件末尾的 `EXAMPLES`。
"""
import argparse
import json
import math
import os
import socket
import sys
import time

# ---- 环境前导: 必须在 import torch / transformers 之前 ----
# 代理改成条件探测(与 run_grounding.py 同样的做法): 端口没在监听就不设,
# 免得把 HF 请求送进黑洞。云端根本没有这个代理。
_PROXY_PORT = 7892


def _proxy_alive(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


if _proxy_alive(_PROXY_PORT):
    os.environ["ALL_PROXY"] = f"http://127.0.0.1:{_PROXY_PORT}"
    os.environ["all_proxy"] = os.environ["ALL_PROXY"]
else:
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for _v in ("no_proxy", "NO_PROXY"):
    if "hf-mirror.com" not in os.environ.get(_v, ""):
        os.environ[_v] = "hf-mirror.com," + os.environ.get(_v, "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.utils import clip_grad_norm_  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mm_data import MultiModalReferDataset, collate_fn, load_items, split_image_ids  # noqa: E402
from mm_loss import build_criterion  # noqa: E402

_REPO = os.path.join(_HERE, "GroundingDINO")
sys.path.insert(0, _REPO)

from groundingdino.models import build_model  # noqa: E402
from groundingdino.models.GroundingDINO.groundingdino import DEFAULT_LR_GROUPS  # noqa: E402
from groundingdino.util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402

DEFAULT_CFG = os.path.join(_REPO, "groundingdino/config/GroundingDINO_MultiModal_SwinT.py")
DEFAULT_WEIGHTS = os.path.join(_HERE, "weights/groundingdino_swint_ogc.pth")


# ================================================================ 参数

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="多模态 GroundingDINO 微调",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- 用户点名要的核心超参 ----
    p.add_argument("--batch-size", type=int, default=2, help="每次前向的样本数")
    p.add_argument("--epochs", type=int, default=15, help="本 stage 的轮数")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="辅助分支(ir/depth/fusion/adapter)的学习率; 其余组按 config 里的比例同步缩放")
    p.add_argument("--stage", type=int, default=1, choices=[0, 1, 2, 3],
                   help="0=RGB baseline(只训检测头) / 1=多模态预热 / 2=部分解冻 / 3=联合微调")

    # ---- 优化器 ----
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="只作用于 ndim>=2 的参数(排除 bias / norm / beta)")
    p.add_argument("--betas", default="0.9,0.999")
    p.add_argument("--momentum", type=float, default=0.9, help="仅 --optimizer sgd 时用")
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=1)

    # ---- 数据 ----
    p.add_argument("--data-dir", default=os.path.join(_HERE, "../TrainSet."))
    p.add_argument("--val-ratio", type=float, default=0.1, help="按**图**留出的验证集比例")
    p.add_argument("--resize", type=int, default=800, help="最短边; 与 run_grounding.py 的 800 对齐")
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--hflip-p", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # ---- 运行 ----
    p.add_argument("--config", default=DEFAULT_CFG)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="RGB 预训练权重; --resume 时忽略")
    p.add_argument("--resume", default=None, help="从本脚本存出的 checkpoint 续训")
    p.add_argument("--out-dir", default=os.path.join(_HERE, "runs/stage1"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", help="CUDA 上开混合精度")
    p.add_argument("--no-checkpointing", action="store_true",
                   help="关掉全部 gradient checkpointing(冒烟测试用, 开了只会拖慢)")
    p.add_argument("--val-every", type=int, default=1, help="每多少轮验证一次; 0 = 不验证")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--print-freq", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0, help="每轮最多多少 iter, 0 = 不限")
    p.add_argument("--val-max", type=int, default=0, help="验证最多用多少条 query, 0 = 全部")
    p.add_argument("--smoke", action="store_true", help="CPU 端到端冒烟测试(覆盖上面一批参数)")
    return p.parse_args(argv)


def apply_smoke(opt):
    """把冒烟测试需要的覆盖集中在一处, 免得散落在各分支里。"""
    opt.device = "cpu"
    opt.batch_size = 1
    opt.epochs = 1
    opt.stage = 1
    opt.num_workers = 0
    opt.resize, opt.max_size = 400, 666
    opt.amp = False
    opt.grad_accum = 1
    opt.warmup_iters = 2
    opt.no_checkpointing = True
    opt.val_every = 1
    opt.save_every = 1
    opt.print_freq = 1
    if opt.max_steps <= 0:
        opt.max_steps = 4          # 每轮 4 个 iter 足够证明管道通了
    if opt.val_max <= 0:
        opt.val_max = 4            # 验证只跑 4 条 query
    if opt.out_dir.endswith("stage1"):
        opt.out_dir = os.path.join(_HERE, "runs/smoke")
    return opt


# ================================================================ 模型

def build_and_load(opt):
    args = SLConfig.fromfile(opt.config)
    args.device = opt.device
    if opt.no_checkpointing:
        # 冒烟测试追求「快」, gradient checkpointing 只会让它更慢
        args.use_checkpoint = False
        args.use_transformer_ckpt = False
        args.ir_use_checkpoint = False

    model = build_model(args)
    model.to(opt.device)

    if opt.resume:
        ckpt = torch.load(opt.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[ckpt] 已从 {opt.resume} 恢复 (stage={ckpt.get('stage')}, "
              f"epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')})")
        return model, ckpt

    if opt.weights and os.path.exists(opt.weights):
        ckpt = torch.load(opt.weights, map_location="cpu", weights_only=True)
        res = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        print(f"[ckpt] RGB 预训练权重 {opt.weights}")
        print(f"       missing={list(res.missing_keys)[:6]} ({len(res.missing_keys)} 个)")
        print(f"       unexpected={list(res.unexpected_keys)}")
        # IR Swin 的 warm-start 由模型内部的 load_state_dict post-hook 完成, 这里无需干预
    else:
        print(f"[ckpt] ⚠️ 没有找到 {opt.weights}, 用随机初始化跑")
    return model, None


def build_optimizer(model, opt):
    """按 config 的比例缩放各分组学习率, 并拆出 decay / no-decay。"""
    base = DEFAULT_LR_GROUPS["ir"]          # --lr 对应的就是这一组
    scale = opt.lr / base
    lr_groups = {k: v * scale for k, v in DEFAULT_LR_GROUPS.items()}
    groups = model.get_param_groups(lr_groups)

    param_groups, n_decay, n_nodecay = [], 0, 0
    for g in groups:
        decay = [p for p in g["params"] if p.ndim >= 2]
        no_decay = [p for p in g["params"] if p.ndim < 2]
        # ⚠️ weight decay 必须排除 ndim<2 的参数: bias / norm 之外, fusion 的
        # beta_ir / beta_depth 也是 1 维的 —— 对它们做 weight decay 等于把融合门慢慢关死。
        if decay:
            param_groups.append({"name": g["name"], "lr": g["lr"],
                                 "weight_decay": opt.weight_decay, "params": decay})
            n_decay += len(decay)
        if no_decay:
            param_groups.append({"name": g["name"] + ":no_decay", "lr": g["lr"],
                                 "weight_decay": 0.0, "params": no_decay})
            n_nodecay += len(no_decay)

    print("[opt] 分组: " + ", ".join(f"{g['name']}@{g['lr']:.2e}" for g in param_groups))
    print(f"[opt] 张量数: decay={n_decay}, no_decay={n_nodecay}")

    if opt.optimizer == "sgd":
        return torch.optim.SGD(param_groups, lr=opt.lr, momentum=opt.momentum), param_groups
    betas = tuple(float(x) for x in opt.betas.split(","))
    return torch.optim.AdamW(param_groups, lr=opt.lr, betas=betas,
                            weight_decay=opt.weight_decay), param_groups


def build_scheduler(optimizer, opt, total_iters):
    if opt.scheduler == "none":
        return None
    min_ratio = opt.min_lr / opt.lr if opt.lr > 0 else 0.0

    def factor(it):
        if it < opt.warmup_iters:
            return (it + 1) / max(1, opt.warmup_iters)
        prog = (it - opt.warmup_iters) / max(1, total_iters - opt.warmup_iters)
        prog = min(1.0, max(0.0, prog))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ================================================================ 评估

def iou_xyxy(a, b):
    """归一化 xyxy 的 IoU(与像素空间 IoU 等价, 因为缩放一致)。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, opt, max_items=0):
    """复刻 run_grounding.py 的 top-1 指代协议, 只是输入换成三模态。

    打分口径逐字对齐: pred_logits 先 sigmoid(**模型输出的已经是 logits**),
    取真实 token 位置 arange(1, n_tok-1)(即排除 [CLS]/[SEP])的最大值作为该框的分数。
    """
    model.eval()
    ious = []
    n = 0
    for samples, targets, ir, depth in loader:
        samples = samples.to(opt.device)
        ir, depth = ir.to(opt.device), depth.to(opt.device)
        captions = [t["caption"] for t in targets]
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=opt.amp):
            # 不传 targets -> 走推理路径(out 里不会有 aux_outputs / text_token_mask),
            # 与 run_grounding.py 的调用方式一致; captions 得显式给。
            outputs = model(samples, captions=captions, ir_samples=ir, depth_samples=depth)

        logits = outputs["pred_logits"].float().sigmoid()   # [B,nq,T]
        boxes = outputs["pred_boxes"].float()               # [B,nq,4] 归一化 cxcywh

        for i, t in enumerate(targets):
            caption = t["caption"]
            n_tok = model.tokenizer(caption, return_tensors="pt")["input_ids"].shape[1]
            valid_pos = torch.arange(1, max(1, n_tok - 1))
            scores = logits[i][:, valid_pos].max(dim=1)[0]  # [nq]
            q = int(scores.argmax().item())
            cx, cy, w, h = boxes[i][q].tolist()
            pred = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

            gt_cxcywh = t["boxes"][0].tolist()              # [1,4] 归一化 cxcywh
            gx, gy, gw, gh = gt_cxcywh
            gt = [gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2]
            ious.append(iou_xyxy(pred, gt))

            n += 1
            if max_items and n >= max_items:
                break
        if max_items and n >= max_items:
            break

    model.train()
    arr = np.array(ious) if ious else np.zeros(1)
    return {
        "n": len(ious),
        "acc@0.25": float((arr >= 0.25).mean()),
        "acc@0.5": float((arr >= 0.5).mean()),
        "acc@0.75": float((arr >= 0.75).mean()),
        "mean_iou": float(arr.mean()),
        "median_iou": float(np.median(arr)),
    }


# ================================================================ checkpoint

def save_ckpt(path, model, optimizer, scaler, epoch, global_step, opt, split,
              best_metric=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "global_step": global_step,
            "stage": opt.stage,
            # 见 main 里「best_metric 继承」那段: 续训同一个 stage 时不能让 best.pth 被倒退覆盖
            "best_metric": best_metric,
            "args": vars(opt),
            "train_image_ids": split[0],
            "val_image_ids": split[1],
        },
        path,
    )


# ================================================================ 主流程

def main(argv=None):
    opt = parse_args(argv)
    if opt.smoke:
        opt = apply_smoke(opt)

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if opt.device.startswith("cuda") and not torch.cuda.is_available():
        print("[env] ⚠️ CUDA 不可用, 回退到 cpu")
        opt.device = "cpu"
    os.makedirs(opt.out_dir, exist_ok=True)

    print("=" * 78)
    print(f"stage={opt.stage}  batch_size={opt.batch_size}  epochs={opt.epochs}  lr={opt.lr:g}")
    print(f"device={opt.device}  amp={opt.amp}  accum={opt.grad_accum}  "
          f"resize={opt.resize}/{opt.max_size}  out={opt.out_dir}")
    print("=" * 78)

    # ---- 数据 ----
    items = load_items(opt.data_dir)
    train_ids, val_ids = split_image_ids(items, opt.val_ratio, opt.seed)
    train_set = MultiModalReferDataset(opt.data_dir, image_ids=train_ids, train=True,
                                       resize=opt.resize, max_size=opt.max_size,
                                       hflip_p=opt.hflip_p, items=items)
    val_set = MultiModalReferDataset(opt.data_dir, image_ids=val_ids, train=False,
                                     resize=opt.resize, max_size=opt.max_size, items=items)
    print(f"[data] {opt.data_dir}")
    print(f"[data] train {len(train_ids)} 图 / {len(train_set)} query   "
          f"val {len(val_ids)} 图 / {len(val_set)} query")

    common = dict(num_workers=opt.num_workers, collate_fn=collate_fn,
                  pin_memory=opt.device.startswith("cuda"))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                              drop_last=False, **common)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, **common)

    # ---- 模型 / 冻结策略 ----
    model, resume_ckpt = build_and_load(opt)
    info = model.set_train_stage(opt.stage)
    print(f"[stage] {info['stage']}: 可训练 {info['num_trainable_tensors']} / "
          f"{info['num_total_tensors']} 个张量")

    if opt.smoke and model.modality_augment is not None:
        # 冒烟测试要可复现, 关掉所有随机增强
        model.modality_augment.p_ir_drop = 0.0
        model.modality_augment.p_depth_drop = 0.0
        model.modality_augment.p_rgb_only = 0.0
        model.modality_augment.p_ir_degrade = 0.0
        model.modality_augment.p_depth_hole = 0.0
        print("[stage] smoke: modality_augment 全部关闭")

    criterion = build_criterion().to(opt.device)
    optimizer, param_groups = build_optimizer(model, opt)

    iters_per_epoch = len(train_loader)
    if opt.max_steps:
        iters_per_epoch = min(iters_per_epoch, opt.max_steps)
    total_iters = max(1, iters_per_epoch * opt.epochs)
    scheduler = build_scheduler(optimizer, opt, total_iters)
    scaler = torch.amp.GradScaler("cuda", enabled=opt.amp)

    start_epoch, global_step = 0, 0
    if resume_ckpt:
        if resume_ckpt.get("optimizer"):
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if resume_ckpt.get("scaler") and opt.amp:
            scaler.load_state_dict(resume_ckpt["scaler"])
        start_epoch = int(resume_ckpt.get("epoch", 0))
        global_step = int(resume_ckpt.get("global_step", 0))
        if scheduler:
            for _ in range(global_step):
                scheduler.step()
        print(f"[ckpt] 续训: 从 epoch {start_epoch} / step {global_step} 开始")

    split = (train_ids, val_ids)
    # best.pth 的判据要能跨续训延续, 否则「同一个 stage 断点续训」会把已经更好的
    # best.pth 用更差的权重覆盖掉(实测: 0.4159 被 0.4158 覆盖)。
    # 只在 **同一个 stage** 内继承: 换 stage 时指标口径不同(比如 stage 2 起点就是
    # stage 1 的终点), 继承反而会让新 stage 一直存不下 best.pth。
    best_metric = -1.0
    if resume_ckpt and resume_ckpt.get("stage") == opt.stage:
        best_metric = float(resume_ckpt.get("best_metric") or -1.0)
        if best_metric >= 0:
            print(f"[ckpt] 继承 best_metric={best_metric:.4f} (同 stage 续训)")
    model.train()

    for epoch in range(start_epoch, opt.epochs):
        t0 = time.perf_counter()
        running, n_run = {}, 0
        optimizer.zero_grad(set_to_none=True)

        for it, (samples, targets, ir, depth) in enumerate(train_loader):
            if opt.max_steps and it >= opt.max_steps:
                break
            # NestedTensor.to() 不接受 non_blocking 参数(它是 namedtuple 的自定义实现)
            samples = samples.to(opt.device)
            ir = ir.to(opt.device, non_blocking=True)
            depth = depth.to(opt.device, non_blocking=True)
            targets = [
                {"caption": t["caption"], "boxes": t["boxes"].to(opt.device)}
                for t in targets
            ]

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=opt.amp):
                # captions 由 forward 自己从 targets 里取, 不要再传 captions=
                out = model(samples, targets, ir_samples=ir, depth_samples=depth)
            # criterion 在 autocast 之外调用, 内部再把张量转 fp32(见 mm_loss.py)
            loss_dict = criterion(out, targets)
            loss = sum(loss_dict.values()) / opt.grad_accum

            scaler.scale(loss).backward()

            if (it + 1) % opt.grad_accum == 0:
                if opt.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]],
                        opt.grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                global_step += 1

            with torch.no_grad():
                for k, v in loss_dict.items():
                    running[k] = running.get(k, 0.0) + float(v.item())
            n_run += 1

            if (it + 1) % opt.print_freq == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                avg = running.get("loss_ce", 0.0) / n_run
                print(f"  e{epoch} [{it + 1}/{iters_per_epoch}] "
                      f"ce={avg:.4f} lr={cur_lr:.2e} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

        dt = time.perf_counter() - t0
        head = " ".join(f"{k}={v / max(1, n_run):.4f}"
                        for k, v in list(running.items())[:3])
        print(f"[epoch {epoch}] {n_run} iters, {dt:.0f}s, {head}", flush=True)

        # ---- 验证 ----
        metrics = None
        if opt.val_every and (epoch + 1) % opt.val_every == 0 and len(val_set):
            metrics = evaluate(model, val_loader, opt, max_items=opt.val_max)
            print(f"[val   {epoch}] n={metrics['n']} "
                  f"Acc@0.25={metrics['acc@0.25']:.4f} Acc@0.5={metrics['acc@0.5']:.4f} "
                  f"Acc@0.75={metrics['acc@0.75']:.4f} meanIoU={metrics['mean_iou']:.4f}")

        # ---- 存档 ----
        if opt.save_every and (epoch + 1) % opt.save_every == 0:
            save_ckpt(os.path.join(opt.out_dir, "last.pth"),
                      model, optimizer, scaler, epoch + 1, global_step, opt, split,
                      best_metric=best_metric)
        if metrics is not None:
            cur = metrics["mean_iou"]
            if cur > best_metric:
                best_metric = cur
                save_ckpt(os.path.join(opt.out_dir, "best.pth"),
                          model, optimizer, scaler, epoch + 1, global_step, opt, split,
                          best_metric=best_metric)
                print(f"[save  {epoch}] best.pth (meanIoU={cur:.4f})")

    # ---- 收尾 ----
    save_ckpt(os.path.join(opt.out_dir, "last.pth"),
              model, optimizer, scaler, opt.epochs, global_step, opt, split,
              best_metric=best_metric)
    result = {"stage": opt.stage, "epochs": opt.epochs, "global_step": global_step,
              "best_mean_iou": best_metric, "out_dir": opt.out_dir}
    with open(os.path.join(opt.out_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n完成: {result}")
    if opt.smoke:
        print("SMOKE OK")


EXAMPLES = """
# ---- 本地冒烟(CPU, 1~2 分钟) ----
python train_multimodal.py --smoke

# ---- 本机 5060 8G: 短程试跑, 确认 loss 在降 ----
# 实测显存: 这个组合 3908 MiB; 直接 --batch-size 2 也跑得动(6134 MiB)但余量只剩 2 GB
python train_multimodal.py --stage 1 --batch-size 1 --epochs 1 --grad-accum 4 \\
    --max-steps 50 --device cuda --amp --out-dir runs/probe

# ---- 云端 4090 24G: 三段式(每段一条命令, 用 --resume 串起来) ----
# 显存实测外推: 固定 ~1.97 GB + 每样本 ~2.08 GB(fp16/800x1333)
#   -> bs=4 约 10.3 GB, bs=8 约 18.6 GB。bs=4 时 1800 query -> 450 iter/epoch
python train_multimodal.py --stage 1 --batch-size 4 --epochs 15 --lr 1e-4 --amp \\
    --out-dir runs/stage1
python train_multimodal.py --stage 2 --resume runs/stage1/best.pth --batch-size 4 \\
    --epochs 8 --lr 5e-5 --amp --out-dir runs/stage2
python train_multimodal.py --stage 3 --resume runs/stage2/best.pth --batch-size 4 \\
    --epochs 5 --lr 2e-5 --amp --out-dir runs/stage3

# ---- 复现论文里的 Stage-0 对照(只训检测头) ----
python train_multimodal.py --stage 0 --batch-size 4 --epochs 5 --lr 5e-5 --amp \\
    --out-dir runs/stage0

# ---- 中途被打断: 同一个 --out-dir + --resume 即可接着跑 ----
python train_multimodal.py --stage 1 --resume runs/stage1/last.pth --batch-size 4 \\
    --epochs 15 --lr 1e-4 --amp --out-dir runs/stage1
"""


if __name__ == "__main__":
    main()
