#!/usr/bin/env python3
"""GroundingDINO 指代表达理解 —— 通用命令行评估 / 推理工具。

数据集格式(TrainSet.):`queries/queries.json` 内每条 query 自带归一化 GT bbox
`bbox = [x1, y1, x2, y2]`(xyxy,数值 ∈ [0,1]);图片路径相对于 data_dir
(如 `"Images/visible/001.png"`),三种模态对齐:visible(RGB)/ infrared / depth。

设计要点(对应可扩展接口):
  - `--modality`     选 visible / infrared / depth 单模态输入 → 为后续「多模态融合」留接口;
  - `--start`/`--num-images` 按「图」粒度切片 → 分批续跑,跑出剩余图片;
  - `--box-threshold`/`--text-threshold`/`--device`/`--config`/`--weights` → 常规超参数;
  - `load_dataset` / `load_model` / `predict_top1` / `iou` 拆成独立函数,便于被 import 复用。

用法示例:
  python run_grounding.py                                   # 全量,RGB 模态,默认 TrainSet.
  python run_grounding.py --num-images 50                   # 只跑前 50 张图
  python run_grounding.py --start 50 --num-images 50        # 从第 50 张图续跑 50 张
  python run_grounding.py --modality infrared --box-threshold 0.3
"""
import argparse
import json
import os
import sys
import time

# ---- 代理环境变量必须在 import torch / transformers 之前设好 ----
# 本机的 clash/v2ray 混合端口不固定, 且云端根本没有这个代理。原来无条件写死
# 127.0.0.1:7892 会让「有网但没开代理」和「云端」两种情况都把 HF 请求送进黑洞。
# 改成条件探测: 端口真在监听才设, 否则保持环境原样。
_PROXY_PORT = 7892
_PROXY = f"http://127.0.0.1:{_PROXY_PORT}"


def _proxy_alive(port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


if _proxy_alive(_PROXY_PORT):
    os.environ["ALL_PROXY"] = _PROXY
    os.environ["all_proxy"] = _PROXY
    print(f"[env] 检测到本地代理 {_PROXY}, 已启用")
else:
    os.environ.pop("ALL_PROXY", None)   # 清掉外部可能继承进来的失效代理
    os.environ.pop("all_proxy", None)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for _v in ("no_proxy", "NO_PROXY"):
    if "hf-mirror.com" not in os.environ.get(_v, ""):
        os.environ[_v] = "hf-mirror.com," + os.environ.get(_v, "")

import numpy as np
import torch

# ---------------- 默认路径(本机固定项,可被命令行覆盖)----------------
REPO = "./GroundingDINO"
DEFAULT_DATA = "../TrainSet."
DEFAULT_CFG = f"{REPO}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_WEIGHTS = "./weights/groundingdino_swint_ogc.pth"

sys.path.insert(0, REPO)

from groundingdino.models import build_model  # noqa: E402
from groundingdino.util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from demo.inference_on_a_image import load_image  # noqa: E402

# 可选的输入模态 —— 这是后续多模态融合的扩展点(现在逐模态单跑)。
MODALITIES = ("visible", "infrared", "depth")


# ================================================================ 数据集

def load_dataset(data_dir: str, modality: str = "visible"):
    """读取 queries.json,返回按 (img_id, qid) 排序的 item 列表。

    每个 item:
      qid      查询 id
      query    自然语言指代句
      img_id   图片 id(如 "001")
      img_path 该模态的绝对图片路径
      bbox     归一化 xyxy [x1,y1,x2,y2] 或 None(无 GT 时仅做推理 dump)
    """
    if modality not in MODALITIES:
        raise ValueError(f"未知模态 {modality!r},可选 {MODALITIES}")

    qfile = os.path.join(data_dir, "queries", "queries.json")
    queries = json.load(open(qfile))

    items = []
    for qid, v in queries.items():
        rel = v[modality]  # e.g. "Images/visible/001.png"
        img_path = os.path.join(data_dir, rel)
        if not os.path.exists(img_path):
            # 兼容:有的数据集路径不挂 "Images" 前缀,退化用 basename 找
            alt = os.path.join(data_dir, modality, os.path.basename(rel))
            img_path = alt if os.path.exists(alt) else img_path
        items.append(
            {
                "qid": qid,
                "query": v["query"],
                "img_id": os.path.splitext(os.path.basename(rel))[0],
                "img_path": img_path,
                "bbox": v.get("bbox"),  # 归一化 xyxy 或 None
            }
        )

    items.sort(key=lambda it: (it["img_id"], it["qid"]))
    return items


def slice_by_images(items, start: int, num_images: int):
    """按「图」粒度切片:先对去重排序后的 img_id 取 [start, start+num),再回选 item。

    num_images <= 0 表示不限(跑全部)。start 用于续跑剩余图片。
    """
    img_ids = []
    for it in items:
        if not img_ids or img_ids[-1] != it["img_id"]:
            img_ids.append(it["img_id"])

    selected = set(img_ids[start:]) if num_images <= 0 else set(img_ids[start:start + num_images])
    return [it for it in items if it["img_id"] in selected]


# ================================================================ 模型

def load_model(cfg_path: str, weights_path: str, device: str):
    args = SLConfig.fromfile(cfg_path)
    args.device = device
    model = build_model(args)
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval().to(device)
    return model


def predict_top1(model, image_tensor, caption: str, box_threshold: float = 0.0):
    """返回 (归一化 xyxy, score);若最高分低于 box_threshold 返回 (None, score)。

    单 phrase 模式:整句 query 当作一个指代短语,分数 = 该框对 query 词 token
    (排除 [CLS]/[SEP])的最大 logit,与官方 demo 的过滤逻辑一致。
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4) cxcywh 归一化

    tokenized = model.tokenizer(caption, return_tensors="pt")
    n_tok = tokenized["input_ids"].shape[1]
    valid_pos = torch.arange(1, max(1, n_tok - 1))
    scores = logits[:, valid_pos].max(dim=1)[0]  # (nq,)

    i = int(scores.argmax().item())
    best = float(scores[i].item())
    if best < box_threshold:
        return None, best

    cx, cy, w, h = boxes[i].tolist()
    xyxy = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    return xyxy, best


# ================================================================ 指标

def iou(a, b):
    """归一化 xyxy 的 IoU(与像素空间 IoU 等价,因缩放一致)。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ================================================================ 主流程

def parse_args():
    p = argparse.ArgumentParser(
        description="GroundingDINO 指代表达理解通用评估/推理工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据 / 数量控制
    p.add_argument("--data-dir", default=DEFAULT_DATA, help="数据集根目录(含 queries/queries.json)")
    p.add_argument("--start", type=int, default=0, help="从第几张图开始(按 img_id 排序,续跑用)")
    p.add_argument("--num-images", type=int, default=0, help="跑多少张图,0=全部;配合 --start 分批续跑")
    # 模型 / 设备
    p.add_argument("--modality", default="visible", choices=MODALITIES, help="输入模态")
    p.add_argument("--config", default=DEFAULT_CFG, help="GroundingDINO config .py")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="checkpoint .pth")
    p.add_argument("--device", default="cuda", help="推理设备 cuda/cpu")
    # 常规超参数
    p.add_argument("--box-threshold", type=float, default=0.0,
                   help="低于此分数的 top-1 框判为无检测;0.0=REC 模式总返回 top-1")
    p.add_argument("--text-threshold", type=float, default=0.25,
                   help="(预留)多 phrase 模式的文本匹配阈值,当前单 phrase 模式未用")
    # 输出
    p.add_argument("--out", default=None, help="结果 JSON 输出路径,默认写到 data-dir 下")
    return p.parse_args()


def main():
    args = parse_args()

    items = load_dataset(args.data_dir, args.modality)
    items = slice_by_images(items, args.start, args.num_images)
    n_imgs = len({it["img_id"] for it in items})
    print(f"数据集: {args.data_dir}")
    print(f"模态  : {args.modality}   设备: {args.device}")
    print(f"待跑  : {n_imgs} 张图 / {len(items)} 条 query (start={args.start}, num_images={args.num_images})")
    if not items:
        print("没有要跑的数据,退出")
        return

    model = load_model(args.config, args.weights, args.device)
    print("模型就绪\n", flush=True)

    # 同一张图多个 query,缓存图片张量避免重复读盘/变换
    img_cache = {}
    results = []
    n_detected = 0
    t0 = time.perf_counter()

    for idx, it in enumerate(items):
        key = it["img_path"]
        if key not in img_cache:
            if not os.path.exists(key):
                print(f"  ⚠️ 图片不存在,跳过: {key}")
                continue
            pil, tensor = load_image(key)
            img_cache[key] = (pil.size, tensor.to(args.device))
        size, tensor = img_cache[key]
        W, H = size

        pred_norm, score = predict_top1(model, tensor, it["query"], args.box_threshold)
        if pred_norm is not None:
            n_detected += 1
            iou_val = iou(pred_norm, it["bbox"]) if it["bbox"] else None
        else:
            iou_val = 0.0 if it["bbox"] else None

        results.append(
            {
                "qid": it["qid"],
                "img_id": it["img_id"],
                "query": it["query"],
                "img_size": [W, H],
                "gt_box": it["bbox"],  # 归一化 xyxy
                "pred_box": pred_norm,  # 归一化 xyxy 或 None
                "score": score,
                "iou": iou_val,
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"  ...{idx + 1}/{len(items)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    dt = time.perf_counter() - t0

    # ---- 汇总 ----
    with_gt = [r for r in results if r["gt_box"] is not None]
    summary = {
        "num_images": n_imgs,
        "num_queries": len(results),
        "num_detected": n_detected,
        "detect_rate": n_detected / len(results) if results else 0.0,
        "elapsed_s": round(dt, 1),
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

    out = args.out or os.path.join(args.data_dir, f"grounding_{args.modality}_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "data_dir": args.data_dir,
                    "modality": args.modality,
                    "start": args.start,
                    "num_images": args.num_images,
                    "box_threshold": args.box_threshold,
                    "text_threshold": args.text_threshold,
                    "weights": args.weights,
                    "device": args.device,
                },
                "summary": summary,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n结果已存: {out}")


if __name__ == "__main__":
    main()
