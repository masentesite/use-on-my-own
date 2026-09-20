#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三模态训练**函数**的冒烟测试 —— 验证 `train_multimodal(opt)` 的可编程契约。

为什么要单独一个文件, 而不是并进 `test_training.py`:
  那个文件是「纯 CPU 的 `mm_data`/`mm_loss` 数据与损失测试」, 秒级完成; 这里的每条用例都要
  真的建模型(BERT + 双 Swin)并走完 forward/backward/step, 单条约 1 分钟。混在一起会毁掉
  那个套件的「快」, 所以各自独立。

覆盖三件事(前两件 `--smoke` 做不到):
  1. **返回值契约**: `train_multimodal(opt)` 直接返回 train_summary.json 里那份 dict,
     上层(§11 的消融矩阵 V2-E0..E6)不必起子进程、不必解析文件就能读结果。
  2. **两个 Fusion 版本都能编程调用**: v2 主方案与 v1(V2-E4 对照)各跑一条。
  3. **`--seed` 真的可复现**: 同 seed 两次运行的结果逐位一致。
     这一条是回归护栏 —— 2026-09-20 实测发现训练脚本只播了 torch/np 的种子, 而数据集水平
     翻转用的是 stdlib `random.random()`, 于是同 seed 会跑出 best_mean_iou 0.3069 / 0.3735
     两个值。修法是 `train_multimodal` 里补一句 `random.seed(opt.seed)`。

用法(仓库根目录, 需要在有数据集的机器上跑):
    .venv/bin/python test_train_mm.py              # 全部 3 条, 约 3~4 分钟(CPU)
    .venv/bin/python test_train_mm.py --only repro # 只跑可复现性那一条
"""

import argparse
import os
import shutil
import sys
import traceback

# ---- 代理环境变量必须在 import torch / transformers 之前设好(与 run_grounding.py 一致) ----
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for _v in ("no_proxy", "NO_PROXY"):
    if "hf-mirror.com" not in os.environ.get(_v, ""):
        os.environ[_v] = "hf-mirror.com," + os.environ.get(_v, "")

import torch  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from train_multimodal import (  # noqa: E402
    DEFAULT_CFG,
    apply_smoke,
    parse_args,
    train_multimodal,
)

DEFAULT_DATA = os.path.join(REPO, "../TrainSet.")
#: 测试产物一律写在这里, 绝不碰 runs/smoke(run_grounding_mm.py 的默认权重)
TEST_RUNS = os.path.join(REPO, "runs/_test_train_mm")


# ======================================================================
#  基础设施(与 test_training.py 同款)
# ======================================================================
class Failure(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Failure(msg)


class Ctx(dict):
    __getattr__ = dict.get


def smoke_opt(data_dir, name, config=None):
    """一份冒烟参数; `out_dir` 必须在 `apply_smoke` **之后**改 —— 它会重写这个键。

    另外把 `print_freq` 置 0(函数调用时不该往 stdout 刷逐 iter 日志)。

    ⚠️ 这里**刻意不设** `opt.smoke = True`。`apply_smoke()` 只覆盖参数, 而
    `train_multimodal` 另有一处 `if opt.smoke: 关掉 modality_augment`
    (train_multimodal.py:631)。也就是说:
        CLI `--smoke`        : 参数覆盖 + **无**模态增强  -> best_mean_iou ≈ 0.3074
        本函数的调用          : 参数覆盖 + **有**模态增强  -> best_mean_iou ≈ 0.3683
    两者都逐位可复现(下面第 3 条验证的就是后者)。这里留增强**开着**, 因为那才是真实训练
    路径: modality dropout / IR 退化 / depth 挖洞自己也要吃 RNG, 关掉就少测一半随机性。
    """
    opt = apply_smoke(parse_args([]))
    opt.data_dir = data_dir
    opt.out_dir = os.path.join(TEST_RUNS, name)
    opt.print_freq = 0
    if config:
        opt.config = config
    return opt


def v1_config(tmp_dir):
    """造一份 `fusion_type = "language_guided_residual"` 的 config(V2-E4 对照档)。

    仓库里只留了 v2 那一份 config(V2 §11 消融矩阵用 `fusion_type` 切换), 所以这里按行改写
    生成临时副本 —— 与 `--config` 手动指一份的做法等价, 只是自动化了。
    """
    os.makedirs(tmp_dir, exist_ok=True)
    out = os.path.join(tmp_dir, "cfg_v1.py")
    src = open(DEFAULT_CFG, encoding="utf-8").read()
    lines = [
        'fusion_type = "language_guided_residual"' if ln.startswith("fusion_type =") else ln
        for ln in src.splitlines()
    ]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out


def check_contract(result, opt, expect_fusion, tag):
    """`train_multimodal` 的返回值契约 + 落盘产物。"""
    check(isinstance(result, dict), f"{tag}: 返回值不是 dict, 而是 {type(result)}")
    for k in ("stage", "epochs", "global_step", "best_mean_iou", "out_dir",
              "fusion_type", "gate_stats", "modality_eval", "eval_gate_stats",
              "missing_modality_delta"):
        check(k in result, f"{tag}: result 缺键 {k}")
    check(result["fusion_type"] == expect_fusion,
          f"{tag}: fusion_type 应为 {expect_fusion!r}, 实际 {result['fusion_type']!r}")
    check(result["global_step"] > 0, f"{tag}: global_step 没动({result['global_step']})")

    # V2 §7.4 第 3 条: 四种模态组合各评一遍; §9: 三模态档相对 RGB-only 的增益
    check(set(result["modality_eval"]) == {"rgb", "rgb+ir", "rgb+depth", "rgb+ir+depth"},
          f"{tag}: modality_eval 的组合不全: {sorted(result['modality_eval'])}")
    check(set(result["missing_modality_delta"]) == {"ir", "depth", "both"},
          f"{tag}: missing_modality_delta 缺档: {sorted(result['missing_modality_delta'])}")
    check(len(result["eval_gate_stats"]) > 0, f"{tag}: eval_gate_stats 为空")

    # 落盘: last.pth / best.pth / train_summary.json, 且 summary 与返回值一致
    for fn in ("last.pth", "best.pth", "train_summary.json"):
        p = os.path.join(opt.out_dir, fn)
        check(os.path.exists(p), f"{tag}: 缺文件 {p}")
    import json

    disk = json.load(open(os.path.join(opt.out_dir, "train_summary.json"), encoding="utf-8"))
    check(disk["best_mean_iou"] == result["best_mean_iou"],
          f"{tag}: 返回的 best_mean_iou 与落盘的 summary 不一致 "
          f"({result['best_mean_iou']} vs {disk['best_mean_iou']})")
    check(disk["fusion_type"] == result["fusion_type"], f"{tag}: summary 里的 fusion_type 不一致")


# ======================================================================
#  测试
# ======================================================================
def test_function_v2(ctx):
    """v2 主方案: 函数式调用一次, 返回值与产物都要对。"""
    opt = smoke_opt(ctx.data_dir, "v2")
    result = train_multimodal(opt)
    check_contract(result, opt, "local_cross_attention_spatial_gate", "v2")

    # V2 §9 的五个诊断量必须都有, 且辅助通道不能被节流成 0 —— 这正是 V2 相对 v1 的立身之本。
    # 注意冒烟用的是随机初始化的融合(只加载了 RGB 预训练权重), 所以这里断言的是
    # 「gate 机制在工作」而不是「融合有效」: 数值应当接近 init 的 0.786/0.107/0.107。
    gs = result["gate_stats"]
    for k in ("gate_rgb_mean", "gate_ir_mean", "gate_depth_mean",
              "aux_ir_ratio", "aux_depth_ratio"):
        check(k in gs, f"v2: gate_stats 缺 {k}(V2 §9 要求这五个诊断量)")
    check(gs["gate_ir_mean"] > 1e-3,
          f"v2: gate_ir_mean={gs['gate_ir_mean']:.3e} ≈ 0 ⇒ 辅助模态的入口被堵死了")
    check(gs["gate_depth_mean"] > 1e-3,
          f"v2: gate_depth_mean={gs['gate_depth_mean']:.3e} ≈ 0 ⇒ 辅助模态的入口被堵死了")
    check(gs["aux_ir_ratio"] > 1e-3 and gs["aux_depth_ratio"] > 1e-3,
          f"v2: aux 强度比塌到 0(ir={gs['aux_ir_ratio']:.3e}, "
          f"depth={gs['aux_depth_ratio']:.3e}) ⇒ 候选被 adapter/attention 压没了")
    check(abs(gs["gate_rgb_mean"] + gs["gate_ir_mean"] + gs["gate_depth_mean"] - 1.0) < 5e-3,
          f"v2: gate 三个通道之和应为 1(softmax), 实际 "
          f"{gs['gate_rgb_mean'] + gs['gate_ir_mean'] + gs['gate_depth_mean']:.4f}")


def test_function_v1(ctx):
    """v1 Fusion(V2-E4 对照)也必须能被同一个函数跑起来, 只是诊断量换成逐 level 的 beta。"""
    cfg = v1_config(ctx.tmp_dir)
    opt = smoke_opt(ctx.data_dir, "v1", config=cfg)
    result = train_multimodal(opt)
    check_contract(result, opt, "language_guided_residual", "v1")

    gs = result["gate_stats"]
    check(any(k.startswith("beta_ir/") for k in gs),
          f"v1: gate_stats 里应有逐 level 的 beta_ir/*, 实际 {sorted(gs)}")
    check(not any(k.startswith("gate_rgb_mean") for k in gs),
          f"v1: 不该出现 v2 的 gate_*_mean 键, 实际 {sorted(gs)}")


def test_reproducible(ctx):
    """同 `--seed` 两次运行必须逐位一致 —— 训练脚本的种子要覆盖到数据增强。

    回归点: 数据集的水平翻转走 stdlib `random.random()`, 只播 torch/np 的种子管不到它。
    """
    keys = ("global_step", "best_mean_iou", "gate_stats", "eval_gate_stats",
            "modality_eval", "missing_modality_delta")
    a = train_multimodal(smoke_opt(ctx.data_dir, "rep_a"))
    b = train_multimodal(smoke_opt(ctx.data_dir, "rep_b"))
    diff = [k for k in keys if a[k] != b[k]]
    check(not diff,
          f"同 seed 两次运行结果不一致: {diff}\n"
          f"  best_mean_iou: {a['best_mean_iou']} vs {b['best_mean_iou']}\n"
          f"  多半是某个增强用了 stdlib random 而 train_multimodal 没 random.seed(opt.seed)")


# ======================================================================
#  驱动
# ======================================================================
# 名字里带 [v2] / [v1] / [repro] 这种 ASCII 标记, 是为了让 `--only` 能用纯 ASCII 子串选中
# (中文名字没法从命令行敲)。这三个标记同时也是稳定的标识, 改描述时别把它们删了。
TESTS = [
    ("1. [v2] 训练函数: 主方案的返回值与落盘契约", test_function_v2),
    ("2. [v1] 训练函数: v1(V2-E4)也走同一个函数", test_function_v1),
    ("3. [repro] 训练函数: 同 seed 逐位可复现", test_reproducible),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--only", default=None,
                    help="按名字子串只跑一条: v2 / v1 / repro, 默认全跑")
    ap.add_argument("--keep", action="store_true", help="保留测试产物目录")
    args = ap.parse_args()

    if not args.keep and os.path.isdir(TEST_RUNS):
        shutil.rmtree(TEST_RUNS)          # 每次从干净目录开始, 免得旧的 best.pth 干扰
    os.makedirs(TEST_RUNS, exist_ok=True)

    tests = [(n, f) for n, f in TESTS if not args.only or args.only in n]
    if not tests:
        raise SystemExit(f"--only={args.only} 没匹配到任何用例: {[n for n, _ in TESTS]}")

    print("=" * 78)
    print(f"data_dir={os.path.abspath(args.data_dir)}   cuda={torch.cuda.is_available()}")
    print(f"产物目录={TEST_RUNS}   (冒烟强制 CPU —— 见 apply_smoke)")
    print("=" * 78)

    ctx = Ctx(data_dir=args.data_dir, tmp_dir=os.path.join(TEST_RUNS, "_cfg"),
              runs=TEST_RUNS)
    passed, failed = 0, []
    for name, fn in tests:
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
    print(f"通过 {passed} / {len(tests)}")
    for n in failed:
        print(f"  ✗ {n}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
