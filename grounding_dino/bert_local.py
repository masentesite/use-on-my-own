"""把 BERT 变成项目里的本地资产 —— 让 `from_pretrained` 彻底不碰 HuggingFace。

## 为什么需要它

GroundingDINO 建模型时会执行 `BertModel.from_pretrained("bert-base-uncased")`,
这句话会**联网**去 HF 拉 420MB。可它拉来的权重在几十行之后就被
`weights/groundingdino_swint_ogc.pth` 里的 `module.bert.*`(**199 个张量**:
embeddings + 12 层 encoder + pooler + position_ids)**整个覆盖**掉 —— 白拉。

HF 那边真正不可替代的只有 4 个小文件,共 **696KB**:

    config.json            570 B
    tokenizer_config.json   48 B
    vocab.txt            231 KB   ← 分词器要的
    tokenizer.json       466 KB

所以本模块干的事是:从**本来就已经在项目里的 .pth** 拆出 BERT 权重,
配上那 4 个小文件,凑成一个 `from_pretrained` 认的完整目录:

    weights/bert-base-uncased/
      ├── config.json            570 B   ← 从 HF 缓存拷(只拷一次)
      ├── tokenizer_config.json   48 B   ← 同上
      ├── vocab.txt            231 KB    ← 同上
      ├── tokenizer.json       466 KB    ← 同上
      └── model.safetensors    438 MB    ← 从 .pth 拆, 不占上传流量

之后 `resolve()` 会把 config 里的 `text_encoder_type` 指到这个目录,于是
**HF、代理、hf-mirror、Xet 存储全都与训练无关了** —— 断网也能跑,
换新实例只要 .pth 在就只差那 696KB。

目录缺失时 `resolve()` **直接报错**,不会回落到联网(见它的 docstring)。

## 用法

    python bert_local.py                     # 造目录(幂等, 缺什么补什么)
    python bert_local.py --force             # 重拆权重
    python bert_local.py --small-files-from /path/to/dir   # 4 个小文件不从 HF 缓存找

拆出来的权重与 .pth 逐位一致(2026-09-19 本机实测:word_embeddings / pooler 均 True)。
"""

import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: 造出来的本地 BERT 目录。`resolve()` 只认这个路径。
DEFAULT_DIR = os.path.join(HERE, "weights", "bert-base-uncased")

#: 权重来源。必须含 `module.bert.*`(官方 RGB 预训练权重就是这个形状)。
DEFAULT_CKPT = os.path.join(HERE, "weights", "groundingdino_swint_ogc.pth")

#: 本地 HF 缓存, 4 个小文件的默认来源。
_HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/*")

#: from_pretrained 认的目录里必须有的小文件(不含权重)。
_SMALL_FILES = ("config.json", "tokenizer_config.json", "vocab.txt", "tokenizer.json")

_WEIGHT_FILE = "model.safetensors"


def local_dir(path: str = DEFAULT_DIR) -> str | None:
    """目录可用就返回路径,否则 None。

    「可用」= 权重文件在 **且** 4 个小文件齐 —— 少一个都算没造好,宁可回落到联网,
    也不要用一个残缺目录把报错推到模型构造深处。
    """
    if not os.path.isfile(os.path.join(path, _WEIGHT_FILE)):
        return None
    if not all(os.path.isfile(os.path.join(path, f)) for f in _SMALL_FILES):
        return None
    return path


def resolve(
    text_encoder_type: str,
    path: str = DEFAULT_DIR,
    *,
    strict: bool = True,
    verbose: bool = True,
) -> str:
    """把 `bert-base-uncased` 换成本地目录;目录缺了就**报错**,不偷偷联网。

    在 `SLConfig.fromfile(...)` 之后、`build_model(...)` 之前调用一次即可:

        args.text_encoder_type = bert_local.resolve(args.text_encoder_type)

    `strict=True`(默认)是有意的:本项目的既定方针是**永不从 HF 下载**,所以目录缺失时
    宁可当场停下并告诉你怎么补,也不要回落到联网 —— 那种「悄悄又去拉 420MB」正是
    2026-09-19 在云端折腾半天的根源。真要临时走网络,传 `strict=False`。
    """
    if text_encoder_type not in ("bert-base-uncased", "roberta-base"):
        return text_encoder_type  # 已经是路径/别的模型, 不插手

    hit = local_dir(path)
    if hit is not None:
        if verbose:
            print(f"[bert] 使用本地 BERT 目录, 不联网: {hit}")
        return hit

    if not strict:
        if verbose:
            print(f"[bert] ⚠️ 无本地目录({path}), 按 strict=False 回落到联网拉 {text_encoder_type}")
        return text_encoder_type

    raise SystemExit(
        f"本地 BERT 目录不可用: {path}\n"
        f"  本项目不走 HuggingFace, 所以这里必须是一个完整的本地目录。\n"
        f"  修法(一条命令):\n"
        f"      python {os.path.basename(__file__)}\n"
        f"  它从 {os.path.basename(DEFAULT_CKPT)} 拆出 BERT 权重, 再把 config/分词器补齐。\n"
        f"  如果它抱怨 4 个小文件找不到(极简环境), 从有缓存的机器上 scp 过来即可:\n"
        f"      4 个文件共 696KB —— config.json, tokenizer_config.json, vocab.txt, tokenizer.json"
    )


def _copy_small_files(out_dir: str, src: str | None) -> list[str]:
    """把 4 个小文件补齐,返回这次实际拷了哪些。已存在的不动。"""
    missing = [f for f in _SMALL_FILES if not os.path.isfile(os.path.join(out_dir, f))]
    if not missing:
        return []
    if src is None:
        cands = sorted(glob.glob(_HF_CACHE))
        if not cands:
            raise SystemExit(
                f"找不到 {_SMALL_FILES} 的来源。\n"
                f"  HF 缓存里没有({_HF_CACHE}),请二选一:\n"
                f"    1) 从有缓存的机器上把这 4 个文件 scp 到 {out_dir}/, 再重跑本脚本;\n"
                f"    2) 用 --small-files-from <含这 4 个文件的目录> 指定来源。"
            )
        src = cands[-1]

    for f in missing:
        s = os.path.join(src, f)
        if not os.path.isfile(s):
            raise SystemExit(f"{src} 里没有 {f} —— 4 个小文件必须齐全(见模块文档串)")
        shutil.copy(s, os.path.join(out_dir, f))  # copy 会跟随软链, 正好把 HF 缓存里的软链解成实体
    return missing


def _extract_bert(ckpt_path: str):
    """从 checkpoint 里取出 BERT 部分, 键名去掉 `module.` 前缀。

    去掉 `bert.embeddings.position_ids`: 它是 `arange` 生成的固定 buffer, 不是学出来的,
    新版 transformers 也不把它注册成 buffer(带着它加载会被报成 unexpected key)。
    """
    import torch

    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"找不到权重来源 {ckpt_path} —— 先用 --ckpt 指定")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    state = blob.get("model", blob)

    for prefix in ("module.bert.", "bert."):
        picked = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if picked:
            break
    else:
        raise SystemExit(
            f"{ckpt_path} 里没有 bert.* 参数 —— 这不是 GroundingDINO 的 RGB 预训练权重?"
        )

    dropped = [k for k in picked if k.endswith("position_ids")]
    picked = {k: v.contiguous().clone() for k, v in picked.items() if k not in dropped}
    n_param = sum(v.numel() for v in picked.values())
    print(f"[bert] 从 {os.path.basename(ckpt_path)} 拆出 {len(picked)} 个张量 "
          f"({n_param/1e6:.1f}M 参数), 丢掉 {len(dropped)} 个固定 buffer")
    return picked


def build(
    out_dir: str = DEFAULT_DIR,
    ckpt: str = DEFAULT_CKPT,
    small_files_from: str | None = None,
    *,
    force: bool = False,
) -> str:
    """把本地 BERT 目录造出来(或补齐)。返回目录路径。"""
    os.makedirs(out_dir, exist_ok=True)

    copied = _copy_small_files(out_dir, small_files_from)
    print(f"[bert] 小文件: {copied if copied else '已齐, 未动'}")

    weight_path = os.path.join(out_dir, _WEIGHT_FILE)
    if os.path.isfile(weight_path) and not force:
        print(f"[bert] 权重已存在, 跳过({weight_path})")
    else:
        from safetensors.torch import save_file

        save_file(_extract_bert(ckpt), weight_path)
        print(f"[bert] 已写 {weight_path} ({os.path.getsize(weight_path)/1e6:.0f} MB)")

    if local_dir(out_dir) is None:
        raise SystemExit(f"{out_dir} 仍不完整, 请检查上面的报错")
    print(f"[bert] 就绪: {out_dir}")
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="从本地 .pth 造一个 off-line 的 BERT 目录(见本文件模块文档串)",
    )
    ap.add_argument("--out", default=DEFAULT_DIR, help="输出目录")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="BERT 权重的来源 .pth")
    ap.add_argument("--small-files-from", default=None,
                    help="config/分词器的来源目录(默认翻 HF 缓存)")
    ap.add_argument("--force", action="store_true", help="重拆一遍权重")
    opt = ap.parse_args(argv)

    build(opt.out, opt.ckpt, opt.small_files_from, force=opt.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
