#!/usr/bin/env bash
# ============================================================================
#  GroundingDINO 云端环境部署 —— conda 版
#
#  在【云端实例】里执行,git clone 之后跑这一条就够了:
#
#      cd <项目>/grounding_dino
#      bash setup_cloud_conda.sh
#
#  与 setup_cloud_env.sh(uv 版)的区别,以及为什么换成 conda:
#
#  1. CUDA 工具链变成环境内的 conda 包,nvcc 与 torch 天然同源
#     uv 版最痛的是:nvcc 是镜像自带的,是多少全看运气,而 PyTorch 编译 _C 前
#     会拿 nvcc 版本和 torch.version.cuda 严格比对(只特批 11.0/11.1),不一致
#     直接 RuntimeError。uv 管不了 nvcc,只能写八十行代码去"找适配的 nvcc →
#     没有就 apt 装 → 再改 CUDA_HOME"。
#     conda 版里 cuda-nvcc / cuda-cudart-dev / libcublas-dev 都是环境内的包,
#     和 pytorch 的 cuda129 变体由同一个 solver 一次解出,版本必然一致 ——
#     上面那八十行整段消失,也不再去 developer.download.nvidia.com 拉 apt 源。
#
#  2. 所有包都能从同一个国内镜像拿到
#     实测清华 conda-forge 镜像包含:python 3.10.21 / pytorch 2.13.0(cuda129) /
#     cuda-nvcc 12.9.86 / transformers 4.57.6 / supervision / pycocotools /
#     timm / addict / yapf / opencv,共 349 个包的完整依赖树,dry-run 可解。
#
#  3. 只读本地代码,不联网
#     唯一还需要 pip 的一步是本项目自己(GroundingDINO 是纯 setup.py 的仓库,
#     conda 装不了 editable)。用 --no-deps + PIP_NO_INDEX=1 强制它只读本地目录,
#     一个字节都不下载。
#
#  ⚠️ 与 uv 版的取舍(务必知悉):
#     uv.lock 锁的是 torch 2.10.0+cu128;conda-forge 没有 cu128 变体,只有
#     cuda129 / cuda130。所以切过来 = torch 从 (2.10.0, cu128) 变成
#     (2.13.0, cu129),跨了 3 个 minor 版本。这是真实的版本跃迁,不是纯工具替换。
#     编译完 _C 后请务必跑一次实际训练/推理验证,别只看 import 通过。
#
#  可选参数:
#      --check          只体检,不做任何改动
#      --name NAME      conda 环境名(默认 gdino)
#      --cuda {129,130} CUDA 变体(默认 129;130 对应镜像自带 13.x 的机器)
#      --jobs N         编译 _C 的并行数(默认 4)
#      --force          环境已存在也删掉重建
#      --write-condarc  把镜像配置写进 ~/.condarc(默认不动你已有的配置)
#      --mirror URL     覆盖 conda-forge 镜像地址
# ============================================================================

set -uo pipefail

# ------------------------------ 版本锚点 ------------------------------
# 要改版本就改这里。这些值都是实测能在清华 conda-forge 镜像上解出来的。
PY_VER="3.10"
TORCH_VER="2.13.0"
TV_VER="0.28.0"
TRANSFORMERS_VER="4.57.6"      # ⚠️ 必须 <5,见下面 install 段的说明
CONDA_MIRROR="${CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}"

# ------------------------------ 参数 ------------------------------
MODE="deploy"; ENV_NAME="gdino"; CUDA_VARIANT="129"; FORCE=0
WRITE_CONDARC=0; MAX_JOBS="${MAX_JOBS:-4}"

usage() { sed -n '2,/^# ===/p' "$0" | sed '$d' | sed 's/^# \?//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --check)         MODE="check" ;;
    --name)          ENV_NAME="$2"; shift ;;
    --cuda)          CUDA_VARIANT="$2"; shift ;;
    --jobs)          MAX_JOBS="$2"; shift ;;
    --force)         FORCE=1 ;;
    --write-condarc) WRITE_CONDARC=1 ;;
    --mirror)        CONDA_MIRROR="$2"; shift ;;
    -h|--help)       usage ;;
    *) echo "未知参数:$1"; usage ;;
  esac
  shift
done

case "$CUDA_VARIANT" in
  129) CUDA_VER="12.9" ;;
  130) CUDA_VER="13.0" ;;
  *) echo "--cuda 只接受 129 或 130(conda-forge 只发这两个变体)"; exit 1 ;;
esac

# ------------------------------ 输出 ------------------------------
C_R=$'\033[1;31m'; C_G=$'\033[1;32m'; C_Y=$'\033[1;33m'; C_C=$'\033[1;36m'; C_0=$'\033[0m'
say()  { printf '\n%s==> %s%s\n' "$C_C" "$*" "$C_0"; }
ok()   { printf '  %s✓%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '  %s!%s %s\n' "$C_Y" "$C_0" "$*"; }
die()  { printf '\n  %s✗ %s%s\n\n' "$C_R" "$*" "$C_0"; exit 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || { echo "无法进入 $PROJECT_DIR"; exit 1; }

# ============================ 1. 体检 ============================
say "1/7 环境体检"
printf '  主机 : %s\n' "$(hostname)"
printf '  系统 : %s\n' "$( . /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-未知}" )"
printf '  项目 : %s\n' "$PROJECT_DIR"
printf '  磁盘 :\n'; df -h / /root/autodl-tmp 2>/dev/null | sed 's/^/         /'

GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
[ -n "$GPU" ] || die "没有检测到 GPU —— 是不是开了「无卡模式」?编译 _C 必须有卡"
ok "GPU:$GPU"

# 算力直接从卡上读,别写死(换卡不用改脚本)
GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_CAP:-8.9}}"
export MAX_JOBS
ok "算力 $GPU_CAP → TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\"  MAX_JOBS=$MAX_JOBS"

[ -f "$PROJECT_DIR/GroundingDINO/setup.py" ] \
  || die "找不到 GroundingDINO/setup.py —— 请在 grounding_dino/ 下执行,且子模块已 clone"
ok "GroundingDINO/ 就位"

case "$PROJECT_DIR" in
  /root/autodl-tmp/*|/root/autodl-fs/*) ;;
  /root/*) warn "项目在系统盘(30G)。conda 环境含 CUDA 工具链约 6~8G,强烈建议移到 /root/autodl-tmp" ;;
esac

# --- conda 定位 ---
if ! command -v conda >/dev/null 2>&1; then
  # 非交互 shell 里 conda 常常不在 PATH,主动找一遍
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /root/miniconda3; do
    if [ -x "$c/bin/conda" ]; then
      # shellcheck disable=SC1091
      . "$c/etc/profile.d/conda.sh" && break
    fi
  done
fi
command -v conda >/dev/null 2>&1 || die "找不到 conda。云端镜像本该自带,若确实没有请先装 miniconda"
ok "conda $(conda --version 2>/dev/null | awk '{print $2}')"

# ============================ 2. 镜像配置 ============================
say "2/7 conda 镜像"
printf '  当前 channels : %s\n' "$(conda config --show channels 2>/dev/null | tr -d ' ' | paste -sd' ')"
if [ -f "$HOME/.condarc" ]; then
  ok "已有 ~/.condarc(尊重你的配置,本脚本默认不改动)"
else
  warn "没有 ~/.condarc"
fi
# 不管 .condarc 怎么配的,下面一律用 --override-channels + 显式 URL,
# 所以即便你的 .condarc 只配了 defaults(pkgs/main),conda-forge 也照样能拿到。
printf '  本脚本使用的 conda-forge : %s\n' "$CONDA_MIRROR"
if [ "$WRITE_CONDARC" = 1 ] && [ "$MODE" != "check" ]; then
  cp "$HOME/.condarc" "$HOME/.condarc.bak.$$" 2>/dev/null && warn "原配置已备份到 ~/.condarc.bak.$$"
  cat > "$HOME/.condarc" <<EOF
channels:
  - conda-forge
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
  ok "已写入 ~/.condarc"
fi

# 镜像连通性:不只看状态码,直接让 conda 去解一次,失败会立刻报错
if [ "$MODE" != "check" ]; then
  printf '  探测镜像可用性 ... '
  if conda search --override-channels -c "$CONDA_MIRROR" cuda-nvcc 2>/dev/null | grep -q cuda-nvcc; then
    printf '%s✓%s\n' "$C_G" "$C_0"
  else
    die "conda-forge 镜像不可达:$CONDA_MIRROR
       换个镜像(如 https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge)
       或先确认这台机器能不能访问国内镜像。"
  fi
fi

# ============================ 3. 清理历史遗留 ============================
say "3/7 清理 uv 版脚本留下的环境变量"
# 旧的 setup_cloud_env.sh 往 ~/.bashrc 里写过 CUDA_HOME=/usr/local/cuda-12.8 和
# 对应的 PATH。那些值会**覆盖**掉 conda 环境里的 nvcc,导致 torch 又去跟镜像
# 自带的 12.8/13.0 比对,直接回到旧问题。必须删掉。
clean_rc() {
  [ "$MODE" = "check" ] && { printf '  (体检模式,不写入) %s\n' "$1"; return; }
  grep -qF "$1" "$HOME/.bashrc" 2>/dev/null || return 0
  cp "$HOME/.bashrc" "$HOME/.bashrc.bak.$$"
  grep -vF "$1" "$HOME/.bashrc.bak.$$" > "$HOME/.bashrc"
  printf '  (已清理 .bashrc,备份 .bashrc.bak.$$) %s\n' "$1"
}
clean_rc 'export CUDA_HOME=/usr/local/cuda-12.8'
clean_rc '/usr/local/cuda-12.8/bin'
clean_rc 'export UV_PYTHON_INSTALL_MIRROR'
# 当前 shell 里也可能已经有(登录时 source 过)
unset CUDA_HOME
ok "已清除旧 CUDA_HOME 干扰(conda 环境里的 nvcc 才是准的)"

# ============================ 4. 创建环境 ============================
say "4/7 创建 conda 环境 [$ENV_NAME]"

env_exists() { conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; }

if env_exists && [ "$FORCE" = 1 ] && [ "$MODE" != "check" ]; then
  warn "--force:删除已有环境 $ENV_NAME"
  conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
  env_exists() { return 1; }
fi

if [ "$MODE" = "check" ]; then
  say "体检模式(--check):到此为止,未做任何改动"
  echo "  conda      : $(conda --version 2>/dev/null || echo 未装)"
  echo "  环境存在   : $(env_exists && echo 是 || echo 否)"
  echo "  下一步     : bash setup_cloud_conda.sh"
  exit 0
fi

if env_exists; then
  ok "环境已存在,复用(要重建加 --force)"
else
  printf '  创建 python=%s 的干净环境 ...\n' "$PY_VER"
  conda create -y -n "$ENV_NAME" --override-channels -c "$CONDA_MIRROR" \
    "python=$PY_VER" pip >/dev/null || die "创建环境失败"
  ok "环境已创建"
fi

# 之后所有操作都在这个环境里
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME" || die "conda activate $ENV_NAME 失败"
ok "已激活:$CONDA_PREFIX"

# ============================ 5. 装依赖 ============================
say "5/7 安装依赖(全走清华 conda-forge,约 3~5G,耐心等)"

# ── 为什么是这些包 ────────────────────────────────────────────────
# pytorch-gpu         : 元包,拉 pytorch 2.13.0 的 cuda129 变体。**不用 -c pytorch
#                       -c nvidia**:pytorch 官方 conda channel 已停更在 2.5.1,
#                       nvidia channel 清华/中科大都没同步(404)。
# cuda-nvcc           : nvcc 本体。装进环境是为了让它和 torch.version.cuda 同源,
#                       这样 _C 编译时那个严格版本比对必然通过。
# cuda-cudart-dev     : cuda_runtime.h(没有它 nvcc 编不过)。
# libcublas-dev       : torch 的 ATen 头文件无条件 include cublas_v2.h / cublasLt.h。
# libcusparse-dev     : 同理,cusparse.h。
# libcusolver-dev     : 同理,cusolverDn.h。
#                       ↑ 后三个 dev 包缺任何一个,编译都过不去,别省。
# transformers        : ⚠️ 钉 4.57.6(即 <5)。上游 bertwarper.py 抓的是 BertModel
#                       上的 get_extended_attention_mask / get_head_mask,这两个
#                       方法在 transformers v5 被删了。conda-forge 最新的 5.17.0
#                       装上去会直接 AttributeError,所以这里必须显式钉版本。
# opencv              : 提供 cv2。注意 conda-forge 的 opencv 会带 qt6/X11 一堆
#                       图形依赖(实测整个解法 349 个包,大半是它们)。纯 headless
#                       跑可以忽略,想瘦身就把 opencv 换掉。
CONDA_PKGS=(
  "pytorch-gpu=$TORCH_VER"
  "torchvision=$TV_VER"
  "cuda-version=$CUDA_VER"
  cuda-nvcc cuda-cudart-dev libcublas-dev libcusparse-dev libcusolver-dev
  "transformers=$TRANSFORMERS_VER"
  timm supervision pycocotools addict yapf opencv numpy setuptools
)

printf '  %s\n' "${CONDA_PKGS[@]}" | sed 's/^/    /'
echo "  ────────────────────────────────────────────────"
conda install -y -n "$ENV_NAME" --override-channels -c "$CONDA_MIRROR" "${CONDA_PKGS[@]}"
RC=$?
echo "  ────────────────────────────────────────────────"
[ $RC -eq 0 ] || die "conda install 失败(退出码 $RC)。见上面报错:
       · PackagesNotFoundError → 该镜像缺包,试 --mirror 换 USTC/阿里镜像
       · 依赖冲突(UnsatisfiableError) → 多半是某个包版本钉太死,
         先松开对应版本号(改脚本顶部的版本锚点),再跑 --force"

# ============================ 6. 装本项目 ============================
say "6/7 安装 GroundingDINO 本体(editable,只读本地,不联网)"

# GroundingDINO 是纯 setup.py 的上游仓库,conda 装不了 editable,只能 pip。
# 但它 setup.py 顶层会 `import torch`,所以必须 --no-build-isolation —— 否则
# pip 会新建一个隔离构建环境,里面没有 torch,setup.py 会去 pip install torch。
# --no-deps + PIP_NO_INDEX=1 是双重保险:依赖已全部由 conda 装好,pip 不该、也
# 不允许去网上拿任何东西。这保证这一步在无外网时也能成功。
[ -f "$PROJECT_DIR/GroundingDINO/setup.py" ] || die "GroundingDINO/setup.py 不存在"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
ok "CUDA_HOME=$CUDA_HOME（指向 conda 环境，不是 /usr/local/cuda）"
printf '  环境内 nvcc : %s\n' "$(nvcc -V 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo 缺失)"

PIP_NO_INDEX=1 pip install -e "$PROJECT_DIR/GroundingDINO" \
  --no-build-isolation --no-deps -q \
  || die "pip install -e 失败。
       注意这里刻意不联网(--no-deps + PIP_NO_INDEX=1),报 'No matching distribution'
       就说明有依赖没被 conda 装上,把包名补进上面的 CONDA_PKGS 再跑。
       报 nvcc/CUDA 相关的编译错误,先看上面那行 nvcc 版本是不是 $CUDA_VER。"
ok "GroundingDINO 已 editable 安装"

# ============================ 7. 导出锁定文件 ============================
say "7/7 导出环境锁定文件"
# conda 没有 uv.lock 那种东西,等价物是 env export:它把每个包的**精确 build
# 字符串**都写进去,重建时可完全复现。这是你要的"固定各个依赖版本"。
# 去掉 prefix: 那一行 —— 它是绝对路径,留着会让 `conda env create -f` 在别的
# 机器/别的目录下直接拒绝加载。
LOCK="$PROJECT_DIR/environment.lock.yml"
if conda env export -n "$ENV_NAME" 2>/dev/null | grep -v '^prefix:' > "$LOCK"; then
  ok "已写入 environment.lock.yml($(grep -c '^  - ' "$LOCK") 个包,含 build 串)"
else
  warn "env export 失败,跳过(不影响本次部署)"
fi

# ============================ 验证 ============================
say "验证"

nvcc_ver() { "$1" -V 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1; }
printf '  CUDA_HOME : %s\n' "${CUDA_HOME:-未设}"
printf '  nvcc      : %s (%s)\n' "$(nvcc_ver "$(command -v nvcc 2>/dev/null || echo /nonexistent)")" "$(command -v nvcc 2>/dev/null || echo 不在 PATH)"

python - <<PY || die "校验失败"
import sys, torch
print('  torch     :', torch.__version__)
print('  cuda      :', torch.version.cuda)
print('  python    :', sys.version.split()[0])
cap = torch.cuda.get_device_capability()
print('  算力      :', cap)
from torch.utils.cpp_extension import CUDA_HOME
print('  torch 眼中的 CUDA_HOME :', CUDA_HOME)

# 版本必须严丝合缝。PyTorch 编译 _C 前会拿 nvcc 和 torch.version.cuda 比对
# (只特批 11.0/11.1),不一致直接 RuntimeError。conda 装的话这里必然相等 ——
# 这条断言就是来证明这一点的。
assert torch.version.cuda == "$CUDA_VER", f'CUDA 版本不符: torch={torch.version.cuda} 期望=$CUDA_VER'
assert torch.cuda.is_available(), 'CUDA 不可用'

# 真正跑一次 GPU 运算,别只看 is_available()。conda-forge 的 build 若没编进
# 你这张卡的算力,这里才会暴露 "no kernel image is available for execution"。
a = torch.randn(512, 512, device='cuda')
b = a @ a
torch.cuda.synchronize()
print('  GPU 矩阵乘 :', tuple(b.shape), '✓')
PY

# 必须先 import torch:_C.so 是 setuptools 老后端编出来的,不带 RPATH,靠 torch
# 先把 libc10/libtorch 载进进程;单独 import _C 会报
# "libc10.so: cannot open shared object file"。
python -c "
import torch
from groundingdino import _C
print('  _C        :', _C.__file__)
" || die "_C 导入失败 —— 见上面报错;若是找不到 libc10.so 之类,就是漏了先 import torch"

printf '\n%s  环境就绪 ✓%s\n' "$C_G" "$C_0"
cat <<EOF
  环境名 : $ENV_NAME
  激活   : conda activate $ENV_NAME
  锁定   : environment.lock.yml
  重建   : conda env create -f environment.lock.yml   # 同机复现

  ⚠️ 提醒:torch 从 uv.lock 的 2.10.0+cu128 换成了 $TORCH_VER+cu$CUDA_VARIANT,
     跨了 3 个 minor 版本。import 通过 ≠ 能跑,请务必实际跑一次训练/推理:
         python run_grounding.py --num-images 20
EOF
