#!/usr/bin/env bash
# ============================================================================
#  GroundingDINO 云端环境一键部署脚本
#
#  在【云端实例】里执行,git clone 之后跑这一条就够了。不需要任何密码。
#
#      cd <项目>/grounding_dino
#      bash setup_cloud_env.sh
#
#  脚本会核对 nvcc 和 torch 的 CUDA 版本,**不一致就自动补装对应版本的工具链**。
#  为什么必须管这件事:PyTorch 编译 _C 之前会拿 nvcc 的版本和 torch.version.cuda
#  严格比对(只特批 11.0/11.1),不一致直接 RuntimeError;而 uv 只管 Python 依赖,
#  nvcc 是镜像自带的,是多少全看运气(2026-09-19 实测某 AutoDL 镜像是 13.0,
#  项目锁的 torch 是 cu128 → uv sync 必挂在 _C 编译上)。
#
#  可选参数:
#      --check        只体检,不做任何改动(隐含 --no-fix-cuda)
#      --no-fix-cuda  只报告版本不匹配,不自动装(旧行为)
#      --fix-cuda     兼容保留:自动装现在是默认行为,写了也无害
#      --jobs N       并行编译数(默认 4)
#      --force        忽略「已装好」的判断,强制重跑 uv sync
#
#  配套文档:groundingdino技术文档/AutoDL云端部署操作文档.md
#USAGE_END
# ============================================================================

set -uo pipefail

# ------------------------------ 参数 ------------------------------
MODE="deploy"; FIX_CUDA=1; FORCE=0; MAX_JOBS="${MAX_JOBS:-4}"

usage() { sed -n '2,/^#USAGE_END/p' "$0" | sed '$d' | sed 's/^# \?//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --check)        MODE="check" ;;
    --no-fix-cuda)  FIX_CUDA=0 ;;
    --fix-cuda)     FIX_CUDA=1 ;;          # 旧参数,现在是默认行为
    --force)        FORCE=1 ;;
    --jobs)         MAX_JOBS="$2"; shift ;;
    -h|--help)      usage ;;
    *) echo "未知参数:$1"; usage ;;
  esac
  shift
done
[ "$MODE" = "check" ] && FIX_CUDA=0        # 体检模式绝不改机器

# ------------------------------ 输出 ------------------------------
C_R=$'\033[1;31m'; C_G=$'\033[1;32m'; C_Y=$'\033[1;33m'; C_C=$'\033[1;36m'; C_0=$'\033[0m'
say()  { printf '\n%s==> %s%s\n' "$C_C" "$*" "$C_0"; }
ok()   { printf '  %s✓%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '  %s!%s %s\n' "$C_Y" "$C_0" "$*"; }
die()  { printf '\n  %s✗ %s%s\n\n' "$C_R" "$*" "$C_0"; exit 1; }
add_rc() {
  if [ "$MODE" = "check" ]; then printf '  (体检模式,不写入) %s\n' "$1"; return; fi
  grep -qF "$1" "$HOME/.bashrc" 2>/dev/null || echo "$1" >> "$HOME/.bashrc"
}

# 脚本自己所在的目录 = 项目目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || { echo "无法进入 $PROJECT_DIR"; exit 1; }

export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_MIRROR="https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download"

# ============================ 1. 体检 ============================
say "1/6 环境体检"
printf '  主机 : %s\n' "$(hostname)"
printf '  系统 : %s\n' "$( . /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-未知}" )"
printf '  项目 : %s\n' "$PROJECT_DIR"
printf '  磁盘 :\n'; df -h / /root/autodl-tmp 2>/dev/null | sed 's/^/         /'

GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
if [ -n "$GPU" ]; then
  ok "GPU:$GPU"
else
  die "没有检测到 GPU —— 是不是开了「无卡模式」?编译 _C 必须有卡"
fi

# 算力直接从卡上读,别写死(换卡不用改脚本)
GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_CAP:-8.9}}"
export MAX_JOBS
ok "算力 $GPU_CAP → TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\"  MAX_JOBS=$MAX_JOBS"

[ -f "$PROJECT_DIR/pyproject.toml" ] && [ -f "$PROJECT_DIR/uv.lock" ] \
  || die "当前目录不像项目根(缺 pyproject.toml / uv.lock)。请在 grounding_dino/ 下执行"
ok "pyproject.toml + uv.lock 就位"

case "$PROJECT_DIR" in
  /root/autodl-tmp/*|/root/autodl-fs/*) ;;
  /root/*) warn "项目在系统盘(30G)。venv 约 7.5G,建议移到 /root/autodl-tmp" ;;
esac

# ============================ 2. 安装 uv ============================
say "2/6 检查 uv"
if command -v uv >/dev/null 2>&1; then
  ok "已装:$(uv --version)"
else
  warn "未装,用官方脚本安装"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    warn "官方脚本失败,退回 pip(走镜像)"
    PIP="$(command -v pip || command -v pip3 || echo /root/miniconda3/bin/pip)"
    "$PIP" install -q uv || die "uv 安装失败 —— 检查网络"
  fi
  ok "安装完成:$(uv --version)"
fi

# ============================ 3. nvcc 核对 / 修正 ============================
say "3/6 核对 nvcc 与 torch 的 CUDA 版本"

# --- 3a. 从 uv.lock 反推需要的 CUDA 版本:torch 2.10.0+cu128 → 需要 12.8 ---
# 只认 [[package]] 里 torch 那一块的 version,且 source 必须是带 cu 的索引,
# 免得抓到别的包 requires-dist 里的内联 torch(PyPI 那个 2.14.0)。
TORCH_SPEC="$(awk '
  /^name = "torch"$/ { n=1; v=""; next }
  n && /^version = / { v=$0; sub(/^version = "/,"",v); sub(/".*$/,"",v); next }
  n && /^source = /  { if ($0 ~ /cu[0-9]+/) { print v; exit } ; n=0 }
' "$PROJECT_DIR/uv.lock" 2>/dev/null)"
[ -n "$TORCH_SPEC" ] || die "读不出 uv.lock 里 torch 的 CUDA 版本 —— lock 文件不对?"
TORCH_CU="$(printf '%s' "$TORCH_SPEC" | grep -oP 'cu\K[0-9]+' | head -1)"
[ -n "$TORCH_CU" ] || die "uv.lock 里 torch($TORCH_SPEC)不是 CUDA 轮子 —— 检查 pytorch-cu128 索引配置"

CUDA_MAJ="${TORCH_CU%?}"; CUDA_MIN="${TORCH_CU: -1}"
CUDA_VER="$CUDA_MAJ.$CUDA_MIN"                 # 12.8   —— 要求 nvcc 严格等于它
CUDA_PKG="$CUDA_MAJ-$CUDA_MIN"                 # 12-8   —— apt 包名后缀
CUDA_HOME_WANT="/usr/local/cuda-$CUDA_VER"     # /usr/local/cuda-12.8
printf '  torch(锁文件) : %s  → 需要 CUDA %s\n' "$TORCH_SPEC" "$CUDA_VER"

nvcc_ver() { "$1" -V 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1; }

# --- 3b. 三处找 nvcc,优先「版本正好对得上且头文件齐全」的那个 ---
# 注意:torch 的 _find_cuda_home() 依次看 CUDA_HOME → PATH 上的 nvcc → /usr/local/cuda,
# 所以光 PATH 对没用 —— 镜像若把 CUDA_HOME 设成了 13.0 那个目录,torch 照样报错。
NVCC_BIN=""; NVCC_VER=""; NVCC_HOME=""; NVCC_SEEN=""
for c in "$CUDA_HOME_WANT/bin/nvcc" "$(command -v nvcc 2>/dev/null || true)" /usr/local/cuda/bin/nvcc; do
  [ -n "$c" ] && [ -x "$c" ] || continue
  v="$(nvcc_ver "$c")"; h="$(dirname "$(dirname "$c")")"
  printf '  发现 nvcc     : %-6s (%s)\n' "${v:-?}" "$c"
  [ -n "$NVCC_SEEN" ] || NVCC_SEEN="$v"
  if [ "$v" = "$CUDA_VER" ] && [ -f "$h/include/cuda_runtime.h" ]; then
    NVCC_BIN="$c"; NVCC_VER="$v"; NVCC_HOME="$h"; break
  fi
done

if [ -n "$NVCC_BIN" ]; then
  ok "nvcc $NVCC_VER 匹配 torch cu$TORCH_CU → CUDA_HOME=$NVCC_HOME"
  export CUDA_HOME="$NVCC_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  add_rc "export CUDA_HOME=$CUDA_HOME"
  add_rc "export PATH=$CUDA_HOME/bin:\$PATH"
elif [ "$FIX_CUDA" != 1 ]; then
  warn "nvcc ${NVCC_SEEN:-缺失} ≠ 需要的 $CUDA_VER,而自动修复已关闭"
  warn "重跑时去掉 --no-fix-cuda 即可自动装上"
else
  say "3c/6 自动补装 CUDA $CUDA_VER 工具链(镜像自带的是 ${NVCC_SEEN:-无 nvcc})"
  AVAIL="$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
  [ -n "$AVAIL" ] && [ "$AVAIL" -lt 4 ] && warn "系统盘只剩 ${AVAIL}G,工具链约 2~3G,可能装不下"

  if [ -x "$CUDA_HOME_WANT/bin/nvcc" ]; then
    ok "$CUDA_HOME_WANT 已存在,跳过 apt"
  else
    REPO="$( . /etc/os-release 2>/dev/null; echo "${ID:-ubuntu}${VERSION_ID//./}" )"
    printf '  apt 仓库 : %s\n' "$REPO"
    export DEBIAN_FRONTEND=noninteractive
    if ! ls /etc/apt/sources.list.d/cuda-*.list >/dev/null 2>&1; then
      wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-keyring_1.1-1_all.deb" \
        -O /tmp/cuda-keyring.deb \
        || die "下载 cuda-keyring 失败(网络到不了 developer.download.nvidia.com)—— 换镜像实例,或手工 conda install -c nvidia cuda-toolkit=$CUDA_VER"
      dpkg -i /tmp/cuda-keyring.deb >/dev/null 2>&1 || die "安装 cuda-keyring 失败"
    fi
    apt-get update -qq >/dev/null 2>&1 || warn "apt-get update 有告警,继续"
    # cudart-dev 给 cuda_runtime.h;后三个 dev 包必须装 —— torch 的 ATen 头文件
    # 无条件 include cusparse.h / cublas_v2.h / cublasLt.h / cusolverDn.h,只装 nvcc 编不过。
    apt-get install -y "cuda-nvcc-$CUDA_PKG" "cuda-cudart-dev-$CUDA_PKG" \
        "libcusparse-dev-$CUDA_PKG" "libcublas-dev-$CUDA_PKG" "libcusolver-dev-$CUDA_PKG" \
      || die "安装 CUDA $CUDA_VER 工具链失败(apt 报错见上)—— 可换镜像实例,或手工装"
  fi

  # 装完 /usr/local/cuda 这个软链仍指向镜像自带的版本,必须显式切 CUDA_HOME 过去;
  # 追加到 ~/.bashrc 末尾,位置在镜像原有设置之后,能覆盖掉它。
  export CUDA_HOME="$CUDA_HOME_WANT"
  export PATH="$CUDA_HOME/bin:$PATH"
  NOW="$(nvcc_ver "$CUDA_HOME/bin/nvcc")"
  [ "$NOW" = "$CUDA_VER" ] || die "装完 nvcc 版本是 '${NOW:-缺失}',期望 $CUDA_VER —— CUDA_HOME=$CUDA_HOME"
  [ -f "$CUDA_HOME/include/cuda_runtime.h" ] || die "$CUDA_HOME/include/cuda_runtime.h 缺失(cudart-dev 没装上)"
  ok "已切到 nvcc $NOW  CUDA_HOME=$CUDA_HOME"
  add_rc "export CUDA_HOME=$CUDA_HOME"
  add_rc "export PATH=$CUDA_HOME/bin:\$PATH"
fi

# ============================ 4. 环境变量持久化 ============================
say "4/6 持久化环境变量到 ~/.bashrc"
add_rc "export TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\""
add_rc "export MAX_JOBS=$MAX_JOBS"
add_rc 'export UV_PYTHON_INSTALL_MIRROR="https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download"'
ok "TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\"  MAX_JOBS=$MAX_JOBS"

# ============================ 5. uv sync ============================
if [ "$MODE" = "check" ]; then
  say "体检模式(--check):到此为止,未做任何改动"
  echo "  uv   : $(uv --version 2>/dev/null || echo 未装)"
  echo "  下一步: bash setup_cloud_env.sh"
  exit 0
fi

if [ "$FORCE" != 1 ] && [ -x "$PROJECT_DIR/.venv/bin/python" ] \
   && "$PROJECT_DIR/.venv/bin/python" -c 'import torch; from groundingdino import _C' >/dev/null 2>&1; then
  ok "5/6 环境已就绪(.venv + _C 都在),跳过 uv sync。要重跑加 --force"
else
  say "5/6 创建环境(uv sync --frozen)"
  cd "$PROJECT_DIR" || die "无法进入 $PROJECT_DIR"
  source /etc/network_turbo 2>/dev/null || warn "没有 /etc/network_turbo,直连下载可能较慢"
  echo "  要下载数 GB(torch cu128 + 依赖)并现场编译 _C,耐心等"
  echo "  ────────────────────────────────────────────────"
  uv sync --frozen --no-build-isolation-package groundingdino
  RC=$?
  echo "  ────────────────────────────────────────────────"
  [ $RC -eq 0 ] || die "uv sync 失败(退出码 $RC)。上面的 nvcc 版本行若显示不匹配,就是那个问题;报错信息里出现 'CUDA version' / 'mismatch' 同理"
  ok "uv sync 完成"
fi

# ============================ 6. 验证 ============================
say "6/6 验证"
printf '  CUDA_HOME : %s\n' "${CUDA_HOME:-未设}"
printf '  nvcc      : %s (%s)\n' "$(nvcc_ver "$(command -v nvcc 2>/dev/null || echo /nonexistent)")" "$(command -v nvcc 2>/dev/null || echo 不在 PATH)"

uv run python -c "
import torch
print('  torch :', torch.__version__)
print('  cuda  :', torch.version.cuda)
cap = torch.cuda.get_device_capability()
print('  算力  :', cap, '(期望 (8, 9))')
assert cap == (8, 9), f'算力不符: {cap}'
from torch.utils.cpp_extension import CUDA_HOME
print('  torch 眼中的 CUDA_HOME :', CUDA_HOME)
" || die "torch 校验失败"

# 必须先 import torch:_C.so 是 setuptools 老后端编出来的,不带 RPATH,靠 torch 先把
# libc10/libtorch 载进进程;单独 import _C 会报 "libc10.so: cannot open shared object file"。
# 项目入口脚本(run_grounding.py:56 / train_multimodal.py:51)本来就是 torch 在前,照抄即可。
uv run python -c "
import torch
from groundingdino import _C
print('  _C    :', _C.__file__)
" || die "_C 导入失败 —— 见上面报错;若是找不到 libc10.so 之类,就是漏了先 import torch"

printf '\n%s  环境就绪 ✓%s\n' "$C_G" "$C_0"
echo "  下一步:数据集 / 权重传到 /root/autodl-tmp 后解包,再跑"
echo "          .venv/bin/python run_grounding.py --num-images 20   # 先小跑验证"
