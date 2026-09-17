#!/usr/bin/env bash
# ============================================================================
#  GroundingDINO 云端环境一键部署脚本
#
#  在【云端实例】里执行,git clone 之后跑这一条就够了。不需要任何密码。
#
#      cd <项目>/grounding_dino
#      bash setup_cloud_env.sh
#
#  可选参数:
#      --check        只体检,不做任何改动
#      --fix-cuda     发现 nvcc 主版本与 torch 不匹配时,自动补装 CUDA 12.8 工具链
#      --jobs N       并行编译数(默认 4)
#      --force        忽略「已装好」的判断,强制重跑 uv sync
#
#  配套文档:groundingdino技术文档/AutoDL云端部署操作文档.md
# ============================================================================

set -uo pipefail

# ------------------------------ 参数 ------------------------------
MODE="deploy"; FIX_CUDA=0; FORCE=0; MAX_JOBS="${MAX_JOBS:-4}"

usage() { sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --check)     MODE="check" ;;
    --fix-cuda)  FIX_CUDA=1 ;;
    --force)     FORCE=1 ;;
    --jobs)      MAX_JOBS="$2"; shift ;;
    -h|--help)   usage ;;
    *) echo "未知参数:$1"; usage ;;
  esac
  shift
done

# ------------------------------ 输出 ------------------------------
C_R=$'\033[1;31m'; C_G=$'\033[1;32m'; C_Y=$'\033[1;33m'; C_C=$'\033[1;36m'; C_0=$'\033[0m'
say()  { printf '\n%s==> %s%s\n' "$C_C" "$*" "$C_0"; }
ok()   { printf '  %s✓%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '  %s!%s %s\n' "$C_Y" "$C_0" "$*"; }
die()  { printf '\n  %s✗ %s%s\n\n' "$C_R" "$*" "$C_0"; exit 1; }
add_rc() { grep -qF "$1" "$HOME/.bashrc" 2>/dev/null || echo "$1" >> "$HOME/.bashrc"; }

# 脚本自己所在的目录 = 项目目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || { echo "无法进入 $PROJECT_DIR"; exit 1; }

export PATH="$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"          # RTX 4090 = sm_89
export MAX_JOBS
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

# ============================ 3. nvcc 核对 ============================
say "3/6 核对 nvcc 与 torch 的 CUDA 版本"
NVCC_BIN="$(command -v nvcc 2>/dev/null || true)"
[ -z "$NVCC_BIN" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC_BIN=/usr/local/cuda/bin/nvcc

TORCH_SPEC="$(grep -oP 'torch", version = "\K[^"]+' "$PROJECT_DIR/uv.lock" 2>/dev/null | head -1)"
TORCH_CU="$(printf '%s' "$TORCH_SPEC" | grep -oP 'cu\K[0-9]+' | head -1)"
printf '  torch(锁文件) : %s  → CUDA 主版本 %s\n' "${TORCH_SPEC:-未知}" "${TORCH_CU:-?}"

NEED_FIX=0
if [ -z "$NVCC_BIN" ]; then
  warn "PATH 里没有 nvcc"; NEED_FIX=1
else
  NVCC_VER="$("$NVCC_BIN" -V 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)"
  NVCC_MAJOR="${NVCC_VER%%.*}"
  printf '  nvcc         : %s  (%s)\n' "$NVCC_VER" "$NVCC_BIN"
  if [ -n "$TORCH_CU" ] && [ "$NVCC_MAJOR" != "$TORCH_CU" ]; then
    NEED_FIX=1
    warn "主版本不一致:nvcc $NVCC_MAJOR vs torch cu$TORCH_CU"
    warn "PyTorch 编译 _C 时会直接抛 RuntimeError(uv 管不到这一步)"
  else
    ok "版本一致,可直接编译"
  fi
fi

if [ "$NEED_FIX" = 1 ]; then
  if [ "$FIX_CUDA" = 1 ]; then
    say "3b/6 补装 CUDA 12.8 精简工具链(--fix-cuda)"
    OS_ID="$( . /etc/os-release 2>/dev/null; echo "${ID:-ubuntu}${VERSION_ID//./}" )"
    REPO="${OS_ID:-ubuntu2204}"          # 云端是 Ubuntu 22.04 → ubuntu2204
    printf '  apt 仓库 : %s\n' "$REPO"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb" \
      -O /tmp/cuda-keyring.deb || die "下载 cuda-keyring 失败"
    dpkg -i /tmp/cuda-keyring.deb >/dev/null 2>&1 || die "安装 cuda-keyring 失败"
    apt-get update -qq >/dev/null 2>&1 || warn "apt-get update 有告警,继续"
    apt-get install -y cuda-nvcc-12-8 libcusparse-dev-12-8 libcublas-dev-12-8 libcusolver-dev-12-8 \
      || die "安装 CUDA 12.8 工具链失败"
    # 关键:装完 /usr/local/cuda 这个软链仍指向镜像自带的 cuda-13.0,必须显式切过去
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    add_rc 'export CUDA_HOME=/usr/local/cuda-12.8'
    add_rc 'export PATH=/usr/local/cuda-12.8/bin:$PATH'
    add_rc 'export TORCH_CUDA_ARCH_LIST="8.9"'
    add_rc "export MAX_JOBS=$MAX_JOBS"
    add_rc 'export UV_PYTHON_INSTALL_MIRROR="https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download"'
    NVCC_NOW="$(nvcc -V 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)"
    ok "已切到 nvcc $NVCC_NOW ($(command -v nvcc))"
  else
    warn "未启用自动修复。若下一步失败,加 --fix-cuda 重跑:"
    warn "    bash setup_cloud_env.sh --fix-cuda"
  fi
fi

# ============================ 4. 环境变量持久化 ============================
say "4/6 持久化环境变量到 ~/.bashrc"
add_rc 'export TORCH_CUDA_ARCH_LIST="8.9"'
add_rc "export MAX_JOBS=$MAX_JOBS"
add_rc 'export UV_PYTHON_INSTALL_MIRROR="https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download"'
ok "TORCH_CUDA_ARCH_LIST=\"8.9\"  MAX_JOBS=$MAX_JOBS"

# ============================ 5. uv sync ============================
if [ "$MODE" = "check" ]; then
  say "体检模式(--check):到此为止,未做任何改动"
  echo "  uv   : $(uv --version 2>/dev/null || echo 未装)"
  echo "  下一步: bash setup_cloud_env.sh$( [ "$NEED_FIX" = 1 ] && echo ' --fix-cuda' )"
  exit 0
fi

if [ "$FORCE" != 1 ] && [ -x "$PROJECT_DIR/.venv/bin/python" ] \
   && "$PROJECT_DIR/.venv/bin/python" -c 'from groundingdino import _C' >/dev/null 2>&1; then
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
  [ $RC -eq 0 ] || die "uv sync 失败(退出码 $RC)。若是 CUDA 版本报错,加 --fix-cuda 重跑"
  ok "uv sync 完成"
fi

# ============================ 6. 验证 ============================
say "6/6 验证"
uv run python -c "
import torch
print('  torch :', torch.__version__)
print('  cuda  :', torch.version.cuda)
cap = torch.cuda.get_device_capability()
print('  算力  :', cap, '(期望 (8, 9))')
assert cap == (8, 9), f'算力不符: {cap}'
" || die "torch 校验失败"

uv run python -c "
from groundingdino import _C
print('  _C    :', _C.__file__)
" || die "_C 导入失败 —— CUDA 扩展没编出来"

printf '\n%s  环境就绪 ✓%s\n' "$C_G" "$C_0"
echo "  下一步:数据集 / 权重传到 /root/autodl-tmp 后解包,再跑"
echo "          .venv/bin/python run_grounding.py --num-images 20   # 先小跑验证"
