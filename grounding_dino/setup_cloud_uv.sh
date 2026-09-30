#!/usr/bin/env bash
# ============================================================================
#  GroundingDINO 云端环境部署 —— uv 版
#
#  在【云端实例】里执行,进到项目目录跑这一条就够了:
#
#      cd <项目>/grounding_dino
#      bash setup_cloud_uv.sh
#
#  ────────────────────────────────────────────────────────────────────────────
#  为什么用 uv(与 setup_cloud_conda.sh 的区别)
#
#  1. 体积与时间(2026-09-30 实测,两条路都跑通过)
#         uv    : .venv 7.4G,无 solve 阶段,uv.lock 直接下
#         conda : env 12.6G + pkgs 18G,下载 4.3G,Solve 284 秒
#     根因:pip wheel 是自包含的;conda 把 torch 拆成 20 多个外部库
#     (libtorch 2159M / libmagma 1264M / libcudnn 1000M ...),全是 DT_NEEDED
#     级硬依赖,删不掉。而且 environment.lock.yml 只是版本清单,
#     `conda env create -f` 照样要重新 solve —— conda 没有"锁下载计划"的机制。
#
#  2. 版本保真
#     uv.lock 锁的是 torch 2.10.0+cu128(本机已验证过的那个版本)。
#     conda-forge 没有 cu128 变体,走 conda 必然被顶到 2.13.0+cu129,
#     跨 3 个 minor。论复现,uv 复现的是"锁文件里那个确切版本"。
#
#  3. 三个源全部有国内镜像(本脚本第 2 步会逐个探活)
#         PyPI 包      → pypi.tuna.tsinghua.edu.cn   (uv.lock 已改好)
#         torch cu128  → mirrors.nju.edu.cn/pytorch  (PEP503,sha256 与官方一致)
#         CPython 3.10 → mirrors.nju.edu.cn/github-release (GitHub 会 302 到
#                        release-assets.githubusercontent.com,国内常年不通)
#
#  ────────────────────────────────────────────────────────────────────────────
#  ⚠️ uv 路线唯一的真难点:nvcc(第 6 步)
#
#  upstream 的 GroundingDINO/setup.py 里 get_extensions() 有个【静默降级】分支:
#
#      if CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ):
#          ...  extension = CUDAExtension; sources += source_cuda
#      else:
#          print("Compiling without CUDA")
#          return None          # ← 不编译任何扩展,但安装照样"成功"
#
#  而没有 _C 时,ms_deform_attn.py:334 的分派条件是
#      if torch.cuda.is_available() and value.is_cuda:   # 看张量在不在 GPU,不是看 _C 导没导进来
#  所以在 4090 上【不会】走 multi_scale_deformable_attn_pytorch 那个纯 PyTorch 回退,
#  而是直接 NameError: name '_C' is not defined —— 而且失败点在第一次前向,
#  不在安装期。第 8 步因此必须验证 _C 真的编出来了,不能只看 import 通过。
#
#  版本比对规则(torch 2.10 实测,别再信"只特批 11.0/11.1"那套旧说法):
#      cpp_extension._check_cuda_version:
#          if cuda_ver != torch_cuda_version:
#              if cuda_ver.major != torch_cuda_version.major:
#                  raise RuntimeError(...)    # 只有 MAJOR 不同才炸
#              logger.warning(...)            # major 相同、minor 不同 → 只警告,照编
#  所以只要 nvcc 的 major 是 12 就能用;镜像自带的 13.0 会炸(13≠12)。
#
#  ────────────────────────────────────────────────────────────────────────────
#  可选参数:
#      --check         只体检,不做任何改动
#      --jobs N        编译 _C 的并行数(默认 4)
#      --force         已存在的 .venv 也删掉重建
#      --pip-nvcc      没有合适 nvcc 时,尝试用 PyPI 的 nvidia-cuda-nvcc-cu12
#                      搭一个假 CUDA_HOME(⚠️ 本方案【未经验证】,见第 6 步)
#      --mirror-tuna URL / --mirror-pytorch URL / --mirror-python URL
#                      覆盖三个镜像地址
#      --foreground    不要自动进 screen(默认会自动进,见第 0 步)
# ============================================================================

# ⚠️ 刻意【不】加 -u(set -u)。
# 理由与 setup_cloud_conda.sh 相同,但这里还有一个更直接的原因:
# uv 会去跑上游 setup.py,而 setup.py 内部又会 subprocess 调 pip。
# 这条链上任何一个环节引用了未定义变量,-u 都会让【整个 shell】退出,
# 而不是只让那个子进程失败 —— 现场表现是"装到一半静默死亡"。
# pipefail 已经足够暴露真实错误码。
set -o pipefail

# ------------------------------ 版本锚点 ------------------------------
TORCH_CUDA_MAJOR="12"            # torch 是 cu128 → nvcc 的 major 必须是 12
MIN_UV_VERSION="0.12.10"         # pyproject.toml 的 build-system 要求 uv_build>=0.12.10
PY_VER="3.10"                    # 与 .python-version 一致
MIRROR_TUNA="${MIRROR_TUNA:-https://pypi.tuna.tsinghua.edu.cn/simple}"
MIRROR_PYTORCH="${MIRROR_PYTORCH:-https://mirrors.nju.edu.cn/pytorch/whl/cu128}"
MIRROR_PYTHON="${MIRROR_PYTHON:-https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone}"

# ------------------------------ 参数 ------------------------------
MODE="deploy"; FORCE=0; USE_PIP_NVCC=0; MAX_JOBS="${MAX_JOBS:-4}"; FOREGROUND=0
ORIG_ARGS=("$@")

usage() { sed -n '2,/^# ===/p' "$0" | sed '$d' | sed 's/^# \?//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --check)          MODE="check" ;;
    --jobs)           MAX_JOBS="$2"; shift ;;
    --force)          FORCE=1 ;;
    --pip-nvcc)       USE_PIP_NVCC=1 ;;
    --mirror-tuna)    MIRROR_TUNA="$2"; shift ;;
    --mirror-pytorch) MIRROR_PYTORCH="$2"; shift ;;
    --mirror-python)  MIRROR_PYTHON="$2"; shift ;;
    --foreground)     FOREGROUND=1 ;;
    -h|--help)        usage ;;
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

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || { echo "无法进入 $PROJECT_DIR"; exit 1; }
LOG="$PROJECT_DIR/setup_uv.log"
SCREEN_NAME="gdino-setup-uv"
DATA_DISK="/root/autodl-tmp"

# ============================ 0. 防断线 ============================
# 这个脚本要下 ~3G(uv 缓存)+ 7.4G 落盘,还要编译 _C,合计 20 分钟以上。
# 2026-09-30 在 conda 那条路上实测过一次事故:脚本裸跑在 SSH 会话里,
# 会话一断 SIGHUP 广播给整个进程组,5 路并行下载在 0.3 秒内一起定格
# (6 个 .partial 的 mtime 落在 0.28 秒内),854M 白下。
# 同时停下 = 父进程没了;网络问题会让它们一个接一个地失败、时间戳是散的。
# 所以默认把自己塞进 screen。
if [ "$MODE" != "check" ] && [ "$FOREGROUND" != 1 ] && [ "${GDINO_IN_SCREEN:-0}" != 1 ]; then
  if [ -z "${STY:-}" ] && [ -z "${TMUX:-}" ]; then
    if command -v screen >/dev/null 2>&1; then
      _cmd="GDINO_IN_SCREEN=1 bash $(printf '%q' "$0")"
      for _a in "${ORIG_ARGS[@]}"; do _cmd="$_cmd $(printf '%q' "$_a")"; done
      screen -dmS "$SCREEN_NAME" bash -c "$_cmd > $(printf '%q' "$LOG") 2>&1"
      printf '\n%s 已在 screen 会话 [%s] 里启动%s\n' "$C_G" "$SCREEN_NAME" "$C_0"
      cat <<EOF

  这一步要跑 20 分钟以上,裸跑在 SSH 里的话,断线会把已下载的包一起带走。
  所以放进 screen 了,断线也不影响。

      看进度 : tail -f $LOG
      进会话 : screen -r $SCREEN_NAME
      退出会话: 先按 Ctrl-A 再按 D(不会中断脚本)

  不想用 screen(请自己挂 nohup):bash $0 --foreground

EOF
      exit 0
    else
      warn "没有 screen 也没有 tmux —— 请务必自己用 nohup 跑,否则断线会前功尽弃"
    fi
  fi
fi

# ============================ 1. 体检 ============================
say "1/8 环境体检"
printf '  主机 : %s\n' "$(hostname)"
printf '  系统 : %s\n' "$( . /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-未知}" )"
printf '  项目 : %s\n' "$PROJECT_DIR"
printf '  磁盘 :\n'; df -h / "$DATA_DISK" 2>/dev/null | sed 's/^/         /'

GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
[ -n "$GPU" ] || die "没有检测到 GPU —— 是不是开了「无卡模式」?编译 _C 必须有卡"
ok "GPU:$GPU"

# 算力直接从卡上读,别写死(换卡不用改脚本)
GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_CAP:-8.9}}"
export MAX_JOBS
ok "算力 $GPU_CAP → TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\"  MAX_JOBS=$MAX_JOBS"

[ -f "$PROJECT_DIR/uv.lock" ] || die "找不到 uv.lock —— 请在 grounding_dino/ 下执行"
[ -f "$PROJECT_DIR/GroundingDINO/setup.py" ] \
  || die "找不到 GroundingDINO/setup.py —— 子模块没 clone 全?"
ok "uv.lock 与 GroundingDINO/ 就位"

# 锁文件里若还有官方主机名,说明用的是没换过源的旧锁 —— 在国内会非常慢。
# ⚠️ 别写 `|| echo 0`:grep -c 无匹配时会【既打印 0 又返回非零】,
#    再 echo 一个 0 就得到 "0\n0",后面那个 != 0 的判断会白白成立。
_off="$(grep -cE 'pythonhosted|pypi\.org|download.*pytorch\.org' "$PROJECT_DIR/uv.lock" 2>/dev/null)"
_off="${_off:-0}"
if [ "$_off" != 0 ]; then
  warn "uv.lock 里还有 $_off 处官方源地址(这份锁没换过镜像),国内会明显变慢"
else
  ok "uv.lock 已全部指向国内镜像(官方主机名 0 残留)"
fi

# 上游 .so 是编进【源码树】的(editable + inplace build)。若上一台机器留下的
# .so 还在,而这次编译又被 setuptools 判定为"无需重编",就会静默用错架构的库。
# 本机的 .so 编进了 sm_35..sm_90 + sm_120,所以搬到 4090 上大概率还能跑 ——
# 正因为它"还能跑",错了也不会报错,所以这里显式提醒。
_so="$(find "$PROJECT_DIR/GroundingDINO" -maxdepth 3 -name '_C*.so' 2>/dev/null | head -1)"
if [ -n "$_so" ]; then
  warn "源码树里已有 _C.so:$(basename "$_so")"
  printf '      它是上一台机器编的。本脚本会重新编译覆盖;若想确保干净可先删:\n'
  printf '          rm -f %s\n' "$_so"
fi

case "$PROJECT_DIR" in
  /root/autodl-tmp/*|/root/autodl-fs/*) ok "项目在数据盘" ;;
  /root/*) warn "项目在系统盘(30G)。.venv 要占 7.4G,见第 3 步的兜底" ;;
esac

# ============================ 2. 镜像 ============================
say "2/8 镜像探活"
# 【为什么是 HEAD 而不是"真去解析一次"】
# conda 那条路的旧版用 `conda search` 验镜像,为了找一个包会把整个 channel 的
# repodata 拉下来解析,实测耗时 4 分钟,而屏幕上一行 `...working...` 看着像卡死。
# 一个 HEAD 请求就能验完 DNS + TLS + HTTP + 路径存在,实测 0.66 秒。
# 老实说这不省字节(装的时候反正要下同一份),只是让"镜像不通"在一秒内暴露。
probe_url() {  # $1 = 完整 URL → 打印 http code,可达则返回 0
  local url="$1" code=""
  code="$(curl -sI -o /dev/null -w '%{http_code}' --max-time 10 "$url" 2>/dev/null)"
  case "$code" in 200) printf '%s' "$code"; return 0 ;; esac
  # 有些镜像禁 HEAD,退化成只取 1 字节的 GET(206)
  code="$(curl -s -o /dev/null -r 0-0 -w '%{http_code}' --max-time 10 "$url" 2>/dev/null)"
  case "$code" in 200|206) printf '%s' "$code"; return 0 ;; esac
  printf '%s' "${code:-000}"; return 1
}

# 探针是只读的,所以 --check 下也照跑 —— 体检的意义正在于提前发现镜像不通。
_fail=0
check_one() {  # $1 = 标签, $2 = 完整 URL
  printf '  %-14s ' "$1"
  if _c="$(probe_url "$2")"; then
    printf '%s✓%s HTTP %s\n' "$C_G" "$C_0" "$_c"
  else
    printf '%s✗%s HTTP %s\n' "$C_R" "$C_0" "$_c"; _fail=1
  fi
}
# 注意三个 URL 的拼法各不相同:$MIRROR_TUNA 已经含 /simple,
# $MIRROR_PYTORCH 已经含 /whl/cu128,所以后面只接包名;
# python 那个探的是 base 本身(它的下一级是 <tag>/,tag 随时间变,不能写死)。
check_one "PyPI(tuna)"  "$MIRROR_TUNA/addict/"
check_one "torch(nju)"  "$MIRROR_PYTORCH/torch/"
check_one "python(nju)" "${MIRROR_PYTHON%/}/"
[ "$_fail" = 0 ] || die "有镜像不可达。换一个再试:
       · PyPI     : --mirror-tuna https://mirrors.aliyun.com/pypi/simple
       · torch    : --mirror-pytorch https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
       · CPython  : --mirror-python https://mirror.sjtu.edu.cn/github-release/astral-sh/python-build-standalone"
printf '  (注:python 那个路径形状是 <base>/<tag>/<文件名>,NJU 没有 /releases/download/ 段,\n'
printf '   正好对上 uv 拼 URL 的方式 —— 所以 UV_PYTHON_INSTALL_MIRROR 要填 base 本身)\n'

# ============================ 3. 目录规划 ============================
# 【为什么要把 uv 的东西也挪走】
# uv 缓存默认在 ~/.cache/uv:torch 那个 wheel 单个就 916M,全部下完约 3G。
# uv 托管的 CPython 默认在 ~/.local/share/uv/python(约 100M)。
# 系统盘只有 30G overlay,而 .venv 还要 7.4G —— 光靠"项目放数据盘"躲不掉缓存。
say "3/8 缓存目录"
export UV_CACHE_DIR="$DATA_DISK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$DATA_DISK/uv-python"

if [ ! -d "$DATA_DISK" ]; then
  warn "没有 $DATA_DISK,只能用系统盘 —— 注意 30G 要装 ~11G,会很紧"
  export UV_CACHE_DIR="$HOME/.cache/uv"
  export UV_PYTHON_INSTALL_DIR="$HOME/.local/share/uv/python"
fi

[ "$MODE" = "check" ] || mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" 2>/dev/null
ok "UV_CACHE_DIR        = $UV_CACHE_DIR"
ok "UV_PYTHON_INSTALL_DIR = $UV_PYTHON_INSTALL_DIR"

# .venv 按 uv 的默认规则落在项目目录下。项目在系统盘时,把 venv 也指到数据盘。
if [ "$FORCE" = 1 ] && [ "$MODE" != "check" ] && [ -d "$PROJECT_DIR/.venv" ]; then
  warn "--force:删除已有 .venv"
  rm -rf "$PROJECT_DIR/.venv"
fi

if [ ! -d "$DATA_DISK" ]; then :; elif [ "${PROJECT_DIR#/root/autodl-tmp/}" = "$PROJECT_DIR" ] \
   && [ "${PROJECT_DIR#/root/autodl-fs/}" = "$PROJECT_DIR" ]; then
  export UV_PROJECT_ENVIRONMENT="$DATA_DISK/gdino-venv"
  warn "项目在系统盘 → .venv 改到 $UV_PROJECT_ENVIRONMENT(否则 7.4G 压系统盘)"
fi

# ============================ 4. 装 uv ============================
say "4/8 安装 uv"
if command -v uv >/dev/null 2>&1; then
  ok "已有 uv $(uv --version 2>/dev/null | awk '{print $2}')"
else
  # uv 本身就在 PyPI 上,所以走 tuna 即可,不必碰 astral.sh / GitHub release。
  # (astral.sh 的安装脚本最终也是去拉 GitHub release,国内不一定通)
  command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 \
    || die "镜像里连 python 都没有,没法引导安装 uv"
  _py="$(command -v python3 || command -v python)"
  printf '  用 %s 从 tuna 装 uv ...\n' "$_py"
  if [ "$MODE" = "check" ]; then
    warn "(体检)未安装。实际执行会跑:$_py -m pip install uv -i $MIRROR_TUNA"
  else
    "$_py" -m pip install -q --upgrade uv -i "$MIRROR_TUNA" \
      || die "从 tuna 装 uv 失败。可换 --mirror-tuna https://mirrors.aliyun.com/pypi/simple"
    # pip 装完可能落在 ~/.local/bin,当前 shell 的 PATH 未必刷新
    command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
    ok "uv 已安装:$(uv --version 2>/dev/null || echo '但不在 PATH')"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  # pyproject.toml 的 build-system 要 uv_build>=0.12.10;太老的 uv 会直接拒绝这个项目
  _uvv="$(uv --version 2>/dev/null | awk '{print $2}')"
  if [ "$(printf '%s\n%s\n' "$MIN_UV_VERSION" "$_uvv" | sort -V | head -1)" = "$MIN_UV_VERSION" ]; then
    ok "uv $_uvv ≥ $MIN_UV_VERSION"
  else
    die "uv $_uvv 太老,项目要求 ≥ $MIN_UV_VERSION(pyproject.toml 的 build-system)"
  fi
fi

# ============================ 5. 装 CPython ============================
say "5/8 准备 CPython $PY_VER"
# 【为什么这一节不能省】
# uv 会按 .python-version 下载【托管】的 CPython,默认源是 GitHub releases。
# 实测 GitHub 会 302 到 release-assets.githubusercontent.com —— 国内常年不通,
# 表现是 uv sync 卡在 "Downloading cpython-3.10.x" 不动。
# 改成 NJU 镜像(它镜像了 astral-sh/python-build-standalone 的全部 release)。
export UV_PYTHON_INSTALL_MIRROR="$MIRROR_PYTHON"
ok "UV_PYTHON_INSTALL_MIRROR=$UV_PYTHON_INSTALL_MIRROR"
printf '  (NJU 布局是 <base>/<tag>/<文件名>,实测 cpython-3.10.21 x86_64 linux 直出 42MB 无跳转)\n'

if [ "$MODE" = "check" ]; then
  warn "(体检)未安装 Python。实际执行会跑:uv python install $PY_VER"
else
  uv python install "$PY_VER" || die "安装 CPython $PY_VER 失败 —— 多半是镜像地址不对,用 --mirror-python 换"
  ok "CPython $PY_VER 就位:$(uv python find "$PY_VER" 2>/dev/null)"
fi

# ============================ 6. nvcc ============================
# uv 路线唯一的真难点。详见文件头 ⚠️:nvcc 的 major 必须等于 torch 的 CUDA major(12)。
say "6/8 定位 nvcc"

nvcc_major() {  # $1 = nvcc 路径 → 打印 major(如 12)
  "$1" --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1
}
nvcc_full() {   # $1 = nvcc 路径 → 打印 x.y
  "$1" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1
}

CUDA_ROOT=""
# 逐个候选看过去,取第一个 major 匹配的
for _cand in ${CUDA_HOME:+"$CUDA_HOME"} /usr/local/cuda /usr/local/cuda-12 \
             /usr/local/cuda-12.8 /usr/local/cuda-12.9 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
  [ -x "$_cand/bin/nvcc" ] || continue
  _m="$(nvcc_major "$_cand/bin/nvcc")"
  printf '  候选 %-26s nvcc %s\n' "$_cand" "$(nvcc_full "$_cand/bin/nvcc")"
  if [ "$_m" = "$TORCH_CUDA_MAJOR" ]; then CUDA_ROOT="$_cand"; break; fi
done

if [ -n "$CUDA_ROOT" ]; then
  export CUDA_HOME="$CUDA_ROOT"
  export PATH="$CUDA_HOME/bin:$PATH"
  ok "CUDA_HOME=$CUDA_HOME (nvcc $(nvcc_full "$CUDA_HOME/bin/nvcc"),major 匹配 torch 的 cu$TORCH_CUDA_MAJOR)"
  # 编 _C 需要的不只是 nvcc,还有头文件 —— ATen 无条件 include cublas_v2.h 等
  for _h in cuda_runtime.h cublas_v2.h cublasLt.h cusparse.h cusolverDn.h; do
    if [ -f "$CUDA_HOME/include/$_h" ]; then ok "  include/$_h"
    else warn "  缺 include/$_h —— 编译大概率过不去(需要完整 toolkit,不是只有 nvcc)"
    fi
  done
else
  # 没找到 major 匹配的。把实际看到的版本说清楚,别让人以为"没装 CUDA"。
  _seen="$(for c in /usr/local/cuda*; do [ -x "$c/bin/nvcc" ] && printf '%s ' "$(nvcc_full "$c/bin/nvcc")"; done)"
  warn "没有找到 major=$TORCH_CUDA_MAJOR 的 nvcc。镜像里实际有:${_seen:-无}"

  if [ "$USE_PIP_NVCC" = 1 ] && [ "$MODE" != "check" ]; then
    # ⚠️⚠️ 以下路径 2026-09-30 时【尚未实测通过】—— 交接文档 §7.1 把它列为待验证项。
    # 思路:nvidia-cuda-nvcc-cu12 提供 12.x 的 nvcc;头文件靠 torch wheel 自带的
    # nvidia-*-cu12 包;把两者拼成一个假 CUDA_HOME 喂给 cpp_extension。
    warn "--pip-nvcc:尝试用 PyPI 的 nvidia-cuda-nvcc-cu12 拼一个假 CUDA_HOME(未验证方案)"
    uv pip install --python "$PROJECT_DIR/.venv/bin/python" nvidia-cuda-nvcc-cu12 2>/dev/null \
      || "$(command -v python3 || command -v python)" -m pip install -q nvidia-cuda-nvcc-cu12 -i "$MIRROR_TUNA" \
      || die "装 nvidia-cuda-nvcc-cu12 失败"
    _sp="$("$(command -v python3 || command -v python)" -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null)"
    FAKE="$DATA_DISK/fake-cuda"
    mkdir -p "$FAKE/bin" "$FAKE/include" "$FAKE/lib"
    ln -sf "$_sp/nvidia/cuda_nvcc/bin/nvcc" "$FAKE/bin/nvcc" 2>/dev/null
    # 头文件从 torch 自带的 nvidia-*-cu12 里凑
    for d in cuda_runtime cublas cusparse cusolver; do
      [ -d "$_sp/nvidia/${d}/include" ] && cp -n "$_sp/nvidia/${d}/include/"* "$FAKE/include/" 2>/dev/null
    done
    export CUDA_HOME="$FAKE"; export PATH="$FAKE/bin:$PATH"
    if [ -n "$(nvcc_full "$FAKE/bin/nvcc")" ]; then
      ok "假 CUDA_HOME 已搭好:$FAKE (nvcc $(nvcc_full "$FAKE/bin/nvcc"))"
      warn "这是未验证路径。若第 8 步 _C 没编出来,请改用 conda 版脚本。"
    else
      die "假 CUDA_HOME 没搭起来(nvcc 不可执行)"
    fi
  else
    cat <<EOF

      两条出路:

      1) 装一个 CUDA 12.x toolkit,让 nvcc 与 torch 的 cu$TORCH_CUDA_MAJOR 同 major。
         AutoDL 的镜像通常自带 /usr/local/cuda-12.x(交接文档实测实例 1 上有);
         换镜像时优先挑带 12.x 的那个。

      2) 加 --pip-nvcc,用 PyPI 的 nvidia-cuda-nvcc-cu12 拼假 CUDA_HOME。
         ⚠️ 这条路【未经实测】,而且 nvidia-cuda-nvcc-cu12 只给 nvcc 不给头文件,
         需要从 torch 自带的 nvidia-*-cu12 里凑 include/,未必凑得齐。

      3) 直接用 setup_cloud_conda.sh —— nvcc 与 torch 由同一个 solver 解出,
         必然同源。代价是 12.6G env + 4.3G 下载 + 284 秒 solve,以及 torch 被
         顶到 2.13.0+cu129。

      不想现在决定,可以先跑:bash $(basename "$0") --check

EOF
    die "没有可用的 nvcc"
  fi
fi
export MAX_JOBS

# ============================ 7. uv sync ============================
say "7/8 uv sync --frozen"
# --frozen:严格按 uv.lock 来,一个字节都不重新解析。锁里是完整 URL + sha256,
#          所以这一步同时还是一次完整性校验 —— 镜像若被篡改会直接失败。
# no-build-isolation-package:已写进 pyproject.toml 的 [tool.uv],这里再显式带一次,
#          是为了防止有人用没更新过的旧 pyproject 跑这个脚本。
#          上游 setup.py 顶层会 import torch,隔离构建环境里没有 torch,它会自己去
#          pip install torch 然后失败(实测:uv lock --refresh 就是这么炸的)。
if [ "$MODE" = "check" ]; then
  warn "(体检)未执行。实际执行会跑:uv sync --frozen --no-build-isolation-package groundingdino"
  say "体检模式(--check):到此为止,未做任何改动"
  echo "  uv      : $(uv --version 2>/dev/null || echo 未装)"
  echo "  nvcc    : ${CUDA_HOME:-未定位}"
  echo "  下一步  : bash $(basename "$0")"
  exit 0
fi

warn "接下来会长时间静默,不是卡死:"
printf '       · 下载 ~3G(torch 单个 wheel 就 916M)\n'
printf '       · 编译 _C(唯一真正耗 CPU 的一步,MAX_JOBS=%s)\n' "$MAX_JOBS"
printf '     想确认它还活着:另开终端看网卡/CPU,别 Ctrl-C。\n'

uv sync --frozen --no-build-isolation-package groundingdino
RC=$?
[ $RC -eq 0 ] || die "uv sync 失败(退出码 $RC)。常见原因:
       · 校验失败(sha256 不符) → 镜像内容与官方不一致,换 --mirror-* 重试
       · No solution / lock 过期  → 说明 uv.lock 与 pyproject.toml 不同步,
         在【本机】跑 `uv lock` 后重新上传锁文件
       · 编译报错              → 看第 6 步的 nvcc 与头文件;major 必须等于 $TORCH_CUDA_MAJOR"

VENV_PY="$PROJECT_DIR/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="${UV_PROJECT_ENVIRONMENT:-$PROJECT_DIR/.venv}/bin/python"
[ -x "$VENV_PY" ] || die "uv sync 报成功但找不到 venv 里的 python,检查 .venv 位置"
ok "环境已就绪:$VENV_PY"

# ============================ 8. 验证 ============================
say "8/8 验证"

"$VENV_PY" - <<PY || die "校验失败"
import sys, torch
print('  torch     :', torch.__version__)
print('  cuda      :', torch.version.cuda)
print('  python    :', sys.version.split()[0])
print('  算力      :', torch.cuda.get_device_capability())
from torch.utils.cpp_extension import CUDA_HOME
print('  torch 眼中的 CUDA_HOME :', CUDA_HOME)
assert torch.cuda.is_available(), 'CUDA 不可用'

# 真正跑一次 GPU 运算,别只看 is_available()。
# 若 wheel 没编进你这张卡的算力,这里才会暴露
# "no kernel image is available for execution"。
# (cu128 的 wheel 不带 sm_89 内核,4090 上靠 PTX JIT 兜 —— 所以这一步必须真跑。)
a = torch.randn(512, 512, device='cuda'); b = a @ a
torch.cuda.synchronize()
print('  GPU 矩阵乘 :', tuple(b.shape), '✓')
PY

# ── _C 必须真的编出来了 ────────────────────────────────────────────
# 这是本脚本最要紧的一条断言。upstream setup.py 在找不到 nvcc 时会
# 【静默 return None】不编译任何扩展,而 pip/uv 都会报"安装成功"。
# 而运行期 ms_deform_attn.py:334 只看张量在不在 GPU,不看 _C 导没导进来,
# 所以症状会延迟到第一次前向才炸成 NameError —— 那时你已经在跑训练了。
# 必须先 import torch:_C.so 是 setuptools 老后端编出来的,不带 RPATH,
# 靠 torch 先把 libc10/libtorch 载进进程;单独 import _C 会报
# "libc10.so: cannot open shared object file"。
"$VENV_PY" - <<'PY' || die "_C 没编出来 —— 见上面报错。
       若提示 Failed to load custom C++ ops / NameError,说明 nvcc 没被认到。
       回第 6 步看 CUDA_HOME;实在不行用 setup_cloud_conda.sh。"
import torch
from groundingdino import _C
print('  _C        :', _C.__file__)
PY

# 最后确认一遍 .so 里真的含本机算力,而不是捡了上一台机器留下的
_so_now="$(find "$PROJECT_DIR/GroundingDINO" -maxdepth 3 -name '_C*.so' 2>/dev/null | head -1)"
if [ -n "$_so_now" ] && command -v cuobjdump >/dev/null 2>&1; then
  if cuobjdump --list-elf "$_so_now" 2>/dev/null | grep -q "sm_${GPU_CAP}"; then
    ok "_C.so 含 sm_${GPU_CAP}(与当前卡匹配)"
  else
    warn "_C.so 里没看到 sm_${GPU_CAP} —— 若训练时报 no kernel image,删掉重编:
         rm -f $_so_now && bash $(basename "$0")"
  fi
fi

printf '\n%s  环境就绪 ✓%s\n' "$C_G" "$C_0"
cat <<EOF
  venv   : $VENV_PY
  激活   : source $(dirname "$(dirname "$VENV_PY")")/bin/activate
  锁定   : uv.lock(完整 URL + sha256)
  重建   : uv sync --frozen --no-build-isolation-package groundingdino

  ⚠️ 还没完:本脚本只证明了环境能装、_C 能编、GPU 能算。
     训练本身【至今没有验证过】,请务必实际跑一次:
         $VENV_PY run_grounding.py --num-images 20
EOF
