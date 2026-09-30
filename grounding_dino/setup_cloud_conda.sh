#!/usr/bin/env bash
# ============================================================================
#  GroundingDINO 云端环境部署 —— conda 版
#
#  在【云端实例】里执行,git clone 之后跑这一条就够了:
#
#      cd <项目>/grounding_dino
#      bash setup_cloud_conda.sh
#
#  ────────────────────────────────────────────────────────────────────────────
#  为什么用 conda 而不是 uv(与 setup_cloud_env.sh 的区别)
#
#  1. CUDA 工具链变成环境内的 conda 包,nvcc 与 torch 天然同源
#     uv 版最痛的是:nvcc 是镜像自带的,是多少全看运气,而 PyTorch 编译 _C 前
#     会拿 nvcc 版本和 torch.version.cuda 严格比对(只特批 11.0/11.1),不一致
#     直接 RuntimeError。uv 管不了 nvcc,只能写八十行代码去"找适配的 nvcc →
#     没有就 apt 装 → 再改 CUDA_HOME"。
#     conda 版里 cuda-nvcc / cuda-cudart-dev / libcublas-dev 都是环境内的包,
#     和 pytorch 的 cuda129 变体由同一个 solver 一次解出,版本必然一致。
#
#  2. 所有包都能从同一个国内镜像拿到
#     实测清华 conda-forge 镜像包含:python 3.10.21 / pytorch 2.13.0(cuda129) /
#     cuda-nvcc 12.9.86 / transformers 4.57.6 / supervision / pycocotools /
#     timm / addict / yapf / opencv,完整依赖树 dry-run 可解。
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
#  ⚠️ 体积(2026-09-30 实测,别被吓到):
#     conda-forge 把 pytorch 编成"链接外部 CUDA 库"的形态,所以
#     nccl / libmagma / mkl / cublas / cudnn ... 每一个都是 libtorch 的硬依赖
#     (DT_NEEDED),删不掉。代价是:
#         压缩包 4.3G(412 个) / pkgs 落盘 18G(含解压镜像) / env ≈ 12.6G
#     作为对照,uv 那条路(自包含 pip 轮子)的 .venv 实测是 7.4G。
#     换来的是"nvcc 与 torch 同源、编译零折腾"。这是明码标价的取舍,不是 bug。
#
#  ────────────────────────────────────────────────────────────────────────────
#  可选参数:
#      --check          只体检,不做任何改动
#      --name NAME      conda 环境名(默认 gdino)
#      --cuda {129,130} CUDA 变体(默认 129;130 对应镜像自带 13.x 的机器)
#      --jobs N         编译 _C 的并行数(默认 4)
#      --force          环境已存在也删掉重建
#      --mirror URL     覆盖 conda-forge 镜像地址
#      --write-condarc  把镜像配置写进 ~/.condarc(默认不动你已有的配置)
#      --foreground     不要自动进 screen(默认会自动进,见下)
#      --no-data-disk   不把缓存/环境放到数据盘(默认会放,见下)
# ============================================================================

# ⚠️ 刻意【不】加 -u(set -u),这不是疏忽。
# conda 的 activate/deactivate hook 是在【当前 shell 里 source】的,所以会继承
# 这里的 set 选项。而 conda-forge 的 libblas_mkl_deactivate.sh 第 1 行是
#     if [ "${CONDA_MKL_INTERFACE_LAYER_BACKUP}" = "" ]     # 少写了 :-
# 引用了可能未定义的变量。set -u 下这会报 unbound variable,而【非交互 shell
# 遇到这个错会直接退出整个 shell】—— 后果是脚本在 conda install 刚跑完、
# 正准备进第 7 步时装 GroundingDINO 时静默死掉。
# 2026-09-30 实测踩到:env 装好了(12G),但 groundingdino 没装、_C 没编、
# lock 没导出,日志停在一条 MKL 报错上,看不出是脚本自己退出了。
# 所以这一行只能是 pipefail,-u 换不来什么,雷却是实打实的。
set -o pipefail

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
FOREGROUND=0; NO_DATA_DISK=0
ORIG_ARGS=("$@")

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
    --foreground)    FOREGROUND=1 ;;
    --no-data-disk)  NO_DATA_DISK=1 ;;
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
LOG="$PROJECT_DIR/setup_conda.log"
SCREEN_NAME="gdino-setup"

# ============================ 0. 防断线 ============================
# 【为什么有这一节】
# 这个脚本要跑 20 分钟以上,其中大部分时间是 conda 在下 4.3G 的包。
# 2026-09-30 实测过一次事故:脚本裸跑在 SSH 会话里(没有 nohup/screen),
# 会话一断,SIGHUP 广播给整个进程组,5 路并行下载在 0.3 秒内被一起掐死 ——
# 现场留下 6 个 .partial(libtorch 220M / libcudnn 264M / libmagma 215M /
# nccl 100M / libcublas 93M / libglx-devel 0 字节),合计 854M 白下,
# 而当时其实已经下到 ~80% 了。
#
# 那次事故的文件系统签名很有辨识度:六个互相独立的 TCP 连接在同一瞬间定格。
# 如果是网络问题,它们会一个接一个地失败,时间戳是散的。同时停下 = 父进程没了。
# 所以这里默认把自己塞进 screen:断了也照跑,回来 tail 日志就行。
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

case "$PROJECT_DIR" in
  /root/autodl-tmp/*|/root/autodl-fs/*) ;;
  /root/*) warn "项目在系统盘(30G)。真正吃盘的是 conda 缓存和 env(合计约 18G),见下一步" ;;
esac

# ============================ 2. 镜像 ============================
say "2/8 conda 镜像"
printf '  当前 channels : %s\n' "$(conda config --show channels 2>/dev/null | tr -d ' ' | paste -sd' ')"
[ -f "$HOME/.condarc" ] && ok "已有 ~/.condarc(尊重你的配置,本脚本默认不改动)" || warn "没有 ~/.condarc"
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

# 【为什么探测方式是 HEAD 而不是 conda search】
# 旧版这里是:
#     conda search --override-channels -c "$CONDA_MIRROR" cuda-nvcc
# 为了找一个包,conda 会先把该 channel 的【完整 repodata】全拉下来并解析。
# 2026-09-30 实测,conda-forge 的 repodata 解压后是:
#     linux-64  430M
#     noarch    181M
# 缓存里 *.info.json 记的 URL 证实了这一点。用户那次从 14:37 走到 14:41,
# 四分钟基本都花在这,而且屏幕上一行 `...working...` 看着像卡死。
#
# 一个 HEAD 请求就能验完 DNS + TLS + HTTP + 路径存在 —— 实测 0.66 秒。
# ⚠️ 老实说:这 430M 后面 conda install 时也要用,所以删掉检查【并不省字节】,
#    只是把这笔开销挪回它该在的地方,并且让"镜像不通"在一秒内暴露,而不是几分钟后。
probe_mirror() {
  local base="$1" code="" path
  for path in linux-64/repodata.json noarch/repodata.json; do
    code="$(curl -sI -o /dev/null -w '%{http_code}' --max-time 10 "$base/$path" 2>/dev/null)"
    case "$code" in 200) printf '%s' "$code"; return 0 ;; esac
    # 有些镜像禁 HEAD,退化成只取 1 字节的 GET(206)
    code="$(curl -s -o /dev/null -r 0-0 -w '%{http_code}' --max-time 10 "$base/$path" 2>/dev/null)"
    case "$code" in 200|206) printf '%s' "$code"; return 0 ;; esac
  done
  printf '%s' "${code:-000}"; return 1
}

if [ "$MODE" != "check" ]; then
  printf '  探测镜像(HTTP HEAD) ... '
  _t0=$(date +%s%3N)
  if _code="$(probe_mirror "$CONDA_MIRROR")"; then
    _el="$(awk -v ms="$(( $(date +%s%3N) - _t0 ))" 'BEGIN{printf "%.2f", ms/1000}')"
    printf '%s✓%s HTTP %s(%ss)\n' "$C_G" "$C_0" "$_code" "$_el"
  else
    die "conda-forge 镜像不可达:$CONDA_MIRROR (HTTP $_code)
       换个镜像(如 https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge)
       或先确认这台机器能不能访问国内镜像。"
  fi
fi

# ============================ 3. 缓存/环境目录 ============================
# 【为什么有这一节】
# conda 的 pkgs_dirs 默认是 <conda根>/pkgs,【完全不看项目放在哪】。
# 而实测:这个环境要下 4.3G 压缩包,落盘 18G(pkgs 里同时存压缩包和解压镜像),
# env 约 12.6G。系统盘只有 30G —— 光靠"把项目放数据盘"是躲不掉的。
#
# ⚠️ pkgs_dirs 和 envs_dirs 必须【一起搬】。只搬 pkgs 的话,conda 在 env 侧
#    无法 hardlink,会退化成 copy,env 那 12.6G 反而实打实压在系统盘上。
#
# 但如果这台机器【已经】在系统盘下过一批包了,就别搬 —— 搬了等于把已下载的
# 几个 G 丢掉重下。所以下面的规则是:缓存还很小(全新机器)→ 搬去数据盘;
# 已经有料 → 原地沿用,只核对剩余空间够不够。
say "3/8 缓存与环境目录"
DATA_DISK="/root/autodl-tmp"
CONDA_ROOT=""

CUR_PKGS="$(conda config --show pkgs_dirs 2>/dev/null | awk '/^ *- /{print $2; exit}')"
CUR_PKGS="${CUR_PKGS:-$HOME/miniconda3/pkgs}"
CUR_MB=0
[ -d "$CUR_PKGS" ] && CUR_MB="$(du -sm "$CUR_PKGS" 2>/dev/null | cut -f1)"
CUR_MB="${CUR_MB:-0}"
printf '  现有 pkgs_dirs : %s (%s MB)\n' "$CUR_PKGS" "$CUR_MB"

if [ "$NO_DATA_DISK" = 1 ]; then
  warn "--no-data-disk:缓存和环境都留在默认位置"
elif [ ! -d "$DATA_DISK" ]; then
  warn "没有 $DATA_DISK,只能用系统盘 —— 注意系统盘 30G 要装 ~18G,会很紧"
elif [ "$CUR_MB" -ge 2048 ]; then
  ok "已有 ${CUR_MB}MB 缓存在系统盘 —— 原地沿用(搬走等于重下,不划算)"
  _free="$(df -Pm "$CUR_PKGS" | awk 'NR==2{print $4}')"
  if [ "${_free:-0}" -lt 8000 ]; then
    warn "但该盘只剩 ${_free}MB。装完 env 可能不够,建议先清:conda clean -a -y"
  else
    ok "该盘剩 ${_free}MB,够装(env 与 pkgs 同盘,走 hardlink,不额外占空间)"
  fi
else
  CONDA_ROOT="$DATA_DISK/conda"
  mkdir -p "$CONDA_ROOT/pkgs" "$CONDA_ROOT/envs"
  ok "缓存与环境改到数据盘:$CONDA_ROOT"
  if [ "$MODE" != "check" ]; then
    cp "$HOME/.condarc" "$HOME/.condarc.bak.$$" 2>/dev/null
    conda config --show pkgs_dirs 2>/dev/null | grep -qF "$CONDA_ROOT/pkgs" \
      || conda config --add pkgs_dirs "$CONDA_ROOT/pkgs"
    conda config --show envs_dirs 2>/dev/null | grep -qF "$CONDA_ROOT/envs" \
      || conda config --add envs_dirs "$CONDA_ROOT/envs"
    ok "已写入 ~/.condarc(两个一起,缺一个就会退化成 copy)"
  fi
fi

# ============================ 4. uv 残留 ============================
# 【为什么这里不再是无条件打印"已清除"】
# 旧版不管有没有删到东西,都打印
#     ✓ 已清除旧 CUDA_HOME 干扰(conda 环境里的 nvcc 才是准的)
# 在全新实例上这是【假消息】——2026-09-30 实测那台:command -v uv 无、
# /root/.local/bin/uv 不存在、.bashrc 里三个关键字零匹配。什么都没清,
# 却报"已清除",还白占一个步骤号让人以为有事发生。现在改成有才报。
#
# 旧 setup_cloud_env.sh 往 ~/.bashrc 写过 CUDA_HOME=/usr/local/cuda-12.8 和
# 对应 PATH。那些值会【覆盖】conda 环境里的 nvcc,导致 torch 又去跟镜像自带
# 的 12.8/13.0 比对,直接回到旧问题。所以真有的话必须删。
say "4/8 清理 uv 版脚本留下的环境变量"
LEFTOVER=0
clean_rc() {
  grep -qF "$1" "$HOME/.bashrc" 2>/dev/null || return 0
  LEFTOVER=1
  if [ "$MODE" = "check" ]; then printf '  (体检)发现残留: %s\n' "$1"; return 0; fi
  cp "$HOME/.bashrc" "$HOME/.bashrc.bak.$$"
  grep -vF "$1" "$HOME/.bashrc.bak.$$" > "$HOME/.bashrc"
  printf '  已从 .bashrc 删除: %s(备份 .bashrc.bak.$$)\n' "$1"
}
clean_rc 'export CUDA_HOME=/usr/local/cuda-12.8'
clean_rc '/usr/local/cuda-12.8/bin'
clean_rc 'export UV_PYTHON_INSTALL_MIRROR'
[ "$LEFTOVER" = 1 ] || ok "无 uv 版残留(这台没装过 uv,不需要清理)"
# 当前 shell 里也可能已经有(登录时 source 过),无条件抹掉:下一节会设成 CONDA_PREFIX
unset CUDA_HOME

# ============================ 5. 创建环境 ============================
say "5/8 创建 conda 环境 [$ENV_NAME]"

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

# ============================ 6. 装依赖 ============================
say "6/8 安装依赖(全走清华 conda-forge)"

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
#                       图形依赖(实测 qt6-main 一项解压就 349M)。纯 headless
#                       跑可以忽略,想瘦身就把 opencv 换掉。
#
# ⚠️ 体积实测(2026-09-30):压缩包 4.3G / pkgs 落盘 18G / env ≈ 12.6G。
#    最大的几项:libtorch 2159M、libmagma 1264M、libcudnn 1000M、libcublas 816M、
#    nccl 750M、mkl 684M、libcusolver 471M、libcusparse 464M、sysroot 460M。
#    这些都是 libtorch 的硬依赖,删不掉(详见文件头 ⚠️)。
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

# 【为什么要先查一遍标志包】
# 已经装好的环境再跑一次 conda install,conda 仍会完整 solve 一遍(实测 284s),
# 什么都没变却要等五分钟。而重跑这个脚本是常态(第 7 步编译失败、断线、
# 只想补跑验证段……)。所以先看两个标志包在不在,在就跳过。--force 可强制重装。
if [ "$FORCE" != 1 ] \
   && conda list -n "$ENV_NAME" libtorch 2>/dev/null | grep -q '^libtorch ' \
   && conda list -n "$ENV_NAME" cuda-nvcc 2>/dev/null | grep -q '^cuda-nvcc '; then
  ok "libtorch 与 cuda-nvcc 已在环境里 —— 跳过 conda install(省掉一次 ~5 分钟的 solve)"
  printf '     要强制重装:bash %s --force\n' "$(basename "$0")"
else

# 【为什么要有这句提示】
# 2026-09-30 实测:"Solving environment" 这一步花【284 秒(4.7 分钟)】,
# 期间屏幕上一行 `Solving environment: ...working...` 完全不动、零下载。
# 用户反馈的"下载速度有点慢",绝大部分其实是在等这个 —— 它看起来和卡死
# 没有任何区别。先说清楚,省得又被 Ctrl-C 掉。
_wait_note=""
[ -d "$CUR_PKGS" ] && [ "$CUR_MB" -lt 2048 ] && _wait_note="(首次运行还要额外下 ~4.3G,合计约 15~20 分钟)"
warn "接下来会有两步长时间静默,都不是卡死:"
printf '       · Solving environment  ≈ 5 分钟(实测 284s),无进度条\n'
printf '       · 下载 4.3G            ≈ 8 分钟(实测 8~11 MB/s,已是这台机的带宽上限)\n'
printf '     %s\n' "${_wait_note:-合计约 15 分钟。}"
printf '     想确认它还活着:另开一个终端看网卡/磁盘,别 Ctrl-C。\n'
echo "  ────────────────────────────────────────────────"

conda install -y -n "$ENV_NAME" --override-channels -c "$CONDA_MIRROR" "${CONDA_PKGS[@]}"
RC=$?
[ $RC -eq 0 ] || die "conda install 失败(退出码 $RC)。见上面报错:
       · PackagesNotFoundError → 该镜像缺包,试 --mirror 换 USTC/阿里镜像
       · 依赖冲突(UnsatisfiableError) → 多半是某个包版本钉太死,
         先松开对应版本号(改脚本顶部的版本锚点),再跑 --force
       · 磁盘满(No space left) → 看第 3 步的提示,把缓存/环境挪到数据盘"
# 失败的 conda install 会留下 *.partial(见文件头那次事故),重跑时反正要重下,
# 先清掉免得白占几 G。实际用的 pkgs 目录按第 3 步的选择定。
REAL_PKGS="${CONDA_ROOT:+$CONDA_ROOT/pkgs}"; REAL_PKGS="${REAL_PKGS:-$CUR_PKGS}"
if compgen -G "$REAL_PKGS/*.partial" >/dev/null 2>&1; then
  _p="$(du -ch "$REAL_PKGS"/*.partial 2>/dev/null | tail -1 | cut -f1)"
  warn "清掉上次中断留下的 .partial(共 $_p,重跑时反正要重下)"
  rm -f "$REAL_PKGS"/*.partial
fi

fi   # ← 跳过检查的 else

# ============================ 7. 装本项目 ============================
say "7/8 安装 GroundingDINO 本体(editable,只读本地,不联网)"

# GroundingDINO 是纯 setup.py 的上游仓库,conda 装不了 editable,只能 pip。
# 但它 setup.py 顶层会 `import torch`,所以必须 --no-build-isolation —— 否则
# pip 会新建一个隔离构建环境,里面没有 torch,setup.py 会去 pip install torch。
# --no-deps + PIP_NO_INDEX=1 是双重保险:依赖已全部由 conda 装好,pip 不该、也
# 不允许去网上拿任何东西。这保证这一步在无外网时也能成功。
[ -f "$PROJECT_DIR/GroundingDINO/setup.py" ] || die "GroundingDINO/setup.py 不存在"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
ok "CUDA_HOME=$CUDA_HOME(指向 conda 环境,不是 /usr/local/cuda)"
printf '  环境内 nvcc : %s\n' "$(nvcc -V 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo 缺失)"

PIP_NO_INDEX=1 pip install -e "$PROJECT_DIR/GroundingDINO" \
  --no-build-isolation --no-deps -q \
  || die "pip install -e 失败。
       注意这里刻意不联网(--no-deps + PIP_NO_INDEX=1),报 'No matching distribution'
       就说明有依赖没被 conda 装上,把包名补进上面的 CONDA_PKGS 再跑。
       报 nvcc/CUDA 相关的编译错误,先看上面那行 nvcc 版本是不是 $CUDA_VER。"
ok "GroundingDINO 已 editable 安装"

# ============================ 8. 导出锁定文件 ============================
say "8/8 导出环境锁定文件"
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
