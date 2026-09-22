#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
SDK_WHEEL="${ARM_SDK_WHEEL:-}"
ARM_DEB="${AIRBOT_ARM_DEB:-}"
SKIP_SYSTEM=false
SKIP_MODEL=false

usage() {
    cat <<'EOF'
用法：
  ./install.sh [--sdk-wheel FILE] [--arm-deb FILE] [--skip-system] [--skip-model]

推荐把 5.2.2 软件包放到项目 packages/ 目录后直接运行 ./install.sh。
也可通过 ARM_SDK_WHEEL、AIRBOT_ARM_DEB、PYTHON_BIN、PIP_INDEX_URL 设置路径。
EOF
}

while (($# > 0)); do
    case "$1" in
        --sdk-wheel) SDK_WHEEL="${2:?--sdk-wheel 缺少路径}"; shift 2 ;;
        --arm-deb) ARM_DEB="${2:?--arm-deb 缺少路径}"; shift 2 ;;
        --skip-system) SKIP_SYSTEM=true; shift ;;
        --skip-model) SKIP_MODEL=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "错误：未知参数 $1" >&2; usage >&2; exit 2 ;;
    esac
done

find_one() {
    local pattern="$1"
    find "$PROJECT_DIR/packages" -maxdepth 1 -type f -name "$pattern" \
        -print -quit 2>/dev/null || true
}

[[ -n "$SDK_WHEEL" ]] || SDK_WHEEL="$(find_one 'arm_sdk-5.2.2-*.whl')"
[[ -n "$ARM_DEB" ]] || ARM_DEB="$(find_one 'airbot-arm_5.2.2_*.deb')"

SDK_ALREADY_INSTALLED=false
if [[ -x "$PROJECT_DIR/venv/bin/python" ]] \
        && "$PROJECT_DIR/venv/bin/python" -c \
        "import arm_sdk; raise SystemExit(arm_sdk.__version__ != '5.2.2')" \
        >/dev/null 2>&1; then
    SDK_ALREADY_INSTALLED=true
fi
if [[ ! -f "$SDK_WHEEL" && "$SDK_ALREADY_INSTALLED" == false ]]; then
    echo "错误：缺少 arm-sdk 5.2.2 wheel。请放入 packages/ 或使用 --sdk-wheel 指定。" >&2
    exit 1
fi
if [[ ! -f "$ARM_DEB" && "$SKIP_SYSTEM" == false ]]; then
    echo "错误：缺少 airbot-arm 5.2.2 deb。请放入 packages/、使用 --arm-deb 指定，" >&2
    echo "或在已安装服务的机器上增加 --skip-system。" >&2
    exit 1
fi

if [[ "$SKIP_SYSTEM" == false ]]; then
    sudo apt-get update
    sudo apt-get install -y python3.12-venv libportaudio2 portaudio19-dev \
        libsndfile1 libxcb-cursor0
    sudo apt-get install -y "$ARM_DEB"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "错误：找不到 Python 3.12：$PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -x "$PROJECT_DIR/venv/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$PROJECT_DIR/venv"
fi

PYTHON="$PROJECT_DIR/venv/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel -i "$PIP_INDEX_URL"
"$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt" -i "$PIP_INDEX_URL"
if [[ -f "$SDK_WHEEL" ]]; then
    "$PYTHON" -m pip install --force-reinstall "$SDK_WHEEL"
fi

if [[ "$SKIP_MODEL" == false ]]; then
    PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
from voice_asr_worker import load_asr
load_asr("paraformer-zh", "cpu")
print("FunASR 模型：OK")
PY
fi

PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
import arm_sdk
import cv2
import funasr
import mobile_sam
import PyQt6
import sounddevice
import soundfile
import torch
import torchaudio
import ultralytics

assert arm_sdk.__version__ == "5.2.2", arm_sdk.__version__
print("统一环境安装完成：Python、视觉、语音、arm-sdk 5.2.2 均可导入")
PY

echo "下一步：./run_grasp.sh --check"
