#!/bin/bash
# =============================================================
# RDK X5 智能语音助手 - 启动脚本
# =============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 加载配置文件（API 密钥等）
CONFIG_FILE="$SCRIPT_DIR/.env"
if [ -f "$CONFIG_FILE" ]; then
  set -a
  source "$CONFIG_FILE"
  set +a
fi

# 检查必要环境变量
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "[WARN] 未设置 DEEPSEEK_API_KEY"
    echo "  请在 .env 文件中添加: DEEPSEEK_API_KEY=sk-your-key"
    echo "  或 export DEEPSEEK_API_KEY=sk-your-key"
    echo ""
fi

# 检查 VOSK 模型
MODEL_DIR="$HOME/.vosk/models/vosk-model-small-cn-0.22"
if [ ! -d "$MODEL_DIR" ]; then
    echo "[INFO] 正在下载 VOSK 中文语音模型 (~42MB)..."
    mkdir -p "$(dirname "$MODEL_DIR")"
    cd "$(dirname "$MODEL_DIR")"
    wget -q --show-progress https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
    echo "[INFO] 解压中..."
    unzip -q vosk-model-small-cn-0.22.zip
    rm vosk-model-small-cn-0.22.zip
    echo "[OK] VOSK 模型就绪"
fi

echo "================================"
echo " RDK X5 智能语音助手"
echo "================================"
echo ""
echo "说 \"你好小智\" 唤醒"
echo "Ctrl+C 退出"
echo ""

exec python3 "$SCRIPT_DIR/voice_assistant.py" "$@"
