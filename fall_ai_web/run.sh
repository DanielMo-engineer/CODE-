#!/bin/bash
# =============================================================
# RDK X5 综合服务平台 — 启动脚本
# 主页: http://<IP>:5050
# =============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 加载 .env 配置
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

echo "═" × 50
echo " 🚀 RDK X5 综合服务平台"
echo "   端口: 5050"
echo "   主页: http://\$(hostname -I | cut -d' ' -f1):5050"
echo "═" × 50

exec python3 "$SCRIPT_DIR/app.py"
