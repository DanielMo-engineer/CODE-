#!/bin/bash
# =============================================================
# 摔倒检测优雅重启（不会触发看门狗误报）
#   1. 创建 .restarting 标记 → 看门狗自动闭嘴
#   2. 停旧程序
#   3. 启动新程序
#   4. 等端口就绪 → 删除 .restarting 标记
#   5. 看门狗恢复工作
# =============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESTART_FLAG="$SCRIPT_DIR/.restarting"

echo "=== 优雅重启 摔倒检测 ==="

# 1. 创建重启标记（看门狗检测到它就不会发警报）
touch "$RESTART_FLAG"
echo "[1/4] 已设置重启标记，看门狗已闭嘴"

# 2. 停旧程序
echo "[2/4] 停止旧进程..."
sudo pkill -f "fall_detection.py" 2>/dev/null || true
sleep 2

# 3. 启动新程序
echo "[3/4] 启动新进程..."
sudo bash "$SCRIPT_DIR/run_fall_detection.sh" --bpu-cores 0 1 &

# 4. 等端口就绪 + 删除标记
echo -n "[4/4] 等待 8080 就绪..."
for i in $(seq 1 20); do
    if ss -tlnp | grep -q 8080; then
        echo " ✅"
        rm -f "$RESTART_FLAG"
        echo "     已清除重启标记，看门狗恢复监控"
        echo ""
        echo "=== 重启完成 ==="
        exit 0
    fi
    sleep 1
done

# 超时：也清除标记，避免看门狗永远沉默
echo " ⚠️ 超时（但仍会在后台继续启动）"
rm -f "$RESTART_FLAG"
exit 1
