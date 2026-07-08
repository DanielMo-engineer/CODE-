#!/bin/bash
# =============================================================
# 摔倒检测启动（不会触发看门狗误报）
#   1. 创建 .restarting 标记 → 看门狗自动闭嘴
#   2. 启动程序
#   3. 等端口就绪 → 删除标记
#   4. 看门狗恢复工作
# =============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESTART_FLAG="$SCRIPT_DIR/.restarting"

echo "=== 启动 摔倒检测 ==="

# 1. 创建启动标记（看门狗检测到它就不会发警报）
touch "$RESTART_FLAG"
echo "[1/3] 已设置启动标记，看门狗已闭嘴"

# 2. 启动程序
echo "[2/3] 启动程序..."
sudo bash "$SCRIPT_DIR/run_fall_detection.sh" --bpu-cores 0 1 &

# 3. 等端口就绪 + 删除标记
echo -n "[3/3] 等待 8080 就绪..."
for i in $(seq 1 20); do
    if ss -tlnp | grep -q 8080; then
        echo " ✅"
        rm -f "$RESTART_FLAG"
        echo "     已清除启动标记，看门狗恢复监控"
        echo ""
        echo "=== 启动完成 ==="
        echo "远程查看: https://52b04c9c.r11.vip.cpolar.cn"
        exit 0
    fi
    sleep 1
done

# 超时：也清除标记，避免看门狗永远沉默
echo " ⚠️ 超时（但仍会在后台继续启动）"
rm -f "$RESTART_FLAG"
exit 1
