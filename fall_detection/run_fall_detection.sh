#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 从 .env 配置文件读取所有环境变量（环境变量优先）
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

# YOLO11n-pose 模型：同时检测人体框 + 17个COCO关键点
POSE_MODEL="/app/pydev_demo/04_pose_sample/01_ultralytics_yolo11_pose/yolo11n_pose_bayese_640x640_nv12.bin"

echo "================================"
echo " RDK X5 摔倒检测 — 姿态骨骼版"
echo " 模型: $(basename $POSE_MODEL)"
echo "================================"
echo ""

if [ ! -f "$POSE_MODEL" ]; then
    echo "[ERROR] 姿态模型未找到: $POSE_MODEL"
    echo "  >>> 请先部署 YOLO11n-pose 模型到该路径 <<<"
    ls /app/pydev_demo/04_pose_sample/01_ultralytics_yolo11_pose/*.bin 2>/dev/null
    exit 1
fi

echo "[OK] 模型: $(basename $POSE_MODEL)"
echo "[OK] 输出: 人体检测框 + 17个骨骼关键点"
echo ""

# 关键点置信度说明：
#   --kpt-conf 0.3  → 显示30%以上置信度的关键点（默认）
#   --kpt-conf 0.5  → 只显示50%以上（更干净）
#   --kpt-conf 0.1  → 显示所有点（可能杂点较多）
#
# 关键点颜色说明：
#   头部(0-4): 青色  肩膀(5-6): 橙色
#   手臂(7-10): 绿色  躯干(11-12): 蓝色
#   腿(13-16): 品红色

exec /usr/bin/python3.10 "$SCRIPT_DIR/fall_detection.py" \
    --model "$POSE_MODEL" \
    --camera-width 640 \
    --camera-height 640 \
    --camera-port 0 \
    --fps 30 \
    --score 0.25 \
    --kpt-conf 0.3 \
    --confirm 10 \
    --cooldown 5 \
    --bpu-cores 0 1 \
    "$@"
