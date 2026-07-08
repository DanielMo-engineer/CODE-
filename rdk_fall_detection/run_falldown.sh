#!/bin/bash
# ============================================================
# RDK GS130W 人体摔倒检测 + 10秒无动静报警
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/fall_monitor.py"

# 1. 设置摄像头类型为 USB（GS130W 是 USB 接口）
export CAM_TYPE=usb

# 2. 配置摄像头设备节点（默认 /dev/video0，可按需修改）
CAM_DEVICE="${1:-/dev/video0}"
echo "🔧 使用摄像头: $CAM_DEVICE"

# 3. 检查摄像头是否存在
if [ ! -e "$CAM_DEVICE" ]; then
    echo "❌ 错误: 摄像头设备 $CAM_DEVICE 不存在！"
    echo "  请检查摄像头是否连接，或指定其他设备节点："
    echo "  $0 /dev/video8"
    exit 1
fi

# 4. 检测可用的视频设备
echo "📷 系统中的视频设备："
ls -la /dev/video* 2>/dev/null || echo "  (无)"

# 5. source ROS2 环境
source /opt/tros/humble/setup.bash

echo ""
echo "============================================"
echo "  RDK GS130W 人体摔倒检测系统"
echo "============================================"
echo ""

# 6. 启动 ROS2 节点 — 后台运行
echo "🚀 启动 USB 摄像头..."
ros2 run hobot_usb_cam hobot_usb_cam --ros-args \
  -p video_device:="$CAM_DEVICE" \
  -p image_width:=960 \
  -p image_height:=544 \
  -p framerate:=30 \
  -p pixel_format:=mjpeg \
  -p io_method:=mmap \
  --log-level warn &
PID_USB_CAM=$!

sleep 1

echo "🚀 启动 JPEG→NV12 编解码..."
ros2 launch hobot_codec hobot_codec_decode.launch.py \
  codec_in_mode:=ros codec_out_mode:=shared_mem \
  codec_sub_topic:=/image codec_pub_topic:=/hbmem_img &
PID_CODEC=$!

sleep 1

echo "🚀 启动体态关键点检测..."
ros2 run mono2d_body_detection mono2d_body_detection --ros-args \
  -p model_file_name:="config/multitask_body_head_face_hand_kps_960x544.hbm" \
  -p model_type:=0 \
  -p ai_msg_pub_topic_name:=/hobot_mono2d_body_detection \
  --log-level warn &
PID_BODY_DET=$!

sleep 1

echo "🚀 启动摔倒检测..."
ros2 run hobot_falldown_detection hobot_falldown_detection --ros-args \
  -p paramSensivity:=3 \
  -p body_kps_topic_name:=hobot_mono2d_body_detection \
  -p pub_smart_topic_name:=/hobot_falldown_detection \
  --log-level info &
PID_FALL_DET=$!

sleep 1

echo "🚀 启动 Web 可视化 (http://<本机IP>:8000)..."
ros2 launch websocket websocket.launch.py \
  websocket_image_topic:=/image \
  websocket_smart_topic:=/hobot_falldown_detection &
PID_WEB=$!

sleep 1

echo ""
echo "🚀 启动摔倒监测节点..."
echo "   → 检测到摔倒后，持续 10 秒无动静将触发终端警告"
echo ""
python3 "$MONITOR_SCRIPT" &
PID_MONITOR=$!

# 7. 捕获退出信号，清理所有后台进程
cleanup() {
    echo ""
    echo "🛑 正在停止所有节点..."
    kill $PID_MONITOR 2>/dev/null || true
    kill $PID_FALL_DET 2>/dev/null || true
    kill $PID_BODY_DET 2>/dev/null || true
    kill $PID_CODEC 2>/dev/null || true
    kill $PID_WEB 2>/dev/null || true
    kill $PID_USB_CAM 2>/dev/null || true
    wait 2>/dev/null || true
    echo "✅ 已停止所有节点"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo ""
echo "✅ 系统已启动，按 Ctrl+C 停止"
echo "============================================"
echo ""

# 等待任一进程退出
wait
