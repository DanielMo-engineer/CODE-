#!/bin/bash
# RDK X5 摔倒检测 — 一键启动脚本
set -e

SCRIPT="/root/.openclaw/workspace/fall_detection.py"

echo "==================================="
echo "  RDK X5 摔倒检测 — 一键启动"
echo "==================================="

# 检查是否 root
if [ "$EUID" -ne 0 ]; then
    echo "请用 sudo 运行: sudo bash $0"
    exit 1
fi

# 1. 检查 config.txt 是否已配摄像头
if ! grep -q "dtoverlay.*cam0\|camera_power" /boot/config.txt 2>/dev/null; then
    echo ""
    echo "📷 第一次运行：配置摄像头电源..."
    echo 'dtoverlay=dtoverlay_cam0_imx219' >> /boot/config.txt
    echo "   已写入 /boot/config.txt ✅"
    echo ""
    echo "⚠️  需要重启才能生效！是否现在重启？(y/n)"
    read -r ans
    if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
        echo "重启中..."
        reboot
        exit 0
    else
        echo "请手动重启后重新运行本脚本: sudo bash $0"
        exit 0
    fi
fi

# 2. 检查脚本语法
echo "🔍 检查脚本..."
/usr/bin/python3.10 -c "import py_compile; py_compile.compile('$SCRIPT', doraise=True)" 2>/dev/null
echo "   语法正确 ✅"

# 3. 检测可用摄像头
echo "🔍 检测摄像头..."
CAM_FOUND=""

# 先试 GS130W (MIPI channel 0)
if /usr/bin/python3.10 -c "
import sys, time
sys.path.append('/app/pydev_demo')
from hobot_vio import libsrcampy as srcampy
cam = srcampy.Camera()
cam.open_cam(0, -1, -1, [1280, 1280], [1088, 1088], 1088, 1280)
time.sleep(2)
raw = cam.get_img(2, 1280, 1088)
cam.close_cam()
exit(0 if raw and len(raw) > 50 else 1)
" 2>/dev/null; then
    echo "   ✅ 检测到 MIPI 摄像头 (GS130W 双目)"
    CAM_FOUND="gs130w"
fi

# 如果没找到，试普通 MIPI (1920x1080)
if [ -z "$CAM_FOUND" ]; then
    if /usr/bin/python3.10 -c "
import sys, time
sys.path.append('/app/pydev_demo')
from hobot_vio import libsrcampy as srcampy
cam = srcampy.Camera()
cam.open_cam(0, -1, -1, [1920, 1920], [1080, 1080], 1080, 1920)
time.sleep(1)
raw = cam.get_img(2, 1920, 1080)
cam.close_cam()
exit(0 if raw and len(raw) > 50 else 1)
" 2>/dev/null; then
        echo "   ✅ 检测到 MIPI 摄像头 (F37/OV5647)"
        CAM_FOUND="mipi"
    fi
fi

# 如果还没找到，试 USB 摄像头
if [ -z "$CAM_FOUND" ]; then
    for dev in 0 1 2; do
        if [ -e "/dev/video$dev" ]; then
            if /usr/bin/python3.10 -c "
import cv2
cap = cv2.VideoCapture($dev)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
import time; time.sleep(0.5)
ret, f = cap.read()
cap.release()
exit(0 if ret and f is not None else 1)
" 2>/dev/null; then
                echo "   ✅ 检测到 USB 摄像头 (/dev/video$dev)"
                CAM_FOUND="usb:$dev"
                break
            fi
        fi
    done
fi

# 没找到任何摄像头
if [ -z "$CAM_FOUND" ]; then
    echo ""
    echo "❌ 未检测到任何摄像头！"
    echo ""
    echo "请检查:"
    echo "  1. 摄像头FPC排线是否插紧"
    echo "  2. 是否已重启 (刚配完 config.txt 需要重启)"
    echo "  3. USB摄像头是否已插入"
    echo ""
    echo "也可手动指定:"
    echo "  GS130W双目:  sudo /usr/bin/python3.10 $SCRIPT --gs130w"
    echo "  USB摄像头:   sudo /usr/bin/python3.10 $SCRIPT --usb 0"
    exit 1
fi

# 4. 启动检测
echo ""
echo "==================================="
echo "  🚀 启动摔倒检测..."
echo "==================================="
echo ""

case "$CAM_FOUND" in
    gs130w)
        echo "模式: RDKGS130W 双目摄像头"
        echo "分辨率: 1280x1088 | BPU推理: YOLO11n-pose"
        echo "通知: Server酱 微信推送 (10秒冷却)"
        echo ""
        /usr/bin/python3.10 "$SCRIPT" --gs130w
        ;;
    mipi)
        echo "模式: MIPI 单目摄像头"
        echo "分辨率: 1920x1080 | BPU推理: YOLO11n-pose"
        echo "通知: Server酱 微信推送 (10秒冷却)"
        echo ""
        /usr/bin/python3.10 "$SCRIPT"
        ;;
    usb:*)
        dev="${CAM_FOUND#usb:}"
        echo "模式: USB摄像头 /dev/video$dev"
        echo "分辨率: 640x480 | BPU推理: YOLO11n-pose"
        echo "通知: Server酱 微信推送 (10秒冷却)"
        echo ""
        /usr/bin/python3.10 "$SCRIPT" --usb "$dev" --usb-width 640 --usb-height 480
        ;;
esac

echo ""
echo "摔倒检测已停止"
