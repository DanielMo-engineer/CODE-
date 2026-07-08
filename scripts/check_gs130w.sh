#!/bin/bash
# rdkGS130W 双目摄像头一键检测脚本
# 用法：bash check_gs130w.sh

set +e

echo "===== 1. /dev/video* 节点 ====="
ls -l /dev/video* 2>&1

echo
echo "===== 2. v4l2-ctl 设备清单 ====="
v4l2-ctl --list-devices 2>&1

echo
echo "===== 3. 摄像头相关 dmesg（最近 80 条）====="
dmesg | grep -iE "uvc|csi|cam|sensor|mipi|gs130|sc230|sc132|isi" | tail -n 80

echo
echo "===== 4. /sys/class/video4linux ====="
ls /sys/class/video4linux/ 2>&1

echo
echo "===== 5. board 端 FPC 物理接口状态 ====="
if [ -f /sys/devices/platform/soc/35000000.hsio_apb/3d0b0000.cam_pulse/status ]; then
  cat /sys/devices/platform/soc/35000000.hsio_apb/3d0b0000.cam_pulse/status
else
  echo "(n/a)"
fi

echo
echo "===== 6. mipi 相关 sysfs 节点 ====="
find /sys -name "*mipi*" -type d 2>/dev/null | head -n 20
