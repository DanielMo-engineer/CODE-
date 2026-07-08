# RDK GS130W 人体摔倒检测 + 10秒无动静报警

## 硬件要求
- RDK X5 开发板
- GCGS130W USB 全局快门摄像头

## 使用步骤

### 1. 连接摄像头
将 GS130W 插入 RDK X5 的 USB 口，确认设备节点：

```bash
ls -la /dev/video*
```

GS130W 通常出现在 `/dev/video0` 或 `/dev/video8`。

### 2. 查看摄像头支持的格式（可选）
```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```

### 3. 运行摔倒检测系统

```bash
cd ~/.openclaw/workspace/rdk_fall_detection

# 使用默认 /dev/video0
./run_falldown.sh

# 指定其他设备节点
./run_falldown.sh /dev/video8
```

### 4. 查看检测结果
- **终端实时输出**：检测到摔倒会打印 `⚠️ 人体摔倒检测`，10秒无动静则触发 `警告`
- **Web 可视化**：浏览器打开 `http://<RDK_IP>:8000` 查看实时画面 + 检测框

### 5. 停止系统
按 `Ctrl+C` 即可停止所有节点。

---

## 架构说明

```
GS130W (USB)
    ↓
hobot_usb_cam  →  /image (jpeg)
    ↓
hobot_codec_decode  →  /hbmem_img (nv12)
    ↓
mono2d_body_detection  →  体态关键点检测
    ↓
hobot_falldown_detection  →  摔倒判断
    ↓                               ↓
fall_monitor.py (10秒报警)     websocket (Web:8000)
```

## 灵敏度调节

摔倒检测敏感度参数 `paramSensivity`（1-5，默认3，值越大越敏感）：

```bash
# 在 run_falldown.sh 中找到这一行，修改参数值
ros2 run hobot_falldown_detection hobot_falldown_detection --ros-args \
  -p paramSensivity:=3 \
  ...
```

## 报警延迟调节

在 `fall_monitor.py` 中修改 `ALARM_DELAY_SEC` 变量即可。
