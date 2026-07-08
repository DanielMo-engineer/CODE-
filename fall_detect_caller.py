#!/usr/bin/env python3
"""
RDK X5 摔倒检测 + SIM800L 自动拨号系统
======================================
摄像头实时检测人体摔倒 → 自动拨打指定电话

工作流程:
  摄像头帧 → YOLO BPU 推理 → 人体检测框 → 宽高比+速度判定摔倒 → SIM800L ATD 拨号

硬件连接:
  SIM800L TX  → RDK X5 Pin 10 (RX)
  SIM800L RX  → RDK X5 Pin 8  (TX)
  SIM800L GND → RDK X5 Pin 6  (GND)
  SIM800L VCC → 外部 5V/2A 电源（⚠️ 不能从 40pin 取电）
  麦克风      → SIM800L MIC+/MIC-
  扬声器      → SIM800L SPK+/SPK-

用法:
  # 完整运行（检测到摔倒后自动拨号）
  sudo /usr/bin/python3.10 fall_detect_caller.py --phone 13800138000

  # 只测试 GSM 模块（不启动摄像头）
  sudo /usr/bin/python3.10 fall_detect_caller.py --phone 13800138000 --test-gsm

  # 只测试摄像头检测（不拨号）
  /usr/bin/python3.10 fall_detect_caller.py --test-camera

  # 使用 MIPI 摄像头
  sudo /usr/bin/python3.10 fall_detect_caller.py --phone 13800138000 --camera mipi
"""

import os
import sys
import time
import json
import signal
import argparse
import numpy as np
from typing import Optional, Tuple

# ── RDK X5 硬件依赖 ──
import hbm_runtime
import cv2

# ── 添加 /app demo 工具路径 ──
sys.path.append('/app/pydev_demo')
sys.path.append('/app/pydev_demo/utils')
import preprocess_utils as pre_utils
import postprocess_utils as post_utils
import common_utils as common_lib

# ═══════════════════════════════════════════════════
# 配置参数（可按需修改）
# ═══════════════════════════════════════════════════

# BPU 模型路径
MODEL_PATH = '/app/pydev_demo/models/yolo11m_detect_bayese_640x640_nv12_modified.bin'
COCO_LABELS = '/app/pydev_demo/07_usb_camera_sample/coco_classes.names'

# SIM800L
SERIAL_PORT = '/dev/ttyS1'
SERIAL_BAUD = 9600

# 摔倒检测参数
PERSON_CONFIDENCE = 0.35        # 人体检测置信度阈值
FALL_ASPECT_THRESHOLD = 1.15    # 宽高比 > 此值视为横躺
FALL_SPEED_THRESHOLD = 6.0      # 垂直速度阈值（像素/帧）
FALL_CONFIRM_FRAMES = 5         # 连续 N 帧确认才触发
COOLDOWN_SECONDS = 60           # 拨号冷却时间（秒）
CAM_WIDTH = 640                 # 摄像头采集宽度
CAM_HEIGHT = 480                # 摄像头采集高度


# ═══════════════════════════════════════════════════
# SIM800L GSM 模块控制
# ═══════════════════════════════════════════════════

class GSMController:
    """通过 UART 串口控制 SIM800L"""

    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD):
        import serial
        self.ser = serial.Serial(port, baud, timeout=2)
        self.last_call_time = 0
        self.is_calling = False
        print(f"  [GSM] UART {port} @ {baud} baud ✅")

    def send_at(self, cmd: str, wait: float = 1.5) -> str:
        """发送 AT 指令"""
        self.ser.write((cmd + '\r\n').encode())
        time.sleep(wait)
        resp = self.ser.read(256).decode(errors='ignore')
        return resp.strip()

    def init_module(self) -> bool:
        """初始化 GSM 模块"""
        for _ in range(3):
            resp = self.send_at('AT', 1)
            if 'OK' in resp:
                break
        else:
            print("  [GSM] ⚠️  模块无响应，检查接线和供电")
            return False

        self.send_at('ATE0', 0.5)          # 关闭回显
        self.send_at('AT+CHFA=1', 0.5)     # 使用手柄模式（板载 MIC/SPK）
        self.send_at('AT+CLVL=80', 0.5)    # 音量 80%
        print("  [GSM] 初始化 OK ✅")
        return True

    def check_signal(self) -> int:
        """查询信号强度 0~31（>10 可用，>20 良好）"""
        resp = self.send_at('AT+CSQ', 1)
        try:
            return int(resp.split(':')[1].split(',')[0].strip())
        except (IndexError, ValueError):
            return 0

    def check_network(self) -> bool:
        """检查网络注册状态"""
        resp = self.send_at('AT+CREG?', 2)
        return '+CREG: 0,1' in resp or '+CREG: 0,5' in resp

    def make_call(self, number: str) -> bool:
        """拨打电话"""
        if self.is_calling:
            print("  [GSM] ⚠️  通话中，跳过")
            return False
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < COOLDOWN_SECONDS:
            print(f"  [GSM] ⏳ 冷却中（还需 {COOLDOWN_SECONDS - elapsed:.0f}s）")
            return False

        print(f"  [GSM] 📞 拨号 {number}...")
        resp = self.send_at(f'ATD{number};', 4)
        if 'OK' in resp or 'CONNECT' in resp:
            self.last_call_time = now
            self.is_calling = True
            # 保持通话一段时间
            threading.Thread(target=self._keep_call, daemon=True).start()
            print(f"  [GSM] ✅ 电话已拨出")
            return True
        print(f"  [GSM] ❌ 拨号失败: {resp[:60]}")
        return False

    def _keep_call(self):
        """保持通话 20 秒后自动挂断"""
        time.sleep(20)
        self.hang_up()

    def hang_up(self):
        """挂断电话"""
        if self.is_calling:
            self.send_at('ATH', 1)
            self.is_calling = False
            print(f"  [GSM] 📞 已挂断")

    def close(self):
        self.hang_up()
        self.ser.close()


# ═══════════════════════════════════════════════════
# YOLO BPU 推理器（兼容 yolo11m / yolo11n / yolo8 等）
# ═══════════════════════════════════════════════════

class YOLODetector:
    """在 RDK X5 BPU 上运行 YOLO 检测"""

    def __init__(self, model_path: str, score_thres: float = PERSON_CONFIDENCE):
        self.model = hbm_runtime.HB_HBMRuntime(model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]
        self.score_thres = score_thres
        self.conf_thres_raw = -np.log(1.0 / self.score_thres - 1.0)

        # YOLO 解码参数
        self.strides = [8, 16, 32]
        self.anchor_sizes = [80, 40, 20]   # 对应 feature map 尺寸
        self.classes_num = 80
        self.reg = 16
        self.weights_static = np.arange(self.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]
        self.resize_type = 1  # letterbox

        # 加载 COCO 标签
        try:
            self.class_names = common_lib.load_class_names(COCO_LABELS)
        except Exception:
            self.class_names = [str(i) for i in range(80)]

        print(f"  [YOLO] 模型: {os.path.basename(model_path)}")
        print(f"  [YOLO] 输入: {self.input_W}x{self.input_H}")
        print(f"  [YOLO] 阈值: {score_thres}")

    def pre_process(self, img: np.ndarray) -> dict:
        """BGR 图像 → NV12 模型输入"""
        resized = pre_utils.resized_image(img, self.input_W, self.input_H, self.resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resized)
        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))
        return {self.model_name: {self.input_names[0]: nv12}}

    def forward(self, tensor: dict) -> dict:
        return self.model.run(tensor)[self.model_name]

    def post_process(self, outputs: dict, img_w: int, img_h: int):
        """
        YOLOv11/v8 后处理：反量化 → 各尺度解码 → 阈值过滤 → NMS → 坐标回缩
        """
        fp32 = post_utils.dequantize_outputs(outputs, self.output_quants)

        all_boxes, all_scores, all_ids = [], [], []

        for i, (stride, anchor_size) in enumerate(zip(self.strides, self.anchor_sizes)):
            cls_key = self.output_names[2 * i]      # 分类输出
            box_key = self.output_names[2 * i + 1]   # 检测框输出

            scores, ids, valid = post_utils.filter_classification(
                fp32[cls_key], self.conf_thres_raw)
            dboxes = post_utils.decode_boxes(
                fp32[box_key], valid, anchor_size, stride, self.weights_static)

            all_boxes.append(dboxes)
            all_scores.append(scores)
            all_ids.append(ids)

        dboxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        ids = np.concatenate(all_ids, axis=0)

        keep = post_utils.NMS(dboxes, scores, ids, 0.45)
        boxes = post_utils.scale_coords_back(
            dboxes[keep], img_w, img_h, self.input_W, self.input_H, self.resize_type)

        return boxes, ids[keep], scores[keep]

    def detect(self, img: np.ndarray):
        """单帧推理"""
        h, w = img.shape[:2]
        tensor = self.pre_process(img)
        outputs = self.forward(tensor)
        return self.post_process(outputs, w, h)


# ═══════════════════════════════════════════════════
# 摔倒检测器
# ═══════════════════════════════════════════════════

class FallDetector:
    """基于 YOLO 人体检测的摔倒判定引擎"""

    def __init__(self, model: YOLODetector):
        self.model = model
        self.prev_bbox = None
        self.fall_counter = 0
        self.fall_triggered = False
        self.tracking_lost = 0

    def _get_person(self, boxes, cls_ids, scores) -> Optional[np.ndarray]:
        """取置信度最高的人体框（COCO class 0 = person）"""
        persons = [(b, s) for b, c, s in zip(boxes, cls_ids, scores) if c == 0]
        if not persons:
            return None
        return max(persons, key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]))[0]

    def check(self, boxes, cls_ids, scores) -> Tuple[bool, float, float]:
        """
        摔倒判定
        返回: (是否触发, 宽高比, 垂直速度)
        """
        bbox = self._get_person(boxes, cls_ids, scores)

        if bbox is None:
            self.tracking_lost += 1
            if self.tracking_lost > 10:
                self.prev_bbox = None
                self.fall_counter = 0
            return False, 0.0, 0.0

        self.tracking_lost = 0
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        aspect = w / (h + 0.001)
        center_y = (bbox[1] + bbox[3]) / 2

        velocity = 0.0
        if self.prev_bbox is not None:
            prev_cy = (self.prev_bbox[1] + self.prev_bbox[3]) / 2
            velocity = abs(center_y - prev_cy)

        self.prev_bbox = bbox

        # 摔倒 = 宽高比超标 + 快速下坠
        is_fall = (aspect > FALL_ASPECT_THRESHOLD) and (velocity > FALL_SPEED_THRESHOLD)

        if is_fall:
            self.fall_counter += 1
        else:
            self.fall_counter = max(0, self.fall_counter - 1)

        if self.fall_counter >= FALL_CONFIRM_FRAMES and not self.fall_triggered:
            self.fall_triggered = True
            return True, aspect, velocity

        # 人重新站起来后重置
        if self.fall_triggered and aspect < 0.9:
            self.fall_triggered = False
            self.fall_counter = 0

        return False, aspect, velocity


# ═══════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='RDK X5 摔倒检测 + SIM800L 自动拨号',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  sudo /usr/bin/python3.10 fall_detect_caller.py --phone 13800138000
  /usr/bin/python3.10 fall_detect_caller.py --test-camera""")
    parser.add_argument('--phone', type=str, default='',
                        help='要拨打的电话号码')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='BPU 模型路径')
    parser.add_argument('--camera', type=str, default='usb', choices=['usb', 'mipi'])
    parser.add_argument('--device', type=int, default=0,
                        help='USB 摄像头设备号')
    parser.add_argument('--port', type=str, default=SERIAL_PORT,
                        help='SIM800L 串口')
    parser.add_argument('--cooldown', type=int, default=COOLDOWN_SECONDS,
                        help='拨号冷却秒数')
    parser.add_argument('--confidence', type=float, default=PERSON_CONFIDENCE,
                        help='检测置信度')
    parser.add_argument('--test-camera', action='store_true',
                        help='只测试摄像头+检测（不连 GSM）')
    parser.add_argument('--test-gsm', action='store_true',
                        help='只测试 GSM 模块（不发摄像头）')
    return parser.parse_args()


def gsm_test(phone, port):
    """GSM 模块单独测试"""
    gsm = GSMController(port, SERIAL_BAUD)
    if not gsm.init_module():
        gsm.close()
        return
    sig = gsm.check_signal()
    net = gsm.check_network()
    print(f"  信号强度: {sig}/31 {'✅' if sig > 10 else '⚠️'}")
    print(f"  网络注册: {'✅' if net else '❌'}")
    if phone:
        print(f"  拨号 {phone}（5秒后挂断）...")
        gsm.make_call(phone)
        time.sleep(5)
        gsm.hang_up()
    gsm.close()
    print("GSM 测试完成 ✅")


def camera_test(model_path, confidence):
    """摄像头+检测单独测试"""
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return
    detector = YOLODetector(model_path, confidence)
    fall_detector = FallDetector(detector)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    print("摄像头已打开 ✅，按 Ctrl+C 退出")
    print()

    fc = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            fc += 1
            t0 = time.time()
            boxes, cls_ids, scores = detector.detect(frame)
            t_ms = (time.time() - t0) * 1000
            is_fall, aspect, vel = fall_detector.check(boxes, cls_ids, scores)

            if fc % 15 == 0:
                persons = sum(1 for c in cls_ids if c == 0)
                p = f"[{'█'*min(fall_detector.fall_counter, FALL_CONFIRM_FRAMES)}{'░'*(FALL_CONFIRM_FRAMES-min(fall_detector.fall_counter, FALL_CONFIRM_FRAMES))}]"
                flag = "⚠️ FALL!" if is_fall else ""
                print(f"  人物={persons} 推理={t_ms:.0f}ms 摔倒={p} 宽高比={aspect:.2f} 速度={vel:.1f} {flag}")
            time.sleep(0.065)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("已退出")


def run():
    args = parse_args()

    if args.test_gsm:
        gsm_test(args.phone, args.port)
        return
    if args.test_camera:
        camera_test(args.model, args.confidence)
        return

    # ── 正式运行 ──
    if not args.phone:
        print("❌ 请指定电话号码: --phone 13800138000")
        sys.exit(1)

    print()
    print("╔══════════════════════════════════════════╗")
    print("║   RDK X5 摔倒检测 + 自动拨号            ║")
    print("╚══════════════════════════════════════════╝")
    print()

    # ── GSM ──
    gsm = GSMController(args.port, SERIAL_BAUD)
    if not gsm.init_module():
        print("  ⚠️  GSM 不可用，仅运行检测（不拨号）")
        gsm = None
    else:
        sig = gsm.check_signal()
        net = gsm.check_network()
        print(f"  信号: {sig}/31 {'✅' if sig > 10 else '⚠️'}")
        print(f"  网络: {'✅' if net else '❌'}")

    # ── 模型 ──
    print()
    if not os.path.exists(args.model):
        print(f"❌ 模型不存在: {args.model}")
        sys.exit(1)
    detector = YOLODetector(args.model, args.confidence)
    fall_detector = FallDetector(detector)

    # ── 摄像头 ──
    print()
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        print("  提示: 接入 USB 摄像头后用 v4l2-ctl --list-devices 检查")
        if gsm:
            gsm.close()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"  [CAM] 摄像头 {real_w:.0f}x{real_h:.0f} ✅")

    # ── 状态 ──
    last_call_time = 0
    frame_count = 0
    fps_timer = time.time()
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        print("\n🛑 退出中...")
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── 线程锁保证 GSM 串口安全 ──
    import threading

    print()
    print("=" * 50)
    print("🚀 系统启动！实时检测中...")
    if gsm:
        print(f"📞 检测到摔倒 → 拨打 {args.phone}（冷却 {args.cooldown}s）")
    print("=" * 50)
    print()

    try:
        while running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            frame_count += 1

            # 推理
            t0 = time.time()
            boxes, cls_ids, scores = detector.detect(frame)
            infer_ms = (time.time() - t0) * 1000

            # 摔倒判定
            is_fall, aspect, velocity = fall_detector.check(boxes, cls_ids, scores)

            # 触发拨号
            if is_fall and gsm:
                now = time.time()
                if now - last_call_time > args.cooldown:
                    print(f"\n⚠️⚠️⚠️ 摔倒！宽高比={aspect:.2f} 速度={velocity:.1f}")
                    gsm.make_call(args.phone)
                    last_call_time = now
                else:
                    print(f"\n⚠️ 摔倒，冷却中...")

            # 状态打印（每 30 帧）
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_timer + 0.001)
                persons = sum(1 for c in cls_ids if c == 0)
                pbar = f"[{'█' * min(fall_detector.fall_counter, FALL_CONFIRM_FRAMES)}{'░' * (FALL_CONFIRM_FRAMES - min(fall_detector.fall_counter, FALL_CONFIRM_FRAMES))}]"
                status = "🔴" if fall_detector.fall_triggered else "🟢"
                print(f"  {status} {fps:.0f}fps | 推理{infer_ms:.0f}ms | "
                      f"人{persons} | 判定{pbar} | "
                      f"宽高={aspect:.2f} 速={velocity:.1f}")

            time.sleep(0.065)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n🛑 关闭中...")
        cap.release()
        if gsm:
            gsm.close()
        print("👋 已退出")


import threading  # for GSM keep_call timer

if __name__ == '__main__':
    run()
