#!/usr/bin/env python3
"""
RDK X5 人体摔倒检测系统 (VP管道采集帧)
========================================

硬件:
  - rdkGS130W 双目摄像头 (SC132GS x2)
  - 蜂鸣器 (低电平触发): VCC→Pin1, GND→Pin6, IO→Pin11

逻辑:
  检测人体摔倒 → 倒地 3 秒不恢复 → 蜂鸣器响 3 秒 → 停止

用法:
  sudo /usr/bin/python3.10 fall_detection_final.py
"""

import os
import sys
import time
import signal
import struct
import subprocess
import numpy as np

# ─── 库导入 ──────────────────────────────────────────────────
try:
    import Hobot.GPIO as GPIO
except ImportError:
    GPIO = None

import hbm_runtime

sys.path.append('/app/pydev_demo')
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils

# ─── 配置 ────────────────────────────────────────────────────

BUZZER_PIN = 11
MODEL_PATH = '/opt/hobot/model/x5/basic/yolov5s_672x672_nv12.bin'
CAM_W, CAM_H = 1088, 1280
SCORE_THRESH = 0.35
NMS_THRESH = 0.45

# 摔倒判定: 宽/高 > 0.7 视为摔倒，宽/高 < 0.71 视为站立
FALL_RATIO = 0.70
STAND_RATIO = 1.0 / 1.4

# 滤波
FALL_FRAMES = 3    # 连续 N 帧确认摔倒
STAND_FRAMES = 5   # 连续 N 帧确认恢复站立

# 计时
WAIT_SECS = 3      # 摔倒后等待秒数
BUZZ_SECS = 3      # 蜂鸣持续秒数

FRAME_GRABBER = '/app/multimedia_samples/sample_pipeline/single_pipe_vin_isp_vse/frame_grabber'

# ─── YOLOv5s ────────────────────────────────────────────────

STRIDES = np.array([8, 16, 32], dtype=np.int32)
ANCHORS = np.array([
    [10, 13], [16, 30], [33, 23],
    [30, 61], [62, 45], [59, 119],
    [116, 90], [156, 198], [373, 326]
], dtype=np.float32).reshape(3, 3, 2)


class YoloPersonDetector:
    def __init__(self, model_path, score_thres=SCORE_THRESH, nms_thres=NMS_THRESH):
        self.model = hbm_runtime.HB_HBMRuntime(model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]
        self.score_thres = score_thres
        self.nms_thres = nms_thres
        self.resize_type = 1
        self.model.set_scheduling_params(
            priority={self.model_name: 0},
            bpu_cores={self.model_name: [0]},
        )

    def detect(self, nv12_bytes, img_w, img_h):
        y, uv = pre_utils.split_nv12_bytes(nv12_bytes, img_w, img_h)
        yr, uvr = pre_utils.resize_nv12_yuv(y, uv, self.input_H, self.input_W)
        yi = yr[..., None][None, ...]
        uvi = uvr[None, ...]
        inp = np.concatenate((yi.reshape(-1), uvi.reshape(-1)))
        inp = inp.reshape((1, self.input_H * 3 // 2, self.input_W, 1))
        tensor = {self.model_name: {self.input_names[0]: inp}}
        outputs = self.model.run(tensor)[self.model_name]

        fp32 = post_utils.dequantize_outputs(outputs, self.output_quants)
        pred = post_utils.decode_outputs(self.output_names, fp32, STRIDES, ANCHORS, 80)
        xyxy, scores, cls = post_utils.filter_predictions(pred, self.score_thres)
        keep = post_utils.NMS(xyxy, scores, cls, self.nms_thres)
        xyxy = post_utils.scale_coords_back(xyxy[keep], img_w, img_h,
                                            self.input_W, self.input_H, self.resize_type)
        mask = cls[keep] == 0  # class 0 = person
        return xyxy[mask], scores[keep][mask]


# ─── 主程序 ──────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════╗")
    print("║  RDK X5 摔倒检测系统                       ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  蜂鸣器: Pin {BUZZER_PIN}               ║")
    print(f"║  逻辑: 摔倒→{WAIT_SECS}s→蜂鸣{BUZZ_SECS}s         ║")
    print(f"║  模型: {os.path.basename(MODEL_PATH)}    ║")
    print("╚══════════════════════════════════════════╝")

    if not os.path.exists(FRAME_GRABBER):
        print(f"[ERROR] frame_grabber 未找到: {FRAME_GRABBER}")
        sys.exit(1)

    # ─── GPIO ──────────────────────────────────────────
    if GPIO is not None:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.output(BUZZER_PIN, GPIO.HIGH)

    def buzz(on):
        if GPIO is not None:
            GPIO.output(BUZZER_PIN, GPIO.LOW if on else GPIO.HIGH)

    # ─── 加载模型 ─────────────────────────────────────
    print("[INIT] 加载 YOLO 模型...")
    detector = YoloPersonDetector(MODEL_PATH)
    print(f"[INIT] 模型就绪 {detector.input_W}x{detector.input_H}")

    # ─── 启动帧采集器 ────────────────────────────────
    print("[INIT] 启动摄像头管线...")
    proc = subprocess.Popen(
        [FRAME_GRABBER, '-s', '5'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)

    if proc.poll() is not None:
        print(f"[ERROR] frame_grabber 启动失败 (exit={proc.returncode})")
        sys.exit(1)

    print("[INIT] ✅ 摄像头已启动")

    # ─── 状态机 ───────────────────────────────────────
    state = "NORMAL"
    fall_count = 0
    stand_count = 0
    fallen_ts = None
    alert_ts = None
    total_frames = 0
    fall_events = 0
    buzz_events = 0
    running = True

    def cleanup(s, f):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    fp = proc.stdout  # type: ignore
    fp_buf = b''

    print("[RUN] 🟢 检测中 (Ctrl+C 停止)")
    fps_cnt = 0
    fps_ts = time.time()

    # ── 第一阶段：跳过输出中的文本头，定位第一个有效帧 ──
    print("[INIT] 等待摄像头就绪...")
    found_frame = False
    while running and not found_frame:
        chunk = fp.read(8192)
        if not chunk:
            running = False
            break
        fp_buf += chunk
        # 搜索第一个完整帧
        FRAME_SZ = 8 + CAM_W * CAM_H * 3 // 2
        for offset in range(max(0, len(fp_buf) - FRAME_SZ + 1)):
            w, h = struct.unpack_from('II', fp_buf, offset)
            if w == CAM_W and h == CAM_H and offset + FRAME_SZ <= len(fp_buf):
                fp_buf = fp_buf[offset:]
                found_frame = True
                print("[INIT] ✅ 摄像头就绪，开始检测")
                break

    if not running or not found_frame:
        print("[ERROR] 未能读取到摄像头帧数据")
        if proc.poll() is None:
            proc.terminate()
        sys.exit(1)

    # 预计算帧尺寸
    FRAME_SZ = 8 + CAM_W * CAM_H * 3 // 2  # header + YUV

    # ── 第二阶段：定长读取每一帧 ──
    while running:
        # 确保缓存中至少有 8 字节 (width+height)
        if len(fp_buf) < 8:
            chunk = fp.read(8 - len(fp_buf))
            if not chunk:
                break
            fp_buf += chunk
            continue

        # 读取完整帧 (header + Y + UV)
        need = FRAME_SZ - len(fp_buf)
        while need > 0:
            chunk = fp.read(need)
            if not chunk:
                running = False
                break
            fp_buf += chunk
            need = FRAME_SZ - len(fp_buf)

        if not running:
            break

        # 解析
        w = struct.unpack_from('I', fp_buf, 0)[0]
        h = struct.unpack_from('I', fp_buf, 4)[0]
        nv12_bytes = fp_buf[8:FRAME_SZ]
        fp_buf = fp_buf[FRAME_SZ:]

        # ── YOLO 检测 ────────────────────────────────
        total_frames += 1
        boxes, scores = detector.detect(nv12_bytes, w, h)

        # ── 摔倒分析 ─────────────────────────────────
        current_state = "NORMAL"
        if len(boxes) > 0:
            best = boxes[scores.argmax()]
            bw, bh = best[2] - best[0], best[3] - best[1]
            ratio = bw / max(bh, 1)
            falling = ratio > FALL_RATIO
            standing = ratio < STAND_RATIO

            if state == "NORMAL":
                if falling:
                    fall_count += 1
                else:
                    fall_count = 0
                if fall_count >= FALL_FRAMES:
                    state = "FALLEN"
                    fallen_ts = time.time()
                    fall_events += 1
                    print(f"[⚠️  FALL] 摔倒检测! ratio={ratio:.2f}")
                current_state = "NORMAL"

            elif state == "FALLEN":
                elapsed = time.time() - fallen_ts
                if standing:
                    stand_count += 1
                    if stand_count >= STAND_FRAMES:
                        print(f"[✅  RECOVER] 已起身 ({elapsed:.1f}s)")
                        state = "NORMAL"
                        fall_count = 0
                        stand_count = 0
                else:
                    stand_count = 0
                    remain = WAIT_SECS - elapsed
                    if total_frames % 15 == 0:
                        print(f"  [⏱] 已倒地 {elapsed:.0f}s，还差 {remain:.0f}s 触发")
                    if elapsed >= WAIT_SECS:
                        state = "ALERTING"
                        alert_ts = time.time()
                        buzz(True)
                        buzz_events += 1
                        print(f"[🚨  ALERT] 蜂鸣器启动!")
                current_state = "FALLEN"

            elif state == "ALERTING":
                elapsed = time.time() - alert_ts
                if total_frames % 10 == 0:
                    print(f"  [🔊] 蜂鸣 {elapsed:.0f}/{BUZZ_SECS}s")
                if elapsed >= BUZZ_SECS:
                    buzz(False)
                    state = "NORMAL"
                    fall_count = 0
                    stand_count = 0
                    print(f"[🔄  RESET] 蜂鸣结束，系统复位")
                current_state = "ALERTING"
        else:
            # 无人检测
            if state == "FALLEN":
                remain = WAIT_SECS - (time.time() - fallen_ts)
                if total_frames % 15 == 0:
                    print(f"  [⏱] 无人可见，计时继续 {remain:.0f}s")
            current_state = state

        # ── FPS ──────────────────────────────────────
        fps_cnt += 1
        el = time.time() - fps_ts
        if el >= 5.0:
            print(f"[STATS] FPS={fps_cnt/el:.1f}  "
                  f"state={state} falls={fall_events} buzz={buzz_events}")
            fps_cnt = 0
            fps_ts = time.time()

    # ─── 清理 ─────────────────────────────────────────
    buzz(False)
    if GPIO is not None:
        GPIO.cleanup()
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except Exception:
        proc.kill()

    s = f"总帧={total_frames} 摔倒={fall_events} 报警={buzz_events}"
    print(f"\n📊 {s}")
    print("👋 系统关闭")


if __name__ == '__main__':
    main()
