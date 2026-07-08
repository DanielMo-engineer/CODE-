#!/usr/bin/env python3
"""
RDK X5 摔倒检测系统 (使用 mipi_cam 后台管线)
=============================================
 
原理：mipi_cam 是唯一能在 X5 上正确初始化 SC132GS 双目摄像头的程序。
     本脚本启动 mipi_cam 作为后台进程，通过共享内存获取帧，运行 YOLO 检测。
 
用法:
  sudo /usr/bin/python3.10 fall_detection_mipi.py
"""

import os
import sys
import time
import signal
import numpy as np
import argparse
import subprocess
import struct
import select

# ─── 导入 ────────────────────────────────────────────────────
try:
    import Hobot.GPIO as GPIO
except ImportError:
    GPIO = None

import hbm_runtime

MIPI_CAM_BIN = '/opt/tros/humble/lib/mipi_cam/mipi_cam'
sys.path.append('/app/pydev_demo')
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils

# ─── 配置 ────────────────────────────────────────────────────

BUZZER_PIN = 11                # BOARD 编号
MODEL_PATH = '/opt/hobot/model/x5/basic/yolov5s_672x672_nv12.bin'
CAM_W, CAM_H = 1088, 1280      # SC132GS 传感器尺寸
CAM_FPS = 10
SCORE_THRESH = 0.35
NMS_THRESH = 0.45
HEIGHT_FALL_RATIO = 0.70
STANDING_RECOVER_RATIO = 1.4
FALL_CONFIRM_FRAMES = 3
RECOVER_CONFIRM_FRAMES = 5
FALL_TIMEOUT = 10
BUZZER_DURATION = 3

# ─── YOLOv5s 模型 ────────────────────────────────────────────

STRIDES = np.array([8, 16, 32], dtype=np.int32)
ANCHORS = np.array([
    [10, 13], [16, 30], [33, 23],
    [30, 61], [62, 45], [59, 119],
    [116, 90], [156, 198], [373, 326]
], dtype=np.float32).reshape(3, 3, 2)

PERSON_CLASS = 0


class YoloV5PersonDetector:
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
        self.classes_num = 80
        self.model.set_scheduling_params(
            priority={self.model_name: 0},
            bpu_cores={self.model_name: [0]},
        )

    def pre_process(self, nv12_bytes, img_w, img_h):
        y, uv = pre_utils.split_nv12_bytes(nv12_bytes, img_w, img_h)
        y_resized, uv_resized = pre_utils.resize_nv12_yuv(y, uv,
                                                           self.input_H, self.input_W)
        y_input = y_resized[..., None][None, ...]
        uv_input = uv_resized[None, ...]
        nv12 = np.concatenate((y_input.reshape(-1), uv_input.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))
        return {self.model_name: {self.input_names[0]: nv12}}

    def forward(self, input_tensor):
        outputs = self.model.run(input_tensor)
        return outputs[self.model_name]

    def post_process(self, outputs, img_w, img_h):
        fp32_outputs = post_utils.dequantize_outputs(outputs, self.output_quants)
        pred = post_utils.decode_outputs(self.output_names, fp32_outputs,
                                         STRIDES, ANCHORS, self.classes_num)
        xyxy_boxes, score, cls = post_utils.filter_predictions(pred, self.score_thres)
        keep = post_utils.NMS(xyxy_boxes, score, cls, self.nms_thres)
        xyxy = post_utils.scale_coords_back(xyxy_boxes[keep], img_w, img_h,
                                            self.input_W, self.input_H, self.resize_type)
        person_mask = cls[keep] == PERSON_CLASS
        return xyxy[person_mask], score[keep][person_mask]

    def detect(self, nv12_bytes, img_w, img_h):
        inputs = self.pre_process(nv12_bytes, img_w, img_h)
        outputs = self.forward(inputs)
        return self.post_process(outputs, img_w, img_h)


# ─── 摔倒检测器 ──────────────────────────────────────────────

class FallDetector:
    STATE_NORMAL = "NORMAL"
    STATE_FALLEN = "FALLEN"
    STATE_ALERTING = "ALERTING"

    def __init__(self):
        if GPIO is not None:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(BUZZER_PIN, GPIO.OUT)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)

        self.state = self.STATE_NORMAL
        self.fall_count = 0
        self.recover_count = 0
        self.fallen_ts = None
        self.alert_ts = None
        self.total_frames = 0
        self.fall_events = 0
        self.buzzer_events = 0

    def _buzzer_on(self):
        if GPIO is not None:
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            print("[BUZZER] 🔔 ON")

    def _buzzer_off(self):
        if GPIO is not None:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            print("[BUZZER] 🔕 OFF")

    def analyze(self, boxes, scores):
        self.total_frames += 1
        if len(boxes) == 0:
            return self.state

        best = scores.argmax()
        x1, y1, x2, y2 = boxes[best]
        w, h = x2 - x1, y2 - y1
        if h == 0:
            h = 1
        ratio = w / h
        falling = ratio > HEIGHT_FALL_RATIO
        standing = ratio < (1.0 / STANDING_RECOVER_RATIO)

        if self.state == self.STATE_NORMAL:
            if falling:
                self.fall_count += 1
            else:
                self.fall_count = 0
            if self.fall_count >= FALL_CONFIRM_FRAMES:
                self.state = self.STATE_FALLEN
                self.fallen_ts = time.time()
                self.fall_events += 1
                print(f"[⚠️  FALL] 摔倒! ratio={ratio:.2f}")

        elif self.state == self.STATE_FALLEN:
            elapsed = time.time() - self.fallen_ts
            if standing:
                self.recover_count += 1
                if self.recover_count >= RECOVER_CONFIRM_FRAMES:
                    print(f"[✅  RECOVER] 已起身 ({elapsed:.1f}s)")
                    self._reset()
            else:
                self.recover_count = 0
                remaining = FALL_TIMEOUT - elapsed
                if self.total_frames % 10 == 0:
                    print(f"  [⏱] fallen, {remaining:.0f}s → alert")
                if elapsed >= FALL_TIMEOUT:
                    self.state = self.STATE_ALERTING
                    self.alert_ts = time.time()
                    self._buzzer_on()
                    self.buzzer_events += 1
                    print(f"[🚨  ALERT] 蜂鸣!")

        elif self.state == self.STATE_ALERTING:
            buzzed = time.time() - self.alert_ts
            if self.total_frames % 5 == 0:
                print(f"  [🔊] buzzing {buzzed:.0f}/{BUZZER_DURATION}s")
            if buzzed >= BUZZER_DURATION:
                self._buzzer_off()
                self._reset()
                print(f"[🔄  RESET] 蜂鸣结束")

        return self.state

    def _reset(self):
        self.state = self.STATE_NORMAL
        self.fall_count = 0
        self.recover_count = 0
        self.fallen_ts = None
        self.alert_ts = None

    def cleanup(self):
        self._buzzer_off()
        if GPIO is not None:
            GPIO.cleanup()


# ─── 主程序 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--score', type=float, default=SCORE_THRESH)
    parser.add_argument('--camera-id', type=int, default=0,
                        help='摄像头编号 (0 或 1)')
    parser.add_argument('--fall-timeout', type=int, default=FALL_TIMEOUT)
    parser.add_argument('--buzzer-duration', type=int, default=BUZZER_DURATION)
    parser.add_argument('--no-buzzer', action='store_true')
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════╗")
    print("║  RDK X5 摔倒检测 (mipi_cam 引擎)             ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  摄像头: #{args.camera_id} ({CAM_W}x{CAM_H})            ║")
    print(f"║  蜂鸣器: Pin {BUZZER_PIN} {'✅' if not args.no_buzzer else '❌'}              ║")
    print(f"║  逻辑: 摔倒 → {args.fall_timeout}s → 蜂鸣 {args.buzzer_duration}s    ║")
    print("╚══════════════════════════════════════════════╝")

    # ─── 加载 YOLO 模型 ────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERROR] 模型不存在: {args.model}")
        sys.exit(1)
    print("[INIT] 加载 YOLO 模型...")
    detector = YoloV5PersonDetector(args.model, score_thres=args.score)
    print(f"[INIT] 模型就绪 {detector.input_W}x{detector.input_H}")

    # ─── 启动 mipi_cam 后台进程 ────────────────────────
    print("[INIT] 启动 mipi_cam 后台管线...")
    ld_path = '/opt/ros/humble/lib:/opt/tros/humble/lib'
    
    # 先杀掉旧的 mipi_cam
    subprocess.run(['sudo', 'pkill', '-9', 'mipi_cam'],
                   capture_output=True, timeout=3)
    time.sleep(1)

    mipi_proc = subprocess.Popen(
        [MIPI_CAM_BIN, '--ros-args',
         '-p', 'device_mode:=dual',
         '-p', 'dual_combine:=1',
         '-p', f'image_width:={CAM_W}',
         '-p', f'image_height:={CAM_H}',
         '-p', f'framerate:={CAM_FPS}',
         '-p', 'lpwm_enable:=True',
         '-p', 'channel:=2',
         '-p', 'channel2:=0',
         '-p', 'out_format:=nv12',
         '-p', 'gdc_enable:=False',
         '-p', 'rotation:=0.0',
         '-p', 'log_level:=fatal'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, 'LD_LIBRARY_PATH': ld_path}
    )
    time.sleep(5)  # 等待 mipi_cam 完成初始化

    if mipi_proc.poll() is not None:
        print(f"[ERROR] mipi_cam 启动失败 (exit={mipi_proc.returncode})")
        sys.exit(1)
    print("[INIT] ✅ mipi_cam 运行中 (PID=%d)" % mipi_proc.pid)

    # ─── 打开画面捕获管线 ─────────────────────────────
    # 使用 /dev/vin*_cap 设备读取帧
    # 当 mipi_cam 在 dual_combine 模式下运行时,
    # vcon@0 → VIN0, vcon@2 → VIN2
    # 我们需要捕获 VIN0 或 VIN2 的画面
    # 使用 spcdev 库通过 ctypes 读取
    
    import ctypes
    spcdev = ctypes.cdll.LoadLibrary('/usr/lib/libspcdev.so')
    spcdev.sp_init_vio_module.restype = ctypes.c_void_p
    vio = spcdev.sp_init_vio_module()
    if not vio:
        print("[ERROR] sp_init_vio_module 失败")
        mipi_proc.terminate()
        sys.exit(1)
    
    # 尝试用已知参数打开
    from ctypes import byref, c_int32, c_char_p, create_string_buffer
    
    class SpParams(ctypes.Structure):
        _fields_ = [('raw_height', c_int32), ('raw_width', c_int32), ('fps', c_int32)]
    
    params = SpParams(CAM_H, CAM_W, CAM_FPS)
    iw, ih = c_int32(CAM_W), c_int32(CAM_H)
    ret = spcdev.sp_open_camera_v2(vio, 0, -1, 1, byref(params), byref(iw), byref(ih))
    print(f"[INIT] sp_open_camera_v2: ret={ret}")
    
    if ret != 0:
        # fallback: 尝试视频索引 0
        iw, ih = c_int32(CAM_W), c_int32(CAM_H)
        ret = spcdev.sp_open_camera_v2(vio, 0, args.camera_id, 1,
                                       byref(params), byref(iw), byref(ih))
        print(f"[INIT] sp_open_camera_v2(cam={args.camera_id}): ret={ret}")
    
    if ret != 0:
        # 最终 fallback: 使用基础 open_camera
        iw, ih = c_int32(CAM_W), c_int32(CAM_H)
        spcdev.sp_open_camera.argtypes = [ctypes.c_void_p, c_int32, c_int32, c_int32,
                                          ctypes.POINTER(c_int32), ctypes.POINTER(c_int32)]
        spcdev.sp_open_camera.restype = c_int32
        ret = spcdev.sp_open_camera(vio, 0, -1, 1, byref(iw), byref(ih))
        print(f"[INIT] sp_open_camera(-1): ret={ret}")

    if ret != 0:
        print("[ERROR] 无法打开画面管线！")
        print("原因: mipi_cam 占用了摄像头管线，SP API 无法同时打开。")
        mipi_proc.terminate()
        spcdev.sp_release_vio_module(vio)
        sys.exit(1)

    # ─── 初始化摔倒检测器 ──────────────────────────────
    fall = FallDetector()
    if args.no_buzzer:
        fall._buzzer_on = lambda: None
        fall._buzzer_off = lambda: None

    global FALL_TIMEOUT, BUZZER_DURATION
    FALL_TIMEOUT = args.fall_timeout
    BUZZER_DURATION = args.buzzer_duration

    # ─── 信号处理 ──────────────────────────────────────
    running = True
    def handler(sig, frame):
        nonlocal running
        print("\n[SHUTDOWN] 关闭中...")
        running = False
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # ─── 主循环 ────────────────────────────────────────
    w_use, h_use = iw.value, ih.value
    frame_size = w_use * h_use * 3 // 2
    print(f"[RUN] 🟢 检测中 ({w_use}x{h_use}) Ctrl+C 停止")
    
    buf = create_string_buffer(frame_size)
    fps_cnt, fps_ts = 0, time.time()

    try:
        while running:
            r = spcdev.sp_vio_get_yuv(vio, buf, w_use, h_use, 2000)
            if r != 0:
                time.sleep(0.1)
                continue

            nv12 = np.frombuffer(buf.raw, dtype=np.uint8).copy()
            boxes, scores = detector.detect(nv12, w_use, h_use)
            fall.analyze(boxes, scores)

            fps_cnt += 1
            el = time.time() - fps_ts
            if el >= 5.0:
                s = fall
                print(f"[STATS] FPS={fps_cnt/el:.1f}  "
                      f"state={s.state} falls={s.fall_events} buzz={s.buzzer_events}")
                fps_cnt, fps_ts = 0, time.time()

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[CLEANUP] 清理...")
        fall.cleanup()
        spcdev.sp_vio_close(vio)
        spcdev.sp_release_vio_module(vio)
        mipi_proc.terminate()
        try:
            mipi_proc.wait(timeout=3)
        except:
            mipi_proc.kill()
        print(f"📊 总帧={fall.total_frames} 摔倒={fall.fall_events} 报警={fall.buzzer_events}")
        print("👋 已关闭")


if __name__ == '__main__':
    main()
