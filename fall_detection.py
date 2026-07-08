#!/usr/bin/env python3
"""
RDK X5 摔倒检测系统 — 支持 RDKGS130W 双目摄像头
=================================================
功能：
- 支持 MIPI 单/双目摄像头 + USB 摄像头
- 实时人体姿态检测（YOLO11n-pose，17个关键点）
- 骨架连接线 + 关节编号可视化
- 摔倒检测（人体框高宽比 + 肩髋水平度双判据）
- 摔倒时边框变红 + Server酱微信推送
- 通知每10秒最多一次

双目摄像头模式：
  RDKGS130W 输出为 1280x1088 NV12 图像（单目画面，GDC 校正后）
  检出人体后直接在该画面绘制框与骨架

用法：
  # 单 MIPI 摄像头（F37/OV5647等）
  sudo /usr/bin/python3.10 fall_detection.py

  # USB 摄像头
  sudo /usr/bin/python3.10 fall_detection.py --usb 0

  # 无 HDMI 显示（纯后台检测+通知）
  sudo /usr/bin/python3.10 fall_detection.py --no-display

首次连接 GS130W 相机前请先：
  sudo nano /boot/config.txt
  在末尾添加：
    dtoverlay=dtoverlay_cam0_imx219
  保存后重启
"""

import os
import sys
import time
import json
import signal
import argparse
import threading
import numpy as np
import cv2

import hbm_runtime
sys.path.append('/app/pydev_demo')
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils

# ============================================================
# COCO 17个关键点骨架
# ============================================================
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# 摔倒检测阈值
FALL_ASPECT_THRESH = 0.70
FALL_KPTS_Y_THRESH = 0.25
NOTIFY_COOLDOWN = 10
FALL_CONSECUTIVE = 5

# 默认 SendKey
SENDKEY = "SCT374376TS007AItuKe5CGm6kUnnfeXrs"


# ============================================================
# YOLO11n-Pose
# ============================================================
class YoloPose:
    def __init__(self, model_path, score=0.5, nms=0.7, kpt_conf=0.5):
        self.model = hbm_runtime.HB_HBMRuntime(model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]
        self.inH = self.input_shapes[self.input_names[0]][2]
        self.inW = self.input_shapes[self.input_names[0]][3]
        self.score_thres = score
        self.conf_raw = -np.log(1.0 / score - 1.0)
        self.nms = nms
        self.kpt_conf = kpt_conf
        self.resize_type = 1
        self.reg = 16
        self.strides = [8, 16, 32]
        self.anchors = [80, 40, 20]
        self.weights = np.arange(self.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]

    def pre_process(self, bgr):
        img = pre_utils.resized_image(bgr, self.inW, self.inH, self.resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(img)
        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.inH * 3 // 2, self.inW, 1))
        return {self.model_name: {self.input_names[0]: nv12}}

    def forward(self, tensor):
        return self.model.run(tensor)[self.model_name]

    def post_process(self, outputs, img_h, img_w):
        all_boxes, all_scores, all_ids = [], [], []
        all_kxy, all_ks = [], []
        fp32 = post_utils.dequantize_outputs(outputs, self.output_quants)
        for i, (stride, a_size) in enumerate(zip(self.strides, self.anchors)):
            ck = self.output_names[3*i]
            bk = self.output_names[3*i+1]
            kk = self.output_names[3*i+2]
            s, ids, valid = post_utils.filter_classification(fp32[ck], self.conf_raw)
            boxes = post_utils.decode_boxes(fp32[bk], valid, a_size, stride, self.weights)
            kxy, ks = post_utils.decode_kpts(fp32[kk], valid, a_size, stride)
            all_boxes.append(boxes)
            all_scores.append(s)
            all_ids.append(ids)
            all_kxy.append(kxy)
            all_ks.append(ks)
        dboxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        ids = np.concatenate(all_ids)
        kxy = np.concatenate(all_kxy)
        ks = np.concatenate(all_ks)
        keep = post_utils.NMS(dboxes, scores, ids, self.nms)
        xyxy = post_utils.scale_coords_back(dboxes[keep], img_w, img_h,
                                            self.inW, self.inH, self.resize_type)
        kxy, ks = post_utils.scale_keypoints_to_original_image(
            kxy[keep], ks[keep], xyxy,
            img_w, img_h, self.inW, self.inH, self.resize_type)
        return ids[keep], scores[keep], xyxy, kxy, ks


# ============================================================
# 摔倒检测
# ============================================================
def check_fall(box, kxy, ks, kpt_thresh):
    x1, y1, x2, y2 = map(int, box)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 2 or bh <= 2:
        return False, ""

    # ── 关键点数量过滤：有效节点 ≤10 时禁止摔倒检测 ──
    valid_kpts = int(np.sum(ks[:, 0] > kpt_thresh)) if ks is not None and len(ks) > 0 else 0
    if valid_kpts <= 10:
        return False, f"kpts:{valid_kpts}"

    aspect = bh / bw
    if aspect < FALL_ASPECT_THRESH:
        return True, f"aspect:{aspect:.2f}"

    conf = -np.log(1.0 / 0.3 - 1.0)
    sy = [kxy[i, 1] for i in (5, 6) if i < len(ks) and ks[i, 0] > conf]
    hy = [kxy[i, 1] for i in (11, 12) if i < len(ks) and ks[i, 0] > conf]
    if sy and hy:
        y_ratio = abs(np.mean(hy) - np.mean(sy)) / bh
        if y_ratio < FALL_KPTS_Y_THRESH:
            return True, f"horizontal:{y_ratio:.3f}"
    return False, ""


def draw_skeleton(img, kxy, ks, thresh, color=(0, 255, 255)):
    h, w = img.shape[:2]
    for i, j in SKELETON:
        if i < len(kxy) and j < len(kxy) and ks[i, 0] > thresh and ks[j, 0] > thresh:
            p1 = (int(kxy[i, 0]), int(kxy[i, 1]))
            p2 = (int(kxy[j, 0]), int(kxy[j, 1]))
            if 0 <= p1[0] < w and 0 <= p2[0] < w and 0 <= p1[1] < h and 0 <= p2[1] < h:
                cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)


def draw_kpts(img, kxy, ks, thresh):
    h, w = img.shape[:2]
    for j in range(len(kxy)):
        if ks[j, 0] < thresh:
            continue
        x, y = int(kxy[j, 0]), int(kxy[j, 1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
        cv2.putText(img, str(j), (x+6, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# ============================================================
# Server酱通知
# ============================================================
def send_notify(title, body, sendkey):
    url = f"https://sctapi.ftqq.com/{sendkey}.send"
    try:
        import requests
        resp = requests.post(url, json={"title": title, "desp": body}, timeout=10)
        r = resp.json()
        if r.get("code") == 0:
            print(f"\n[通知] ✅ {title}")
        else:
            print(f"\n[通知] ⚠️ {r.get('message', r)}")
    except ImportError:
        import urllib.request
        data = json.dumps({"title": title, "desp": body}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            r = json.loads(resp.read())
            if r.get("code") == 0:
                print(f"\n[通知] ✅ {title}")
            else:
                print(f"\n[通知] ⚠️ {r}")
        except Exception as e:
            print(f"\n[通知] ❌ {e}")
    except Exception as e:
        print(f"\n[通知] ❌ {e}")


def disp_color(r, g, b):
    return (255 << 24) | (r << 16) | (g << 8) | b


# ============================================================
# 主函数
# ============================================================
STOP = False

def sigint(sig, frame):
    global STOP
    print("\n正在停止...")
    STOP = True


def main():
    global STOP
    signal.signal(signal.SIGINT, sigint)

    parser = argparse.ArgumentParser(description="RDK X5 摔倒检测系统")
    parser.add_argument('--model', default='/app/pydev_demo/04_pose_sample/01_ultralytics_yolo11_pose/yolo11n_pose_bayese_640x640_nv12.bin')
    parser.add_argument('--score', type=float, default=0.5)
    parser.add_argument('--nms', type=float, default=0.7)
    parser.add_argument('--kpt-conf', type=float, default=0.5)
    parser.add_argument('--sendkey', default=SENDKEY)
    parser.add_argument('--no-display', action='store_true', help='无HDMI显示')
    parser.add_argument('--usb', type=int, default=None, help='USB摄像头设备号')
    parser.add_argument('--usb-width', type=int, default=640)
    parser.add_argument('--usb-height', type=int, default=480)
    # GS130W 双目参数
    parser.add_argument('--gs130w', action='store_true',
                        help='RDKGS130W 双目模式 (1280x1088, LPWM)')
    parser.add_argument('--gs130w-split', action='store_true',
                        help='GS130W 使用双目拼接画面: 仅用左半/右半边')
    parser.add_argument('--gs130w-split-side', choices=['left', 'right'], default='left',
                        help='GS130W 使用哪一边 (默认 left)')
    args = parser.parse_args()

    # ================================================================
    # 1. 加载模型
    # ================================================================
    print("[1/4] 加载 YOLO11n-pose 模型...")
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        sys.exit(1)
    yolo = YoloPose(args.model, args.score, args.nms, args.kpt_conf)
    kpt_thresh = -np.log(1.0 / args.kpt_conf - 1.0)
    print(f"  ✅ 输入: {yolo.inW}x{yolo.inH}")

    # ================================================================
    # 2. 初始化摄像头
    # ================================================================
    print("[2/4] 初始化摄像头...")
    display = None

    if args.usb is not None:
        # ---- USB 摄像头 ----
        print(f"  USB /dev/video{args.usb}")
        cap = cv2.VideoCapture(args.usb)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.usb_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.usb_height)
        time.sleep(0.5)
        ret, test = cap.read()
        if not ret or test is None:
            print(f"  ❌ USB摄像头无画面!")
            cap.release()
            sys.exit(1)
        H, W = test.shape[:2]
        print(f"  ✅ USB摄像头 {W}x{H}")
        is_mipi, srcampy = False, None
    elif args.gs130w:
        # ---- RDKGS130W 双目摄像头 ----
        try:
            from hobot_vio import libsrcampy as srcampy
        except ImportError:
            try:
                from hobot_vio_rdkx5 import libsrcampy as srcampy
            except ImportError:
                print("  ❌ 找不到 libsrcampy")
                sys.exit(1)
        cam = srcampy.Camera()
        # GS130W 参数: 1280x1088, 10FPS, LPWM enable
        # cam.open_cam(channel, -1, -1, [width, disp_width], [height, disp_height], sensor_h, sensor_w)
        W, H = 1280, 1088
        if not args.no_display:
            disp_w, disp_h = 1920, 1080
            if os.path.exists("/usr/bin/get_hdmi_res"):
                import subprocess
                p = subprocess.Popen(["/usr/bin/get_hdmi_res"], stdout=subprocess.PIPE)
                res = p.communicate()[0].split(b',')
                disp_w = max(min(int(res[1]), 1920), 0)
                disp_h = max(min(int(res[0]), 1080), 0)
            print(f"  GS130W 双目摄像头, HDMI: {disp_w}x{disp_h}")
            cam.open_cam(0, -1, -1, [W, disp_w], [H, disp_h], H, W)
            display = srcampy.Display()
            display.display(0, disp_w, disp_h)
            srcampy.bind(cam, display)
        else:
            print(f"  GS130W 双目摄像头 (后台模式)")
            cam.open_cam(0, -1, -1, [W, W], [H, H], H, W)

        time.sleep(2)  # GS130W 需要更长时间初始化
        test = cam.get_img(2, W, H)
        if test is None or len(test) == 0:
            print("  ❌ GS130W 无画面！")
            print("  💡 请检查:")
            print("     1. FPC排线是否插紧")
            print("     2. 是否已配置 /boot/config.txt")
            print("     3. 摄像头供电是否正常")
            cam.close_cam()
            sys.exit(1)
        print(f"  ✅ GS130W 就绪 ({W}x{H})")
        cap = None
        is_mipi = True
    else:
        # ---- 标准 MIPI 单摄像头 ----
        try:
            from hobot_vio import libsrcampy as srcampy
        except ImportError:
            try:
                from hobot_vio_rdkx5 import libsrcampy as srcampy
            except ImportError:
                print("  ❌ 找不到 libsrcampy")
                sys.exit(1)
        cam = srcampy.Camera()
        W, H = 1920, 1080
        if not args.no_display:
            dw, dh = 1920, 1080
            if os.path.exists("/usr/bin/get_hdmi_res"):
                import subprocess
                p = subprocess.Popen(["/usr/bin/get_hdmi_res"], stdout=subprocess.PIPE)
                res = p.communicate()[0].split(b',')
                dw = max(min(int(res[1]), 1920), 0)
                dh = max(min(int(res[0]), 1080), 0)
            print(f"  MIPI摄像头, HDMI: {dw}x{dh}")
            cam.open_cam(0, -1, -1, [W, dw], [H, dh], 1080, 1920)
            display = srcampy.Display()
            display.display(0, dw, dh)
            srcampy.bind(cam, display)
        else:
            print("  MIPI摄像头 (后台模式)")
            cam.open_cam(0, -1, -1, [W, W], [H, H], 1080, 1920)
        time.sleep(1)
        test = cam.get_img(2, W, H)
        if test is None or len(test) == 0:
            print("  ❌ MIPI摄像头无画面！请检查FPC连接或尝试USB模式")
            cam.close_cam()
            sys.exit(1)
        print(f"  ✅ MIPI摄像头 {W}x{H}")
        cap = None
        is_mipi = True

    # ================================================================
    # 3. 状态变量
    # ================================================================
    last_notify = 0.0
    notified_this_fall = False  # 当前摔倒事件是否已推送过
    normal_recovery_frames = 0  # 连续正常帧计数，用于复位 notified_this_fall
    fall_cnt = 0
    fps_n, fps_t0 = 0, time.time()
    fps_val = 0

    print(f"[3/4] Server酱 推送 {'已配置' if args.sendkey else '未配置'}")
    print(f"[4/4] 🚀 摔倒检测系统运行中！")
    print("=" * 55)

    # ================================================================
    # 4. 主循环
    # ================================================================
    while not STOP:
        t0 = time.time()

        # ---- 取帧 ----
        if is_mipi:
            raw = cam.get_img(2, W, H)
            if raw is None or len(raw) == 0:
                time.sleep(0.02)
                continue
            nv12 = np.frombuffer(raw, dtype=np.uint8).reshape((H * 3 // 2, W))
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        else:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                time.sleep(0.02)
                continue

        # GS130W 双目裁剪模式: 只取左半或右半边
        if args.gs130w and args.gs130w_split:
            h, w = bgr.shape[:2]
            half_w = w // 2
            if args.gs130w_split_side == 'left':
                bgr = bgr[:, :half_w]
            else:
                bgr = bgr[:, half_w:]
            H, W = bgr.shape[:2]

        # ---- 推理 ----
        inp = yolo.pre_process(bgr)
        out = yolo.forward(inp)
        ids, scores, boxes, kxy, ks = yolo.post_process(out, H, W)

        # ---- 逐人体处理 ----
        now = time.time()
        any_fall = False
        fall_reason = ""

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)
            fallen, reason = check_fall(box, kxy[i], ks[i], kpt_thresh)
            if fallen:
                any_fall = True
                fall_reason = reason

            color = (0, 0, 255) if fallen else (0, 255, 0)
            label = f"FALL {scores[i]:.2f}" if fallen else f"Person {scores[i]:.2f}"

            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 3)
            cv2.putText(bgr, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            draw_skeleton(bgr, kxy[i], ks[i], kpt_thresh,
                          color=(0, 200, 255) if not fallen else (0, 100, 255))
            draw_kpts(bgr, kxy[i], ks[i], kpt_thresh)

        # ---- 摔倒计数 ----
        if any_fall:
            fall_cnt += 1
            normal_recovery_frames = 0  # 还在摔倒，复位恢复计数
        else:
            fall_cnt = 0
            normal_recovery_frames += 1
            if normal_recovery_frames >= 30:
                notified_this_fall = False  # 确认已恢复站立，允许下次摔倒重新推送

        # ---- 触发通知 ----
        new_incident = not notified_this_fall
        if fall_cnt >= FALL_CONSECUTIVE and now - last_notify >= NOTIFY_COOLDOWN and new_incident:
            last_notify = now
            notified_this_fall = True  # 标记已推送，同一次摔倒不再发
            tstr = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[🚨 摔倒] {tstr} | 持续{fall_cnt}帧 | {len(boxes)}人 | {fall_reason}")
            if args.sendkey and args.sendkey != SENDKEY:
                threading.Thread(target=send_notify, args=(
                    "🚨 摔倒检测告警",
                    f"**检测到有人摔倒！**\n\n"
                    f"- 时间：{tstr}\n"
                    f"- 持续帧数：{fall_cnt}\n"
                    f"- 画面中 {len(boxes)} 人\n"
                    f"- 判据：{fall_reason}\n\n"
                    f"请及时查看！",
                    args.sendkey
                ), daemon=True).start()
            elif args.sendkey:
                # 默认已配置
                threading.Thread(target=send_notify, args=(
                    "🚨 摔倒检测告警",
                    f"**检测到有人摔倒！**\n\n"
                    f"- 时间：{tstr}\n"
                    f"- 持续帧数：{fall_cnt}\n"
                    f"- 画面中 {len(boxes)} 人\n"
                    f"- 判据：{fall_reason}\n\n"
                    f"请及时查看！",
                    args.sendkey
                ), daemon=True).start()

        # ---- HDMI 叠加 ----
        if is_mipi and display is not None:
            display.set_graph_rect(0, 0, 0, 0, 2, 1, 0, 1)
            display.set_graph_word(0, 0, b"", 2, 1, 0, 10)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                fallen, _ = check_fall(boxes[i], kxy[i], ks[i], kpt_thresh)
                c = disp_color(255, 0, 0) if fallen else disp_color(0, 255, 0)
                display.set_graph_rect(x1, y1, x2, y2, 2, 0, c, 3)
                lbl = b"FALL!" if fallen else f"Person {scores[i]:.2f}".encode()
                display.set_graph_word(x1, max(y1-22, 0), lbl, 2, 0, c, 15)
            display.set_graph_word(10, 10, f"FPS:{fps_val}".encode(), 2, 0,
                                   disp_color(255, 255, 255), 15)

        # ---- OpenCV窗口（USB模式） ----
        if not is_mipi and not args.no_display:
            cv2.imshow("Fall Detection", bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ---- FPS ----
        fps_n += 1
        if now - fps_t0 >= 1.0:
            fps_val = fps_n
            fps_n = 0
            fps_t0 = now
            status = "🟢" if not any_fall else "🔴"
            cam_label = "GS130W" if args.gs130w else ("USB" if not is_mipi else "MIPI")
            print(f"\r[{cam_label}] FPS={fps_val} | 人体={len(boxes)} | {status}    ", end="")

        # ---- 限速 ----
        dt = time.time() - t0
        if dt < 1/30:
            time.sleep(1/30 - dt)

    # ================================================================
    # 清理
    # ================================================================
    print("\n\n[退出] 释放资源...")
    if is_mipi:
        if display is not None:
            srcampy.unbind(cam, display)
            display.close()
        cam.close_cam()
    else:
        cap.release()
        cv2.destroyAllWindows()
    print("[退出] ✅ 已安全停止")


if __name__ == "__main__":
    main()
