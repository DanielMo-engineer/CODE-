#!/usr/bin/env python3
"""
RDK X5 人体摔倒检测系统 — 带姿态骨骼点版
========================================
- YOLO11n-pose: 单模型同时输出人体框 + 17个COCO关键点
- 骨架骨骼点绘制（彩色圆点 + 人体骨架连线）
- 摔倒检测逻辑不变（框宽高比 + 垂直速度 + 三级过滤）
- Web MJPEG 流 480p
- Server酱微信推送 + 腾讯云语音电话

Author: OpenClaw | D-Robotics
"""

import os
import sys
import time
import math
import signal
import argparse
import socket
import threading
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

import hbm_runtime

sys.path.append('/app/pydev_demo')
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils
import utils.common_utils as common

try:
    from hobot_vio import libsrcampy
except ImportError:
    from hobot_vio_rdkx5 import libsrcampy

# ── 表情识别 ──
_EMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion")
sys.path.insert(0, _EMO_DIR)
from emotion_recognizer import EmotionRecognizer, MoodLogger


# ============================================================
# COCO 17 关键点定义 + 骨架连线
# ============================================================

COCO_KEYPOINT_NAMES = [
    "鼻子", "左眼", "右眼", "左耳", "右耳",
    "左肩", "右肩", "左肘", "右肘",
    "左手腕", "右手腕", "左髋", "右髋",
    "左膝", "右膝", "左踝", "右踝",
]

# 骨架连线（两端点索引）
SKELETON_CONNECTIONS = [
    # 头部
    (0, 1), (0, 2),        # 鼻子 → 左右眼
    (1, 3), (2, 4),        # 左右眼 → 左右耳
    # 肩部
    (5, 6),                 # 左肩 ↔ 右肩
    # 手臂
    (5, 7), (7, 9),        # 左肩 → 左肘 → 左手腕
    (6, 8), (8, 10),       # 右肩 → 右肘 → 右手腕
    # 躯干
    (5, 11), (6, 12),      # 左右肩 → 左右髋
    (11, 12),              # 左髋 ↔ 右髋
    # 腿部
    (11, 13), (13, 15),    # 左髋 → 左膝 → 左踝
    (12, 14), (14, 16),    # 右髋 → 右膝 → 右踝
]

# 每段骨架连线的颜色（BGR）
SKELETON_COLORS = {
    # 头部连线（青色）
    (0, 1): (255, 255, 0), (0, 2): (255, 255, 0),
    (1, 3): (255, 255, 0), (2, 4): (255, 255, 0),
    # 肩部（橙色）
    (5, 6): (0, 165, 255),
    # 手臂（绿色系）
    (5, 7): (0, 255, 100), (7, 9): (0, 255, 100),
    (6, 8): (0, 255, 100), (8, 10): (0, 255, 100),
    # 躯干（蓝色系）
    (5, 11): (255, 128, 0), (6, 12): (255, 128, 0),
    (11, 12): (255, 128, 0),
    # 腿部（紫色/粉色系）
    (11, 13): (255, 0, 255), (13, 15): (255, 0, 255),
    (12, 14): (255, 0, 255), (14, 16): (255, 0, 255),
}

# 关键点颜色（BGR）
KPT_COLORS = [
    (0, 255, 255),   # 0  鼻子      — 黄色
    (255, 255, 0),   # 1  左眼      — 青色
    (255, 255, 0),   # 2  右眼      — 青色
    (255, 255, 0),   # 3  左耳      — 青色
    (255, 255, 0),   # 4  右耳      — 青色
    (0, 165, 255),   # 5  左肩      — 橙色
    (0, 165, 255),   # 6  右肩      — 橙色
    (0, 255, 100),   # 7  左肘      — 亮绿
    (0, 255, 100),   # 8  右肘      — 亮绿
    (0, 255, 100),   # 9  左手腕    — 亮绿
    (0, 255, 100),   # 10 右手腕    — 亮绿
    (255, 128, 0),   # 11 左髋      — 深蓝
    (255, 128, 0),   # 12 右髋      — 深蓝
    (255, 0, 255),   # 13 左膝      — 品红
    (255, 0, 255),   # 14 右膝      — 品红
    (255, 0, 255),   # 15 左踝      — 品红
    (255, 0, 255),   # 16 右踝      — 品红
]


def draw_skeleton(img: np.ndarray, kpts_xy: np.ndarray,
                  kpts_score: np.ndarray, conf_thresh: float = 0.3,
                  draw_labels: bool = False) -> None:
    """
    在图像上绘制骨架（骨骼连线 + 关键点圆点）。
    kpts_xy: shape (17, 2)
    kpts_score: shape (17, 1)
    """
    h, w = img.shape[:2]

    # 1) 绘制骨架连线
    for (i, j) in SKELETON_CONNECTIONS:
        if (kpts_score[i, 0] < conf_thresh or
            kpts_score[j, 0] < conf_thresh):
            continue
        xi, yi = int(kpts_xy[i, 0]), int(kpts_xy[i, 1])
        xj, yj = int(kpts_xy[j, 0]), int(kpts_xy[j, 1])
        if not (0 <= xi < w and 0 <= yi < h and
                0 <= xj < w and 0 <= yj < h):
            continue
        color = SKELETON_COLORS.get((i, j), SKELETON_COLORS.get((j, i), (0, 255, 0)))
        cv2.line(img, (xi, yi), (xj, yj), color, 2, cv2.LINE_AA)

    # 2) 绘制关键点圆点
    for kpt_idx in range(17):
        if kpts_score[kpt_idx, 0] < conf_thresh:
            continue
        x, y = int(kpts_xy[kpt_idx, 0]), int(kpts_xy[kpt_idx, 1])
        if not (0 <= x < w and 0 <= y < h):
            continue

        color = KPT_COLORS[kpt_idx]

        # 外圈（白色粗描边）
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
        # 内圈（部位专属颜色）
        cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)

        # 可选：标注关键点编号
        if draw_labels:
            cv2.putText(img, str(kpt_idx), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


# ============================================================
# YOLO11n-pose 检测器（人体检测 + 姿态估计）
# ============================================================

class PoseDetector:
    """YOLO11n-pose 人体检测+姿态估计（BPU 加速）"""

    def __init__(self, model_path: str, score_thres: float = 0.25,
                 kpt_conf_thres: float = 0.3):
        self.model = hbm_runtime.HB_HBMRuntime(model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]

        self.score_thres = score_thres
        self.kpt_conf_thres = kpt_conf_thres
        self.conf_thres_raw = -np.log(1 / self.score_thres - 1)  # sigmoid 逆
        self.nms_thresh = score_thres
        self.resize_type = 1  # letterbox
        self.reg = 16
        self.strides = [8, 16, 32]
        self.anchor_sizes = [80, 40, 20]
        self.weights_static = np.arange(self.reg, dtype=np.float32)[
            np.newaxis, np.newaxis, :]

    def set_scheduling(self, priority: int = 0,
                       bpu_cores: Optional[list] = None):
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def infer(self, img_bgr: np.ndarray) -> Tuple:
        """
        单帧推理：BGR → letterbox NV12 → BPU → 后处理
        返回 (person_boxes, person_kpts_xy, person_kpts_score, max_score)
        """
        img_h, img_w = img_bgr.shape[:2]

        # 1) 预处理：BGR → letterbox resize → NV12
        resize_img = pre_utils.resized_image(
            img_bgr, self.input_W, self.input_H, self.resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))
        inp = {self.model_name: {self.input_names[0]: nv12}}

        # 2) BPU 推理
        outputs = self.model.run(inp)[self.model_name]

        # 3) 后处理
        fp32 = post_utils.dequantize_outputs(outputs, self.output_quants)

        all_dbboxes, all_scores, all_ids = [], [], []
        all_kpts_xy, all_kpts_score = [], []

        for i, (stride, anchor_size) in enumerate(
                zip(self.strides, self.anchor_sizes)):
            cls_key = self.output_names[3 * i]
            box_key = self.output_names[3 * i + 1]
            kpts_key = self.output_names[3 * i + 2]

            scores, ids, valid_indices = \
                post_utils.filter_classification(
                    fp32[cls_key], self.conf_thres_raw)

            dbboxes = post_utils.decode_boxes(
                fp32[box_key], valid_indices,
                anchor_size, stride, self.weights_static)

            kpts_xy, kpts_score = post_utils.decode_kpts(
                fp32[kpts_key], valid_indices, anchor_size, stride)

            all_dbboxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)
            all_kpts_xy.append(kpts_xy)
            all_kpts_score.append(kpts_score)

        # 合并所有尺度
        dbboxes = np.concatenate(all_dbboxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        ids = np.concatenate(all_ids, axis=0)
        kpts_xy = np.concatenate(all_kpts_xy, axis=0)
        kpts_score = np.concatenate(all_kpts_score, axis=0)

        # NMS
        keep = post_utils.NMS(dbboxes, scores, ids, self.nms_thresh)

        # 缩放到原图坐标
        xyxy = post_utils.scale_coords_back(
            dbboxes[keep], img_w, img_h,
            self.input_W, self.input_H, self.resize_type)

        kpts_xy, kpts_score = post_utils.scale_keypoints_to_original_image(
            kpts_xy[keep], kpts_score[keep], xyxy,
            img_w, img_h, self.input_W, self.input_H, self.resize_type)

        # 只保留 person（class 0）
        person_mask = ids[keep] == 0
        person_boxes = xyxy[person_mask]
        person_kpts_xy = kpts_xy[person_mask]
        person_kpts_score = kpts_score[person_mask]

        max_score = float(max(scores)) if len(scores) > 0 else 0.0

        return person_boxes, person_kpts_xy, person_kpts_score, max_score


# ============================================================
# 摔倒检测器（与纯检测版逻辑完全一致）
# ============================================================

class FallDetector:
    """
    混合摔倒检测（BBox + 姿态关键点双仲裁）

    ╔═══════════════════════════════════════════════════╗
    ║  条件A（BBox 初筛）：宽高比 > 1.2 或 垂直下落      ║
    ║  条件B（姿态仲裁）：身体倾角 < 30° 且 脚离地      ║
    ║  最终决策：连续确认后，三级过滤器「三选二」投票     ║
    ╚═══════════════════════════════════════════════════╝

    COCO 关键点索引:
       5=左肩, 6=右肩, 11=左髋, 12=右髋, 15=左踝, 16=右踝
    """

    def __init__(self, fps=30, confirm_frames=10, cooldown=5):
        self.frame_interval = 1.0 / fps
        self.confirm_frames = confirm_frames
        self.cooldown = cooldown
        self.history = deque(maxlen=20)
        self.consecutive = 0
        self.in_alarm = False
        self.alarm_until = 0.0  # 时间戳：ALARM 状态持续到何时
        self.min_interval = 10  # 两次报警最小间隔（秒）

    # ── 姿态关键点辅助函数 ──

    @staticmethod
    def get_midpoint(kpts_xy, idx_a, idx_b, conf_thresh=0.3):
        """返回两个关键点的中点 (x, y, valid)，若置信度不足则 valid=False"""
        if kpts_xy is None or len(kpts_xy) < max(idx_a, idx_b) + 1:
            return 0, 0, False
        sa = kpts_xy[idx_a, 2] if kpts_xy.shape[1] > 2 else 1.0
        sb = kpts_xy[idx_b, 2] if kpts_xy.shape[1] > 2 else 1.0
        if sa < conf_thresh or sb < conf_thresh:
            return 0, 0, False
        mx = (kpts_xy[idx_a, 0] + kpts_xy[idx_b, 0]) / 2
        my = (kpts_xy[idx_a, 1] + kpts_xy[idx_b, 1]) / 2
        return mx, my, True

    @staticmethod
    def calc_body_angle(kpts_xy, conf_thresh=0.3):
        """
        计算身体与水平面的夹角（度）。
        肩部中心 → 髋部中心 向量与水平方向夹角。
        躺平时接近 0°，站立时接近 90°。
        """
        sx, sy, sv = FallDetector.get_midpoint(kpts_xy, 5, 6, conf_thresh)
        hx, hy, hv = FallDetector.get_midpoint(kpts_xy, 11, 12, conf_thresh)
        if not sv or not hv:
            return None, False
        dx = sx - hx
        dy = sy - hy  # y向下为正，站立时dy<0
        # 与水平面夹角（0°=水平，90°=垂直）
        angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
        return angle, True

    @staticmethod
    def is_ankle_above_hip(kpts_xy, conf_thresh=0.3):
        """
        脚踝关键点Y是否高于髋关节中心（脚离地支持）。
        在图像坐标系中 y↓，所以 ankle_y < hip_y 表示脚踝在髋关节上方。
        """
        hx, hy, hv = FallDetector.get_midpoint(kpts_xy, 11, 12, conf_thresh)
        if not hv:
            return False
        # 任一脚踝高于髋关节即判定为脚离地
        for ankle_idx in (15, 16):
            _, ay, av = FallDetector.get_single_kpt(kpts_xy, ankle_idx, conf_thresh)
            if av and ay < hy:
                return True
        return False

    @staticmethod
    def get_single_kpt(kpts_xy, idx, conf_thresh=0.3):
        if kpts_xy is None or len(kpts_xy) < idx + 1:
            return 0, 0, False
        sc = kpts_xy[idx, 2] if kpts_xy.shape[1] > 2 else 1.0
        if sc < conf_thresh:
            return 0, 0, False
        return kpts_xy[idx, 0], kpts_xy[idx, 1], True

    @staticmethod
    def estimate_close_range(area_px, frame_h, frame_w):
        """
        粗略判断人是否离摄像头很近（框面积占比 > 40%）。
        近距时关键点抖动大，需要更保守的判定。
        """
        ratio = area_px / (frame_h * frame_w)
        return ratio > 0.40, ratio

    def detect(self, frame_idx, box, prev_area, kpts_xy=None,
               kpts_score=None, frame_h=1280, frame_w=1088):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        aspect = w / h if h > 0 else 0
        area = w * h
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # ── 判断是否近距离 ──
        is_close, close_ratio = self.estimate_close_range(area, frame_h, frame_w)

        # ── 关键点数量过滤：有效节点 ≤10 时禁止摔倒检测 ──
        if kpts_score is not None and len(kpts_score) > 0:
            valid_kpts = int(np.sum(kpts_score[:, 0] > 0.3))
        elif kpts_xy is not None and kpts_xy.shape[-1] >= 3:
            # 如果未传 kpts_score，尝试从 kpts_xy 第三列读置信度
            valid_kpts = int(np.sum(kpts_xy[:, 2] > 0.3))
        elif kpts_xy is not None:
            valid_kpts = len(kpts_xy)  # 无置信度时默认全部有效
        else:
            valid_kpts = 0

        if valid_kpts <= 10:
            return ("NORMAL", {
                "aspect": float(aspect), "avg_aspect": float(aspect),
                "vert_vel": 0, "horiz_vel": 0,
                "body_angle": 0, "feet_off": False,
                "close_ratio": float(close_ratio), "triggered": False,
                "consecutive": 0, "pose_valid": False,
                "valid_kpts": int(valid_kpts),
                "reason": "too_few_keypoints",
            })

        # ── 姿态关键点分析（条件B） ──
        body_angle = None
        angle_valid = False
        feet_off = False
        if kpts_xy is not None:
            body_angle, angle_valid = self.calc_body_angle(kpts_xy)
            feet_off = self.is_ankle_above_hip(kpts_xy)
        body_horizontal = (angle_valid and body_angle is not None
                           and body_angle < 30.0)

        # ── Metrics ──

        # Metric 1: Aspect ratio jump
        aspect_jump = False
        if len(self.history) >= 5:
            past_aspects = [d[0] for d in list(self.history)[-5:]]
            if all(a < 0.5 for a in past_aspects) and aspect > 1.2:
                aspect_jump = True

        # Metric 2: Vertical velocity
        vert_vel = 0.0
        if len(self.history) >= 2:
            prev_cy = self.history[-1][2]
            vert_vel = (cy - prev_cy) / self.frame_interval
        vel_body_per_sec = vert_vel / h if h > 0 else 0

        # Metric 3: Horizontal velocity
        horiz_vel = 0.0
        if len(self.history) >= 2:
            prev_cx = self.history[-1][1]
            horiz_vel = abs(cx - prev_cx) / self.frame_interval

        self.history.append((aspect, cx, cy, area))

        avg_aspect = np.mean([d[0] for d in self.history])

        # ── 条件A（BBox）：是否疑似摔倒 ──
        tilt_trigger = avg_aspect > 1.0
        fall_trigger = vel_body_per_sec > 0.7
        jump_trigger = aspect_jump
        bbox_triggered = tilt_trigger or fall_trigger or jump_trigger

        # ── 条件B（姿态）：身体已躺平 ──
        pose_triggered = body_horizontal or (angle_valid and body_angle is not None
                                             and body_angle < 45.0)

        # ── 综合触发：条件A且条件B（有姿态时），或条件A仅BBox ──
        if angle_valid:
            triggered = bbox_triggered and pose_triggered
        else:
            triggered = bbox_triggered

        # 近距离时，置信度要求更高
        if is_close and angle_valid:
            # 近距：必须同时BBox触发 + 身体明显躺平(<25°) + 脚离地
            triggered = triggered and body_angle is not None and body_angle < 25.0

        # ── 连续计数 ──
        if triggered:
            self.consecutive += 1
        else:
            if self.in_alarm and time.time() < self.alarm_until:
                # 冷却中（时间制），继续返回 ALARM
                return ("ALARM", {
                    "aspect": float(aspect), "avg_aspect": float(avg_aspect),
                    "vert_vel": float(vert_vel), "horiz_vel": float(horiz_vel),
                    "body_angle": float(body_angle) if body_angle is not None else 0,
                    "feet_off": feet_off, "close_ratio": float(close_ratio),
                    "triggered": triggered, "consecutive": self.consecutive,
                    "pose_valid": angle_valid,
                })
            self.in_alarm = False
            self.consecutive = 0
            return ("NORMAL", {
                "aspect": float(aspect), "avg_aspect": float(avg_aspect),
                "vert_vel": float(vert_vel), "horiz_vel": float(horiz_vel),
                "body_angle": float(body_angle) if body_angle is not None else 0,
                "feet_off": feet_off, "close_ratio": float(close_ratio),
                "triggered": triggered, "consecutive": self.consecutive,
                "pose_valid": angle_valid,
            })

        # ── 连续确认未达标 → 继续观察 ──
        if self.consecutive < self.confirm_frames:
            return ("NORMAL", {
                "aspect": float(aspect), "avg_aspect": float(avg_aspect),
                "vert_vel": float(vert_vel), "horiz_vel": float(horiz_vel),
                "body_angle": float(body_angle) if body_angle is not None else 0,
                "feet_off": feet_off, "close_ratio": float(close_ratio),
                "triggered": triggered, "consecutive": self.consecutive,
                "pose_valid": angle_valid,
            })

        # ════════════════════════════════════════════════════════
        #  三级过滤器 → 改为「三选二」投票机制
        # ════════════════════════════════════════════════════════
        filters_passed = 0
        filter_details = []

        # Filter 1: 水平速度是否远大于垂直速度（快速横向移动）
        filter1 = abs(vert_vel) <= horiz_vel * 1.5
        # ✅ filter1 = True → 更像是横向移动，可能是误报 → 不计数
        # ❌ filter1 = False → 垂直下降为主 → 是有效摔倒信号
        if not filter1:
            filters_passed += 1
            filter_details.append("F1_vert_dominant")
        else:
            filter_details.append("F1_horiz_motion")

        # Filter 2: 宽高比 > 1.2 时，检查水平位移（躺平但没滑动=坐着）
        filter2 = True
        if aspect > 1.2:
            horiz_body = horiz_vel / h if h > 0 else 0
            # 近距离时阈值加倍（抖动大）
            horiz_thresh = 0.4 if is_close else 0.2
            if horiz_body < horiz_thresh:
                filter2 = False  # 躺平没动 → 可能是坐着，排除
            else:
                filter2 = True   # 躺着也在动 → 有效
        else:
            filter2 = True       # 还没躺平，交给其他过滤器
        if filter2:
            filters_passed += 1
            filter_details.append("F2_moving_while_flat")
        else:
            filter_details.append("F2_static_flat")

        # Filter 3: 面积变化（帧间抖动过大 = 误检）
        filter3 = True
        if prev_area > 0 and area > 0:
            change = abs(area - prev_area) / max(area, prev_area)
            # 近距离时阈值放宽（大框抖动更明显）
            change_thresh = 0.6 if is_close else 0.5
            filter3 = change <= change_thresh
        # filter3 = True → 面积稳定，有效
        if filter3:
            filters_passed += 1
            filter_details.append("F3_area_stable")
        else:
            filter_details.append("F3_area_jitter")

        # ── 三选二投票 ──
        if filters_passed < 2:
            self.consecutive = 0
            return ("NORMAL", {
                "aspect": float(aspect), "avg_aspect": float(avg_aspect),
                "vert_vel": float(vert_vel), "horiz_vel": float(horiz_vel),
                "body_angle": float(body_angle) if body_angle is not None else 0,
                "feet_off": feet_off, "close_ratio": float(close_ratio),
                "triggered": triggered, "consecutive": self.consecutive,
                "pose_valid": angle_valid,
                "filters": filter_details,
                "filters_passed": filters_passed,
            })

        # ── 最小报警间隔检查（防频繁触发） ──
        now_ts = time.time()
        if hasattr(self, '_last_alarm_real_ts') and now_ts - self._last_alarm_real_ts < self.min_interval:
            self.consecutive = 0
            remain = self.min_interval - (now_ts - self._last_alarm_real_ts)
            print(f"  [⏳] 距上次报警 {now_ts - self._last_alarm_real_ts:.0f}s, 还需 {remain:.0f}s")
            return ("NORMAL", {
                "aspect": float(aspect), "avg_aspect": float(avg_aspect),
                "vert_vel": float(vert_vel), "horiz_vel": float(horiz_vel),
                "body_angle": float(body_angle) if body_angle is not None else 0,
                "feet_off": feet_off, "close_ratio": float(close_ratio),
                "triggered": triggered, "consecutive": self.consecutive,
                "pose_valid": angle_valid,
                "filters": filter_details,
                "filters_passed": filters_passed,
            })

        self._last_alarm_real_ts = now_ts

        # ── ALARM ──
        self.in_alarm = True
        self.alarm_until = now_ts + self.cooldown
        now = time.strftime("%H:%M:%S")
        print(f"\033[31m{'='*50}")
        print(f"  ⚠ 摔倒警告 ⚠  {now}")
        print(f"  帧={frame_idx}  宽高比={aspect:.2f}  下落={vel_body_per_sec:.2f}身/s"
              f"  倾角={body_angle}°" if body_angle is not None else "")
        if is_close:
            print(f"  📏 近距离({close_ratio:.0%}), 已收紧阈值")
        if filters_passed >= 2:
            print(f"  ✅ 三选二投票: {filters_passed}/3 通过 → {filter_details}")
        print(f"{'='*50}\033[0m")
        return ("ALARM", {
            "aspect": float(aspect), "avg_aspect": float(avg_aspect),
            "vert_vel": float(vert_vel), "horiz_vel": float(horiz_vel),
            "body_angle": float(body_angle) if body_angle is not None else 0,
            "feet_off": feet_off, "close_ratio": float(close_ratio),
            "triggered": triggered, "consecutive": self.consecutive,
            "pose_valid": angle_valid,
            "filters": filter_details,
            "filters_passed": filters_passed,
        })


# ============================================================
# HTTP MJPEG 流
# ============================================================

class WebStreamer:
    def __init__(self, port=8080, quality=50, disp_w=480):
        self.port = port
        self.quality = quality
        self.disp_w = disp_w
        self._frame = None
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    class _RequestHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, streamer=None, **kw):
            self.streamer = streamer
            super().__init__(*args, **kw)

        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html;charset=utf-8')
                self.end_headers()
                ip = self.streamer.local_ip
                html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>RDK X5 摔倒检测 + 姿态骨骼</title>
<style>
  *{{margin:0;padding:0}}
  body{{background:#111;display:flex;justify-content:center;align-items:center;height:100vh;font-family:sans-serif}}
  #wrap{{text-align:center}}
  img{{max-width:100vw;max-height:90vh;border-radius:8px;box-shadow:0 0 20px rgba(0,0,0,0.5)}}
  #info{{color:#aaa;margin-top:10px;font-size:14px}}
  .alarm-dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
  .alarm{{background:#f44;box-shadow:0 0 8px #f44}}
  .ok{{background:#4f4}}
</style></head><body>
<div id="wrap">
  <img src="/stream" />
  <div id="info">RDK X5 摔倒检测 + 姿态骨骼 · <a href="/api/status" style="color:#888">状态</a></div>
</div>
</body></html>'''
                self.wfile.write(html.encode())
            elif self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=fr')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                try:
                    while True:
                        with self.streamer._lock:
                            frame = self.streamer._frame
                        if frame is not None:
                            self.wfile.write(b'--fr\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                            self.wfile.write(frame)
                            self.wfile.write(b'\r\n')
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            elif self.path == '/api/status':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                s = self.streamer
                self.wfile.write(json.dumps({
                    'fps': s.fps, 'status': s.status,
                    'people': s.people, 'alarm': s.alarm,
                    'aspect': s.aspect, 'vert_vel': s.vert_vel,
                    'emotion': s.emotion,
                    'emotion_cn': s.emotion_cn,
                    'emotion_conf': s.emotion_conf,
                }).encode())
            else:
                self.send_response(404)
                self.end_headers()

    def start(self):
        server = ThreadingHTTPServer(('0.0.0.0', self.port),
                            lambda *a, **kw: self._RequestHandler(*a, streamer=self, **kw))
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('10.254.254.254', 1))
            self.local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            self.local_ip = 'localhost'
        self.fps = 0
        self.status = 'NORMAL'
        self.people = 0
        self.alarm = False
        self.aspect = 0
        self.vert_vel = 0
        self.emotion = ""
        self.emotion_cn = ""
        self.emotion_conf = 0.0
        print(f"[Web] → http://{self.local_ip}:{self.port}")
        return self

    def push(self, img_bgr, fps=0, status='NORMAL', people=0, alarm=False,
             aspect=0, vert_vel=0,
             emotion="", emotion_cn="", emotion_conf=0.0):
        h, w = img_bgr.shape[:2]
        if w > self.disp_w:
            scale = self.disp_w / w
            img = cv2.resize(img_bgr, (self.disp_w, int(h * scale)),
                             interpolation=cv2.INTER_LINEAR)
        else:
            img = img_bgr
        ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ret:
            with self._lock:
                self._frame = jpeg.tobytes()
        self.fps = fps
        self.status = status
        self.people = people
        self.alarm = alarm
        self.aspect = aspect
        self.vert_vel = vert_vel
        self.emotion = emotion
        self.emotion_cn = emotion_cn
        self.emotion_conf = emotion_conf

    def stop(self):
        if self._server:
            self._server.shutdown()


# ============================================================
# 主应用
# ============================================================

class FallDetectionApp:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.cam_w = args.camera_width
        self.cam_h = args.camera_height
        self.fps = args.fps
        self.cam_port = args.camera_port

        # 姿态检测器（取代纯检测版）
        self.detector = PoseDetector(
            args.model,
            score_thres=args.score,
            kpt_conf_thres=args.kpt_conf)
        self.detector.set_scheduling(
            priority=args.priority, bpu_cores=args.bpu_cores)

        self.fall_detector = FallDetector(
            fps=args.fps, confirm_frames=args.confirm,
            cooldown=args.cooldown)

        self.frame_count = 0
        self.prev_area = 0.0
        self.fps_cnt = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0

        self.web_port = args.web_port
        self.debug_scores = args.debug_scores
        self.show_labels = args.show_keypoint_labels
        self.streamer = None
        if self.web_port > 0:
            self.streamer = WebStreamer(
                port=self.web_port, quality=args.web_quality,
                disp_w=args.web_disp_w)

        # 表情识别
        self.emotion_recognizer = EmotionRecognizer()
        self.mood_logger = MoodLogger()
        self.current_emotion = ""
        self.current_emotion_cn = ""
        self.current_emotion_conf = 0.0

        # 🚦 推送节流：同一摔倒事件只发一次通知
        self._last_push_time = 0
        self._push_throttled = False   # 当前摔倒已推送过
        self._normal_confirm = 0        # 连续正常帧计数

        model_name = os.path.basename(args.model)
        print(f"[Init] Model:     {model_name} (pose: box + 17 keypoints)")
        print(f"[Init] Camera:   {self.cam_w}x{self.cam_h} @ {self.fps}fps port={self.cam_port}")
        print(f"[Init] Detect:   score={args.score} nms={args.score}")
        print(f"[Init] Keypoint: conf={args.kpt_conf} labels={self.show_labels}")
        print(f"[Init] FallDet:  confirm={args.confirm} frames cooldown={args.cooldown}")
        print(f"[Init] BPU:      cores={args.bpu_cores}")
        if self.streamer:
            print(f"[Init] Web:      port={self.web_port} q={args.web_quality} w={args.web_disp_w}")

    def setup_camera(self):
        cam = libsrcampy.Camera()
        ret = cam.open_cam(self.cam_port, -1, self.fps, self.cam_w, self.cam_h)
        if ret != 0:
            print(f"[ERROR] Camera open failed on port {self.cam_port}")
            sys.exit(1)
        print(f"[Camera] OK port={self.cam_port} {self.cam_w}x{self.cam_h}")
        return cam

    # ── 绘制叠加层 ──

    def draw_status_bar(self, img, metrics, alarm, people, fps):
        """顶部状态栏"""
        h, w = img.shape[:2]
        overlay = img.copy()
        bar_h = 28

        # 半透明背景
        bg_color = (0, 0, 40) if alarm else (0, 40, 0)
        cv2.rectangle(overlay, (0, 0), (w, bar_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # 状态标签
        label = "⚠ FALL ALARM  摔倒报警" if alarm else "✓ NORMAL  正常"
        label_color = (0, 0, 255) if alarm else (0, 255, 0)
        cv2.putText(img, label, (8, bar_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1, cv2.LINE_AA)

        # 右侧信息
        right_text = f"人数:{people}  FPS:{fps:.0f}"
        if alarm and metrics:
            right_text += f"  W/H:{metrics.get('aspect', 0):.2f}"
        tw = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(img, right_text, (w - tw - 8, bar_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        return img

    def draw_metric_panel(self, img, metrics, alarm):
        """左下角指标面板"""
        if not metrics:
            return img
        h, w = img.shape[:2]
        body_angle = metrics.get('body_angle', 0)
        pose_valid = metrics.get('pose_valid', False)
        angle_str = f"倾角:{body_angle:.0f}°" if pose_valid and body_angle > 0 else "倾角:N/A"
        filters_passed = metrics.get('filters_passed', -1)
        filter_str = f"投票:{filters_passed}/3" if filters_passed >= 0 else ""
        lines = [
            f"Aspect: {metrics.get('aspect', 0):.2f}",
            f"Vert:   {metrics.get('vert_vel', 0):.0f}px/s",
            f"Horiz:  {metrics.get('horiz_vel', 0):.0f}px/s",
            f"C:{metrics.get('consecutive', 0)}/{self.args.confirm}",
            f"{angle_str}  {filter_str}",
        ]
        panel_x, panel_y = 6, 34
        panel_w = 170
        panel_h = len(lines) * 14 + 8

        # 半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        for i, txt in enumerate(lines):
            color = (0, 0, 255) if (alarm and i == 3) else (200, 200, 200)
            cv2.putText(img, txt,
                        (panel_x + 4, panel_y + 11 + i * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        return img

    def draw_detection_box(self, img, box, kpts_xy, kpts_score, alarm):
        """
        绘制检测框 + 骨架 + 关键点
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        # 裁剪到图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        box_color = (0, 0, 255) if alarm else (0, 255, 0)

        # ── 边框：双层矩形（外框 + 发光效果） ──
        # 外框（粗）
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
        # 内框（细，亮色）
        inner = tuple(min(255, c + 60) for c in box_color)
        cv2.rectangle(img, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), inner, 1, cv2.LINE_AA)

        # ── 角标装饰（四个角的小标记） ──
        corner_len = min(20, (x2 - x1) // 5, (y2 - y1) // 5)
        corner_color = (255, 255, 255)
        corner_thick = 2
        # 左上
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), corner_color, corner_thick)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), corner_color, corner_thick)
        # 右上
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), corner_color, corner_thick)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), corner_color, corner_thick)
        # 左下
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), corner_color, corner_thick)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), corner_color, corner_thick)
        # 右下
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), corner_color, corner_thick)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), corner_color, corner_thick)

        # ── 骨架绘制 ──
        if kpts_xy is not None and len(kpts_xy) > 0:
            draw_skeleton(
                img, kpts_xy, kpts_score,
                conf_thresh=self.args.kpt_conf,
                draw_labels=self.show_labels)

        return img

    def draw_on_frame(self, img_bgr, best_box, kpts_xy, kpts_score,
                      status, metrics, alarm, people,
                      emotion_cn="", emotion_conf=0.0):
        """完整的画面叠加绘制"""
        # 1) 检测框 + 骨架
        if best_box is not None and kpts_xy is not None:
            img_bgr = self.draw_detection_box(
                img_bgr, best_box, kpts_xy, kpts_score, alarm)

            # 在框上方显示表情
            if emotion_cn:
                x1, y1, x2, y2 = map(int, best_box)
                emo_label = f"{emotion_cn}"
                if emotion_conf > 0:
                    emo_label += f" {emotion_conf:.0%}"
                tw = cv2.getTextSize(emo_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
                lx = max(0, x1)
                ly = max(28, y1 - 12)
                # 背景
                cv2.rectangle(img_bgr, (lx - 2, ly - 24),
                              (lx + tw + 4, ly + 4),
                              (40, 40, 40), -1)
                cv2.putText(img_bgr, emo_label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img_bgr, "No person detected",
                        (8, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 2) 顶部状态栏
        img_bgr = self.draw_status_bar(img_bgr, metrics, alarm, people,
                                       self.current_fps)

        # 3) 指标面板
        if metrics:
            img_bgr = self.draw_metric_panel(img_bgr, metrics, alarm)

        # 底部水印
        cv2.putText(img_bgr, "RDK X5 Pose Fall Detection",
                    (8, img_bgr.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100),
                    1, cv2.LINE_AA)

        return img_bgr

    def run(self):
        cam = self.setup_camera()
        signal.signal(signal.SIGINT, self._signal_handler)
        print("[App] Running... Ctrl+C to stop.")
        if self.streamer:
            self.streamer.start()

        while self.running:
            try:
                t0 = time.time()

                # 采集 NV12
                nv12 = cam.get_img(2, self.cam_w, self.cam_h)
                if nv12 is None or len(nv12) == 0:
                    continue

                # NV12 → BGR（用于显示 + 姿态模型输入）
                y = np.frombuffer(nv12[:self.cam_w * self.cam_h],
                                  dtype=np.uint8).reshape(self.cam_h, self.cam_w)
                uv = np.frombuffer(nv12[self.cam_w * self.cam_h:],
                                   dtype=np.uint8).reshape(self.cam_h // 2, self.cam_w)
                yuv = np.zeros((self.cam_h * 3 // 2, self.cam_w), dtype=np.uint8)
                yuv[:self.cam_h] = y
                yuv[self.cam_h:] = uv
                img_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

                # 姿态推理（人体框 + 关键点）
                person_boxes, person_kpts, person_kpts_scores, max_score = \
                    self.detector.infer(img_bgr)

                if self.debug_scores and len(person_boxes) == 0 and max_score > 0:
                    print(f"[Debug] 最高置信度={max_score:.3f} (阈值={self.args.score})")

                # 选择最大的人体框（最近的人）
                best_box = None
                best_kpts = None
                best_kpts_score = None
                curr_area = 0.0

                if len(person_boxes) > 0:
                    areas = ((person_boxes[:, 2] - person_boxes[:, 0]) *
                             (person_boxes[:, 3] - person_boxes[:, 1]))
                    best_idx = int(np.argmax(areas))
                    best_box = person_boxes[best_idx]
                    curr_area = areas[best_idx]

                    # 关联关键点
                    if len(person_kpts) > best_idx:
                        best_kpts = person_kpts[best_idx]
                        best_kpts_score = person_kpts_scores[best_idx]

                # 摔倒检测（带姿态关键点仲裁）
                status = "NORMAL"
                metrics = {}
                if best_box is not None:
                    status, metrics = self.fall_detector.detect(
                        self.frame_count, best_box, self.prev_area,
                        kpts_xy=best_kpts, kpts_score=best_kpts_score,
                        frame_h=self.cam_h, frame_w=self.cam_w)
                    self.prev_area = curr_area

                    # ── 推送通知（同原版） ──
                    now = time.time()

                    # 复位机制：连续 30 帧 NORMAL 则清除《已推送》标记
                    if status != "ALARM":
                        self._normal_confirm += 1
                        if self._normal_confirm >= 30:
                            self._push_throttled = False
                            self._normal_confirm = 0
                    else:
                        self._normal_confirm = 0

                    push_allowed = (now - self._last_push_time) >= 10
                    new_incident = not self._push_throttled

                    if status == "ALARM" and push_allowed and new_incident:
                        self._last_push_time = now
                        self._push_throttled = True

                        sendkey = os.environ.get("SC_KEY", "")
                        if sendkey:
                            try:
                                import subprocess, urllib.parse
                                title = urllib.parse.quote("🚨 摔倒警告（带姿态）")
                                desc = urllib.parse.quote(
                                    f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"帧号：{self.frame_count}\n"
                                    f"宽高比：{metrics.get('aspect', 0):.2f}\n"
                                    f"速度：{metrics.get('vert_vel', 0):.0f}px/s\n"
                                    f"请及时查看摄像头画面！"
                                )
                                subprocess.Popen([
                                    "curl", "-s",
                                    f"https://sctapi.ftqq.com/{sendkey}.send",
                                    "-d", f"title={title}&desp={desc}"
                                ])
                                print(f"[Server酱] ✅ 微信推送已发送")
                            except Exception as e:
                                print(f"[Server酱] ❌ {e}")

                        if os.environ.get("TX_SECRET_ID", ""):
                            try:
                                import subprocess
                                script_path = os.path.join(
                                    os.path.dirname(os.path.abspath(__file__)),
                                    "tx_voice_call.py"
                                )
                                subprocess.Popen(
                                    [sys.executable, script_path],
                                    env={**os.environ}
                                )
                                print(f"[腾讯云语音] ✅ 语音呼叫已提交")
                            except Exception as e:
                                print(f"[腾讯云语音] ❌ {e}")

                # 统计
                self.frame_count += 1
                self.fps_cnt += 1
                if time.time() - self.fps_timer >= 1.0:
                    self.current_fps = self.fps_cnt
                    self.fps_cnt = 0
                    self.fps_timer = time.time()

                infer_ms = (time.time() - t0) * 1000

                # 终端输出
                if self.frame_count % 30 == 0:
                    people = len(person_boxes)
                    if status == "ALARM":
                        print(f"\033[31m[摔倒警告] 帧{self.frame_count} "
                              f"宽高比={metrics.get('aspect',0):.2f} "
                              f"速度={metrics.get('vert_vel',0):.0f}px/s "
                              f"FPS={self.current_fps:.0f} "
                              f"推理={infer_ms:.0f}ms "
                              f"人={people} 关键点可见\033[0m")
                    else:
                        print(f"[OK] 帧{self.frame_count} "
                              f"FPS={self.current_fps:.0f} "
                              f"推理={infer_ms:.0f}ms "
                              f"人={people}")

                # Web 推流
                if self.streamer:
                    # ── 表情识别 ──
                    if best_box is not None and best_kpts is not None and best_kpts_score is not None:
                        try:
                            em, em_conf, face_roi, em_cn = self.emotion_recognizer.detect(
                                img_bgr, best_box,
                                kpts_xy=best_kpts, kpts_score=best_kpts_score)
                            self.current_emotion = em
                            self.current_emotion_cn = em_cn
                            self.current_emotion_conf = em_conf
                            # 记录日志
                            if em != "neutral" or (face_roi is not None and em_conf > 0.4):
                                self.mood_logger.log(em, em_conf, em_cn)
                        except Exception as e:
                            pass
                    else:
                        self.current_emotion = ""
                        self.current_emotion_cn = ""
                        self.current_emotion_conf = 0.0

                    # 在完整分辨率上绘制
                    display_img = self.draw_on_frame(
                        img_bgr, best_box, best_kpts, best_kpts_score,
                        status, metrics,
                        alarm=(status == "ALARM"),
                        people=len(person_boxes),
                        emotion_cn=self.current_emotion_cn,
                        emotion_conf=self.current_emotion_conf)
                    self.streamer.push(
                        display_img, fps=self.current_fps,
                        status=status, people=len(person_boxes),
                        alarm=(status == "ALARM"),
                        aspect=metrics.get('aspect', 0),
                        vert_vel=metrics.get('vert_vel', 0),
                        emotion=self.current_emotion,
                        emotion_cn=self.current_emotion_cn,
                        emotion_conf=self.current_emotion_conf)

            except Exception as e:
                print(f"[Error] 帧{self.frame_count}: {e}")
                import traceback
                traceback.print_exc()
                continue

        cam.close_cam()
        if self.streamer:
            self.streamer.stop()
        print("[App] Stopped.")

    def _signal_handler(self, sig, frame):
        print("\n[App] Stopping...")
        self.running = False


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RDK X5 摔倒检测 — 带姿态骨骼点版")

    parser.add_argument('--model', type=str,
                        default='/app/pydev_demo/04_pose_sample/01_ultralytics_yolo11_pose/yolo11n_pose_bayese_640x640_nv12.bin',
                        help='YOLO11n-pose 姿态模型路径')
    parser.add_argument('--camera-width', type=int, default=1088)
    parser.add_argument('--camera-height', type=int, default=1280)
    parser.add_argument('--camera-port', type=int, default=0)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--score', type=float, default=0.25,
                        help='检测置信度')
    parser.add_argument('--kpt-conf', type=float, default=0.3,
                        help='关键点置信度（调低显示更多点）')
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--confirm', type=int, default=10)
    parser.add_argument('--cooldown', type=int, default=5)
    parser.add_argument('--web-port', type=int, default=8080)
    parser.add_argument('--web-quality', type=int, default=50)
    parser.add_argument('--web-disp-w', type=int, default=640,
                        help='Web推流宽度（更高清查看骨骼）')
    parser.add_argument('--debug-scores', action='store_true')
    parser.add_argument('--show-keypoint-labels', action='store_true',
                        help='显示关键点编号（调试用）')

    args = parser.parse_args()
    FallDetectionApp(args).run()


if __name__ == '__main__':
    main()
