#!/usr/bin/env python3
"""
RDK X5 表情识别模块
====================
- 基于面部几何姿态 + 图像分析的表情识别
- 使用 COCO 关键点（鼻、眼、耳）定位面部
- 识别 7 类表情：生气/厌恶/恐惧/开心/悲伤/惊讶/中性
- 支持 MODE A (ONNX 模型, 如可用) 或 MODE B (几何+图像规则)
"""

import os, cv2, json, time, math
import numpy as np
from pathlib import Path

# ── 配置 ──────────────────────────────────────────
EMOTION_DIR = Path(__file__).parent
MODEL_PATH = EMOTION_DIR / "emotion_model.onnx"
LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABELS_CN = {
    "angry": "😠 生气", "disgust": "🤢 厌恶", "fear": "😨 害怕",
    "happy": "😊 开心", "sad": "😢 悲伤", "surprise": "😲 惊讶",
    "neutral": "😐 中性",
}
EMOJI_ONLY = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "sad": "😢", "surprise": "😲", "neutral": "😐",
}


class EmotionRecognizer:
    """
    表情识别器
    MODE A: ONNX 深度学习模型（若模型文件存在）
    MODE B: 基于面部几何 + 图像规则（默认）
    """

    def __init__(self):
        self.session = None
        self.use_onnx = MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100000

        if self.use_onnx:
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    str(MODEL_PATH), providers=['CPUExecutionProvider'])
                print(f"[Emotion] MODE A: ONNX 模型已加载: {MODEL_PATH.name}")
            except Exception as e:
                print(f"[Emotion] ONNX 加载失败, 回退 MODE B: {e}")
                self.use_onnx = False

        if not self.use_onnx:
            print(f"[Emotion] MODE B: 基于姿态 + 图像规则的表情识别")

        # Haar cascade 作为面部检测备用
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # 缓存
        self.last_emotion = "neutral"
        self.last_confidence = 0.0
        self.last_cn = "😐 中性"
        self.last_check = 0
        self.check_interval = 1.0  # 1 秒检测一次

    # ── 从关键点提取面部 ROI ──
    def get_face_roi(self, img, person_box, kpts_xy, kpts_score, conf=0.2):
        """从 pose 关键点提取面部区域，Haar 级联作为备用"""
        h, w = img.shape[:2]

        # 方法1: 从 pose 关键点
        face_roi = self._from_pose_kpts(img, kpts_xy, kpts_score, conf)

        # 方法2: 备用 — Haar cascade
        if face_roi is None or face_roi.size < 200:
            face_roi = self._from_haar(img, person_box)

        return face_roi

    def _from_pose_kpts(self, img, kpts_xy, kpts_score, conf=0.3):
        """从 COCO 面部关键点提取 face ROI"""
        if kpts_xy is None or len(kpts_xy) < 5:
            return None
        h, w = img.shape[:2]

        face_idx = [0, 1, 2, 3, 4]  # 鼻, 左眼, 右眼, 左耳, 右耳
        scores = [kpts_score[i, 0] if i < len(kpts_score) else 0
                  for i in face_idx]

        if not (scores[0] >= conf and (scores[1] >= conf or scores[2] >= conf)):
            return None

        pts = []
        for i in face_idx:
            if scores[face_idx.index(i)] >= conf and i < len(kpts_xy):
                pts.append((int(kpts_xy[i, 0]), int(kpts_xy[i, 1])))

        if len(pts) < 3:
            return None

        pts_arr = np.array(pts)
        cx, cy = int(np.mean(pts_arr[:, 0])), int(np.mean(pts_arr[:, 1]))

        # 根据人脸关键点扩散范围估算面部大小
        spread = max(np.max(pts_arr[:, 0]) - np.min(pts_arr[:, 0]),
                     np.max(pts_arr[:, 1]) - np.min(pts_arr[:, 1]))
        face_size = int(spread * 2.2)

        if face_size < 40:
            return None

        x1 = max(0, cx - face_size // 2)
        y1 = max(0, cy - face_size // 2)
        x2 = min(w, x1 + face_size)
        y2 = min(h, y1 + face_size)

        return img[y1:y2, x1:x2]

    def _from_haar(self, img, person_box):
        """Haar 级联备用"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        px1, py1, px2, py2 = [int(v) for v in person_box]
        best = max(faces, key=lambda r:
            max(0, min(px2, r[0]+r[2]) - max(px1, r[0])) *
            max(0, min(py2, r[1]+r[3]) - max(py1, r[1])))
        fx, fy, fw, fh = best
        return img[fy:fy+fh, fx:fx+fw]

    # ── MODE B: 基于规则的表情识别 ──
    def _predict_rule(self, face_roi):
        """
        保守版表情识别 — 高阈值 + 反特征过滤
        宁"中性"勿误判
        """
        h, w = face_roi.shape[:2]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        overall = np.mean(gray_eq)

        # ── 1. 嘴部特征 ──
        mouth = gray_eq[2*h//3:, w//4:3*w//4]
        if mouth.size < 10:
            mouth = gray_eq[2*h//3:, :]
        mouth_m = np.mean(mouth)
        mouth_std = np.std(mouth)

        # Sobel 梯度分析嘴型
        mouth_dx = cv2.Sobel(mouth, cv2.CV_32F, 1, 0, ksize=3)
        mouth_dy = cv2.Sobel(mouth, cv2.CV_32F, 0, 1, ksize=3)
        mg_h = float(np.mean(np.abs(mouth_dx)))
        mg_v = float(np.mean(np.abs(mouth_dy)))

        # ── 2. 眼部特征 ──
        eye = gray_eq[h//5:2*h//5, w//5:4*w//5]
        if eye.size < 10:
            eye = gray_eq[h//5:2*h//5, :]
        eye_std = float(np.std(eye))
        eye_dx = cv2.Sobel(eye, cv2.CV_32F, 1, 0, ksize=3)
        eye_grad = float(np.mean(np.abs(eye_dx)))

        # ── 3. 鼻部纹理（厌恶/用力表情） ──
        nose = gray_eq[h//3:2*h//3, w//4:3*w//4]
        nose_std = float(np.std(nose)) if nose.size > 5 else 0

        # ── 4. 色彩（LAB/HSV） ──
        lip_a = 128
        hsv_sat, hsv_val = 50, 128
        try:
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            lip_block = lab[2*h//3:, w//4:3*w//4]
            lip_a = float(np.mean(lip_block[:, :, 1])) if lip_block.size > 30 else 128
        except:
            pass
        try:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            hsv_sat = float(np.mean(hsv[:, :, 1]))
            hsv_val = float(np.mean(hsv[:, :, 2]))
        except:
            pass

        # ── 5. 归一化特征值（标准化到面部分区相对值） ──
        top_m = float(np.mean(gray_eq[:h//3, :]))
        bot_m = float(np.mean(gray_eq[2*h//3:, :]))
        mouth_vs_face = mouth_m / max(overall, 1)

        # ============================================
        # 各表情判断 — 严格阈值
        # ============================================
        # 先给 neutral 较高的基础分
        results = {"neutral": 0}
        neutral_score = 30

        # 😊 开心：必须是露齿笑（嘴亮+纹理丰富+水平梯度主导）
        happy_score = 0
        if mouth_vs_face > 1.12 and mouth_std > 40:
            happy_score += 30  # 嘴明显亮且有纹理
        if mg_h > mg_v * 3 and mg_h > 12:
            happy_score += 25  # 水平梯度远大于垂直=微笑咧嘴
        if eye_std > 55:
            happy_score += 15  # 笑纹明显
        if lip_a > 140:
            happy_score += 10  # 唇色红润
        # 反特征：如果嘴部很暗，不可能开心
        if mouth_vs_face < 0.95:
            happy_score = 0
        if happy_score >= 40:
            results["happy"] = happy_score

        # 😢 悲伤：嘴暗、眉眼无神、整体暗
        sad_score = 0
        if mouth_vs_face < 0.88 and mouth_std < 20:
            sad_score += 25
        if eye_std < 22:
            sad_score += 20
        if overall < 90 and mouth_vs_face < 0.92:
            sad_score += 20
        if hsv_val < 90:
            sad_score += 15
        # 反特征：嘴部亮则不可能悲伤
        if mouth_vs_face > 1.05:
            sad_score = 0
        if sad_score >= 35:
            results["sad"] = sad_score

        # 😲 惊讶：嘴张开（垂直梯度大），整体亮，眉眼
        surprise_score = 0
        if mg_v > 12 and mg_h < mg_v * 1.5:
            surprise_score += 30  # 垂直主导=张嘴
        if mouth_vs_face > 1.15 and mg_v > 8:
            surprise_score += 20  # 嘴亮且张开
        if eye_std > 50 and overall > 140:
            surprise_score += 15  # 眼睛睁大+亮
        if eye_grad > 30:
            surprise_score += 10  # 眉眼纹理多=挑眉
        if surprise_score >= 35:
            results["surprise"] = surprise_score

        # 😠 生气：皱眉(眉眼梯度高)、嘴紧(低std)、暗
        angry_score = 0
        if eye_grad > 22 and mouth_std < 18:
            angry_score += 25  # 眉眼纹理多+嘴紧
        if nose_std > 35 and mouth_std < 15:
            angry_score += 20  # 鼻周皱+嘴紧闭
        if overall < 90 and eye_grad > 18:
            angry_score += 15
        if lip_a < 120:
            angry_score += 10
        if angry_score >= 35:
            results["angry"] = angry_score

        # 😨 害怕：眼睁大(高std)+嘴微张、整体偏暗
        fear_score = 0
        if eye_std > 45 and mg_v > 7 and overall < 115:
            fear_score += 25
        if eye_std > 50 and overall < 100:
            fear_score += 20
        if mg_v > 8 and mg_v < 15 and overall < 110:
            fear_score += 15  # 嘴微微张开但不太大
        if fear_score >= 30:
            results["fear"] = fear_score

        # 🤢 厌恶：鼻周+眉间纹理多，嘴紧
        disgust_score = 0
        if nose_std > 40 and mouth_std < 16:
            disgust_score += 25
        if eye_grad > 20 and mouth_std < 14:
            disgust_score += 20
        if lip_a < 115 and mouth_std < 15:
            disgust_score += 15
        if disgust_score >= 30:
            results["disgust"] = disgust_score

        # ── 最终决策 ──
        if not results or (len(results) == 1 and "neutral" in results):
            return "neutral", 0.3

        best = max(results, key=results.get)
        best_score = results[best]
        # 得分太低 → neutral
        if best_score < 40:
            return "neutral", 0.3

        confidence = min(best_score / 90.0, 0.85)
        return best, confidence

    def detect(self, img, person_box, kpts_xy=None, kpts_score=None):
        """
        检测面部表情
        返回: (emotion_str, confidence_float, face_roi_or_None, cn_label_str)
        """
        now = time.time()
        if now - self.last_check < self.check_interval:
            return self.last_emotion, self.last_confidence, None, self.last_cn

        self.last_check = now

        # 提取面部
        face_roi = None
        try:
            face_roi = self.get_face_roi(img, person_box, kpts_xy, kpts_score)
        except Exception:
            pass

        if face_roi is None or face_roi.size < 400:
            self.last_emotion = "neutral"
            self.last_confidence = 0.0
            self.last_cn = "😐 中性"
            return self.last_emotion, self.last_confidence, None, self.last_cn

        # 分类
        if self.use_onnx and self.session:
            try:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.resize(gray, (48, 48))
                # 模型使用 [0,1] 归一化
                inp = gray_r.astype(np.float32) / 255.0
                inp = inp[np.newaxis, np.newaxis, :, :]
                outs = self.session.run(None, {self.input_name: inp})
                probs = np.array(outs[0][0]).flatten()
                idx = int(np.argmax(probs))
                self.last_emotion = LABELS[idx]
                self.last_confidence = float(probs[idx])
            except Exception as e:
                # ONNX 推理失败时回退规则模式
                self.last_emotion, self.last_confidence = self._predict_rule(face_roi)
        else:
            self.last_emotion, self.last_confidence = self._predict_rule(face_roi)

        self.last_cn = LABELS_CN.get(self.last_emotion, f"❓ {self.last_emotion}")
        return self.last_emotion, self.last_confidence, face_roi, self.last_cn


# ── 心情日志 ──────────────────────────────────────

MOOD_LOG_PATH = EMOTION_DIR / "mood_log.json"


class MoodLogger:
    def __init__(self, log_path=None):
        self.log_path = Path(log_path or MOOD_LOG_PATH)
        self.data = self._load()
        self.last_log = 0
        self.interval = 5  # 5 秒记一次（测试友好）
        self.session = []
        self.last_emotion_log_time = {}  # 每种情绪的最后记录时间
        self.emotion_min_interval = 30   # 同一种情绪至少隔 30 秒才记录

    def _load(self):
        if self.log_path.exists():
            try:
                return json.loads(self.log_path.read_text(encoding="utf-8"))
            except:
                pass
        return {"entries": [], "summaries": {}}

    def _save(self):
        # 保存前重新加载文件，避免覆盖外部清空操作
        file_data = self._load()
        # 仅用文件中最新的 entry 来补充 self.data（应对外部清空）
        if len(file_data.get("entries", [])) < len(self.data.get("entries", [])):
            # 外部可能已清空，以文件为准 + 追加本进程的 session 记录
            self.data = file_data
            self.data["entries"].extend(
                e for e in self.session if e not in self.data["entries"]
            )
        if len(self.data["entries"]) > 10000:
            self.data["entries"] = self.data["entries"][-5000:]
        self.log_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def log(self, emotion, confidence, cn_label):
        now = time.time()
        if now - self.last_log < self.interval:
            return None
        self.last_log = now

        # 同一种情绪不能记录太频繁（避免某情绪刷屏）
        last_emo_time = self.last_emotion_log_time.get(emotion, 0)
        if now - last_emo_time < self.emotion_min_interval:
            return None
        self.last_emotion_log_time[emotion] = now

        entry = {
            "time": now,
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "date": time.strftime("%Y-%m-%d"),
            "emotion": emotion,
            "label_cn": cn_label,
            "confidence": round(confidence, 3),
        }
        self.data["entries"].append(entry)
        self.session.append(entry)
        if len(self.data["entries"]) % 5 == 0:
            self._save()
        return entry

    def get_today(self):
        today = time.strftime("%Y-%m-%d")
        return [e for e in self.data["entries"] if e.get("date") == today]

    def get_by_date(self, date_str):
        return [e for e in self.data["entries"] if e.get("date") == date_str]

    def stats(self, entries=None):
        if entries is None:
            entries = self.data["entries"]
        if not entries:
            return {"total": 0, "distribution": {e: 0 for e in LABELS},
                    "dominant": "neutral", "dominant_cn": "😐 中性"}
        dist = {e: 0 for e in LABELS}
        for e in entries:
            em = e.get("emotion", "neutral")
            if em in dist:
                dist[em] += 1
        dom = max(dist, key=dist.get)
        return {"total": len(entries), "distribution": dist,
                "dominant": dom, "dominant_cn": LABELS_CN.get(dom, dom)}

    def summary_prompt(self, entries=None):
        if entries is None:
            entries = self.get_today()
        if not entries:
            return "今天还没有记录到表情数据。"
        s = self.stats(entries)
        timeline = "\n".join(
            f"  {e['time_str']}  {e['label_cn']} (置信:{e.get('confidence',0):.0%})"
            for e in entries)
        return (
            f"以下是一个人今天的心情记录数据，请生成简洁的心情日志总结：\n\n"
            f"## 数据\n- 记录: {s['total']} 条\n- 主要情绪: {s['dominant_cn']}\n\n"
            f"## 时间线\n{timeline}\n\n"
            f"## 要求\n用中文总结情绪变化趋势，给出心情评分(满分10)，"
            f"控制在150字以内，语气温暖，最后一句鼓励的话结尾。"
        )

    def save_summary(self, text, date_str=None):
        if date_str is None:
            date_str = time.strftime("%Y-%m-%d")
        if "summaries" not in self.data:
            self.data["summaries"] = {}
        self.data["summaries"][date_str] = {
            "summary": text,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save()


if __name__ == "__main__":
    r = EmotionRecognizer()
    print(f"模式: {'MODE A (ONNX)' if r.use_onnx else 'MODE B (Rules)'}")
    l = MoodLogger()
    print(f"日志: {l.stats()}")
