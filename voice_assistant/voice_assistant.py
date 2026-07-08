#!/usr/bin/env python3
"""
RDK X5 智能语音助手
=====================
功能: 离线唤醒词 + 语音识别 + DeepSeek对话 + 语音合成

架构:
  VOSK (唤醒词+STT) → DeepSeek API → Edge-TTS → aplay

使用:
  export DEEPSEEK_API_KEY="sk-xxx"
  python3 voice_assistant.py

唤醒词: "你好 小智" (可在 WAKEWORD 变量修改)
"""

import os
import sys
import json
import time
import queue
import struct
import threading
import subprocess as sp
import urllib.request
import urllib.error
from pathlib import Path

# ── 配置 ─────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# 唤醒词（VOSK 关键词列表，可以是单个词或短语）
WAKEWORDS = ["你好 小智", "你好小智", "小智小智"]
# 系统提示词
SYSTEM_PROMPT = "你是一个友好的智能语音助手，请用简洁自然的口语回答，每次回答控制在50字以内。"

# 音频参数
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = "S16_LE"
DEVICE = "plughw:0,0"

# VOSK 模型路径
MODEL_PATH = os.path.join(os.path.expanduser("~"), ".vosk/models/vosk-model-small-cn-0.22")

# TTS 语音
TTS_VOICE = "zh-CN-XiaoxiaoNeural"

# ── 全局状态 ────────────────────────────────────────
is_listening = False       # 用户正在说话
is_speaking = False        # 助手正在播放语音
waiting_for_response = False
audio_queue = queue.Queue()
recording_thread = None

# 会话历史
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


# ── 工具函数 ─────────────────────────────────────────

def play_audio(filepath):
    """播放音频文件"""
    global is_speaking
    is_speaking = True
    try:
        sp.run(["aplay", "-D", DEVICE, filepath],
               stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=30)
    except:
        pass
    is_speaking = False


def play_beep(freq=800, duration=0.15):
    """生成并播放提示音（用ffmpeg生成正弦波）"""
    try:
        sp.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"sine=frequency={freq}:duration={duration}",
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav",
            "/tmp/beep.wav"
        ], stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=5)
        sp.run(["aplay", "-D", DEVICE, "/tmp/beep.wav"],
               stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=5)
    except:
        pass


def record_audio(duration, filename="/tmp/user_speech.wav"):
    """录制音频"""
    try:
        sp.run([
            "arecord", "-D", DEVICE, "-f", FORMAT,
            "-r", str(SAMPLE_RATE), "-c", str(CHANNELS),
            "-d", str(duration), filename
        ], stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=duration + 5)
        return filename
    except:
        return None


def text_to_speech(text, filename="/tmp/tts_output.mp3"):
    """Edge-TTS 文字转语音"""
    try:
        sp.run([
            "edge-tts", "--voice", TTS_VOICE,
            "--text", text,
            "--write-media", filename
        ], stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=30)
        # 转成 WAV 格式
        wav_file = filename.replace(".mp3", ".wav")
        sp.run([
            "ffmpeg", "-y", "-i", filename,
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            wav_file
        ], stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=10)
        return wav_file
    except Exception as e:
        print(f"[TTS Error] {e}")
        return None


def call_deepseek(user_text):
    """调用 DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return "抱歉，我没有配置 API 密钥，无法回答。请设置 DEEPSEEK_API_KEY 环境变量。"

    messages.append({"role": "user", "content": user_text})

    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": 256,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            reply = result["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": reply})
            return reply
    except urllib.error.HTTPError as e:
        return f"网络错误: {e.code}"
    except Exception as e:
        return f"请求失败: {str(e)}"


# ── VOSK 唤醒词 + 语音识别 ──────────────────────────

def load_vosk():
    """加载 VOSK 模型"""
    from vosk import Model, KaldiRecognizer

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] VOSK 模型未找到: {MODEL_PATH}")
        print("  请先下载: https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip")
        sys.exit(1)

    model = Model(MODEL_PATH)
    # 标准识别器（16kHz 单声道）
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    # 设置关键词列表（唤醒词）
    recognizer.SetWords(True)
    # 设置关键词列表来提升唤醒词准确率
    recognizer.SetPartialWords(True)
    print(f"[VOSK] 模型已加载: {os.path.basename(MODEL_PATH)}")
    return model, recognizer


def wakeword_listen():
    """
    使用 VOSK 持续监听唤醒词
    返回: (text, audio_data) 唤醒后的语音文本和音频
    """
    import pyaudio

    model, recognizer = load_vosk()
    print(f"[WAKE] 唤醒词: {WAKEWORDS[0]}")
    print("[WAKE] 等待唤醒... (Ctrl+C 退出)")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=8000,
        input_device_index=None,
    )

    wake_detected = False
    wake_phrase = ""
    speech_frames = []

    try:
        while not wake_detected:
            data = stream.read(4000, exception_on_overflow=False)
            speech_frames.append(data)

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    print(f"  [识别] {text}")
                    # 检查是否包含唤醒词
                    for ww in WAKEWORDS:
                        if ww in text:
                            wake_detected = True
                            wake_phrase = text.replace(ww, "").strip()
                            print(f"\n  ✅ 唤醒词 '{ww}' 已触发!")
                            break
            else:
                # 实时显示
                partial = json.loads(recognizer.PartialResult())
                ptext = partial.get("partial", "").strip()
                if ptext and len(ptext) > 2:
                    # 检查唤醒词
                    for ww in WAKEWORDS:
                        if ww in ptext:
                            wake_detected = True
                            wake_phrase = ptext.replace(ww, "").strip()
                            print(f"\n  ✅ 唤醒词 '{ww}' 已触发!")
                            break

            # 如果检测到唤醒词，继续录一会儿收尾
            if wake_detected:
                # 再录制 3 秒收尾（如果用户继续说话）
                for _ in range(6):  # 6 x 0.5s = 3s
                    try:
                        data = stream.read(4000, exception_on_overflow=False)
                        speech_frames.append(data)
                        recognizer.AcceptWaveform(data)
                    except:
                        break
                break

    except KeyboardInterrupt:
        print("\n[EXIT] 用户退出")
        stream.close()
        p.terminate()
        return None, None

    stream.close()
    p.terminate()

    # 获取完整识别结果
    final = json.loads(recognizer.FinalResult())
    full_text = final.get("text", "").strip()
    if not full_text and wake_phrase:
        full_text = wake_phrase

    # 保存音频
    audio_file = "/tmp/wake_audio.wav"
    import wave
    with wave.open(audio_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(speech_frames))

    return full_text, audio_file


def continuous_listen():
    """
    只使用 VAD+arecord 进行录音（不需要 VOSK 持续监听）：
    检测到声音开始 → 录音 → 检测到沉默 → 停止 → 发送到 VOSK 识别
    适合作为备用方案
    """
    pass


# ── 主循环 ───────────────────────────────────────────

def main_loop():
    """主循环：唤醒 → 录音 → STT → LLM → TTS → 播放"""
    print("=" * 50)
    print("  RDK X5 智能语音助手")
    print("=" * 50)
    print(f"  唤醒词: {WAKEWORDS[0]}")
    print(f"  LLM: DeepSeek ({DEEPSEEK_MODEL})")
    print(f"  TTS: {TTS_VOICE}")
    print(f"  VOSK: {os.path.basename(MODEL_PATH)}")
    if not DEEPSEEK_API_KEY:
        print("  ⚠️  未设置 DEEPSEEK_API_KEY，对话功能不可用")
    print("=" * 50)
    print()

    # 预下载 VOSK 模型
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 请先下载 VOSK 模型到 {MODEL_PATH}")
        print("  wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip")
        sys.exit(1)

    # 先播放就绪提示音
    play_beep(600, 0.2)
    time.sleep(0.5)

    conversation_count = 0

    try:
        while True:
            print(f"\n{'─'*40}")
            print(f"🎤 正在监听唤醒词... (第{conversation_count+1}轮)")

            # 1) 等待唤醒词
            text, audio_file = wakeword_listen()
            if text is None:
                break

            # 2) 播放提示音（唤醒成功）
            play_beep(1000, 0.1)
            time.sleep(0.3)

            # 3) 提取用户实际语音（去掉唤醒词部分）
            user_text = text.strip()
            for ww in WAKEWORDS:
                user_text = user_text.replace(ww, "").strip()
            if not user_text:
                user_text = ""
                # 如果没有额外语音，再录 5 秒
                print("[WAIT] 请说出您的问题...")
                record_audio(5, "/tmp/question.wav")
                # 用 VOSK 转文字
                from vosk import Model, KaldiRecognizer
                model = Model(MODEL_PATH)
                rec = KaldiRecognizer(model, SAMPLE_RATE)
                with wave.open("/tmp/question.wav", "rb") as wf:
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        rec.AcceptWaveform(data)
                final = json.loads(rec.FinalResult())
                user_text = final.get("text", "").strip()

            print(f"🗣️ 你说: {user_text}")
            if not user_text:
                print("[SKIP] 未检测到语音")
                continue

            # 4) DeepSeek 回答
            print("🤖 思考中...")
            reply = call_deepseek(user_text)
            print(f"🤖 回答: {reply}")

            # 5) TTS 合成
            print("🔊 语音合成...")
            wav_file = text_to_speech(reply)
            if wav_file and os.path.exists(wav_file):
                print(f"🔊 播放中...")
                play_audio(wav_file)
            else:
                print(f"[WARN] TTS 播放失败")

            conversation_count += 1

    except KeyboardInterrupt:
        print("\n[EXIT] 语音助手已停止")

    play_beep(400, 0.3)


if __name__ == "__main__":
    # 检查依赖
    missing = []
    try:
        import wave
    except:
        missing.append("wave")
    try:
        import pyaudio
    except:
        missing.append("pyaudio")

    try:
        from vosk import Model
    except:
        missing.append("vosk")

    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print("请安装: pip3 install vosk pyaudio")
        sys.exit(1)

    main_loop()
