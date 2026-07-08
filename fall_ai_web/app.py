#!/usr/bin/env python3
"""
RDK X5 摔倒检测综合平台 — Flask 后端（端口 5050）
===================================================
重构版 v2.0:
- 仪表盘: 实时系统性能曲线 + 今日检测统计
- 摄像头: iframe 直接嵌入 + 状态叠加
- AI 对话: DeepSeek API + 语音输入/输出
- 告警记录: 摔倒事件持久化 + 确认标注
- 心情日志: 表情数据可视化 + AI 总结
- 架构说明: 系统设计文档页
"""

import os
import json
import time
import socket
import threading
import urllib.request
import urllib.error
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import requests
from flask import (
    Flask, render_template, jsonify, request,
    Response, stream_with_context
)

logging.basicConfig(level=logging.INFO, format="[Web] %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def read_sysfs(path: str, default=""):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

PORT = 5050
FALL_POLL_INTERVAL = 2
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

CONFIG_FILE = Path(__file__).parent / "config.json"

_default_config = {
    "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
    "CAMERA_URL": "",
    "SERVER_PORT": 5050,
}

config_lock = threading.Lock()
config = dict(_default_config)


def default_camera_url():
    return f"http://{get_local_ip()}:8080"


def load_config():
    global config
    with config_lock:
        config = dict(_default_config)
        if CONFIG_FILE.exists():
            try:
                user_cfg = json.loads(CONFIG_FILE.read_text())
                config.update(user_cfg)
            except Exception as e:
                log.warning(f"配置读取失败: {e}")
        if not config.get("CAMERA_URL"):
            config["CAMERA_URL"] = default_camera_url()


def save_config(updates: dict):
    with config_lock:
        config.update(updates)
        CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2))


def get_camera_url():
    url = config.get("CAMERA_URL", "")
    return url if url else default_camera_url()


def get_deepseek_key():
    return config.get("DEEPSEEK_API_KEY", "")


load_config()

# ═══════════════════════════════════════════════════════════
# 对话历史
# ═══════════════════════════════════════════════════════════

conversation_history = [
    {"role": "system", "content": "你是一个友善的智能助手，用中文回答用户的问题。回答要简洁准确，语气自然。可以回答日常问题，也可以提供帮助和建议。"}
]
history_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════
# 摔倒状态缓存
# ═══════════════════════════════════════════════════════════

fall_status = {"status": "NORMAL", "alarm": False, "people": 0, "fps": 0}
fall_lock = threading.Lock()
last_alert_time = 0.0  # 上次告警时间（防频繁触发）

# ═══════════════════════════════════════════════════════════
# 告警记录持久化
# ═══════════════════════════════════════════════════════════

ALERTS_FILE = Path(__file__).parent / "alerts.json"
alerts_lock = threading.Lock()


def _load_alerts():
    if ALERTS_FILE.exists():
        try:
            return json.loads(ALERTS_FILE.read_text())
        except Exception:
            pass
    return {"alerts": [], "next_id": 1}


def _save_alerts(data):
    ALERTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def add_alert(info: dict):
    """添加一条告警记录"""
    with alerts_lock:
        data = _load_alerts()
        alert = {
            "id": data["next_id"],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": info.get("type", "fall"),
            "detail": info.get("detail", "检测到摔倒事件"),
            "people": info.get("people", 0),
            "fps": info.get("fps", 0),
            "acknowledged": False,
            "ack_time": None,
        }
        data["alerts"].append(alert)
        data["next_id"] += 1
        # 只保留最近 500 条
        if len(data["alerts"]) > 500:
            data["alerts"] = data["alerts"][-500:]
        _save_alerts(data)
    return alert


def get_alerts(limit=100, offset=0, unread_only=False):
    with alerts_lock:
        data = _load_alerts()
        alerts = data["alerts"]
        if unread_only:
            alerts = [a for a in alerts if not a["acknowledged"]]
        alerts = list(reversed(alerts))  # 最新的在前
        total = len(alerts)
        page = alerts[offset:offset + limit]
        unread = sum(1 for a in data["alerts"] if not a["acknowledged"])
        return {"alerts": page, "total": total, "unread": unread}


def ack_alert(alert_id: int):
    with alerts_lock:
        data = _load_alerts()
        for a in data["alerts"]:
            if a["id"] == alert_id and not a["acknowledged"]:
                a["acknowledged"] = True
                a["ack_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _save_alerts(data)
                return True
    return False


# ═══════════════════════════════════════════════════════════
# SSE 客户端列表（用于推送）
# ═══════════════════════════════════════════════════════════

sse_clients = []
sse_lock = threading.Lock()


def notify_alarm(data):
    """通过 SSE 推送摔倒报警并保存记录"""
    # 保存告警记录
    add_alert({
        "type": "fall",
        "detail": f"检测到有人摔倒 ({data.get('people', 0)}人)",
        "people": data.get("people", 0),
        "fps": data.get("fps", 0),
    })
    # 推送给所有 SSE 客户端
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put(json.dumps({"type": "fall_alarm", "data": data}))
            except Exception:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)


def poll_fall_status():
    """轮询摔倒检测状态（每 10 秒最多记录一条告警）"""
    global fall_status, last_alert_time
    while True:
        try:
            resp = requests.get(f"{get_camera_url()}/api/status", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                with fall_lock:
                    was_alarm = fall_status.get("alarm", False)
                    fall_status = data
                    if data.get("alarm") and not was_alarm:
                        now = time.time()
                        if now - last_alert_time >= 10:
                            last_alert_time = now
                            notify_alarm(data)
        except Exception:
            with fall_lock:
                fall_status = {"status": "OFFLINE", "alarm": False, "people": 0, "fps": 0}
        time.sleep(1)


# ═══════════════════════════════════════════════════════════
# 系统硬件监测
# ═══════════════════════════════════════════════════════════

def get_system_info():
    """采集 RDK X5 实时系统数据"""
    info = {}

    # CPU
    try:
        with open("/proc/stat") as f:
            fields = f.readline().split()
            total = sum(int(x) for x in fields[1:])
            idle = int(fields[4])
        info["cpu_idle"] = idle
        info["cpu_total"] = total
    except Exception:
        info["cpu_idle"] = 0
        info["cpu_total"] = 1

    # CPU 使用率 = 1 - idle/total (需要两次采样, 简化版用瞬时值)
    # 实际上用 /proc/stat 需要两次采样算差值，这里简化

    # CPU 频率
    freq = read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "0")
    info["cpu_freq"] = int(freq) // 1000 if freq else 0  # MHz

    # CPU 温度
    temp = read_sysfs("/sys/class/thermal/thermal_zone0/temp", "0")
    info["temperature"] = round(int(temp) / 1000, 1) if temp.isdigit() and int(temp) > 0 else 0

    # 内存
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                if parts[0].startswith("MemTotal"):
                    mem["total"] = int(parts[1])
                elif parts[0].startswith("MemAvailable"):
                    mem["available"] = int(parts[1])
                elif parts[0].startswith("MemFree"):
                    mem.setdefault("available", int(parts[1]))
        info["mem_total_mb"] = round(mem.get("total", 0) / 1024, 0)
        info["mem_avail_mb"] = round(mem.get("available", 0) / 1024, 0)
        info["mem_used_pct"] = round(
            (1 - mem.get("available", 1) / max(mem.get("total", 1), 1)) * 100, 1
        )
    except Exception:
        info["mem_total_mb"] = 0
        info["mem_avail_mb"] = 0
        info["mem_used_pct"] = 0

    # 磁盘
    try:
        st = os.statvfs("/")
        total = st.f_frsize * st.f_blocks
        free = st.f_frsize * st.f_bfree
        info["disk_total_gb"] = round(total / (1024**3), 1)
        info["disk_free_gb"] = round(free / (1024**3), 1)
        info["disk_used_pct"] = round((1 - free / total) * 100, 1)
    except Exception:
        info["disk_total_gb"] = 0
        info["disk_free_gb"] = 0
        info["disk_used_pct"] = 0

    # 网络
    info["ip"] = get_local_ip()

    # 在线时长
    try:
        with open("/proc/uptime") as f:
            uptime_sec = float(f.readline().split()[0])
            info["uptime_hours"] = round(uptime_sec / 3600, 1)
    except Exception:
        info["uptime_hours"] = 0

    # 进程数
    try:
        info["processes"] = len(os.listdir("/proc")) - 2
    except Exception:
        info["processes"] = 0

    # GPU / BPU 信息 (RDK X5 specific)
    try:
        r = subprocess.run(["hbm_runtime", "info"], capture_output=True, text=True, timeout=2)
        info["bpu_info"] = r.stdout.strip() or "N/A"
    except Exception:
        info["bpu_info"] = "N/A"

    return info


# CPU 使用率追踪（两次采样计算）
_cpu_prev = {"idle": 0, "total": 0}
_cpu_lock = threading.Lock()


def get_cpu_usage():
    """计算 CPU 使用率（基于两次采样差值）"""
    global _cpu_prev
    try:
        with open("/proc/stat") as f:
            fields = f.readline().split()
            total = sum(int(x) for x in fields[1:])
            idle = int(fields[4])
        with _cpu_lock:
            if _cpu_prev["total"] > 0:
                delta_total = total - _cpu_prev["total"]
                delta_idle = idle - _cpu_prev["idle"]
                usage = round((1 - delta_idle / max(delta_total, 1)) * 100, 1)
            else:
                usage = 0
            _cpu_prev = {"idle": idle, "total": total}
        return usage
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════
# 心情日志
# ═══════════════════════════════════════════════════════════

MOOD_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "fall_detection", "emotion", "mood_log.json"
)


def _load_mood_log():
    path = Path(MOOD_LOG_PATH)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except:
            pass
    return {"entries": [], "summaries": {}}


# ═══════════════════════════════════════════════════════════
# Flask 应用
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)


# ═══════════════════════════════════════════════════════════
# 页面路由
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", server_ip=get_local_ip(), port=PORT)


@app.route("/camera")
def camera_page():
    return render_template("camera.html", server_ip=get_local_ip(), port=PORT)


@app.route("/chat")
def chat_page():
    return render_template("chat.html", server_ip=get_local_ip(), port=PORT)


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html", server_ip=get_local_ip(), port=PORT)


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html", server_ip=get_local_ip(), port=PORT)


@app.route("/mood")
def mood_page():
    return render_template("mood.html", server_ip=get_local_ip(), port=PORT)


@app.route("/about")
def about_page():
    return render_template("about.html", server_ip=get_local_ip(), port=PORT)


@app.route("/settings")
def settings_page():
    return render_template("settings.html", server_ip=get_local_ip(), port=PORT)


@app.route("/weather")
def weather_page():
    return render_template("weather.html", server_ip=get_local_ip(), port=PORT)


@app.route("/offline")
def offline_page():
    return render_template("offline.html", server_ip=get_local_ip(), port=PORT)


@app.route("/api/icon")
def pwa_icon():
    svg = """<svg xmlns="https://www.w3.org/2000/svg" width="192" height="192" viewBox="0 0 192 192">
      <defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#667eea"/>
        <stop offset="100%" style="stop-color:#764ba2"/>
      </linearGradient></defs>
      <rect width="192" height="192" rx="32" fill="url(#g)"/>
      <text x="96" y="108" text-anchor="middle" font-size="80" fill="white">R</text>
      <text x="96" y="168" text-anchor="middle" font-size="28" fill="rgba(255,255,255,0.6)">X5</text>
    </svg>"""
    return Response(svg, mimetype="image/svg+xml")


# ═══════════════════════════════════════════════════════════
# API: 基础
# ═══════════════════════════════════════════════════════════

@app.route("/api/ip")
def api_ip():
    return jsonify({"ip": get_local_ip(), "port": PORT})


# ═══════════════════════════════════════════════════════════
# API: 设置读写
# ═══════════════════════════════════════════════════════════

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        settings = dict(config)
        key = settings.get("DEEPSEEK_API_KEY", "")
        settings["DEEPSEEK_API_KEY_MASKED"] = (
            key[:8] + "****" + key[-4:] if len(key) > 12 else "****"
        ) if key else ""
        return jsonify(settings)

    data = request.get_json(force=True)
    updates = {}
    if "DEEPSEEK_API_KEY" in data:
        updates["DEEPSEEK_API_KEY"] = data["DEEPSEEK_API_KEY"].strip()
    if "CAMERA_URL" in data:
        updates["CAMERA_URL"] = data["CAMERA_URL"].strip().rstrip("/")
    save_config(updates)
    return jsonify({"ok": True, "message": "设置已保存 ✅"})


@app.route("/api/settings/test", methods=["POST"])
def api_settings_test():
    data = request.get_json(force=True)
    target = data.get("target", "camera")

    if target == "camera":
        url = data.get("url", get_camera_url())
        try:
            resp = requests.get(f"{url}/api/status", timeout=5)
            if resp.status_code == 200:
                return jsonify({"ok": True, "message": f"✅ 摄像头连接成功！"})
            return jsonify({"ok": False, "message": f"❌ 摄像头返回状态码: {resp.status_code}"})
        except Exception as e:
            return jsonify({"ok": False, "message": f"❌ 连接失败: {str(e)[:60]}"})

    elif target == "deepseek":
        key = data.get("api_key", get_deepseek_key())
        if not key:
            return jsonify({"ok": False, "message": "❌ 请先输入 API Key"})
        try:
            payload = json.dumps({
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": "回复OK即可"}],
                "stream": False, "max_tokens": 10
            }).encode()
            req = urllib.request.Request(
                DEEPSEEK_URL, data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                reply = result["choices"][0]["message"]["content"]
                return jsonify({"ok": True, "message": f"✅ DeepSeek 连接成功！回复: {reply[:50]}"})
        except Exception as e:
            return jsonify({"ok": False, "message": f"❌ DeepSeek 连接失败: {str(e)[:60]}"})

    return jsonify({"ok": False, "message": "未知测试目标"})


# ═══════════════════════════════════════════════════════════
# API: 系统状态 / 性能监测
# ═══════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    """摔倒检测状态"""
    with fall_lock:
        return jsonify(fall_status)


@app.route("/api/system/info")
def api_system_info():
    """实时系统信息（CPU/内存/温度等）"""
    info = get_system_info()
    info["cpu_usage"] = get_cpu_usage()
    # 加入检测系统状态
    with fall_lock:
        info["detector"] = {
            "status": fall_status.get("status", "UNKNOWN"),
            "alarm": fall_status.get("alarm", False),
            "people": fall_status.get("people", 0),
            "fps": fall_status.get("fps", 0),
        }
    return jsonify(info)


@app.route("/api/system/history")
def api_system_history():
    """返回最近 N 秒系统性能历史"""
    # 从内存 buffer 读取
    with history_lock:
        snapshots = list(system_history)
    return jsonify({"points": snapshots[-60:]})  # 最近 60 个点


# 系统性能采样缓冲区（最多 120 个点 = 2 分钟 @ 1s）
system_history = []
max_history = 120


def sample_system():
    """后台线程：每秒采样系统性能"""
    global system_history
    while True:
        try:
            cpu = get_cpu_usage()
            info = get_system_info()
            point = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cpu": cpu,
                "mem": info.get("mem_used_pct", 0),
                "temp": info.get("temperature", 0),
                "disk": info.get("disk_used_pct", 0),
            }
            with history_lock:
                system_history.append(point)
                if len(system_history) > max_history:
                    system_history = system_history[-max_history:]
        except Exception:
            pass
        time.sleep(1)


# ═══════════════════════════════════════════════════════════
# API: SSE 实时推送
# ═══════════════════════════════════════════════════════════

@app.route("/api/events")
def sse_events():
    def event_stream():
        q = __import__("queue").Queue()
        with sse_lock:
            sse_clients.append(q)
        try:
            with fall_lock:
                yield f"data: {json.dumps({'type': 'status', 'data': fall_status})}\n\n"
            while True:
                data = q.get(timeout=30)
                yield f"data: {data}\n\n"
        except Exception:
            pass
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ═══════════════════════════════════════════════════════════
# API: 告警记录
# ═══════════════════════════════════════════════════════════

@app.route("/api/alerts")
def api_alerts():
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    unread_only = request.args.get("unread_only", 0, type=int)
    return jsonify(get_alerts(limit=limit, offset=offset, unread_only=bool(unread_only)))


@app.route("/api/alerts/ack", methods=["POST"])
def api_alert_ack():
    data = request.get_json(force=True)
    alert_id = data.get("id")
    if not alert_id:
        return jsonify({"ok": False, "message": "缺少 id"})
    ok = ack_alert(int(alert_id))
    return jsonify({"ok": ok, "message": "已确认" if ok else "未找到或已确认"})


@app.route("/api/alerts/ack-all", methods=["POST"])
def api_alert_ack_all():
    """确认所有未读告警"""
    data = get_alerts(limit=1000, unread_only=True)
    count = 0
    for a in data.get("alerts", []):
        if ack_alert(a["id"]):
            count += 1
    return jsonify({"ok": True, "count": count})


@app.route("/api/alerts/clear", methods=["POST"])
def api_alert_clear():
    """一键删除所有告警记录"""
    with alerts_lock:
        _save_alerts({"alerts": [], "next_id": 1})
    return jsonify({"ok": True, "message": "所有告警记录已删除 ✅"})


# ═══════════════════════════════════════════════════════════
# API: AI 对话
# ═══════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "请说点什么吧～"})

    if not get_deepseek_key():
        return jsonify({
            "reply": "⚠️ 我还没有配置 DeepSeek API 密钥。\n\n请前往 **设置** 页面配置 API Key。"
        })

    with history_lock:
        conversation_history.append({"role": "user", "content": user_msg})
        messages = conversation_history[-20:]

    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        DEEPSEEK_URL, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {get_deepseek_key()}",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            reply = result["choices"][0]["message"]["content"]
            with history_lock:
                conversation_history.append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})
    except urllib.error.HTTPError as e:
        return jsonify({"reply": f"😓 API 请求失败 ({e.code})。请检查 API 密钥和网络。"})
    except Exception as e:
        return jsonify({"reply": f"😓 网络错误: {str(e)}"})


@app.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    with history_lock:
        conversation_history.clear()
        conversation_history.append(
            {"role": "system", "content": "你是一个友善的智能助手，用中文回答用户的问题。回答要简洁准确，语气自然。"}
        )
    return jsonify({"ok": True})


@app.route("/api/chat/set-system", methods=["POST"])
def api_chat_set_system():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if prompt:
        with history_lock:
            for i, msg in enumerate(conversation_history):
                if msg["role"] == "system":
                    conversation_history[i]["content"] = prompt
                    break
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════
# API: 天气
# ═══════════════════════════════════════════════════════════

@app.route("/api/weather/locate")
def api_weather_locate():
    """通过服务端 IP 获取大致位置（比浏览器定位稳定）"""
    try:
        resp = requests.get(
            "http://ip-api.com/json/?fields=status,lat,lon,city,regionName,country",
            timeout=5
        )
        data = resp.json()
        if data.get("status") == "success":
            return jsonify({
                "lat": data["lat"],
                "lon": data["lon"],
                "city": data.get("city", ""),
                "region": data.get("regionName", ""),
                "country": data.get("country", ""),
            })
    except Exception as e:
        log.warning(f"天气定位失败: {e}")
    return jsonify({"error": "无法定位"}), 502


@app.route("/api/tts", methods=["POST"])
def api_tts():
    """文字转语音 — 返回 WAV 音频"""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "no text"}), 400

    import subprocess as sp
    import uuid

    filename = f"/tmp/tts_{uuid.uuid4().hex}.mp3"
    wav_file = filename.replace(".mp3", ".wav")

    try:
        sp.run([
            "edge-tts", "--voice", "zh-CN-XiaoxiaoNeural",
            "--text", text[:200],
            "--write-media", filename
        ], stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=30)

        sp.run([
            "ffmpeg", "-y", "-i", filename,
            "-ar", "16000", "-ac", "1",
            wav_file
        ], stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=10)

        with open(wav_file, "rb") as f:
            audio_data = f.read()

        for f in [filename, wav_file]:
            try:
                os.remove(f)
            except Exception:
                pass

        return Response(audio_data, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# API: 心情日志
# ═══════════════════════════════════════════════════════════

@app.route("/api/mood/today")
def api_mood_today():
    log_data = _load_mood_log()
    today = time.strftime("%Y-%m-%d")
    entries = [e for e in log_data.get("entries", []) if e.get("date") == today]
    return jsonify({"entries": entries, "total": len(entries)})


@app.route("/api/mood/stats")
def api_mood_stats():
    log_data = _load_mood_log()
    entries = log_data.get("entries", [])
    from collections import Counter
    today = time.strftime("%Y-%m-%d")
    today_entries = [e for e in entries if e.get("date") == today]

    dist = Counter(e.get("emotion", "neutral") for e in today_entries)
    total = len(today_entries)

    labels_cn = {
        "angry": "😠 生气", "disgust": "🤢 厌恶", "fear": "😨 害怕",
        "happy": "😊 开心", "sad": "😢 悲伤", "surprise": "😲 惊讶",
        "neutral": "😐 中性",
    }

    distribution = {}
    for k in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
        distribution[labels_cn.get(k, k)] = dist.get(k, 0)

    dominant = max(dist, key=dist.get) if dist else "neutral"

    return jsonify({
        "total": total,
        "distribution": distribution,
        "dominant": labels_cn.get(dominant, dominant),
        "date": today,
    })


@app.route("/api/mood/summary", methods=["POST"])
def api_mood_summary():
    from collections import Counter
    log_data = _load_mood_log()
    today = time.strftime("%Y-%m-%d")
    entries = [e for e in log_data.get("entries", []) if e.get("date") == today]

    if not entries:
        return jsonify({"summary": "今天还没有记录到表情数据。"})

    summaries = log_data.get("summaries", {})
    if today in summaries:
        return jsonify({"summary": summaries[today]["summary"]})

    labels_cn = {
        "angry": "😠 生气", "disgust": "🤢 厌恶", "fear": "😨 害怕",
        "happy": "😊 开心", "sad": "😢 悲伤", "surprise": "😲 惊讶",
        "neutral": "😐 中性",
    }
    timeline = "\n".join(
        f"  {e['time_str']}  {labels_cn.get(e.get('emotion','neutral'), e.get('label_cn','??'))}"
        for e in entries[-50:]
    )

    dist = Counter(e.get("emotion", "neutral") for e in entries)
    dominant = max(dist, key=dist.get) if dist else "neutral"

    prompt = (
        f"以下是一个人今天的心情记录数据，请生成一份简洁的心情日志总结：\n\n"
        f"## 数据\n"
        f"- 记录: {len(entries)} 条\n"
        f"- 主要情绪: {labels_cn.get(dominant, dominant)}\n\n"
        f"## 时间线\n{timeline}\n\n"
        f"## 要求\n"
        f"1. 用中文总结情绪变化趋势\n"
        f"2. 指出情绪变化的关键时段\n"
        f"3. 给出一个心情评分（满分10分）\n"
        f"4. 语气温暖轻松\n"
        f"5. 控制在 150 字以内\n"
        f"6. 最后用一句鼓励的话结尾\n"
        f"7. 不带任何 markdown 格式标记"
    )

    api_key = get_deepseek_key()
    if not api_key:
        return jsonify({"summary": "请先在设置页面配置 DeepSeek API Key 😊"})

    try:
        payload = json.dumps({
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是温暖的心理分析师，用中文简洁总结心情日志。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }).encode()
        req = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            summary = result["choices"][0]["message"]["content"]

        def _save_summary():
            try:
                p = Path(MOOD_LOG_PATH)
                if p.exists():
                    d = json.loads(p.read_text(encoding="utf-8"))
                    d.setdefault("summaries", {})[today] = {
                        "summary": summary,
                        "at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            except:
                pass
        threading.Thread(target=_save_summary, daemon=True).start()

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"summary": f"😓 生成总结失败: {str(e)[:80]}"})


@app.route("/api/mood/clear", methods=["POST"])
def api_mood_clear():
    """一键删除所有心情日志"""
    try:
        path = Path(MOOD_LOG_PATH)
        if path.exists():
            path.write_text(
                json.dumps({"entries": [], "summaries": {}}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        return jsonify({"ok": True, "message": "心情日志已清空 ✅"})
    except Exception as e:
        return jsonify({"ok": False, "message": f"清空失败: {str(e)}"}), 500


if __name__ == "__main__":
    print("═" * 50)
    print("  RDK X5 综合服务平台 v2.0")
    print("═" * 50)
    ip = get_local_ip()
    print(f"  局域网: http://{ip}:{PORT}")
    print(f"  本机:   http://127.0.0.1:{PORT}")
    if not get_deepseek_key():
        print("  ⚠️  AI 对话未配置 API Key")
    print()

    # 启动后台轮询
    t = threading.Thread(target=poll_fall_status, daemon=True)
    t.start()

    # 启动系统采样
    t2 = threading.Thread(target=sample_system, daemon=True)
    t2.start()

    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
