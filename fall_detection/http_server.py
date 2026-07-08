#!/usr/bin/env python3
"""
独立 HTTP 服务器 — 从共享文件读取最新的 JPEG 帧并推流
与 fall_detection.py 配合使用
"""
import os
import sys
import json
import time
import socket
import struct
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ── 配置 ──
PORT = 8080
FRAME_FILE = "/tmp/fall_frame.jpg"
META_FILE = "/tmp/fall_meta.json"

DEFAULT_META = json.dumps({
    "fps": 0,
    "status": "WAITING",
    "people": 0,
    "alarm": False,
    "aspect": 0,
    "vert_vel": 0,
    "emotion": "",
    "emotion_cn": "",
    "emotion_conf": 0,
}).encode()


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默

    def do_GET(self):
        if self.path == '/':
            self._serve_html()
        elif self.path == '/stream':
            self._serve_mjpeg()
        elif self.path == '/api/status':
            self._serve_status()
        elif self.path == '/favicon.ico':
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html;charset=utf-8')
        self.end_headers()
        ip = get_local_ip()
        html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>RDK X5 摔倒检测 + 姿态骨骼</title>
<style>
  *{{margin:0;padding:0}}
  body{{background:#111;display:flex;justify-content:center;align-items:center;height:100vh;font-family:sans-serif}}
  #wrap{{text-align:center}}
  img{{max-width:100vw;max-height:90vh;border-radius:8px;box-shadow:0 0 20px rgba(0,0,0,0.5)}}
  #info{{color:#aaa;margin-top:10px;font-size:14px}}
</style></head><body>
<div id="wrap">
  <img src="/stream" />
  <div id="info">RDK X5 摔倒检测 + 姿态骨骼 · <a href="/api/status" style="color:#888">状态</a></div>
</div>
</body></html>'''
        self.wfile.write(html.encode())

    def _serve_status(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            data = Path(META_FILE).read_bytes()
            self.wfile.write(data)
        except Exception:
            self.wfile.write(DEFAULT_META)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=fr')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        last_mtime = 0
        try:
            while True:
                try:
                    st = os.stat(FRAME_FILE)
                    if st.st_mtime > last_mtime:
                        jpeg = Path(FRAME_FILE).read_bytes()
                        last_mtime = st.st_mtime
                        self.wfile.write(b'--fr\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                except (FileNotFoundError, OSError):
                    pass
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def send_header(self, keyword, value):
        """Override to avoid duplicate Server/Date headers."""
        super().send_header(keyword, value)


def main():
    # 清理旧文件
    try:
        os.unlink(FRAME_FILE)
    except FileNotFoundError:
        pass

    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    print(f"[Web] → http://{get_local_ip()}:{PORT}")
    print(f"[Web] 等待 fall_detection.py 写入帧到 {FRAME_FILE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    main()
