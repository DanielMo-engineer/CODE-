#!/usr/bin/env python3
"""
快速查询/打印当前 fall_detection 的三种访问入口：
  1. 公网 URL（cpolar website 隧道）—— 适合在外面用
  2. 本机 IP（局域网内用手机/PC 看）
  3. localhost（板子自己调试）
同时打印 SSH 公网 TCP 地址，便于应急登录。
"""
import re, subprocess, socket, sys, os
from pathlib import Path

CPOLAR_LOG = "/var/log/cpolar/access.log"

def cpolar_url():
    p = Path(CPOLAR_LOG)
    if not p.exists():
        return None
    txt = p.read_text(errors="ignore")
    # 抓最后一次 http/https 隧道建立
    matches = re.findall(r"Tunnel established at (https?://[\w\.\-]+)", txt)
    return matches[-1] if matches else None

def cpolar_tcp():
    p = Path(CPOLAR_LOG)
    if not p.exists():
        return None
    txt = p.read_text(errors="ignore")
    # 公网 TCP 隧道通常以 "tcp://x.x.x.x:PORT" 或 "tcp://region.cpolar.io:PORT" 形式登记
    # cpolar 的 ssh 隧道日志字段不同，回退用 cpolar api
    try:
        out = subprocess.check_output(
            ["cpolar", "status"], stderr=subprocess.DEVNULL, timeout=5
        ).decode()
        m = re.search(r"(tcp://[\w\.:\-]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None

def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())

def web_alive():
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8080/api/status", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

def main():
    print("=" * 60)
    print(" RDK X5 Fall Detection 访问入口")
    print(f" 本机时间: {subprocess.check_output(['date'], text=True).strip()}")
    print(f" 本机 web 8080: {'✅ 正常' if web_alive() else '❌ 无响应'}")
    print("-" * 60)
    print(f" 🏠 局域网  : http://{local_ip()}:8080/")
    u = cpolar_url()
    print(f" 🌍 公网 URL : {u if u else '(cpolar 日志暂无，请稍候或重启 cpolar)'}")
    t = cpolar_tcp()
    print(f" 🔐 SSH 公网 : {t if t else '(ssh 隧道尚未拉起)'}")
    print("=" * 60)
    print(" 提示:")
    print("   - cpolar 免费版 URL 24h 变化；上面是当前最近的 URL。")
    print("   - 若需要新的 URL，运行:  sudo systemctl restart cpolar")
    print("   - 若 8080 无响应，运行:  sudo bash /root/.openclaw/workspace/fall_detection/run_fall_detection.sh --bpu-cores 0 1")
    print()

if __name__ == "__main__":
    main()
