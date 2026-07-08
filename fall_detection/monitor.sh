#!/bin/bash
# =============================================================
# Fall Detection 看门狗
#   • 每 30s 检测 http://localhost:8080/api/status 是否在 5s 内返回
#   • 不自动 kill BPU 长任务：只推 Server 酱通知
#   • 重启期间不打搅：检测到 .restarting 标记文件就跳过本轮
#   • 同时确保 cpolar 在跑 + 解析官网 URL
# =============================================================
set -u

LOG=/var/log/fall-detection-monitor.log
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
RESTART_FLAG="$SCRIPT_DIR/.restarting"

# 加载 Server 酱推送目标（你的 .env 里已有 SC_KEY）
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

mkdir -p "$(dirname "$LOG")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

send_sct() {
  local title="$1"; local body="$2"
  if [ -n "${SC_KEY:-}" ]; then
    (
      curl -sS -m 5 -X POST "https://sctapi.ftqq.com/${SC_KEY}.send" \
        --data-urlencode "title=${title}" \
        --data-urlencode "desp=${body}" >/dev/null 2>&1 || true
    ) &
  fi
}

check_8080() {
  local code
  code=$(curl -sS -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/api/status 2>/dev/null || echo 000)
  echo "$code"
}

check_cpolar() {
  pgrep -x cpolar >/dev/null && echo 1 || echo 0
}

ensure_cpolar() {
  if [ "$(check_cpolar)" != "1" ]; then
    log "[WARN] cpolar 未运行，尝试拉起"
    pkill -9 cpolar 2>/dev/null
    sleep 1
    nohup /usr/bin/cpolar start-all \
      -config=/usr/local/etc/cpolar/cpolar.yml \
      -log=/var/log/cpolar/access.log \
      >>/var/log/cpolar/stdout.log 2>&1 &
    sleep 4
    if [ "$(check_cpolar)" = "1" ]; then
      log "[OK] cpolar 已重启"
      send_sct "🔧 cpolar 已重启" "看门狗检测到 cpolar 异常退出，已自动恢复"
    else
      log "[ERROR] cpolar 拉起失败"
    fi
  fi
}

LAST_NOTIFY_URL=""
notify_url_if_changed() {
  local logfile="/var/log/cpolar/access.log"
  [ -f "$logfile" ] || return
  local url
  url=$(grep -oE 'Tunnel established at https?://[^ ]+' "$logfile" | tail -1 | awk '{print $NF}')
  [ -z "$url" ] && return
  if [ "$url" != "$LAST_NOTIFY_URL" ]; then
    LAST_NOTIFY_URL="$url"
    log "[INFO] 公网 URL 更新: $url"
    send_sct "📷 摄像头画面已上线" "最新公网地址: ${url}\n（cpolar 免费版 URL 会变化，需要时再来拿）"
  fi
}

main_loop() {
  log "=== monitor 启动（PID $$） ==="
  while true; do
    # 重启中 → 跳过本轮检查（不发警报）
    if [ -f "$RESTART_FLAG" ]; then
      log "[SKIP] 检测到 .restarting 标记，跳过本轮"
      sleep 10
      continue
    fi

    code=$(check_8080)
    if [ "$code" != "200" ]; then
      log "[WARN] /api/status 返回 ${code}（仅通知，不自动 kill）"
      send_sct "❌ fall_detection 8080 无响应 (HTTP ${code})" "请手动重启: sudo bash ${SCRIPT_DIR}/run_fall_detection.sh --bpu-cores 0 1"
    fi
    ensure_cpolar
    notify_url_if_changed
    sleep 30
  done
}

main_loop
