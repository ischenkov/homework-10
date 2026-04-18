#!/usr/bin/env bash
# Запуск SearchMCP, ReportMCP і ACP у фоні (логи в logs/). З кореня репозиторію.
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Немає виконуваного $PY"
  echo "Створи venv: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
export KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p logs .pids

wait_port() {
  local host=$1 port=$2 tries=${3:-30}
  local i=0
  while (( i < tries )); do
    if "$PY" -c "import socket; s=socket.socket(); s.settimeout(0.3); s.connect(('$host',$port)); s.close()" 2>/dev/null; then
      return 0
    fi
    sleep 0.2
    ((i++)) || true
  done
  return 1
}

start_one() {
  local name=$1
  shift
  local pidfile=".pids/${name}.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "[skip] ${name} уже працює (pid $(cat "$pidfile"))"
    return 0
  fi
  rm -f "$pidfile"
  local log="logs/${name}.log"
  : >"$log"
  nohup "$PY" "$@" >>"$log" 2>&1 &
  echo $! >"$pidfile"
  echo "[start] ${name} pid=$(cat "$pidfile")  log=${log}"
}

fail_log() {
  echo ""
  echo "!!! Помилка: $1"
  echo "--- останні рядки $2 ---"
  tail -n 40 "$2" 2>/dev/null || true
  echo "---"
  exit 1
}

start_one search_mcp mcp_servers/search_mcp.py
if ! wait_port 127.0.0.1 8901 40; then
  fail_log "SearchMCP не відкрив порт 8901" logs/search_mcp.log
fi
echo "[check] SearchMCP 8901 OK"

start_one report_mcp mcp_servers/report_mcp.py
if ! wait_port 127.0.0.1 8902 40; then
  fail_log "ReportMCP не відкрив порт 8902" logs/report_mcp.log
fi
echo "[check] ReportMCP 8902 OK"

start_one acp_server acp_server.py
if ! wait_port 127.0.0.1 8903 60; then
  fail_log "ACP не відкрив порт 8903" logs/acp_server.log
fi
echo "[check] ACP 8903 OK"

echo ""
echo "Готово. Запусти REPL: .venv/bin/python main.py"
echo "Зупинити: ./stop_backends.sh"
