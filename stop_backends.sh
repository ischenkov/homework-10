#!/usr/bin/env bash
# Зупиняє процеси, запущені start_backends.sh
set -euo pipefail
cd "$(dirname "$0")"
shopt -s nullglob
for f in .pids/*.pid; do
  name=$(basename "$f" .pid)
  pid=$(cat "$f" 2>/dev/null || true)
  [[ -n "${pid:-}" ]] || continue
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" && echo "Зупинено ${name} (pid ${pid})"
  else
    echo "Процес ${name} (pid ${pid}) уже не працює — прибираю pid-файл"
  fi
  rm -f "$f"
done
echo "Готово."
