#!/bin/bash
cd "$(dirname "$0")"
# Avoid OpenMP conflict between faiss/numpy (macOS)
export KMP_DUPLICATE_LIB_OK=TRUE
#
# Start in separate terminals first (Python >= 3.11):
#   python mcp_servers/search_mcp.py
#   python mcp_servers/report_mcp.py
#   python acp_server.py
#
.venv/bin/python main.py
