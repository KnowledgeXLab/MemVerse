#!/usr/bin/env bash
set -e

echo "Starting MCP server..."
python mcp_server.py &

sleep 2

echo "Starting FastAPI..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
