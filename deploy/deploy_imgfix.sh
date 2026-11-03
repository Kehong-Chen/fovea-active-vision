#!/bin/bash
set -e
cp /tmp/server.py /opt/fovea/fovea_service/server.py
cp /tmp/worker.py /opt/fovea/fovea_service/worker.py
rm -rf /opt/fovea/fovea_service/__pycache__
systemctl restart fovea
sleep 2
systemctl is-active fovea
curl -s -o /dev/null -w "GET / -> %{http_code}\n" http://127.0.0.1:8765/
curl -s -o /dev/null -w "GET /api/image/xxx (无会话无token) -> %{http_code}\n" http://127.0.0.1:8765/api/image/xxx
