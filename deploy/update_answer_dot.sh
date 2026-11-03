#!/bin/bash
# 更新：答题放大 + 红点落点 + 自洽校验
set -e
cp /tmp/backends.py /opt/fovea/fovea_service/backends.py
cp /tmp/worker.py   /opt/fovea/fovea_service/worker.py
cp /tmp/server.py   /opt/fovea/fovea_service/server.py
cp /tmp/index.html  /opt/fovea/fovea_service/static/index.html
rm -rf /opt/fovea/fovea_service/__pycache__
systemctl restart fovea
sleep 2
systemctl is-active fovea
curl -s --max-time 5 http://127.0.0.1:8765/api/health
echo
