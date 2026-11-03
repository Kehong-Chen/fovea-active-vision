#!/bin/bash
# 更新：合成场景走本地 sim，前端显示来源
set -e
cp /tmp/server.py  /opt/fovea/fovea_service/server.py
cp /tmp/index.html /opt/fovea/fovea_service/static/index.html
rm -rf /opt/fovea/fovea_service/__pycache__
systemctl restart fovea
sleep 2
echo "--- 服务状态 ---"
systemctl is-active fovea
echo "--- health ---"
curl -s --max-time 5 http://127.0.0.1:8765/api/health
echo
