#!/bin/bash
set -e
cp /tmp/backends.py /opt/fovea/fovea_service/backends.py
rm -rf /opt/fovea/fovea_service/__pycache__
systemctl restart fovea
sleep 2
echo -n "service: "; systemctl is-active fovea
echo -n "health: "; curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/api/health
echo -n "locate* 残留: "; grep -c "def locate" /opt/fovea/fovea_service/backends.py || true
echo -n "pick_point: "; grep -c "def pick_point" /opt/fovea/fovea_service/backends.py || true
