#!/bin/bash
# 部署访问控制：登录分流 / 密钥 / 访客配额 / 管理后台
set -e
cp /tmp/auth.py    /opt/fovea/fovea_service/auth.py
cp /tmp/db.py      /opt/fovea/fovea_service/db.py
cp /tmp/server.py  /opt/fovea/fovea_service/server.py
cp /tmp/login.html /opt/fovea/fovea_service/static/login.html
cp /tmp/admin.html /opt/fovea/fovea_service/static/admin.html
cp /tmp/index.html /opt/fovea/fovea_service/static/index.html
if [ -f /tmp/keys.json ]; then
  cp /tmp/keys.json /opt/fovea/.keys.json
  chown root:root /opt/fovea/.keys.json
  chmod 600 /opt/fovea/.keys.json
fi
rm -rf /opt/fovea/fovea_service/__pycache__
systemctl restart fovea
sleep 2
echo "--- 服务 ---"
systemctl is-active fovea
echo "--- 首页行为（未登录应返回登录页）---"
curl -s http://127.0.0.1:8765/ | grep -o "<title>[^<]*</title>"
echo "--- 未登录访问 API（应 401）---"
curl -s -o /dev/null -w "  /api/demo -> %{http_code}\n" http://127.0.0.1:8765/api/demo
echo "--- 密钥 ---"
cat /opt/fovea/.keys.json | python3 -c "import json,sys; d=json.load(sys.stdin); [print(' ', k['key'], k['role']) for k in d['keys']]"
