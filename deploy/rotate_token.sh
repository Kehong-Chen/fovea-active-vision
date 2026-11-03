#!/bin/bash
# 轮换 worker token：旧 worker 进程会因 token 失效而无法再领任务
set -e
OLD=$(grep -o 'token [a-f0-9]*' /etc/systemd/system/fovea.service | awk '{print $2}')
NEW=$(openssl rand -hex 16)
echo "$NEW" > /opt/fovea/.token
chmod 600 /opt/fovea/.token
sed -i "s/--token $OLD/--token $NEW/" /etc/systemd/system/fovea.service
systemctl daemon-reload
systemctl restart fovea
sleep 2
echo "服务：$(systemctl is-active fovea)"
echo "新 token：$NEW"
curl -s -o /dev/null -w "健康检查 %{http_code}\n" http://127.0.0.1:8765/api/health
