#!/bin/bash
# 把 fovea 服务切到 queue 模式（只排队，计算交给 worker）
set -e
sed -i 's/--mode inline/--mode queue/; s/--backend sim/--backend vlm/' /etc/systemd/system/fovea.service
systemctl daemon-reload
systemctl restart fovea
sleep 2
echo "--- systemd 状态 ---"
systemctl is-active fovea
echo "--- 当前启动参数 ---"
grep ExecStart /etc/systemd/system/fovea.service
echo "--- health ---"
curl -s --max-time 5 http://127.0.0.1:8765/api/health
echo
