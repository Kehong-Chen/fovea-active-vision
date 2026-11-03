#!/bin/bash
TOKEN=<WORKER_TOKEN>
echo "=== server.py 里的 image 路由 ==="
grep -n "api/image" /opt/fovea/fovea_service/server.py | head -6
echo
echo "=== 取一个真实图片 id ==="
IMG=$(ls /opt/fovea/fovea_service/data/images 2>/dev/null | tail -1 | sed 's/\.jpg$//')
echo "id=$IMG"
echo
echo "=== 本机直连测试 ==="
curl -s -o /dev/null -w "  带 token   -> %{http_code}\n" "http://127.0.0.1:8765/api/image/$IMG?token=$TOKEN"
curl -s -o /dev/null -w "  不带 token -> %{http_code}\n" "http://127.0.0.1:8765/api/image/$IMG"
echo
echo "=== systemd 里的 token ==="
grep ExecStart /etc/systemd/system/fovea.service | sed 's/--token [a-f0-9]*/--token ***/'
