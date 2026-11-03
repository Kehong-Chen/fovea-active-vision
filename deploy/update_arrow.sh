#!/bin/bash
# 更新前端：答案红点 → 指向性箭头（index.html 是静态文件，按请求读盘，不用重启服务）
set -e
D=/opt/fovea/fovea_service/static/index.html
cp /tmp/index.html "$D"
echo "已写入 $D"
echo "drawAnswerArrow 出现 $(grep -c 'drawAnswerArrow' "$D") 次 · 残留红点注释 $(grep -c '答案位置：红点' "$D" || true) 处"
curl -s --max-time 5 http://127.0.0.1:8765/api/health; echo
