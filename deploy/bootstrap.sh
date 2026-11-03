#!/usr/bin/env bash
# =====================================================================
# fovea_service + PDF 下载站 一键部署
# 目标机器：阿里云轻量应用服务器 2核2G3M（Ubuntu 22.04 / 24.04）
# 用法：把整个 fovea_deploy 目录传上去，然后
#       sudo bash bootstrap.sh
# 结果：http://<IP>:8765  fovea 演示页
#       http://<IP>:8080/perception-bandwidth-zh.pdf  论文 PDF（点了就下）
# =====================================================================
set -euo pipefail

APP=/opt/fovea
DL=/srv/dl
SERVICE_PORT=8765
DL_PORT=8080
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> 1/7 安装系统包（python3 + Pillow 用 apt 版本，绕开 PEP668）"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-pil nginx openssl tar >/dev/null
timedatectl set-timezone Asia/Shanghai 2>/dev/null || true

# Pillow 兜底：apt 装不上（universe 未启用）就走 pip
if ! python3 -c "import PIL" 2>/dev/null; then
  echo "    apt 版 Pillow 缺失，改用 pip 安装"
  python3 -m pip install -q --break-system-packages pillow \
    -i https://mirrors.cloud.tencent.com/pypi/simple/ \
    || python3 -m pip install -q --break-system-packages pillow \
       -i https://mirrors.aliyun.com/pypi/simple/
fi
python3 -c "import PIL, sys; print('   Pillow', PIL.__version__)"

echo "==> 2/7 swap（2G 内存必做，否则 Pillow 解大图会 OOM）"
if ! swapon --show 2>/dev/null | grep -q '/swapfile'; then
  fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> 3/7 解包代码"
mkdir -p "$APP" "$DL"
tar -xzf "$HERE/fovea_payload.tar.gz" -C "$APP"
[ -f "$HERE/perception-bandwidth-zh.pdf" ] && cp -f "$HERE/perception-bandwidth-zh.pdf" "$DL/"

echo "==> 4/7 生成 worker token（默认 dev 必须换掉）"
TOKEN_FILE="$APP/.token"
[ -f "$TOKEN_FILE" ] || openssl rand -hex 16 > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"
TOKEN="$(cat "$TOKEN_FILE")"

echo "==> 5/7 systemd 服务 fovea.service"
cat > /etc/systemd/system/fovea.service <<EOF
[Unit]
Description=FOVEA service (inline/sim)
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP
Environment=PYTHONUNBUFFERED=1
Environment=FOVEA_MAX_SIDE=2048
ExecStart=/usr/bin/python3 -m fovea_service.server --host 0.0.0.0 --port $SERVICE_PORT --mode inline --backend sim --token $TOKEN
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

echo "==> 6/7 nginx：静态 PDF（强制下载、禁目录列表）"
cp -f "$HERE/nginx-dl.conf" /etc/nginx/sites-available/dl
ln -sf /etc/nginx/sites-available/dl /etc/nginx/sites-enabled/dl
rm -f /etc/nginx/sites-enabled/default
nginx -t

echo "==> 7/7 启动"
systemctl daemon-reload
systemctl enable --now fovea >/dev/null
systemctl restart nginx

sleep 1
IP="$(curl -s --max-time 3 http://metadata.tencentyun.com/latest/meta-data/public-ipv4 2>/dev/null \
     || curl -s --max-time 3 http://100.100.100.200/latest/meta-data/eipv4 2>/dev/null \
     || hostname -I | awk '{print $1}')"
echo
echo "======================================================"
echo " FOVEA 演示页 : http://$IP:$SERVICE_PORT/"
echo " 论文 PDF     : http://$IP:$DL_PORT/perception-bandwidth-zh.pdf"
echo " worker token : $TOKEN   (在 $TOKEN_FILE)"
echo "------------------------------------------------------"
echo " 健康检查     : curl -s http://127.0.0.1:$SERVICE_PORT/api/health"
echo " 服务日志     : journalctl -u fovea -n 50 --no-pager"
echo " 切 queue 模式: 改 /etc/systemd/system/fovea.service 的 --mode queue 后 daemon-reload && restart"
echo "======================================================"
