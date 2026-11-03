# -*- coding: utf-8 -*-
"""fovea_service.auth —— 访问控制：访客 / 参与者（密钥）/ 管理员。

设计：
  · 密钥文件 keys.json：{"keys": [{"id": "K1", "key": "FV-XXXX-XXXX", "role": "admin|member", "note": ""}], "secret": "..."}
  · 会话用 HMAC 签名的 cookie（fovea_sess），payload = base64(json) + "." + 签名；
  · 访客用 fovea_guest cookie 存设备号，配额按天记录在 SQLite 里。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Dict, List, Optional, Tuple

DEFAULT_TTL = 7 * 24 * 3600          # 兜底：7 天
SESSION_UNTIL = "2027-01-01T00:00:00+08:00"   # 参与者会话统一有效到这个时刻
GUEST_DAILY_QUOTA = 3                # 访客每天可识别图片数


def session_ttl(now: Optional[float] = None) -> int:
    """距离 SESSION_UNTIL 还剩多少秒（至少 60 秒，避免负数导致立刻过期）。"""
    from datetime import datetime
    import math
    end = datetime.fromisoformat(SESSION_UNTIL).timestamp()
    t = time.time() if now is None else now
    return max(60, int(math.ceil(end - t)))


def gen_keys(n: int = 9, admin: int = 1) -> Dict:
    """生成 n 个密钥，其中前 admin 个是管理员。"""
    keys = []
    for i in range(n):
        raw = secrets.token_hex(4).upper()
        keys.append({
            "id": f"K{i + 1}",
            "key": f"FV-{raw[:4]}-{raw[4:]}",
            "role": "admin" if i < admin else "member",
            "note": "管理员" if i < admin else "参与者",
        })
    return {"keys": keys, "secret": secrets.token_hex(32), "created": int(time.time())}


def load(path: str) -> Dict:
    if not os.path.exists(path):
        data = gen_keys()
        save(path, data)
        return data
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(path, 0o600)
    except Exception:  # noqa: BLE001
        pass


def find_key(store: Dict, key: str) -> Optional[Dict]:
    k = (key or "").strip().upper()
    for item in store.get("keys", []):
        if item.get("key", "").upper() == k:
            return item
    return None


def name_of(store: Optional[Dict], kid: str) -> str:
    """按密钥 id 反查显示名（后台用来标注是谁在操作）。"""
    if not store or not kid:
        return ""
    for item in store.get("keys", []):
        if item.get("id") == kid:
            return item.get("name") or item.get("id") or ""
    return kid


def _sign(secret: str, body: str) -> str:
    return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()


def make_session(secret: str, role: str, kid: str = "", ttl: Optional[int] = None) -> str:
    payload = {"role": role, "kid": kid, "exp": int(time.time()) + (ttl if ttl is not None else session_ttl())}
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    return body + "." + _sign(secret, body)


def read_session(secret: str, token: str) -> Optional[Dict]:
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(_sign(secret, body), sig):
        return None
    pad = "=" * (-len(body) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + pad).decode())
    except Exception:  # noqa: BLE001
        return None
    if int(payload.get("exp", 0)) < time.time():
        return None
    return payload


def parse_cookies(header: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not header:
        return out
    for part in header.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def new_guest_id() -> str:
    return "g-" + secrets.token_hex(8)
