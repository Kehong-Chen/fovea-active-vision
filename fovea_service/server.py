# -*- coding: utf-8 -*-
"""fovea_service.server —— 云端/本地一体的 Web 服务（标准库，零依赖）。

两种运行方式：
  1) 一体化（默认）：--mode inline，提交后服务自己算（适合本地演示）
  2) 前后端分离：--mode queue，只负责「上传 + 排队 + 展示」，
     计算由 fovea_service.worker 在任意机器上主动外连领取——云端不需要 GPU、
     不需要公网可达，按需开机即可。

接口：
  GET  /                    前端页面
  GET  /api/health          健康检查 + 队列统计
  GET  /api/demo[?new=1]    合成示例图
  GET  /api/image/<id>      原图
  POST /api/upload          {data:"data:image/jpeg;base64,..."}
  POST /api/query           {image_id, question, mode, seed} -> task_id
  GET  /api/task/<id>       任务状态/步骤/结果
  POST /api/claim           {worker, token, lease} -> 任务（worker 专用）
  POST /api/complete        {task_id, token, steps, result, elapsed, error?}
  POST /api/heartbeat       {task_id, worker, token}
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple
from urllib.parse import parse_qs, urlparse

from . import auth, db, search
from .backends import SimBackend, VLBackend, make_demo_scene
from fovea_lite.geometry import inside

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(DATA, "images")
SCENE_DIR = os.path.join(DATA, "scenes")
STATIC = os.path.join(ROOT, "static")
KEYS_PATH = os.path.join(os.path.dirname(ROOT), ".keys.json")
for d in (DATA, IMG_DIR, SCENE_DIR):
    os.makedirs(d, exist_ok=True)

STATE = {
    "backend": "sim", "mode": "inline", "token": "dev",
    "model": os.environ.get("FOVEA_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
    "demo_id": None, "sem": threading.Semaphore(2),
    "keys": None, "secret": os.environ.get("FOVEA_SECRET", ""),
}


# ------------------------------------------------------------------ 图片
def _save_image(raw: bytes) -> Tuple[str, int, int]:
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    long_side = int(os.environ.get("FOVEA_MAX_SIDE", "4096"))
    if max(im.size) > long_side:
        r = long_side / max(im.size)
        im = im.resize((int(im.size[0] * r), int(im.size[1] * r)))
    iid = uuid.uuid4().hex[:12]
    im.save(os.path.join(IMG_DIR, f"{iid}.jpg"), format="JPEG", quality=90)
    return iid, im.size[0], im.size[1]


def _demo(force_new: bool = False) -> Dict:
    did = STATE["demo_id"]
    if (not force_new) and did and os.path.exists(os.path.join(IMG_DIR, f"{did}.jpg")):
        meta = json.load(open(os.path.join(SCENE_DIR, f"{did}.json"), encoding="utf-8"))
        return {"image_id": did, **meta}
    import random as _r
    seed = _r.randrange(1, 10 ** 6) if force_new else 7
    sc, raw, size = make_demo_scene(seed)
    iid, w, h = _save_image(raw)
    meta = {"width": w, "height": h,
            "note": f"合成场景（seed={seed}）：白色圆环是目标，浅黄色小方块是干扰物（比目标小）",
            "truth": {"target": list(sc.target), "size": sc.size,
                      "distractors": [[list(p), s] for p, s in sc.distractors]}}
    json.dump({**meta, "scene": {"target": list(sc.target), "size": sc.size,
                                 "distractors": [[list(p), s] for p, s in sc.distractors]}},
              open(os.path.join(SCENE_DIR, f"{iid}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    STATE["demo_id"] = iid
    return {"image_id": iid, **meta}


def _scene_of(iid: str):
    p = os.path.join(SCENE_DIR, f"{iid}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8")).get("scene")


def _truth_of(iid: str):
    """真值：合成图用场景里的目标位置；真实图用用户点选的点。没设过就返回 None。"""
    p = os.path.join(SCENE_DIR, f"{iid}.json")
    if not os.path.exists(p):
        return None
    meta = json.load(open(p, encoding="utf-8"))
    sc = meta.get("scene")
    if sc:
        return {"kind": "scene", "target": list(sc["target"]), "size": sc["size"],
                "distractors": [[list(q), s] for q, s in sc["distractors"]]}
    pt = meta.get("user_truth")
    if pt:
        return {"kind": "point", "point": list(pt)}
    return None


def _review(task, result):
    """开了自动审查且任务带真值 → 判定命中。返回 (result, hit)，hit=None 表示未判定。"""
    if not task or not task.get("auto") or not result:
        return result, None
    truth = task.get("truth")
    box = result.get("box")
    if not truth or not box:
        return result, None
    if truth.get("kind") == "point":
        u, v = truth["point"]
        hit = bool(box[0] <= u <= box[0] + box[2] and box[1] <= v <= box[1] + box[3])
    else:
        hit = bool(inside(tuple(box), tuple(truth.get("target") or (0, 0))))
    result["hit"] = hit
    result["truth"] = truth
    return result, (1 if hit else 0)


# ------------------------------------------------------------------ 本地执行（inline 模式）
def _execute(task: Dict, kind: str = "sim") -> None:
    tid = task["id"]
    t0 = time.time()
    steps: list = []
    with STATE["sem"]:
        try:
            if kind == "sim":
                sc = _scene_of(task["image_id"])
                if sc is None:
                    raise RuntimeError(
                        "sim 后端无法评判真实图片：它依赖已知目标位置的合成场景，"
                        "对照片只能瞎猜。请用「用示例图（合成场景）」演示，"
                        "或把服务切到 --backend vlm 接真 VLM。")
                from fovea_lite.world import Scene
                be = SimBackend(Scene(target=tuple(sc["target"]), size=sc["size"],
                                      distractors=[(tuple(p), s) for p, s in sc["distractors"]]),
                                seed=hash(task["id"]) & 0xFFFF)
            else:
                raw = open(os.path.join(IMG_DIR, f"{task['image_id']}.jpg"), "rb").read()
                be = VLBackend(raw, task.get("question", ""))

            def on_step(s):
                s["t"] = round(time.time() - t0, 2)
                steps.append(s)

            result = search.run(task.get("mode", "greedy"), tuple(task.get("seed") or [0.3, 0.3, 0.4, 0.4]),
                                be.probe, on_step)
            if isinstance(be, VLBackend):
                try:
                    a = be.answer(tuple(result["box"]))
                    result["answer"] = a.get("text", "")
                    result["answer_point"] = a.get("point")
                    result["answer_reason"] = a.get("reason", "")
                    result["answer_labels"] = a.get("labels", [])
                    result["answer_verified"] = a.get("verified")
                    if (a.get("text") or "").strip() and result.get("box"):
                        b = result["box"]
                        result["answer_point"] = [round(b[0] + b[2] / 2, 4), round(b[1] + b[3] / 2, 4)]
                        result["answer_point_method"] = "final_box"
                except Exception as e:  # noqa: BLE001
                    result["answer"] = ""
                    result["answer_error"] = f"{type(e).__name__}: {e}"
            if not result.get("answer_point"):
                t = _truth_of(task["image_id"])
                if t and t.get("kind") == "scene":
                    # 合成场景：真值就是答案，直接指到目标本身
                    result["answer_point"] = [round(t["target"][0], 4), round(t["target"][1], 4)]
                    result["answer_point_method"] = "scene_truth"
                elif result.get("box"):
                    b = result["box"]
                    result["answer_point"] = [round(b[0] + b[2] / 2, 4), round(b[1] + b[3] / 2, 4)]
                    result["answer_point_method"] = "final_box"
            result, hit = _review(task, result)
            db.complete(tid, steps, result, round(time.time() - t0, 2), hit=hit)
        except Exception as e:  # noqa: BLE001
            db.complete(tid, steps, None, round(time.time() - t0, 2), f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ 后台/会话辅助
def _me(handler, sess) -> Dict:
    role = sess.get("role", "guest")
    kid = sess.get("kid", "")
    nm = auth.name_of(STATE["keys"], kid)
    out = {"role": role, "kid": kid, "name": nm if (nm and role != "guest") else "访客",
           "expires_at": auth.SESSION_UNTIL if role != "guest" else ""}
    if role == "guest":
        gid = handler._guest_id()
        day = time.strftime("%Y-%m-%d")
        used = db.guest_used(gid, day) if gid else 0
        out.update({"quota_limit": auth.GUEST_DAILY_QUOTA, "quota_used": used,
                    "quota_left": max(0, auth.GUEST_DAILY_QUOTA - used)})
    return out


def _admin_status() -> Dict:
    st = {**db.stats(), **db.review_stats()}
    total = avail = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            mem = {}
            for line in f:
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0])
        total = mem.get("MemTotal", 0) // 1024
        avail = mem.get("MemAvailable", 0) // 1024
    except Exception:  # noqa: BLE001
        pass
    dtotal = dfree = 0.0
    try:
        sv = os.statvfs("/")
        dtotal = sv.f_blocks * sv.f_frsize / 2 ** 30
        dfree = sv.f_bavail * sv.f_frsize / 2 ** 30
    except Exception:  # noqa: BLE001
        pass
    up = 0.0
    try:
        up = float(open("/proc/uptime", encoding="utf-8").read().split()[0])
    except Exception:  # noqa: BLE001
        pass
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0
    return {
        **st,
        "mem_total_mb": total, "mem_used_mb": max(0, total - avail),
        "disk_total_gb": round(dtotal, 1), "disk_used_gb": round(dtotal - dfree, 1),
        "load": round(load, 2),
        "uptime_str": f"{int(up // 3600)}h{int(up % 3600 // 60)}m",
        "now": time.strftime("%Y-%m-%d %H:%M:%S"),
        "guest_today": db.guest_today(time.strftime("%Y-%m-%d")),
    }


def _admin_tasks(limit: int = 30) -> Dict:
    return {"tasks": db.recent(max(1, min(limit, 200)))}


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "fovea-lite/0.2"

    def log_message(self, fmt, *args):
        return

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for ck in getattr(self, "_pending_cookies", []):
            self.send_header("Set-Cookie", ck)
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: str, ctype: str):
        if not os.path.exists(path):
            return self._json({"error": "not found"}, 404)
        body = open(path, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for ck in getattr(self, "_pending_cookies", []):
            self.send_header("Set-Cookie", ck)
        self.end_headers()
        self.wfile.write(body)

    # ---------------- 会话 ----------------
    def _cookie(self, name: str, value: str, max_age: int = 604800):
        self._pending_cookies = getattr(self, "_pending_cookies", [])
        self._pending_cookies.append(f"{name}={value}; Path=/; Max-Age={max_age}; HttpOnly; SameSite=Lax")

    def _cookies(self) -> Dict[str, str]:
        return auth.parse_cookies(self.headers.get("Cookie"))

    def _session(self):
        ck = self._cookies()
        return auth.read_session(STATE["secret"], ck.get("fovea_sess", ""))

    def _guest_id(self) -> str:
        return self._cookies().get("fovea_guest", "")

    def _body(self) -> Dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def _auth(self, body: Dict) -> bool:
        return str(body.get("token", "")) == STATE["token"]

    # ---------------- GET ----------------
    def do_GET(self):
        self._pending_cookies = []
        p = urlparse(self.path).path
        sess = self._session()

        if p in ("/", "/index.html"):
            page = "index.html" if sess else "login.html"
            return self._file(os.path.join(STATIC, page), "text/html; charset=utf-8")
        if p in ("/login", "/login.html"):
            return self._file(os.path.join(STATIC, "login.html"), "text/html; charset=utf-8")
        if p in ("/admin", "/admin.html"):
            if not sess or sess.get("role") != "admin":
                return self._json({"error": "需要管理员密钥"}, 403)
            return self._file(os.path.join(STATIC, "admin.html"), "text/html; charset=utf-8")

        if p == "/api/health":
            return self._json({"ok": True, "backend": STATE["backend"], "mode": STATE["mode"],
                               "model": STATE["model"], **db.stats()})

        if p.startswith("/api/admin/"):
            if not sess or sess.get("role") != "admin":
                return self._json({"error": "需要管理员密钥"}, 403)
            if p == "/api/admin/status":
                return self._json(_admin_status())
            if p == "/api/admin/tasks":
                q = parse_qs(urlparse(self.path).query)
                return self._json(_admin_tasks(int((q.get("limit") or ["30"])[0])))

        if p.startswith("/api/image/"):
            # 用户走会话；worker 走 token（它要下载图片去调模型）
            q = parse_qs(urlparse(self.path).query)
            tok = (q.get("token") or [""])[0]
            if not sess and tok != STATE["token"]:
                return self._json({"error": "未登录", "need_login": True}, 401)
            return self._file(os.path.join(IMG_DIR, p.split("/")[-1] + ".jpg"), "image/jpeg")

        if not sess:
            return self._json({"error": "未登录", "need_login": True}, 401)

        if p == "/api/me":
            return self._json(_me(self, sess))
        if p == "/api/demo":
            q = parse_qs(urlparse(self.path).query)
            return self._json(_demo(force_new=bool(q.get("new"))))
        if p.startswith("/api/task/"):
            t = db.get(p.split("/")[-1])
            if not t:
                return self._json({"error": "no such task"}, 404)
            out = {k: t[k] for k in ("id", "status", "mode", "question", "image_id",
                                     "steps", "result", "elapsed", "error", "note",
                                     "auto", "hit", "truth")}
            if t["status"] == "running" and t.get("started"):
                out["elapsed"] = round(time.time() - t["started"], 1)
            return self._json(out)
        if p == "/api/stats":
            return self._json({**db.stats(), **db.review_stats()})
        self._json({"error": "not found"}, 404)

    # ---------------- POST ----------------
    def do_POST(self):
        self._pending_cookies = []
        p = urlparse(self.path).path
        body = self._body()

        if p == "/api/login":
            mode = str(body.get("mode") or "guest")
            if mode == "guest":
                gid = self._guest_id() or auth.new_guest_id()
                self._cookie("fovea_guest", gid, max_age=86400 * 365)
                self._cookie("fovea_sess", auth.make_session(STATE["secret"], "guest", gid))
                day = time.strftime("%Y-%m-%d")
                used = db.guest_used(gid, day)
                return self._json({"ok": True, "role": "guest",
                                   "quota_limit": auth.GUEST_DAILY_QUOTA,
                                   "quota_used": used,
                                   "quota_left": max(0, auth.GUEST_DAILY_QUOTA - used)})
            item = auth.find_key(STATE["keys"] or {}, body.get("key") or "")
            if not item:
                return self._json({"error": "密钥无效，请检查后重试"}, 403)
            tok = auth.make_session(STATE["secret"], item["role"], item["id"])
            self._cookie("fovea_sess", tok, max_age=auth.session_ttl())
            return self._json({"ok": True, "role": item["role"], "key_id": item["id"],
                               "name": item.get("name") or item["id"]})

        if p == "/api/logout":
            self._cookie("fovea_sess", "", max_age=0)
            return self._json({"ok": True})

        sess = self._session()
        WORKER_PATHS = ("/api/claim", "/api/complete", "/api/progress", "/api/heartbeat")
        if p not in WORKER_PATHS and not sess:
            return self._json({"error": "未登录，请先选择进入方式", "need_login": True}, 401)

        if p == "/api/upload":
            data = (body.get("data") or "").split(",")[-1]
            if not data:
                return self._json({"error": "缺少图片数据"}, 400)
            try:
                raw = base64.b64decode(data)
            except Exception:
                return self._json({"error": "base64 解码失败"}, 400)
            if len(raw) > 25 * 1024 * 1024:
                return self._json({"error": "图片过大"}, 413)
            iid, w, h = _save_image(raw)
            return self._json({"image_id": iid, "width": w, "height": h,
                               "truth": _truth_of(iid),
                               "note": "已上传；真实图片需 vlm 后端才能评判（sim 后端只支持合成场景）"})

        if p == "/api/truth":
            iid = body.get("image_id")
            if not iid or not os.path.exists(os.path.join(IMG_DIR, f"{iid}.jpg")):
                return self._json({"error": "image_id 无效"}, 400)
            pt = body.get("point")
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                return self._json({"error": "point 需为 [u, v]（归一化坐标）"}, 400)
            try:
                u, v = float(pt[0]), float(pt[1])
            except Exception:
                return self._json({"error": "point 必须是数字"}, 400)
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                return self._json({"error": "point 超出画面"}, 400)
            path = os.path.join(SCENE_DIR, f"{iid}.json")
            meta = {}
            if os.path.exists(path):
                meta = json.load(open(path, encoding="utf-8"))
            meta["user_truth"] = [u, v]
            json.dump(meta, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            return self._json({"ok": True, "truth": _truth_of(iid)})

        if p == "/api/query":
            iid = body.get("image_id")
            if not iid or not os.path.exists(os.path.join(IMG_DIR, f"{iid}.jpg")):
                return self._json({"error": "image_id 无效"}, 400)
            mode = body.get("mode", "greedy")
            if mode not in search.POLICIES:
                return self._json({"error": f"未知模式 {mode}"}, 400)
            if sess and sess.get("role") == "guest":
                gid = self._guest_id()
                if not gid:
                    return self._json({"error": "访客身份丢失，请重新进入", "need_login": True}, 401)
                day = time.strftime("%Y-%m-%d")
                allowed, used = db.guest_bump(gid, day, auth.GUEST_DAILY_QUOTA)
                if not allowed:
                    return self._json({
                        "error": f"访客每天最多识别 {auth.GUEST_DAILY_QUOTA} 张图片，今天的次数已用完",
                        "quota_limit": auth.GUEST_DAILY_QUOTA,
                        "quota_used": used}, 429)
            tid = uuid.uuid4().hex[:12]
            auto = bool(body.get("auto"))
            truth = _truth_of(iid) if auto else None
            actor = (auth.name_of(STATE["keys"], sess.get("kid", ""))
                     if sess and sess.get("role") != "guest" else "访客")
            db.create(tid, iid, body.get("question", ""), mode,
                      body.get("seed") or [0.3, 0.3, 0.4, 0.4], body.get("note", ""),
                      auto=auto, truth=truth, actor=actor)
            # 合成场景：自带可解析观测模型，本地直接算，不花 API、不用 worker
            local = _scene_of(iid) is not None
            if local or STATE["mode"] == "inline":
                kind = "sim" if local else STATE["backend"]
                threading.Thread(target=_execute, args=(db.get(tid), kind), daemon=True).start()
            return self._json({"task_id": tid, "mode": STATE["mode"], "auto": auto,
                               "truth_known": bool(truth), "local": local,
                               "backend": "sim" if local else STATE["backend"]})

        if p == "/api/claim":
            if not self._auth(body):
                return self._json({"error": "bad token"}, 403)
            t = db.claim(str(body.get("worker") or "w"), float(body.get("lease") or 300))
            if not t:
                return self._json({"id": None})
            out = {k: t[k] for k in ("id", "mode", "question", "image_id", "seed")}
            out["scene"] = _scene_of(t["image_id"])
            return self._json(out)

        if p == "/api/progress":
            if not self._auth(body):
                return self._json({"error": "bad token"}, 403)
            tid = body.get("task_id")
            if not tid or not db.get(tid):
                return self._json({"error": "no such task"}, 404)
            db.progress(tid, body.get("steps") or [], float(body.get("elapsed") or 0))
            db.heartbeat(tid, str(body.get("worker") or "w"), float(body.get("lease") or 300))
            return self._json({"ok": True})

        if p == "/api/complete":
            if not self._auth(body):
                return self._json({"error": "bad token"}, 403)
            tid = body.get("task_id")
            t = db.get(tid) if tid else None
            if not t:
                return self._json({"error": "no such task"}, 404)
            res, hit = _review(t, body.get("result"))
            db.complete(tid, body.get("steps") or [], res,
                        float(body.get("elapsed") or 0), body.get("error") or "", hit=hit)
            return self._json({"ok": True, "hit": hit})

        if p == "/api/heartbeat":
            if not self._auth(body):
                return self._json({"error": "bad token"}, 403)
            ok = db.heartbeat(body.get("task_id", ""), str(body.get("worker") or "w"),
                              float(body.get("lease") or 300))
            return self._json({"ok": ok})

        self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--backend", default="sim", choices=["sim", "vlm"])
    ap.add_argument("--mode", default="inline", choices=["inline", "queue"],
                    help="inline=服务自己算；queue=只排队，等 worker")
    ap.add_argument("--token", default=os.environ.get("WORKER_TOKEN", "dev"))
    ap.add_argument("--db", default=os.path.join(DATA, "tasks.db"))
    ap.add_argument("--keys", default=os.environ.get("FOVEA_KEYS", KEYS_PATH))
    args = ap.parse_args()

    STATE["backend"], STATE["mode"], STATE["token"] = args.backend, args.mode, args.token
    fresh = not os.path.exists(args.keys)
    STATE["keys"] = auth.load(args.keys)
    STATE["secret"] = STATE["secret"] or STATE["keys"].get("secret", "")
    if fresh:
        print("[fovea] 已生成访问密钥（只显示这一次）：")
        for k in STATE["keys"]["keys"]:
            print(f"        {k['key']}   {k['role']}   {k['id']}")
        print(f"        密钥文件：{args.keys}")
    db.init(args.db)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[fovea] http://{args.host}:{args.port}  backend={args.backend} mode={args.mode}")
    if args.mode == "queue":
        print("[fovea] queue 模式：请另外启动 worker（python -m fovea_service.worker --token ...）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[fovea] bye")


if __name__ == "__main__":
    main()
