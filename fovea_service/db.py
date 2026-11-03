# -*- coding: utf-8 -*-
"""fovea_service.db —— 任务表（SQLite，标准库）。任务必须落盘，否则重启就丢。

状态机：queued → running（带 lease）→ done / error；lease 过期自动回收为 queued。
自动审查：auto=1 的任务才会写入 hit（NULL 表示未判定），统计只认这些。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

LOCK = threading.Lock()
CONN: Optional[sqlite3.Connection] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  image_id TEXT NOT NULL,
  question TEXT DEFAULT '',
  mode TEXT DEFAULT 'greedy',
  seed TEXT DEFAULT '[0.3,0.3,0.4,0.4]',
  steps TEXT DEFAULT '[]',
  result TEXT DEFAULT '',
  error TEXT DEFAULT '',
  note TEXT DEFAULT '',
  created REAL NOT NULL,
  started REAL DEFAULT 0,
  elapsed REAL DEFAULT 0,
  worker TEXT DEFAULT '',
  lease_until REAL DEFAULT 0,
  attempts INTEGER DEFAULT 0,
  auto INTEGER DEFAULT 0,
  truth TEXT DEFAULT '',
  hit INTEGER DEFAULT NULL,
  actor TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_status ON tasks(status, created);
CREATE TABLE IF NOT EXISTS guest_usage (
  guest TEXT NOT NULL,
  day TEXT NOT NULL,
  n INTEGER DEFAULT 0,
  PRIMARY KEY (guest, day)
);
"""

# 老库补列
MIGRATIONS = [
    ("auto", "ALTER TABLE tasks ADD COLUMN auto INTEGER DEFAULT 0"),
    ("truth", "ALTER TABLE tasks ADD COLUMN truth TEXT DEFAULT ''"),
    ("hit", "ALTER TABLE tasks ADD COLUMN hit INTEGER DEFAULT NULL"),
    ("actor", "ALTER TABLE tasks ADD COLUMN actor TEXT DEFAULT ''"),
]


def init(path: str) -> None:
    global CONN
    os.makedirs(os.path.dirname(path), exist_ok=True)
    CONN = sqlite3.connect(path, check_same_thread=False)
    CONN.row_factory = sqlite3.Row
    with LOCK:
        CONN.executescript(SCHEMA)
        cols = {r["name"] for r in CONN.execute("PRAGMA table_info(tasks)").fetchall()}
        for name, sql in MIGRATIONS:
            if name not in cols:
                CONN.execute(sql)
        CONN.commit()


def _row(r) -> Dict:
    d = dict(r)
    d["seed"] = json.loads(d.get("seed") or "[0.3,0.3,0.4,0.4]")
    d["steps"] = json.loads(d.get("steps") or "[]")
    d["result"] = json.loads(d["result"]) if d.get("result") else None
    d["truth"] = json.loads(d["truth"]) if d.get("truth") else None
    return d


def create(tid: str, image_id: str, question: str, mode: str, seed: List[float],
           note: str = "", auto: bool = False, truth: Optional[Dict] = None,
           actor: str = "") -> None:
    with LOCK:
        CONN.execute(
            "INSERT INTO tasks(id,status,image_id,question,mode,seed,note,created,auto,truth,actor) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (tid, "queued", image_id, question, mode, json.dumps(seed), note, time.time(),
             1 if auto else 0, json.dumps(truth, ensure_ascii=False) if truth else "", actor))
        CONN.commit()


def get(tid: str) -> Optional[Dict]:
    with LOCK:
        r = CONN.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
    return _row(r) if r else None


def claim(worker: str, lease: float = 300.0) -> Optional[Dict]:
    now = time.time()
    with LOCK:
        # 先回收超时任务
        CONN.execute("UPDATE tasks SET status='queued', worker='' WHERE status='running' AND lease_until<?",
                     (now,))
        r = CONN.execute("SELECT * FROM tasks WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if r is None:
            CONN.commit()
            return None
        CONN.execute("UPDATE tasks SET status='running', worker=?, started=?, lease_until=?, attempts=attempts+1 "
                     "WHERE id=?", (worker, now, now + lease, r["id"]))
        CONN.commit()
        r2 = CONN.execute("SELECT * FROM tasks WHERE id=?", (r["id"],)).fetchone()
    return _row(r2)


def heartbeat(tid: str, worker: str, lease: float = 300.0) -> bool:
    with LOCK:
        cur = CONN.execute("UPDATE tasks SET lease_until=? WHERE id=? AND worker=? AND status='running'",
                           (time.time() + lease, tid, worker))
        CONN.commit()
        return cur.rowcount > 0


def progress(tid: str, steps: List[Dict], elapsed: float) -> None:
    """worker 中途上报进度：让前端能看到实时步数与耗时。"""
    with LOCK:
        CONN.execute("UPDATE tasks SET steps=?, elapsed=? WHERE id=?",
                     (json.dumps(steps, ensure_ascii=False), elapsed, tid))
        CONN.commit()


def complete(tid: str, steps: List[Dict], result: Optional[Dict], elapsed: float,
             error: str = "", hit: Optional[int] = None) -> None:
    with LOCK:
        CONN.execute(
            "UPDATE tasks SET status=?, steps=?, result=?, elapsed=?, error=?, lease_until=0, "
            "hit=COALESCE(?, hit) WHERE id=?",
            ("error" if error else "done", json.dumps(steps, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False) if result else "", elapsed, error, hit, tid))
        CONN.commit()


def stats() -> Dict:
    with LOCK:
        rows = CONN.execute("SELECT status, COUNT(*) c FROM tasks GROUP BY status").fetchall()
        last = CONN.execute("SELECT worker, MAX(started) t FROM tasks WHERE worker!='' GROUP BY worker").fetchall()
    cutoff = time.time() - 1800          # 30 分钟没干过活的 worker 不算在线
    active = {r["worker"]: r["t"] for r in last if (r["t"] or 0) > cutoff}
    return {"counts": {r["status"]: r["c"] for r in rows},
            "workers": active}


def review_stats() -> Dict:
    """只统计开了自动审查、且已判定的任务。"""
    with LOCK:
        rows = CONN.execute(
            "SELECT mode, COUNT(*) c, SUM(hit) h, AVG(elapsed) e "
            "FROM tasks WHERE auto=1 AND hit IS NOT NULL GROUP BY mode").fetchall()
        tot = CONN.execute(
            "SELECT COUNT(*) c, SUM(hit) h FROM tasks WHERE auto=1 AND hit IS NOT NULL").fetchone()
    by_mode = {}
    for r in rows:
        c = r["c"] or 0
        h = int(r["h"] or 0)
        by_mode[r["mode"]] = {"judged": c, "hits": h,
                              "rate": round(h / c, 3) if c else 0.0,
                              "avg_elapsed": round(r["e"] or 0, 1)}
    c = tot["c"] or 0
    h = int(tot["h"] or 0)
    return {"review": {"judged": c, "hits": h,
                       "rate": round(h / c, 3) if c else 0.0,
                       "by_mode": by_mode}}


# ------------------------------------------------------------------ 访客配额
def guest_used(guest: str, day: str) -> int:
    with LOCK:
        r = CONN.execute("SELECT n FROM guest_usage WHERE guest=? AND day=?", (guest, day)).fetchone()
    return int(r["n"]) if r else 0


def guest_bump(guest: str, day: str, limit: int) -> Tuple[bool, int]:
    """占一次额度。返回 (是否允许, 已用次数)。"""
    with LOCK:
        r = CONN.execute("SELECT n FROM guest_usage WHERE guest=? AND day=?", (guest, day)).fetchone()
        n = int(r["n"]) if r else 0
        if n >= limit:
            return False, n
        CONN.execute(
            "INSERT INTO guest_usage(guest,day,n) VALUES(?,?,1) "
            "ON CONFLICT(guest,day) DO UPDATE SET n=n+1", (guest, day))
        CONN.commit()
    return True, n + 1


def guest_today(day: str) -> int:
    with LOCK:
        r = CONN.execute("SELECT COALESCE(SUM(n),0) s FROM guest_usage WHERE day=?", (day,)).fetchone()
    return int(r["s"] or 0)


def recent(limit: int = 30) -> List[Dict]:
    with LOCK:
        rows = CONN.execute("SELECT * FROM tasks ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = _row(r)
        out.append({
            "id": d["id"], "status": d["status"], "mode": d["mode"],
            "question": d.get("question") or "", "elapsed": d.get("elapsed") or 0,
            "hit": d.get("hit"), "auto": d.get("auto"),
            "actor": d.get("actor") or "",
            "answer": ((d.get("result") or {}) or {}).get("answer") or "",
            "time": time.strftime("%m-%d %H:%M", time.localtime(d.get("created") or 0)),
        })
    return out
