# -*- coding: utf-8 -*-
"""fovea_service.worker —— 主动外连云端的计算进程（本地/租卡都跑它）。

它只做三件事：领任务 → 本地算 → 回传结果。
因此：云端不需要公网可达、不需要常开 GPU、你自己电脑在不开防火墙的情况下也能干活。

用法（两种后端）：
    # 合成场景：不用 GPU，联调用
    python -m fovea_service.worker --server http://127.0.0.1:8765 --token dev

    # 真 VLM：先把 vLLM 起在 8000
    set FOVEA_BASE_URL=http://127.0.0.1:8000/v1
    set FOVEA_MODEL=Qwen/Qwen3-VL-8B-Instruct
    python -m fovea_service.worker --server http://127.0.0.1:8765 --token dev --backend vlm
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import atexit
import io
import json
import sys
import os
import random
import time
import urllib.request
from typing import Dict, Optional

from . import search
from .backends import SimBackend, VLBackend, annotate as _annotate

LOCK_PATH = os.path.join(os.path.expanduser("~"), ".fovea_worker.pid")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(h)
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _release_lock():
    try:
        if os.path.exists(LOCK_PATH):
            with open(LOCK_PATH, encoding="utf-8") as f:
                if int((f.read().strip() or 0)) == os.getpid():
                    os.remove(LOCK_PATH)
    except Exception:  # noqa: BLE001
        pass


def acquire_singleton() -> bool:
    """同一台机器只允许一个 worker，避免出现多个僵尸进程抢任务。"""
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH, encoding="utf-8") as f:
                pid = int((f.read().strip() or 0))
        except Exception:  # noqa: BLE001
            pid = 0
        if pid and pid != os.getpid() and _pid_alive(pid):
            print(f"[worker] 已有 worker 在运行（PID {pid}），本进程退出。")
            print(f"[worker] 需要重启请先结束它：taskkill /PID {pid} /F")
            return False
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    atexit.register(_release_lock)
    return True


def _req(url: str, payload: Optional[Dict] = None, timeout: int = 120):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _backend_for(task: Dict, server: str, kind: str, question: str, token: str = ""):
    if kind == "sim":
        sc = task.get("scene")
        if not sc:
            raise RuntimeError("该任务没有合成场景（用户上传的图），sim 后端无法评判；请用 --backend vlm")
        from fovea_lite.world import Scene
        scene = Scene(target=tuple(sc["target"]), size=sc["size"],
                      distractors=[(tuple(p), s) for p, s in sc["distractors"]])
        return SimBackend(scene, seed=hash(task["id"]) & 0xFFFF)
    raw = _download(f"{server}/api/image/{task['image_id']}?token={token}")
    return VLBackend(raw, question)


def claim_task(server: str, token: str, worker: str, lease: float) -> Optional[Dict]:
    """只领任务，不处理（并发体现在这里：多个线程可以同时领）。"""
    task = _req(f"{server}/api/claim", {"worker": worker, "token": token, "lease": lease})
    return task if task.get("id") else None


def process_task(task: Dict, server: str, token: str, kind: str, worker: str, lease: float) -> None:
    tid, mode, question = task["id"], task.get("mode", "greedy"), task.get("question", "")
    t0 = time.time()
    steps: list = []
    try:
        be = _backend_for(task, server, kind, question, token)

        def probe(box):
            return be.probe(box)

        def on_step(s):
            s["t"] = round(time.time() - t0, 2)
            steps.append(s)
            # 实时上报，前端才能看到"已探测 N 次 / 已用 X 秒"
            try:
                r = _req(f"{server}/api/progress", {"task_id": tid, "token": token, "worker": worker,
                                                   "lease": lease, "steps": steps,
                                                   "elapsed": round(time.time() - t0, 2)})
                if isinstance(r, dict) and r.get("cancelled"):
                    raise RuntimeError("cancelled by user")   # 用户点了重置，立即中止
            except RuntimeError:
                raise
            except Exception:
                pass

        result = search.run(mode, tuple(task.get("seed") or [0.3, 0.3, 0.4, 0.4]), probe, on_step)
        if isinstance(be, SimBackend):
            result["hit"] = be.answer_hit(tuple(result["box"]))
        elif isinstance(be, VLBackend):
            # 探针只回答 Yes/No，最终答案要单独再问一次
            try:
                a = be.answer(tuple(result["box"]))
                result["answer"] = a.get("text", "")
                result["answer_point"] = a.get("point")
                result["answer_reason"] = a.get("reason", "")
                result["answer_labels"] = a.get("labels", [])
                result["answer_verified"] = a.get("verified")
                if (a.get("text") or "").strip() and result.get("box"):
                    b = result["box"]
                    # 先在该框内部做一次局部定位（空间小、更容易指对），失败才退回框心
                    p2 = be.locate_in_box(a["text"].strip(), tuple(b))
                    if p2:
                        result["answer_point"] = p2
                        result["answer_point_method"] = "crop_locate"
                    else:
                        result["answer_point"] = [round(b[0] + b[2] / 2, 4), round(b[1] + b[3] / 2, 4)]
                        result["answer_point_method"] = "final_box"
            except Exception as e:  # noqa: BLE001
                result["answer"] = ""
                result["answer_error"] = f"{type(e).__name__}: {e}"
        # 标注：命中就指真值本身；未命中不给箭头；无真值才指框心
        _annotate(result)
        _req(f"{server}/api/complete", {"task_id": tid, "token": token, "steps": steps,
                                        "result": result, "elapsed": round(time.time() - t0, 2)})
        print(f"[worker] done {tid} mode={mode} steps={len(steps)} {round(time.time()-t0,2)}s")
    except Exception as e:  # noqa: BLE001
        _req(f"{server}/api/complete", {"task_id": tid, "token": token, "steps": steps,
                                        "result": None, "elapsed": round(time.time() - t0, 2),
                                        "error": f"{type(e).__name__}: {e}"})
        print(f"[worker] error {tid}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--token", default=os.environ.get("WORKER_TOKEN", "dev"))
    ap.add_argument("--backend", default="sim", choices=["sim", "vlm"])
    ap.add_argument("--worker", default=f"w-{os.uname().nodename if hasattr(os,'uname') else 'pc'}-{os.getpid()}")
    ap.add_argument("--lease", type=float, default=300.0)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--concurrency", type=int, default=2, help="同时跑几个任务（长任务不再互相堵住）")
    ap.add_argument("--once", action="store_true", help="只领一个任务就退出")
    args = ap.parse_args()

    if not acquire_singleton():
        return

    n = max(1, args.concurrency)
    print(f"[worker] server={args.server} backend={args.backend} worker={args.worker} "
          f"pid={os.getpid()} concurrency={n}")
    print(f"[worker] 单实例锁：{LOCK_PATH}")

    pool = ThreadPoolExecutor(max_workers=n)
    futures = set()
    auth_fail = 0
    try:
        while True:
            futures = {f for f in futures if not f.done()}
            got_any = False
            while len(futures) < n:                     # 有空位就继续领
                try:
                    task = claim_task(args.server, args.token, args.worker, args.lease)
                    auth_fail = 0
                except Exception as e:  # noqa: BLE001
                    code = getattr(e, "code", 0)
                    if code in (401, 403):
                        auth_fail += 1
                        print(f"[worker] 认证失败（{code}），连续 {auth_fail} 次：{e}")
                        if auth_fail >= 3:
                            print("[worker] token 可能已失效，退出以避免僵尸进程。")
                            sys.exit(66)                # 66：让看门狗别再重启
                    else:
                        print(f"[worker] poll error: {e}")
                    break
                if not task:
                    break
                got_any = True
                futures.add(pool.submit(process_task, task, args.server, args.token,
                                        args.backend, args.worker, args.lease))
            if args.once and (futures or got_any):
                for f in list(futures):
                    f.result()
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[worker] bye")
    finally:
        pool.shutdown(wait=True)
        _release_lock()

if __name__ == "__main__":
    main()
