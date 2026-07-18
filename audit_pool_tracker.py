#!/usr/bin/env python3
"""Track concurrent capacity-audit lease demand on the worker pool (Balancer 2).

Sizing goal: how many audit workers do you need? Each worker handles ONE audit at
a time (max_active=1), and an audit holds its worker from warm-up (B_select)
through the proof (B_proof) — roughly 1-2 minutes. So the worker count you need =
the PEAK number of simultaneous audits your proxies ever demand.

This polls Balancer 2's /v1/workers and, per GPU class, records:
  - current concurrent audits  = leased + busy
  - PEAK concurrent seen        (with the timestamp it happened)
  - saturation moments          = polls where idle == 0 (pool fully used)

Reading the result:
  * If saturation NEVER happens, PEAK concurrent = your true peak demand →
    workers needed = PEAK (add +1 margin for safety).
  * If saturation DOES happen (idle hit 0), demand may have EXCEEDED the pool at
    that moment (some proxies got 503 "no worker available"). PEAK is then a
    LOWER bound — add workers and keep measuring until saturation stops, and the
    peak you then see is the real number.

To measure true peak: temporarily over-provision workers so idle never reaches 0.

Usage
-----
    export CAPACITY_AUDIT_BALANCER_API_KEY=...    # or ~/.audit_balancer_key
    ./audit_pool_tracker.py --balancer http://65.108.102.231:8081
    ./audit_pool_tracker.py --balancer ... --interval 3 --state ~/.audit_peak.json
    ./audit_pool_tracker.py --balancer ... --once     # one snapshot, no loop
    ./audit_pool_tracker.py --show   --state ~/.audit_peak.json   # print peak, exit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone


def _key() -> str:
    k = os.environ.get("CAPACITY_AUDIT_BALANCER_API_KEY", "").strip()
    if k:
        return k
    for path in (os.path.expanduser("~/.audit_balancer_key"), os.path.expanduser("~/.vast_api_key")):
        if os.path.exists(path):
            with open(path) as f:
                v = f.read().strip()
                if v:
                    return v
    return ""


def fetch(balancer: str, key: str, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(
        f"{balancer.rstrip('/')}/v1/workers",
        headers={"Accept": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r) or {}


def snapshot(data: dict) -> dict:
    """Normalize one /v1/workers response into per-class concurrency."""
    classes = {}
    for cls, v in (data.get("byGpuClass") or {}).items():
        total = int(v.get("total") or 0)
        leased = int(v.get("leased") or 0)
        busy = int(v.get("busy") or 0)
        idle = int(v.get("idle") if v.get("idle") is not None else total - leased - busy)
        classes[cls] = {
            "total": total,
            "concurrent": leased + busy,   # audits in progress
            "idle": idle,
            "saturated": idle <= 0 and total > 0,
        }
    return {
        "ts": time.time(),
        "workerCount": int(data.get("workerCount") or 0),
        "leaseCount": int(data.get("leaseCount") or 0),
        "classes": classes,
    }


def load_state(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"started_at": time.time(), "polls": 0, "peak": {}, "overall_peak": 0,
            "overall_peak_at": 0.0, "saturation_polls": 0}


def save_state(path: str, state: dict) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def update_peak(state: dict, snap: dict) -> None:
    state["polls"] = int(state.get("polls", 0)) + 1
    overall = 0
    any_sat = False
    for cls, c in snap["classes"].items():
        overall += c["concurrent"]
        if c["saturated"]:
            any_sat = True
        rec = state["peak"].setdefault(cls, {"peak": 0, "peak_at": 0.0, "workers_seen": 0, "saturation_polls": 0})
        rec["workers_seen"] = max(rec["workers_seen"], c["total"])
        if c["saturated"]:
            rec["saturation_polls"] = int(rec.get("saturation_polls", 0)) + 1
        if c["concurrent"] > rec["peak"]:
            rec["peak"] = c["concurrent"]
            rec["peak_at"] = snap["ts"]
    if any_sat:
        state["saturation_polls"] = int(state.get("saturation_polls", 0)) + 1
    if overall > int(state.get("overall_peak", 0)):
        state["overall_peak"] = overall
        state["overall_peak_at"] = snap["ts"]


def _iso(ts: float) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M:%S")


def print_summary(state: dict) -> None:
    dur_h = (time.time() - float(state.get("started_at", time.time()))) / 3600.0
    print(f"\n=== PEAK concurrent audit demand (over {dur_h:.1f}h, {state.get('polls',0)} polls) ===")
    if not state.get("peak"):
        print("  (no data yet)")
    for cls, rec in sorted(state["peak"].items()):
        sat = rec.get("saturation_polls", 0)
        flag = f"  ⚠ SATURATED {sat}x (peak is a LOWER bound — add workers)" if sat else "  (never saturated → peak = true demand)"
        rec_workers = rec.get("workers_seen", 0)
        need = rec["peak"] + (1 if sat else 0)
        print(f"  {cls}")
        print(f"    peak concurrent = {rec['peak']}   at {_iso(rec['peak_at'])}   "
              f"(workers available then: {rec_workers})")
        print(f"    -> workers needed >= {max(need, rec['peak'])}{flag}")
    print(f"  overall peak (all classes) = {state.get('overall_peak',0)} at {_iso(state.get('overall_peak_at',0))}")
    if state.get("saturation_polls"):
        print(f"  ⚠ pool was fully saturated on {state['saturation_polls']} poll(s) — true peak likely higher.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Track peak concurrent capacity-audit demand for worker sizing")
    ap.add_argument("--balancer", default=os.environ.get("CAPACITY_AUDIT_BALANCER_URL", "http://65.108.102.231:8081"))
    ap.add_argument("--interval", type=float, default=3.0, help="poll seconds (default 3; leases last ~1-2min)")
    ap.add_argument("--state", default=os.path.expanduser("~/.audit_peak.json"), help="peak state file (persists)")
    ap.add_argument("--once", action="store_true", help="one snapshot then exit")
    ap.add_argument("--show", action="store_true", help="print recorded peak from --state and exit")
    ap.add_argument("--reset", action="store_true", help="reset the peak state file")
    args = ap.parse_args()

    if args.reset and args.state and os.path.exists(args.state):
        os.remove(args.state)
        print(f"reset {args.state}")
        return
    if args.show:
        print_summary(load_state(args.state))
        return

    key = _key()
    state = load_state(args.state)

    def poll_once():
        data = fetch(args.balancer, key)
        snap = snapshot(data)
        update_peak(state, snap)
        save_state(args.state, state)
        cur = " ".join(f"{c}={v['concurrent']}/{v['total']}{'!' if v['saturated'] else ''}"
                       for c, v in snap["classes"].items()) or "(no workers)"
        peaks = " ".join(f"{c}:{r['peak']}" for c, r in state["peak"].items())
        print(f"{_iso(snap['ts'])}  now[{cur}]  leases={snap['leaseCount']}  peak[{peaks}]")

    if args.once:
        poll_once()
        print_summary(state)
        return

    print(f"tracking {args.balancer} every {args.interval:g}s -> {args.state}  (Ctrl-C to stop)")
    try:
        while True:
            try:
                poll_once()
            except Exception as exc:
                print(f"{_iso(time.time())}  poll error: {exc}", file=sys.stderr)
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        print_summary(state)


if __name__ == "__main__":
    main()
