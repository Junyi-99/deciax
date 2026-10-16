"""Compare trace contents and complete device intervals across TPU trace modes."""

import argparse
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path


def overlap_union(events, start, end):
    intervals = sorted((max(start, e["ts"]), min(end, e["ts"] + e["dur"])) for e in events if e["ts"] < end and e["ts"] + e["dur"] > start)
    total = 0.0
    cursor = start
    for a, b in intervals:
        total += max(0.0, b - max(a, cursor))
        cursor = max(cursor, b)
    return total


def inspect(path, backend, steps):
    events = json.loads(gzip.decompress(path.read_bytes()))["traceEvents"]
    tracks = {(e["pid"], e["tid"]): e.get("args", {}).get("name") for e in events if e.get("name") == "thread_name"}

    def track(e):
        return tracks.get((e.get("pid"), e.get("tid")), "")

    modules = sorted([e for e in events if e.get("ph") == "X" and track(e) == "XLA Modules"], key=lambda e: e["ts"])
    sync = [e for e in events if e.get("ph") == "X" and track(e) == "Tensor Core Sync Flag"]
    waits = [e for e in sync if e["name"].startswith("SyncWait:")]
    scopes = [e for e in events if e.get("ph") == "X" and e.get("name") == "resident_forward"]
    measured = []
    if backend == "vmem":
        assert len(modules) == steps and len(scopes) == steps * 9, (len(modules), len(scopes))
        for m in modules:
            inner = sorted([e for e in scopes if m["ts"] <= e["ts"] and e["ts"] + e["dur"] <= m["ts"] + m["dur"]], key=lambda e: e["ts"])
            assert len(inner) == 9
            measured.extend(inner[1:])
    else:
        assert len(modules) == steps * 8, len(modules)
        measured = modules
    samples = [e["dur"] for e in measured]
    return {
        "events": len(events),
        "tracks": list(tracks.values()),
        "track_event_counts": dict(Counter(track(e) for e in events if e.get("ph") == "X")),
        "module_count": len(modules),
        "resident_scope_count": len(scopes),
        "measured_count": len(samples),
        "p50_us": statistics.median(samples),
        "samples_us": samples,
        "sync_events": len(sync),
        "sync_wait_events": len(waits),
        "sync_names": dict(Counter(e["name"] for e in sync)),
        "sync_wait_overlap_measured_us": sum(overlap_union(waits, e["ts"], e["ts"] + e["dur"]) for e in measured),
        "trace_gzip_bytes": path.stat().st_size,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--steps", type=int, default=3)
    a = p.parse_args()
    result = {
        mode: {backend: inspect(a.root / mode / backend / "trace.json.gz", backend, a.steps) for backend in ("jax", "pallas", "vmem")}
        for mode in ("TRACE_COMPUTE", "TRACE_COMPUTE_AND_SYNC")
    }
    (a.root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                mode: {
                    b: {k: v for k, v in row.items() if k in ("p50_us", "sync_events", "module_count", "resident_scope_count", "sync_wait_overlap_measured_us")}
                    for b, row in rows.items()
                }
                for mode, rows in result.items()
            },
            indent=2,
        )
    )
