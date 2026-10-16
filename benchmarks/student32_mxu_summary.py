"""Instruction-lane coverage over complete kernels and warm resident forwards."""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

from student32_llo_summary import coverage, merge


def summarize(root):
    paths = list(root.rglob("trace.json.gz"))
    assert len(paths) == 1, paths
    events = json.loads(gzip.decompress(paths[0].read_bytes()))["traceEvents"]
    tracks = {(e["pid"], e["tid"]): e.get("args", {}).get("name") for e in events if e.get("name") == "thread_name"}
    grouped = defaultdict(list)
    for e in events:
        if e.get("ph") == "X":
            grouped[tracks.get((e.get("pid"), e.get("tid")), "unknown")].append(e)
    modules = sorted(grouped["XLA Modules"], key=lambda e: e["ts"])
    forwards = [e for e in grouped["XLA TraceMe"] if e["name"] == "resident_forward"]
    warm = []
    for m in modules:
        fs = sorted([e for e in forwards if m["ts"] <= e["ts"] and e["ts"] + e["dur"] <= m["ts"] + m["dur"]], key=lambda e: e["ts"])
        assert len(fs) == 9, len(fs)
        warm.extend(fs[1:])
    assert len(modules) == 3 and len(forwards) == 27 and len(warm) == 24
    units = {k: merge([(e["ts"], e["ts"] + e["dur"]) for e in es]) for k, es in grouped.items() if k.endswith(" Instructions")}
    assert "MXU0 Instructions" in units and "MXU1 Instructions" in units
    mx0, mx1 = units["MXU0 Instructions"], units["MXU1 Instructions"]
    either = merge(mx0 + mx1)
    waits = merge([(e["ts"], e["ts"] + e["dur"]) for e in grouped["Tensor Core Sync Flag"] if e["name"].startswith("SyncWait:")])
    result = {
        "interpretation": "LLO instruction intervals are interpolated estimates, not hardware busy counters. Mean per-MXU coverage and either-MXU coverage are distinct; neither is MFU.",
        "module_count": len(modules),
        "forward_count": len(forwards),
        "warm_count": len(warm),
        "scopes": {},
    }
    for name, es in [("warm_forwards", warm), ("all_forwards", forwards), ("complete_kernels", modules)]:
        windows = sorted([(e["ts"], e["ts"] + e["dur"]) for e in es])
        duration = sum(b - a for a, b in windows)
        per_unit = {k: coverage(spans, windows) for k, spans in units.items()}
        c0, c1 = per_unit["MXU0 Instructions"], per_unit["MXU1 Instructions"]
        cu = coverage(either, windows)
        result["scopes"][name] = {
            "count": len(es),
            "duration_total_us": duration,
            "duration_p50_us": statistics.median(e["dur"] for e in es),
            "unit_coverage_fraction": {k: v / duration for k, v in per_unit.items()},
            "mxu_mean_coverage_fraction": (c0 + c1) / (2 * duration),
            "mxu_either_coverage_fraction": cu / duration,
            "mxu_both_coverage_fraction": (c0 + c1 - cu) / duration,
            "mxu_neither_coverage_fraction": 1 - cu / duration,
            "sync_wait_overlap_us": coverage(waits, windows),
        }
    opcode_spans = defaultdict(list)
    for event in grouped["XLU Instructions"]:
        opcode_spans[event["name"]].append((event["ts"], event["ts"] + event["dur"]))
    warm_windows = sorted((e["ts"], e["ts"] + e["dur"]) for e in warm)
    result["xlu_opcode_coverage_mean_forward_us"] = {name: coverage(spans, warm_windows) / len(warm) for name, spans in opcode_spans.items()}
    result["warm_samples_us"] = [e["dur"] for e in warm]
    (root / "mxu-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
