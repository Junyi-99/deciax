"""Summarize fresh-process custom-call traces; LLO spans are interpolated estimates."""

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def coverage(spans, windows):
    spans = merge(spans)
    total = 0.0
    i = 0
    for a, b in windows:
        while i < len(spans) and spans[i][1] <= a:
            i += 1
        j = i
        while j < len(spans) and spans[j][0] < b:
            total += max(0.0, min(b, spans[j][1]) - max(a, spans[j][0]))
            j += 1
    return total


def inspect(root, mode):
    path = next(root.rglob("trace.json.gz"))
    events = json.loads(gzip.decompress(path.read_bytes()))["traceEvents"]
    tracks = {(e["pid"], e["tid"]): e.get("args", {}).get("name") for e in events if e.get("name") == "thread_name"}
    grouped = defaultdict(list)
    for e in events:
        if e.get("ph") == "X":
            grouped[tracks.get((e.get("pid"), e.get("tid")), "unknown")].append(e)
    modules = sorted(grouped["XLA Modules"], key=lambda e: e["ts"])
    scopes = [e for e in grouped["XLA TraceMe"] if e["name"] == "resident_forward"]
    measured = []
    for m in modules:
        inner = sorted([e for e in scopes if m["ts"] <= e["ts"] and e["ts"] + e["dur"] <= m["ts"] + m["dur"]], key=lambda e: e["ts"])
        assert len(inner) == 9, len(inner)
        measured.extend(inner[1:])
    assert len(modules) == 3 and len(measured) == 24
    windows = [(e["ts"], e["ts"] + e["dur"]) for e in measured]
    duration = sum(b - a for a, b in windows)
    unit = {}
    for name, es in grouped.items():
        if name.endswith(" Instructions"):
            covered = coverage([(e["ts"], e["ts"] + e["dur"]) for e in es], windows)
            unit[name] = {
                "trace_events": len(es),
                "estimated_covered_us": covered,
                "estimated_region_coverage_fraction": covered / duration,
                "opcode_examples": dict(Counter(e["name"] for e in es).most_common(8)),
            }
    waits = [e for e in grouped.get("Tensor Core Sync Flag", []) if e["name"].startswith("SyncWait:")]
    result = {
        "mode": mode,
        "events": len(events),
        "track_event_counts": {k: len(v) for k, v in grouped.items()},
        "module_count": len(modules),
        "resident_scope_count": len(scopes),
        "measured_count": len(measured),
        "p50_us": statistics.median(e["dur"] for e in measured),
        "samples_us": [e["dur"] for e in measured],
        "units": unit,
        "interpretation": "Per-unit instruction spans are compiler-cost interpolation between trace markers, not sampled hardware busy counters. Coverage fractions across units overlap and must not be added.",
        "sync_events": len(grouped.get("Tensor Core Sync Flag", [])),
        "sync_wait_events": len(waits),
        "sync_wait_overlap_measured_us": coverage([(e["ts"], e["ts"] + e["dur"]) for e in waits], windows),
        "trace_gzip_bytes": path.stat().st_size,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    pointer = json.loads((root / "llo-analysis.json").read_text())
    if pointer.get("status") == "SAVED_TO_FILE":
        llo = json.loads(Path(pointer["file_path"]).read_text())
    else:
        llo = pointer
    assert llo.get("success"), llo
    small = {k: v for k, v in llo.items() if k not in ("modules", "source_map_by_kernel")}
    small["student_modules"] = [{k: v for k, v in m.items() if k != "source_map"} for m in llo["modules"] if "student32" in m["kernel_name"]]
    (root / "llo-summary.json").write_text(json.dumps(small, indent=2) + "\n")
    print(mode, result["p50_us"], result["sync_events"], {k: v["estimated_region_coverage_fraction"] for k, v in unit.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    for mode in ("TRACE_COMPUTE", "TRACE_COMPUTE_AND_SYNC"):
        inspect(a.root / mode, mode)
