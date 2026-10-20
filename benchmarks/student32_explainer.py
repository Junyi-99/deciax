"""Build a self-contained HTML model guide with its measured XProf trace embedded."""

import argparse
import base64
import gzip
import hashlib
import heapq
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from student32_llo_summary import coverage


def perfetto_lanes(events, tracks, root):
    """Separate overlapping XProf intervals without changing their timestamps.

    Perfetto JSON thread slices must be disjoint or strictly nested. XProf's
    tracks can partially overlap, so events are spread over parallel display
    lanes. The assignment is nesting-aware: an event stays in a lane when it
    is fully inside that lane's innermost open interval (a child scope under
    its parent) or the lane is idle; only genuinely interleaved events move
    to another lane. Containment is checked with Perfetto's own rules
    (src/trace_processor/importers/json): ts and dur are rounded to integer
    nanoseconds separately with llround, end = ts + dur, a slice is popped once
    end <= new ts, and a slice fits when its end is within the open slice.
    Events are emitted in (ts, -dur) order because Perfetto breaks timestamp
    ties by file order.
    """

    def ns(value):
        return int(value) * 1000 if float(value).is_integer() else math.floor(value * 1000.0 + 0.5)

    grouped = defaultdict(list)
    output = [e for e in events if e.get("ph") == "M" and not e.get("name", "").startswith("thread_")]
    for event in events:
        if event.get("ph") == "X":
            grouped[event["pid"], event["tid"]].append(event)
    next_tid = 100000
    lane_counts = {}
    preferred = [
        "XLA Modules",
        "XLA TraceMe",
        "MXU0 Instructions",
        "MXU1 Instructions",
        "VALU Instructions",
        "EUP Instructions",
        "XLU Instructions",
        "VLD Instructions",
        "VST Instructions",
        "SALU Instructions",
        "Tensor Core Sync Flag",
        "XLA Ops",
        "Framework Name Scope",
        "Framework Ops",
        "Source code",
        "Pallas Primitives",
    ]
    order = {name: i for i, name in enumerate(preferred)}
    for key in sorted(grouped, key=lambda k: (order.get(tracks.get(k), 100), k)):
        values = grouped[key]
        stacks = []  # per lane: open intervals, innermost last
        tids = []
        timed = sorted(((ns(e["ts"]), ns(e["dur"]), e) for e in values), key=lambda x: (x[0], -x[1]))
        for start, dur, event in timed:
            end = start + dur
            for lane, stack in enumerate(stacks):
                while stack and stack[-1] <= start:
                    stack.pop()
                if not stack or end <= stack[-1]:
                    break
            else:
                lane = len(stacks)
                stacks.append([])
                tids.append(next_tid)
                next_tid += 1
            stacks[lane].append(end)
            copy = dict(event)
            copy["tid"] = tids[lane]
            output.append(copy)
        name = tracks.get(key, str(key))
        lane_counts[name] = len(tids)
        for lane, tid in enumerate(tids):
            output.extend(
                [
                    {"ph": "M", "name": "thread_name", "pid": key[0], "tid": tid, "args": {"name": f"{name} / lane {lane:02d}"}},
                    {"ph": "M", "name": "thread_sort_index", "pid": key[0], "tid": tid, "args": {"sort_index": tid}},
                ]
            )
    assert sum(e.get("ph") == "X" for e in output) == sum(e.get("ph") == "X" for e in events)
    payload = gzip.compress(json.dumps({"traceEvents": output}, separators=(",", ":")).encode(), compresslevel=6, mtime=0)
    (root / "perfetto.trace.json.gz").write_bytes(payload)
    return payload, lane_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--template", type=Path, default=Path(__file__).with_name("student32_explainer.html"))
    args = parser.parse_args()
    trace = args.root / "trace.json.gz"
    payload = trace.read_bytes()
    events = json.loads(gzip.decompress(payload))["traceEvents"]
    tracks = {(e.get("pid"), e.get("tid")): e.get("args", {}).get("name", "") for e in events if e.get("name") == "thread_name"}
    grouped = defaultdict(list)
    for event in events:
        if event.get("ph") == "X":
            grouped[tracks.get((event.get("pid"), event.get("tid")), "unknown")].append(event)
    scopes = grouped["XLA TraceMe"]
    forwards = [e for e in scopes if e["name"] == "resident_forward"]
    boundary = "resident_forward"
    if not forwards:
        # The exporter coalesces short loop-boundary scopes. Analyze only the
        # fully preserved range, excluding embedding lookup, and label it.
        starts = sorted((e for e in scopes if e["name"] == "embedding_norm"), key=lambda e: e["ts"])
        ends = sorted((e for e in scopes if e["name"] == "output_pack"), key=lambda e: e["ts"])
        assert len(starts) == len(ends) == 27, (len(starts), len(ends))
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            stop = end["ts"] + end["dur"]
            assert start["ts"] < stop
            assert i == len(starts) - 1 or stop <= starts[i + 1]["ts"]
            forwards.append({"ts": start["ts"], "dur": stop - start["ts"]})
        boundary = "embedding_norm start through output_pack end; EXCLUDES embedding lookup; raw trace unchanged"
    modules = grouped["XLA Modules"]
    warm = []
    for module in modules:
        inside = sorted(
            (e for e in forwards if module["ts"] <= e["ts"] and e["ts"] + e["dur"] <= module["ts"] + module["dur"]),
            key=lambda e: e["ts"],
        )
        assert len(inside) == 9, len(inside)
        warm.extend(inside[1:])
    assert len(warm) == 24 and len(modules) == 3
    windows = [(e["ts"], e["ts"] + e["dur"]) for e in warm]
    duration = sum(b - a for a, b in windows)
    samples = defaultdict(list)
    for event in scopes:
        if any(a <= event["ts"] and event["ts"] + event["dur"] <= b for a, b in windows):
            samples[event["name"]].append(event["dur"])
    required = [f"encoder_{i:02d}" for i in range(10)] + ["decision_00", "decision_01", "scorer_down", "action_down"]
    assert all(len(samples[k]) == 24 for k in required)
    units = []
    for name, values in grouped.items():
        if name.endswith(" Instructions"):
            selected = [e for e in values if any(e["ts"] < b and e["ts"] + e["dur"] > a for a, b in windows)]
            units.append(
                {
                    "name": name,
                    "coverage": coverage([(e["ts"], e["ts"] + e["dur"]) for e in values], windows) / duration,
                    "opcodes": Counter(e["name"] for e in selected).most_common(6),
                }
            )
    summary = {
        "trace_sha256": hashlib.sha256(payload).hexdigest(),
        "trace_bytes": len(payload),
        "events": len(events),
        "modules": len(modules),
        "forwards": len(forwards),
        "forward_boundary": boundary,
        "warm_forwards": len(warm),
        "p50_us": statistics.median(e["dur"] for e in warm),
        "scope_stats": {name: {"count": len(values), "p50_us": statistics.median(values)} for name, values in samples.items()},
        "units": units,
        "environment": json.loads((args.root / "environment.json").read_text()),
        "validation": json.loads((args.root / "scope-validation.json").read_text()),
        "interpretation": "LLO coverage is interpolated instruction-interval coverage, not measured utilization. Unit intervals overlap. Scope instrumentation can perturb scheduling and duration.",
    }
    perfetto_payload, lane_counts = perfetto_lanes(events, tracks, args.root)
    summary["perfetto_sha256"] = hashlib.sha256(perfetto_payload).hexdigest()
    summary["perfetto_bytes"] = len(perfetto_payload)
    summary["perfetto_lanes"] = lane_counts
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    html = (
        args.template.read_text()
        .replace("__SUMMARY__", json.dumps(summary, ensure_ascii=False))
        .replace("__TRACE_BASE64__", base64.b64encode(perfetto_payload).decode())
    )
    (args.root / "index.html").write_text(html)
    print(json.dumps({k: summary[k] for k in ("events", "forwards", "warm_forwards", "p50_us", "trace_bytes")}, indent=2))


if __name__ == "__main__":
    main()
