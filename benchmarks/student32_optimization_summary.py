"""Summarize warm resident forward scopes for the optimization A/B runs."""

import argparse
import gzip
import json
import statistics
from pathlib import Path

from student32_llo_summary import coverage


def summarize(root):
    results = {}
    for run in sorted(root.iterdir()):
        paths = list(run.rglob("trace.json.gz")) if run.is_dir() else []
        if not paths:
            continue
        assert len(paths) == 1, paths
        events = json.loads(gzip.decompress(paths[0].read_bytes()))["traceEvents"]
        tracks = {(e["pid"], e["tid"]): e.get("args", {}).get("name") for e in events if e.get("name") == "thread_name"}
        modules = sorted([e for e in events if e.get("ph") == "X" and tracks.get((e.get("pid"), e.get("tid"))) == "XLA Modules"], key=lambda e: e["ts"])
        scopes = [e for e in events if e.get("ph") == "X" and e.get("name") == "resident_forward"]
        samples = []
        measured = []
        for module in modules:
            inner = sorted([e for e in scopes if module["ts"] <= e["ts"] and e["ts"] + e["dur"] <= module["ts"] + module["dur"]], key=lambda e: e["ts"])
            assert len(inner) == 9, len(inner)
            samples.extend(e["dur"] for e in inner[1:])
            measured.extend(inner[1:])
        environment = json.loads((run / "environment.json").read_text())
        assert len(modules) == environment["steps"]
        assert len(scopes) == len(modules) * 9
        validation = json.loads((run / "variant-validation.json").read_text())
        assert validation["baseline_vmem_bitwise_equal"]
        results[run.name] = {
            "p50_us": statistics.median(samples),
            "samples_us": samples,
            "count": len(samples),
            "min_us": min(samples),
            "max_us": max(samples),
            "validation": validation,
            "environment": environment,
        }
        mxu = [e for e in events if e.get("ph") == "X" and tracks.get((e.get("pid"), e.get("tid"))) in ("MXU0 Instructions", "MXU1 Instructions")]
        if mxu:
            intervals = [(e["ts"], e["ts"] + e["dur"]) for e in mxu]
            windows = [(e["ts"], e["ts"] + e["dur"]) for e in measured]
            prefixes = [min(e["ts"] for e in mxu if a <= e["ts"] < b) - a for a, b in windows]
            results[run.name]["first_mxu_prefix_p50_us"] = statistics.median(prefixes)
            results[run.name]["mxu_instruction_union_fraction"] = coverage(intervals, windows) / sum(samples)
            results[run.name]["coverage_note"] = "Interpolated instruction spans, not hardware busy counters."
    baseline = results["baseline"]["p50_us"]
    for result in results.values():
        result["latency_reduction_pct"] = 100 * (1 - result["p50_us"] / baseline)
    (root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in value.items() if k not in ("samples_us", "environment")} for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
