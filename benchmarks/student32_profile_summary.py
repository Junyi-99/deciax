"""Extract device timing, rejecting missing or truncated resident-forward scopes."""

import argparse
import gzip
import json
import statistics
from pathlib import Path


def summarize(root: Path, steps: int):
    result = {}
    for name in ("jax", "pallas", "vmem"):
        events = json.loads(gzip.decompress((root / name / "trace.json.gz").read_bytes()))["traceEvents"]
        tracks = {(e["pid"], e["tid"]) for e in events if e.get("name") == "thread_name" and e.get("args", {}).get("name") == "XLA Modules"}
        assert len(tracks) == 1, tracks
        modules = sorted([e for e in events if (e.get("pid"), e.get("tid")) in tracks and e.get("ph") == "X"], key=lambda e: e["ts"])
        if name == "vmem":
            assert len(modules) == steps
            scopes = [e for e in events if e.get("name") == "resident_forward" and e.get("ph") == "X"]
            assert len(scopes) == steps * 9
            samples, before, warmup = [], [], []
            for module in modules:
                group = sorted(
                    [e for e in scopes if module["ts"] <= e["ts"] and e["ts"] + e["dur"] <= module["ts"] + module["dur"]],
                    key=lambda e: e["ts"],
                )
                assert len(group) == 9, len(group)
                before.append(group[0]["ts"] - module["ts"])
                warmup.append(group[0]["dur"])
                samples.extend(e["dur"] for e in group[1:])
            result["vmem_pre_forward_us"] = before
            result["vmem_first_warmup_forward_us"] = warmup
        else:
            samples = [e["dur"] for e in modules]
        assert len(samples) == steps * 8, (name, len(samples))
        result[name] = {
            "p50_us": statistics.median(samples),
            "min_us": min(samples),
            "max_us": max(samples),
            "samples_us": samples,
            "n": len(samples),
        }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    report = summarize(args.root, args.steps)
    print(json.dumps({k: v["p50_us"] for k, v in report.items() if isinstance(v, dict)}, indent=2))
