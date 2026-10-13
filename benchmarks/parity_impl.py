"""impl="pallas" vs impl="jax" on the same checkpoint and inputs: max |delta p| over the option
probabilities, argmax agreement, act-logit error. Inputs: the PyTorch-parity fixture cases.

  python benchmarks/parity_impl.py [--model laya]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from deciax.models.laya.model import LayaModel
from deciax.protocol.answers import softmax
from deciax.protocol.types import parse_questions
from deciax.runtime.engine import Engine
from tests.conftest import checkpoint_dir

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "laya"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya")
    args = ap.parse_args()
    path = checkpoint_dir(args.model)
    cases = json.loads((FIXTURES / f"{args.model}.json").read_text())["cases"]

    models = {impl: LayaModel(path, jnp.bfloat16, impl) for impl in ("jax", "pallas")}
    worst_dp, worst_act, agree, total = 0.0, 0.0, 0, 0
    for case in cases:
        outs = {}
        for impl, m in models.items():
            rows = m.encode(case["state"], parse_questions(case["questions"]))
            outs[impl] = Engine(m).run(rows)
        a, b = outs["jax"], outs["pallas"]
        for i in range(a.option_logits.shape[0]):
            k = int(a.option_mask[i].sum())
            pa, pb = softmax(a.option_logits[i, :k].astype(np.float64)), softmax(b.option_logits[i, :k].astype(np.float64))
            worst_dp = max(worst_dp, float(np.abs(pa - pb).max()))
            agree += int(pa.argmax() == pb.argmax())
            total += 1
        worst_act = max(worst_act, float(np.abs(a.act_logits - b.act_logits).max() / (np.abs(a.act_logits).max() + 1e-6)))
    print(f"{args.model}: {len(cases)} cases, {total} questions; argmax agree {agree}/{total}; max |dp| {worst_dp:.4f}; act logits max rel err {worst_act:.2e}")


if __name__ == "__main__":
    main()
