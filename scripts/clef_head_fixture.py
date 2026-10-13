"""Generate the small joint-head fixture from the pinned Cloudflare source.

python scripts/clef_head_fixture.py /path/to/clef/snapshot
The source must be from revision 0b331204bb13fbd2ca93a64df1956f5b55478ce5.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from deciax.models.clef.input import Field, Record, collate


def records():
    fields = (
        Field("choice", 1, (3, 7), ((8, 12), (13, 15), (16, 19)), ("a", "b", "c")),
        Field("yes", 0, (20, 22), ((23, 26), (27, 30)), ("true", "false")),
        Field("score", 2, (31, 34), ((35, 38), (39, 42)), ("0", "1")),
    )
    return [Record(tuple(range(80)), fields), Record(tuple(range(50)), fields[:1]), Record(tuple(range(70)), fields[:2])]


if __name__ == "__main__":
    source = Path(sys.argv[1]) / "joint_schema_model.py"
    spec = importlib.util.spec_from_file_location("clef_reference", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(131)
    head = module.JointSchemaHead(hidden_size=32, width=16, routing_layers=2, layers=2, heads=4, feedforward=32).eval()
    h = torch.randn(4, 128, 32)
    embedding = torch.randn(128, 32)
    rs = records()
    arrays = collate(rs, 0)
    reference_records = [
        module.EncodedRecord(r.ids, tuple(module.EncodedQuestion(f.id, f.kind, f.span, f.option_spans, f.option_ids) for f in r.fields), str(i))
        for i, r in enumerate(rs)
    ]
    with torch.inference_mode():
        out = head(h, torch.tensor(arrays["input_ids"]), torch.tensor(arrays["mask"]), reference_records, embedding)
    saved = {"hidden": h.numpy(), "embedding": embedding.numpy()}
    saved.update({"param." + k: v.numpy() for k, v in head.state_dict().items()})
    saved.update({"array." + k: v for k, v in arrays.items()})
    saved.update({"expected." + str(i): torch.cat(row).numpy() for i, row in enumerate(out)})
    np.savez_compressed("tests/fixtures/clef_head.npz", **saved)
    metadata = {
        "source": "https://huggingface.co/Cloudflare/clef/blob/0b331204bb13fbd2ca93a64df1956f5b55478ce5/joint_schema_model.py",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generator": "scripts/clef_head_fixture.py",
        "seed": 131,
        "torch": torch.__version__,
        "dtype": "float32",
        "config": {"hidden_size": 32, "width": 16, "routing_layers": 2, "layers": 2, "heads": 4, "feedforward": 32},
        "fixture_sha256": hashlib.sha256(Path("tests/fixtures/clef_head.npz").read_bytes()).hexdigest(),
    }
    Path("tests/fixtures/clef_head.json").write_text(json.dumps(metadata, indent=2) + "\n")
