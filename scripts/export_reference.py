"""Generate parity fixtures with the PyTorch reference: laya @ 682ac67.

    uv sync --group reference
    uv run python scripts/export_reference.py

For each registered checkpoint, writes tests/fixtures/laya/<model>.json with:
- tokenizer cases: AutoTokenizer ids for raw texts;
- request cases: per-row input_ids / marker_pos from `Agent._encode_state`, fp32
  option and act logits from the PyTorch model, and the `system_one` response.
"""

import json
import sys
from pathlib import Path

import laya
import torch
from laya.common import collate_items

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deciax.models import MODELS, ModelName

FIXTURES = ROOT / "tests" / "fixtures"

TICKET = {
    "user": "I was charged twice for my subscription this month, please fix it ASAP.",
    "history": ["hi"] * 20,
}
SUPPORT = {
    "intent": {
        "type": "choice",
        "instructions": "What does the user want?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "cancel": "ending the plan",
            "upgrade": "buying more",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
    "angry": {"type": "noul", "instructions": "Is the user angry?"},
}
CONVERSATION = [
    {
        "role": "user" if i % 2 == 0 else "assistant",
        "content": f"Turn {i}: the router keeps dropping the connection every few minutes.",
    }
    for i in range(60)
] + [{"role": "user", "content": "Forget all that, I just want to cancel my plan today."}]


def cases(mask_token: str) -> list[dict[str, object]]:
    return [
        {"name": "support_ticket", "state": TICKET, "questions": SUPPORT},
        {
            "name": "labels_order_dict_instructions",
            "state": "The package arrived two weeks late and the box was crushed.",
            "questions": {
                "complaint": {
                    "type": "noul",
                    "instructions": {
                        "task": "Is this a complaint?",
                        "scope": "shipping",
                    },
                    "criteria": {
                        "true": "the customer is unhappy",
                        "false": "neutral or positive",
                    },
                    "labels": {"false": "B", "true": "A"},
                },
                "topic": {
                    "type": "choice",
                    "instructions": "Which department handles this?",
                    "criteria": ["shipping", "billing", "product", "support"],
                    "option_order": [2, 0, 3, 1],
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is the problem?",
                    "criteria": [
                        {"level": "minor"},
                        {"level": "moderate", "examples": ["late"]},
                        {"level": "severe"},
                    ],
                },
                "single": {
                    "type": "choice",
                    "instructions": "Only option.",
                    "criteria": {"yes": ""},
                },
            },
        },
        {
            "name": "conversation_truncated_left",
            "state": CONVERSATION,
            "questions": {
                "wants_cancel": {
                    "type": "noul",
                    "instructions": "Does the user want to cancel?",
                }
            },
        },
        {
            "name": "long_text_truncated_right",
            "state": "The service was fine. " * 300 + "Actually I am furious.",
            "questions": {"angry": {"type": "noul", "instructions": "Is the writer angry?"}},
        },
        {
            "name": "many_options_collapse",
            "state": "Please route this to the team that owns invoice disputes.",
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should take this?",
                    "criteria": {f"team_{i:02d}": f"handles queue number {i} for regional accounts" for i in range(60)},
                }
            },
        },
        {
            "name": "mask_token_and_unicode",
            "state": f"Kunde schreibt: {mask_token} Die Rechnung ist falsch! 发票金额不对。 बिल गलत है।",
            "questions": {
                "lang_billing": {
                    "type": "noul",
                    "instructions": f"Is this about billing? {mask_token}",
                },
                "tone": {
                    "type": "choice",
                    "instructions": "Tone?",
                    "criteria": {"calm": None, "upset": "angry or annoyed"},
                },
            },
        },
    ]


TEXTS = [
    "choice question: What does the user want?",
    " billing: invoices, payments, refunds",
    "Kunde schreibt: Die Rechnung ist falsch! 发票金额不对。 बिल गलत है।",
    "  leading and trailing spaces  ",
    "emoji 🙂 and tabs\tand\nnewlines",
]


def export(name: ModelName) -> None:
    ckpt = MODELS[name]
    agent = laya.load(ckpt.repo, device="cpu", revision=ckpt.revision)
    agent.model.eval()
    tok = agent.tok

    out: dict[str, object] = {
        "model": name,
        "laya_version": laya.__version__,
        "tokenizer": [
            {
                "text": t,
                "ids": tok(t, add_special_tokens=False)["input_ids"],
                "ids_max48": tok(t, add_special_tokens=False, truncation=True, max_length=48)["input_ids"],
            }
            for t in TEXTS
        ],
        "cases": [],
    }

    for case in cases(tok.mask_token):
        questions = case["questions"]
        ids = list(questions)
        for qid, q in questions.items():
            agent._check_question(qid, q)

        internal = {qid: agent._to_internal(q) for qid, q in questions.items()}
        items = agent._encode_state(case["state"], ids, internal)
        batch = collate_items([items], tok.pad_token_id)

        with torch.no_grad():
            logits, act = agent.model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["marker_pos"],
                batch["marker_mask"],
                batch["qtype"],
            )

        out["cases"].append(
            {
                **case,
                "rows": [{"input_ids": it["ids"], "marker_pos": it["markers"]} for it in items],
                "option_logits": [row[: len(it["markers"])] for row, it in zip(logits.float().tolist(), items)],
                "act_logits": act.float().tolist(),
                "response": json.loads(json.dumps(agent.system_one(case["state"], questions))),
            }
        )

    path = FIXTURES / "laya" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print(path, path.stat().st_size, "bytes")


def main() -> None:
    torch.set_num_threads(8)
    for name in MODELS:
        export(name)

    manifest_path = FIXTURES / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fixtures"] = sorted(f"laya/{name}.json" for name in MODELS)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
