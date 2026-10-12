"""Public entry point: encode a request, run the engine, decode typed answers."""

from collections.abc import Mapping, Sequence
from typing import Any

from deciax.protocol.types import Answer, State


class Decider:
    """One loaded checkpoint bound to a runtime engine."""

    def decide(
        self, state: State, questions: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Answer]:
        """Answer every question about one state."""
        raise NotImplementedError

    def decide_batch(
        self, states: Sequence[State], questions: Mapping[str, Mapping[str, Any]]
    ) -> list[dict[str, Answer]]:
        """Answer the same questions about many states."""
        raise NotImplementedError


def load(model: str = "laya", *, revision: str | None = None) -> Decider:
    """Load a registered checkpoint (see `deciax.models.MODELS`)."""
    raise NotImplementedError
