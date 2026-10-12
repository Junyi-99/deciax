"""Compiled forward passes keyed by shape bucket, with warmup and a device lock."""

from deciax.models.base import DecisionModel, EncodedBatch
from deciax.protocol.output import DecisionOutput


class Engine:
    def __init__(self, model: DecisionModel) -> None:
        self.model = model

    def warmup(self) -> None:
        raise NotImplementedError

    def run(self, batch: EncodedBatch) -> DecisionOutput:
        raise NotImplementedError
