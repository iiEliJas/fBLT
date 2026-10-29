from dataclasses import dataclass


@dataclass
class InferState:
    output: list
    committed: int
    target: int
