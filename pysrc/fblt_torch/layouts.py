from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class PatchLayout:
    patches: List[Tuple[int, int]]
    num_patches: int
    num_clean: int
    block_size: int = 4


def build_patch_layout(bytes_seq, patches):
    seq_len = len(bytes_seq)
    assert patches and patches[0][0] == 0
    pos = 0
    for s, plen in patches:
        assert s == pos
        assert plen >= 1
        pos += plen
    assert pos == seq_len
    return PatchLayout(list(patches), len(patches), seq_len)
