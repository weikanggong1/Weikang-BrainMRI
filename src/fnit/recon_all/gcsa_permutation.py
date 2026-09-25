"""FreeSurfer's seeded VNL vertex permutation for GCSA Gibbs iteration."""

from __future__ import annotations

import numpy as np


class VnlRandom:
    """Marsaglia-Zaman ``vnl_random`` state used by ``OpenRan1``."""

    def __init__(self, seed: int):
        self.position = 0
        self.borrow = 0
        self.array = []
        state = seed
        for _ in range(37):
            state = (state * 1664525 + 1) & 0xFFFFFFFF
            self.array.append(state)
        for _ in range(1000):
            self.lrand32()

    def lrand32(self) -> int:
        p1 = self.array[(37 + self.position - 24) % 37]
        p2 = (p1 - self.array[self.position] - self.borrow) & 0xFFFFFFFF
        if p2 < p1:
            self.borrow = 0
        if p2 > p1:
            self.borrow = 1
        self.array[self.position] = p2
        self.position = (self.position + 1) % 37
        return p2

    def open_ran1(self) -> float:
        high = self.lrand32()
        low = self.lrand32()
        value = high / 0xFFFFFFFF + low / (0xFFFFFFFF * 0xFFFFFFFF)
        return float(np.float32(value))


def vertex_permutation(random: VnlRandom, vertices: int) -> np.ndarray:
    """Match ``MRIScomputeVertexPermutation`` swap order for one iteration."""
    result = np.arange(vertices, dtype=np.int32)
    for i in range(vertices):
        index = int(random.open_ran1() * (vertices - 0.0001))
        result[index], result[i] = result[i], result[index]
    return result
