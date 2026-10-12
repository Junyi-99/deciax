"""Shape buckets: pad (rows, seq_len, options) up to a fixed set so jit compiles once
per bucket rather than once per request shape."""

from collections.abc import Sequence

ROW_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
SEQ_BUCKETS = (64, 128, 256, 512, 1024)
OPTION_BUCKETS = (2, 4, 8, 16, 32, 64, 128, 256)


def bucket(n: int, buckets: Sequence[int]) -> int:
    """Smallest bucket >= n."""
    for b in sorted(buckets):
        if b >= n:
            return b
    raise ValueError(f"{n} exceeds the largest bucket {max(buckets)}")
