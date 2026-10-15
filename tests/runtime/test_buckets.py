import pytest

from deciax.runtime.buckets import SEQ_BUCKETS, bucket


@pytest.mark.parametrize(("n", "want"), [(1, 64), (64, 64), (65, 128), (SEQ_BUCKETS[-1], SEQ_BUCKETS[-1])])
def test_bucket(n: int, want: int) -> None:
    assert bucket(n, SEQ_BUCKETS) == want


def test_bucket_overflow() -> None:
    with pytest.raises(ValueError):
        bucket(SEQ_BUCKETS[-1] + 1, SEQ_BUCKETS)
