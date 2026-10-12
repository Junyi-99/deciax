import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip `tpu` tests unless JAX's default backend is a TPU."""
    if not any("tpu" in item.keywords for item in items):
        return
    import jax

    if jax.default_backend() == "tpu":
        return
    skip = pytest.mark.skip(reason="needs a TPU")
    for item in items:
        if "tpu" in item.keywords:
            item.add_marker(skip)
