"""pytest bootstrap: keep JAX on GPU but bound its memory on small cards.

Without these, JAX preallocates ~75% of device memory up front, which OOMs
the 6 GB RTX 3050 on tiny workloads. Must run before jax is imported.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


def pytest_configure(config):
    import jax

    # jaxlib's XLA runtime error has been renamed across versions:
    #   jaxlib.xla_extension.XlaRuntimeError (old) -> jax.errors.XlaRuntimeError
    #   -> jax.errors.JaxRuntimeError (jax>=0.7+ / jaxlib 0.11). Probe all.
    xla_errors = []
    for mod_name, attr in (
        ("jax.errors", "XlaRuntimeError"),
        ("jax.errors", "JaxRuntimeError"),
        ("jaxlib.xla_extension", "XlaRuntimeError"),
        ("jaxlib", "XlaRuntimeError"),
    ):
        try:
            import importlib
            xla_errors.append(getattr(importlib.import_module(mod_name), attr))
        except Exception:
            pass

    dev = jax.devices()[0]
    print(f"\n[harness] default device: {dev.platform}:{getattr(dev, 'device_kind', 'n/a')} "
          f"id={getattr(dev, 'id', 'n/a')}")
    try:
        _ = jax.numpy.ones((4, 4))
    except Exception as e:
        if any(isinstance(e, err) for err in xla_errors) or not xla_errors:
            raise RuntimeError(f"JAX crashed bringing up device {dev}: {e}") from e
        raise