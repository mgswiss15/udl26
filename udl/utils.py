"""Small helpers: seeding and self-checks."""
import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def check(name: str, ok, hint: str = "") -> None:
    """Print a pass/fail line for a self-check."""
    ok = bool(ok)
    print(("[ OK ] " if ok else "[FAIL] ") + name + ("" if ok or not hint else f"  -> {hint}"))
