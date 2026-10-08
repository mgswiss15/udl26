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


def report(name: str, logp) -> float:
    """Held-out negative log-likelihood: NLL = -mean(log p(x)) over the test examples, in nats. Lower is better.
    logp: tensor with one log density per test example. Infinite NLL = at least one test example has probability zero."""
    logp = torch.as_tensor(logp)
    nll = float(-logp.mean())
    n_inf = int((~torch.isfinite(logp)).sum())
    extra = f"   ({n_inf} of {len(logp)} test examples have probability zero)" if n_inf else ""
    print(f"{name}: NLL = {nll:.3f} nats per example{extra}")
    return nll
