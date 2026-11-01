"""Shared utilities for the toll-learning experiments (CPU-only, numpy)."""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figs"
FIG.mkdir(exist_ok=True)
RES = HERE / "results.json"


def save(key, val):
    data = json.loads(RES.read_text()) if RES.exists() else {}
    data[key] = val
    RES.write_text(json.dumps(data, indent=1, default=float))


def load():
    return json.loads(RES.read_text()) if RES.exists() else {}


def mean_ci(x):
    x = np.asarray(x, dtype=float).ravel()
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(x.size))


def verdict(pred, sim, hw, rel=0.005):
    """Reporting standard: inside the 95% CI, or within 0.5% of the simulated value."""
    err = abs(pred - sim)
    return bool(err <= hw or err <= rel * abs(sim))


def rev(tau, alpha, beta=1.0):
    """True (clipped) toll revenue; arrays with periods on the last axis."""
    return (tau * np.maximum(alpha - beta * tau, 0.0)).sum(-1)


def static_opt_rev(alpha, beta=1.0):
    """Exact revenue-optimal static toll for one profile (any H, O(H log H))."""
    a = np.sort(np.asarray(alpha, float))[::-1]
    H = a.size
    cs = np.cumsum(a)
    k = np.arange(1, H + 1)
    cand = np.concatenate([cs / (2 * beta * k), a / beta])
    asc = a[::-1]
    kk = H - np.searchsorted(asc, beta * cand, side="right")  # number with a > beta*tau
    top = np.where(kk > 0, cs[np.maximum(kk - 1, 0)], 0.0)
    r = cand * (top - kk * beta * cand)
    j = int(np.argmax(r))
    return float(r[j]), float(cand[j])


def static_opt_rev_batch(alpha, beta=1.0):
    """Exact static optimum for a batch [N,H] with small H (candidate enumeration)."""
    alpha = np.asarray(alpha, float)
    H = alpha.shape[-1]
    best = np.full(alpha.shape[:-1], -np.inf)
    cands = [alpha[..., i] / beta for i in range(H)]
    srt = -np.sort(-alpha, axis=-1)
    cs = np.cumsum(srt, axis=-1)
    for k in range(1, H + 1):
        cands.append(cs[..., k - 1] / (2 * beta * k))
    for c in cands:
        r = (c[..., None] * np.maximum(alpha - beta * c[..., None], 0)).sum(-1)
        best = np.maximum(best, r)
    return best


def dyn_opt_rev(alpha, beta=1.0):
    return (np.asarray(alpha, float) ** 2).sum(-1) / (4 * beta)


def kalman_p(nu, sigma):
    """Steady-state posterior variance for local-level model: p^2 + nu^2 p - nu^2 sigma^2 = 0."""
    return (math.sqrt(nu**4 + 4 * nu**2 * sigma**2) - nu**2) / 2


def lbar_batched(m, nu, sigma, beta=1.0, C=0.0):
    m = np.asarray(m, float)
    return (nu**2 * (5 * m**2 + 1) / (6 * m) + sigma**2 / m) / (8 * beta) + C / m


def lbar_kalman_sched(m, nu, sigma, beta=1.0, C=0.0):
    m = np.asarray(m, float)
    p = kalman_p(nu, sigma)
    return (p + (m + 1) * nu**2 / 2) / (8 * beta) + C / m


def ewma_loss(g, nu, sigma, beta=1.0):
    g = np.asarray(g, float)
    V = ((1 - g) ** 2 * nu**2 + g**2 * sigma**2) / (1 - (1 - g) ** 2)
    return (V + nu**2) / (8 * beta)
