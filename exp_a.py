"""E1-E3 and ablations A1, A2, A6, A7: known demand, estimation error, review period."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

SQ2 = math.sqrt(2)
RHO_MIN = (1 + SQ2) / (2 * SQ2)


def rho_formula(r):
    r = np.asarray(r, float)
    return np.where(r <= 1 + SQ2, (1 + r) ** 2 / (2 * (1 + r**2)), r**2 / (1 + r**2))


def e1():
    r = np.unique(np.concatenate([np.linspace(1, 50, 6000), np.linspace(2.3, 2.5, 2001), [1 + SQ2]]))
    alpha = np.stack([np.ones_like(r), r], -1)
    rho = static_opt_rev_batch(alpha) / dyn_opt_rev(alpha)
    err = np.abs(rho - rho_formula(r))
    i = int(np.argmin(rho))
    out = dict(max_abs_err=float(err.max()), argmin_r=float(r[i]), min_rho=float(rho[i]),
               pred_min=RHO_MIN, pred_argmin=1 + SQ2, rho_r50=float(rho[-1]))
    # general H: no constant guarantee without bounding heterogeneity
    rows = []
    for H in [2, 4, 16, 256, 4096, 65536, 1048576]:
        a = np.sqrt(H / np.arange(1, H + 1))
        rs, _ = static_opt_rev(a)
        hh = float(np.sum(1.0 / np.arange(1, H + 1))) if H <= 10**6 + 1 else math.log(H) + 0.5772
        rows.append(dict(H=H, rho=rs / float(dyn_opt_rev(a)), bound=4 / hh, ratio_r=float(a.max() / a.min())))
    out["heavy_tail"] = rows
    # interior lower bound rho >= (sum a)^2/(H sum a^2) >= 4 m M/(m+M)^2 on random profiles
    rng = np.random.default_rng(1)
    viol = 0
    worst_gap = 1.0
    for _ in range(20000):
        H = int(rng.integers(2, 12))
        a = np.exp(rng.normal(0, rng.uniform(0.05, 1.5), H))
        rs, _ = static_opt_rev(a)
        rho_ = rs / float(dyn_opt_rev(a))
        lb1 = a.sum() ** 2 / (H * (a**2).sum())
        m, M = a.min(), a.max()
        lb2 = 4 * m * M / (m + M) ** 2
        viol += (rho_ < lb1 - 1e-12) or (lb1 < lb2 - 1e-12)
        worst_gap = min(worst_gap, rho_ - lb2)
    out["bound_violations"] = int(viol)
    out["bound_min_slack"] = float(worst_gap)
    save("E1", out)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    rr = np.linspace(1, 12, 800)
    ax[0].plot(rr, rho_formula(rr), "k-", label="Theorem 3.2")
    al = np.stack([np.ones_like(rr), rr], -1)
    ax[0].plot(rr, static_opt_rev_batch(al) / dyn_opt_rev(al), "r--", label="exact enumeration")
    ax[0].axhline(RHO_MIN, color="gray", lw=.6)
    ax[0].axvline(1 + SQ2, color="gray", lw=.6, ls=":")
    ax[0].set_xlabel("demand ratio $r$")
    ax[0].set_ylabel("static / dynamic revenue")
    ax[0].legend(frameon=False)
    Hs = [x["H"] for x in rows]
    ax[1].semilogx(Hs, [x["rho"] for x in rows], "o-", label="$\\rho(\\alpha^{(H)})$")
    ax[1].semilogx(Hs, [x["bound"] for x in rows], "s--", label="$4/H_H$")
    ax[1].set_xlabel("number of periods $H$")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "e1.pdf")
    plt.close(fig)
    return out


def welfare(tau, a, b=1.0, c=1.0):
    x = np.clip((a - b * tau) / (1 + b * c), 0, a)
    return ((a * x - x**2 / 2) / b - c * x**2).sum(-1)


def e2(N=2_000_000, seed=2):
    rng = np.random.default_rng(seed)
    beta = 1.0
    out = {}
    # revenue, i.i.d. errors
    alpha = np.array([10.0, 8.0])
    rows = []
    RS = float(static_opt_rev(alpha)[0])
    RD = float(dyn_opt_rev(alpha))
    for s in [0.25, 0.5, 1.0, 1.5]:
        eps = rng.normal(0, s, (N, 2))
        ah = alpha + eps
        tD = ah / (2 * beta)
        tS = np.repeat((ah.sum(-1) / (4 * beta))[:, None], 2, 1)
        LD = RD - rev(tD, alpha)
        LS = RS - rev(tS, alpha)
        d = LD - LS
        mD, hD = mean_ci(LD)
        mS, hS = mean_ci(LS)
        mdf, hdf = mean_ci(d)
        rows.append(dict(sigma=s, LD=mD, LD_hw=hD, LD_pred=s**2 / (2 * beta), LS=mS, LS_hw=hS, LS_pred=s**2 / (4 * beta),
                         diff=mdf, diff_hw=hdf, diff_pred=s**2 / (4 * beta),
                         ok=bool(verdict(s**2 / (2 * beta), mD, hD) and verdict(s**2 / (4 * beta), mS, hS)
                                 and verdict(s**2 / (4 * beta), mdf, hdf))))
    out["iid_revenue"] = rows
    # crossing locus: realised advantage of dynamic over static plug-in
    alpha = np.array([20.0, 12.0])
    G = (alpha[0] - alpha[1]) ** 2 / (8 * beta)
    RSx = float(static_opt_rev(alpha)[0])
    RDx = float(dyn_opt_rev(alpha))
    sc = abs(alpha[0] - alpha[1]) / SQ2
    rows = []
    for fac in [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5]:
        s = fac * sc
        eps = rng.normal(0, s, (N, 2))
        ah = alpha + eps
        tD = ah / (2 * beta)
        tS = np.repeat((ah.sum(-1) / (4 * beta))[:, None], 2, 1)
        adv_true = rev(tD, alpha) - rev(tS, alpha)
        # same comparison under the exact quadratic loss of Lemma 3.1 (no clipping)
        lossD = beta * ((tD - alpha / (2 * beta)) ** 2).sum(-1)
        lossS_dyn_oracle = beta * ((tS - alpha / (2 * beta)) ** 2).sum(-1)
        adv_quad = lossS_dyn_oracle - lossD
        m1, h1 = mean_ci(adv_true)
        m2, h2 = mean_ci(adv_quad)
        pred = G - s**2 / 4 / beta * 1.0
        rows.append(dict(fac=fac, sigma=s, adv_clipped=m1, adv_clipped_hw=h1, adv_quad=m2, adv_quad_hw=h2,
                         pred=pred, ok_quad=bool(verdict(pred, m2, h2)), ok_clipped=bool(verdict(pred, m1, h1))))
    out["crossing"] = dict(alpha=alpha.tolist(), G=G, sigma_crit=sc, rows=rows)
    # A1 welfare arm
    b, c = 1.0, 1.0
    a = np.array([10.0, 8.0])
    lam = b * (1 + 2 * b * c) / (2 * (1 + b * c) ** 2)
    kap = b * c**2 / (4 * (1 + b * c) ** 2 * (1 + 2 * b * c))
    tW = a * c / (1 + 2 * b * c)
    WD = float(welfare(tW, a))
    WS = float(welfare(np.full(2, tW.mean()), a))
    # numerically recover lambda from curvature of W
    h = 0.37
    lam_num = (WD - float(welfare(tW + np.array([h, 0.0]), a))) / h**2
    rows = []
    for s in [0.25, 0.5, 1.0, 1.5]:
        eps = rng.normal(0, s, (N, 2))
        ah = a + eps
        tD = ah * c / (1 + 2 * b * c)
        tS = np.repeat(tD.mean(-1, keepdims=True), 2, 1)
        LD = WD - welfare(tD, a)
        LS = WS - welfare(tS, a)
        d = LD - LS
        mD, hD = mean_ci(LD)
        mS, hS = mean_ci(LS)
        md, hd = mean_ci(d)
        pD, pS = 2 * kap * 2 * s**2, kap * 2 * s**2
        rows.append(dict(sigma=s, LD=mD, LD_pred=pD, LS=mS, LS_pred=pS, diff=md, diff_pred=pD - pS,
                         ok=bool(verdict(pD, mD, hD) and verdict(pS, mS, hS) and verdict(pD - pS, md, hd))))
    out["welfare"] = dict(lam=lam, lam_numeric=float(lam_num), kappa=kap, G=float(WD - WS), G_pred=kap * (a[0] - a[1]) ** 2,
                          rows=rows)
    save("E2", out)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    xs = np.array([r["fac"] for r in out["crossing"]["rows"]])
    ax.plot(xs, [r["pred"] for r in out["crossing"]["rows"]], "k-", label="Corollary 4.4 prediction")
    ax.plot(xs, [r["adv_quad"] for r in out["crossing"]["rows"]], "bo", label="simulation (Lemma 3.1 loss)")
    ax.plot(xs, [r["adv_clipped"] for r in out["crossing"]["rows"]], "r^", label="simulation (clipped revenue)")
    ax.axhline(0, color="gray", lw=.6)
    ax.set_xlabel("$\\sigma_\\alpha/\\sigma_{crit}$")
    ax.set_ylabel("dynamic advantage")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e2.pdf")
    plt.close(fig)
    return out


def a2_a6_a7(N=1_000_000, seed=3):
    """Error structure (A2), corners (A6), number of periods (A7)."""
    rng = np.random.default_rng(seed)
    beta = 1.0
    alpha = np.array([10.0, 8.0])
    RS = float(static_opt_rev(alpha)[0])
    RD = float(dyn_opt_rev(alpha))
    s = 1.0
    out = {}
    rows = []

    def losses(eps, alpha=alpha, RS=RS, RD=RD):
        ah = alpha + eps
        tD = ah / (2 * beta)
        tS = np.repeat((ah.sum(-1) / (4 * beta))[:, None], 2, 1)
        return RD - rev(tD, alpha), RS - rev(tS, alpha), eps / (2 * beta)

    for rho in [-1, -0.5, 0, 0.5, 1]:
        z = rng.normal(0, 1, (N, 2))
        e1_ = z[:, 0]
        e2_ = rho * z[:, 0] + math.sqrt(1 - rho**2) * z[:, 1]
        eps = s * np.stack([e1_, e2_], -1)
        LD, LS, u = losses(eps)
        d, hd = mean_ci(LD - LS)
        pred = beta * 0.5 * np.mean((u[:, 0] - u[:, 1]) ** 2)
        pred_cf = s**2 * (1 - rho) / (4 * beta)
        rows.append(dict(rho=rho, diff=d, hw=hd, pred=float(pred), pred_closed=pred_cf, ok=bool(verdict(pred_cf, d, hd))))
    out["correlation"] = rows
    # heavy tails (Student t, nu=3 scaled to unit variance)
    z = rng.standard_t(3, (N, 2)) / math.sqrt(3.0)
    LD, LS, u = losses(s * z)
    d, hd = mean_ci(LD - LS)
    pred = s**2 / (4 * beta)
    out["student_t3"] = dict(diff=d, hw=hd, pred=pred, ok=bool(verdict(pred, d, hd)))
    # bias: common-mode (no effect on L^D - L^S) and differential
    rows = []
    for b1, b2 in [(0.5, 0.5), (1.0, 1.0), (0.5, -0.5), (1.0, -1.0)]:
        eps = rng.normal(0, s, (N, 2)) + np.array([b1, b2])
        LD, LS, u = losses(eps)
        d, hd = mean_ci(LD - LS)
        pred = (2 * s**2 + (b1 - b2) ** 2) / (8 * beta)
        rows.append(dict(b1=b1, b2=b2, diff=d, hw=hd, pred=pred, ok=bool(verdict(pred, d, hd))))
    out["bias"] = rows
    # A6: corners, large errors relative to demand, clipped true revenue vs Lemma-1 prediction
    alpha2 = np.array([6.0, 3.0])
    RS2, RD2 = float(static_opt_rev(alpha2)[0]), float(dyn_opt_rev(alpha2))
    rows = []
    for s2 in [0.5, 1.0, 2.0, 3.0, 4.0]:
        eps = rng.normal(0, s2, (N, 2))
        LD, LS, _ = losses(eps, alpha2, RS2, RD2)
        d, hd = mean_ci(LD - LS)
        pred = s2**2 / (4 * beta)
        rows.append(dict(sigma=s2, diff=d, hw=hd, pred=pred, rel_err=(d - pred) / pred))
    out["corner_errors"] = rows
    # A7: number of periods
    rows = []
    for H in [2, 4, 8, 16]:
        a = np.linspace(14.0, 10.0, H)
        RSH = float(static_opt_rev(a)[0])
        RDH = float(dyn_opt_rev(a))
        eps = rng.normal(0, s, (N // 4, H))
        ah = a + eps
        tD = ah / (2 * beta)
        tS = np.repeat((ah.sum(-1) / (2 * beta * H))[:, None], H, 1)
        LD = RDH - rev(tD, a)
        LS = RSH - rev(tS, a)
        mD, hD = mean_ci(LD)
        mS, hS = mean_ci(LS)
        G = beta * float(((a / (2 * beta) - a.mean() / (2 * beta)) ** 2).sum())
        rows.append(dict(H=H, LD=mD, LD_pred=H * s**2 / (4 * beta), LS=mS, LS_pred=s**2 / (4 * beta), G=G,
                         thr=(H - 1) * s**2 / (4 * beta), ok=bool(verdict(H * s**2 / (4 * beta), mD, hD)
                                                                 and verdict(s**2 / (4 * beta), mS, hS))))
    out["periods"] = rows
    save("A267", out)
    return out


# ---------------------------------------------------------------- E3
def _rw(P, T, nu, sigma, rng, S0=1000.0):
    S = S0 + np.cumsum(rng.normal(0, nu, (P, T)), 1)
    return S, S + rng.normal(0, sigma, (P, T))


def sim_batched(m, nu, sigma, beta, P, days, rng, mode="rw", phi=1.0, drift=0.0, S0=1000.0):
    ncyc = max(days // m, 20)
    T = (ncyc + 1) * m
    if mode == "rw":
        S, obs = _rw(P, T, nu, sigma, rng, S0)
    elif mode == "ar1":
        S = np.empty((P, T))
        x = np.zeros(P)
        for t in range(T):
            x = phi * x + rng.normal(0, nu, P)
            S[:, t] = S0 + x
        obs = S + rng.normal(0, sigma, (P, T))
    elif mode == "drift":
        S = S0 + drift * np.arange(T)[None, :] + 0 * rng.normal(0, 1, (P, 1))
        obs = S + rng.normal(0, sigma, (P, T))
    cm = obs.reshape(P, ncyc + 1, m).mean(2)
    tau = cm[:, :-1, None] / (4 * beta)
    Sc = S.reshape(P, ncyc + 1, m)[:, 1:, :]
    r = tau * (Sc - 2 * beta * tau)
    opt = Sc**2 / (8 * beta)
    return (opt - r).mean((1, 2))


def sim_kalman(nu, sigma, beta, P, T, rng, ms=(1,), S0=1000.0, K=None, ew=None):
    p = kalman_p(nu, sigma)
    Pp = p + nu**2
    Kg = Pp / (Pp + sigma**2) if K is None else K
    S = S0 + np.zeros(P)
    x = S + rng.normal(0, math.sqrt(p), P)
    Sa = np.empty((P, T))
    xa = np.empty((P, T))  # xa[:,t] = estimate formed after observing day t
    for t in range(T):
        S = S + rng.normal(0, nu, P)
        y = S + rng.normal(0, sigma, P)
        x = x + Kg * (y - x)
        Sa[:, t] = S
        xa[:, t] = x
    res = {}
    for m in ms:
        # toll for days in cycle starting at tc uses estimate after day tc-1
        starts = np.arange(m, T - m + 1, m)
        tot = 0.0
        for k in range(m):
            tau = xa[:, starts - 1] / (4 * beta)
            Sd = Sa[:, starts + k]
            tot = tot + (Sd**2 / (8 * beta) - tau * (Sd - 2 * beta * tau))
        res[m] = (tot / m).mean(1)
    return res


def e3(seed=4):
    rng = np.random.default_rng(seed)
    beta = 1.0
    out = {"settings": []}
    ms = list(range(1, 31))
    for (nu, sigma) in [(0.5, 3.0), (1.0, 1.0), (0.3, 3.0)]:
        sim, hw, pred = [], [], []
        for m in ms:
            L = sim_batched(m, nu, sigma, beta, P=1500, days=3000, rng=rng)
            a, h = mean_ci(L)
            sim.append(a)
            hw.append(h)
            pred.append(float(lbar_batched(m, nu, sigma, beta)))
        ok = [verdict(p, s, h) for p, s, h in zip(pred, sim, hw)]
        mstar_real = math.sqrt((6 * sigma**2 + nu**2) / (5 * nu**2))
        mint = ms[int(np.argmin(pred))]
        msim = ms[int(np.argmin(sim))]
        kal = sim_kalman(nu, sigma, beta, P=2000, T=2500, rng=rng, ms=(1, 2, 4, 8))
        kal_out = {int(m): mean_ci(v) for m, v in kal.items()}
        p = kalman_p(nu, sigma)
        out["settings"].append(dict(nu=nu, sigma=sigma, m=ms, sim=sim, hw=hw, pred=pred, n_ok=int(sum(ok)),
                                    max_rel_err=float(np.max(np.abs(np.array(sim) - np.array(pred)) / np.array(pred))),
                                    mstar_real=mstar_real, mstar_int_pred=mint, mstar_int_sim=msim,
                                    Lstar=float(nu * math.sqrt(5 * (6 * sigma**2 + nu**2)) / (24 * beta)),
                                    kalman_pred=float((p + nu**2) / (8 * beta)),
                                    kalman_sim=kal_out[1][0], kalman_hw=kal_out[1][1],
                                    kalman_sched={m: dict(sim=v[0], hw=v[1], pred=float(lbar_kalman_sched(m, nu, sigma, beta)))
                                                  for m, v in kal_out.items()}))
    # Corollary 4: daily batching optimal iff sigma^2/nu^2 <= 3/2 (integer m)
    rows = []
    for ratio in [0.5, 1.0, 1.4, 1.5, 1.6, 2.0, 4.0, 9.0]:
        nu, sigma = 1.0, math.sqrt(ratio)
        vals = [float(lbar_batched(m, nu, sigma)) for m in range(1, 60)]
        rows.append(dict(ratio=ratio, best_int=int(np.argmin(vals)) + 1, pred_m1=bool(ratio <= 1.5),
                         m_star=math.sqrt((6 * ratio + 1) / 5)))
    out["cor4"] = rows
    # Corollary 5 with revision cost
    nu, sigma, C = 0.5, 3.0, 0.2
    out["cor5"] = dict(m_star=math.sqrt((6 * sigma**2 + nu**2 + 48 * beta * C) / (5 * nu**2)),
                       m_best_int=int(np.argmin([float(lbar_batched(m, nu, sigma, beta, C)) for m in range(1, 80)])) + 1)
    # EWMA family
    nu, sigma = 0.5, 3.0
    gs = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    sims = []
    p = kalman_p(nu, sigma)
    Kstar = (p + nu**2) / (p + nu**2 + sigma**2)
    for g in gs:
        r = sim_kalman(nu, sigma, beta, P=1500, T=2500, rng=rng, ms=(1,), K=g)
        sims.append(mean_ci(r[1]))
    out["ewma"] = dict(g=gs.tolist(), sim=[a for a, _ in sims], hw=[h for _, h in sims],
                       pred=ewma_loss(gs, nu, sigma).tolist(), Kstar=Kstar,
                       Lstar=float(ewma_loss(Kstar, nu, sigma)), Lkal=float((p + nu**2) / (8 * beta)))
    save("E3", out)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for k, st in enumerate(out["settings"]):
        ax[k].errorbar(st["m"], st["sim"], yerr=st["hw"], fmt="o", ms=3, label="simulation")
        ax[k].plot(st["m"], st["pred"], "k-", label="Theorem 7.3")
        ax[k].axhline(st["kalman_pred"], color="g", ls="--", label="Kalman")
        ax[k].set_title(f"$\\nu={st['nu']},\\ \\sigma={st['sigma']}$", fontsize=9)
        ax[k].set_xlabel("review period $m$")
        if k == 0:
            ax[k].set_ylabel("per-day loss")
            ax[k].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e3.pdf")
    plt.close(fig)
    return out


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["e1", "e2", "a267", "e3"]
    for w in which:
        print(w, flush=True)
        {"e1": e1, "e2": e2, "a267": a2_a6_a7, "e3": e3}[w]()
