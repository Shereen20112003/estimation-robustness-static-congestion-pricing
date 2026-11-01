"""E4-E5 and ablations A3, A4, A5, A8; new-result checks E7-E9."""
import math
import numpy as np
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
import exp_a


# ---------------------------------------------------------------- E4
def lag_exact(m, jumps=None):
    """Enumerate every jump offset; return days from the jump until the toll uses only post-jump data."""
    out = []
    n_cyc = 8
    T = n_cyc * m
    for J in (range(2 * m, 3 * m) if jumps is None else jumps):
        S = np.where(np.arange(T + 3 * m) >= J, 11.0, 10.0)  # noise-free, jump of +1
        tau = np.empty_like(S)
        for c in range(1, (T + 3 * m) // m):
            est = S[(c - 1) * m:c * m].mean()
            tau[c * m:(c + 1) * m] = est / 4
        new_opt = 11.0 / 4
        t = J
        while abs(tau[t] - new_opt) > 1e-12:
            t += 1
        out.append(t - J)
    return float(np.mean(out)), out


def sim_cusum(P, H, m, delta, sigma, h, k, w, rng, n0=50, jump=True):
    beta = 1.0
    S0 = 1000.0
    n_burn = 3 * m + 10
    T = n_burn + H
    obs = S0 + rng.normal(0, sigma, (P, T))
    J = n_burn + rng.integers(0, m, P) if jump else np.full(P, 10**9)
    S = np.where(np.arange(T)[None, :] >= J[:, None], S0 + delta, S0)
    obs = S + (obs - S0)
    xhat = np.full(P, S0)
    g = np.zeros(P)
    gm = np.zeros(P)
    wait = np.full(P, -1)
    alarm_day = np.full(P, -1)
    false_alarm = np.zeros(P, bool)
    loss = np.zeros(P)
    lossm = np.zeros(P)
    xm = np.full(P, S0)
    buf = []
    for t in range(T):
        # toll used today is xhat (cusum policy) / xm (plain m-scheduled)
        if t >= n_burn:
            loss += (S[:, t] - xhat) ** 2 / (8 * beta)
            lossm += (S[:, t] - xm) ** 2 / (8 * beta)
        y = obs[:, t]
        # plain m-scheduled: update at cycle starts using previous cycle mean
        if t % m == 0 and t >= m:
            xm = obs[:, t - m:t].mean(1)
        buf.append(y)
        if t % m == 0 and t >= m:
            xhat = np.where(wait < 0, obs[:, t - m:t].mean(1), xhat)
        # cusum
        e = y - xhat
        g = np.maximum(0, g + e - k)
        gm = np.maximum(0, gm - e - k)
        trig = ((g > h) | (gm > h)) & (wait < 0)
        alarm_day = np.where(trig & (alarm_day < 0) & (t >= J), t, alarm_day)
        false_alarm |= trig & (t < J)
        wait = np.where(trig, w, wait)
        g = np.where(trig, 0, g)
        gm = np.where(trig, 0, gm)
        done = wait == 0
        if done.any() and len(buf) >= w:
            est = np.mean(buf[-w:], axis=0)
            xhat = np.where(done, est, xhat)
        wait = np.where(wait > 0, wait - 1, np.where(done, -1, wait))
    return loss / H, lossm / H, alarm_day, J, false_alarm


def e4(seed=5):
    rng = np.random.default_rng(seed)
    out = {}
    rows = []
    for m in [1, 2, 3, 5, 8, 12, 20, 30]:
        mean_lag, lags = lag_exact(m)
        rows.append(dict(m=m, lag=mean_lag, pred=(3 * m - 1) / 2, min=min(lags), max=max(lags)))
    out["lag"] = rows
    # estimate-once never recovers: post-jump per-day loss (Delta^2 + E zeta^2)/(8 beta)
    sigma, n0, delta = 2.0, 25, 3.0
    P = 200000
    zeta = rng.normal(0, sigma / math.sqrt(n0), P)
    S1 = 1000 + delta
    xhat = 1000 + zeta
    tau = xhat / 4
    loss = (S1**2 / 8 - tau * (S1 - 2 * tau))
    a, h = mean_ci(loss)
    pred = (delta**2 + sigma**2 / n0) / 8
    out["estimate_once"] = dict(sim=a, hw=h, pred=pred, ok=bool(verdict(pred, a, h)))
    # A4: CUSUM-triggered re-tolling vs scheduled review after a jump (no drift)
    rows = []
    sigma = 1.0
    for delta in [2.0, 4.0]:
        for m in [10, 30]:
            L, Lm, ad, J, fa = sim_cusum(6000, 300, m, delta, sigma, h=8.0, k=0.5, w=5, rng=rng)
            det = (ad[ad >= 0] - J[ad >= 0])
            rows.append(dict(delta=delta, m=m, loss_cusum=mean_ci(L)[0], loss_sched=mean_ci(Lm)[0],
                             det_rate=float((ad >= 0).mean()), mean_delay=float(det.mean()) if det.size else None, false_alarm=float(fa.mean())))
    L, Lm, ad, J, fa = sim_cusum(6000, 300, 30, 0.0, sigma, h=8.0, k=0.5, w=5, rng=rng, jump=False)
    out["cusum"] = dict(rows=rows, false_alarm_rate_300d=float(((ad >= 0) | fa).mean()), no_jump_loss_cusum=float(L.mean()), no_jump_loss_sched=float(Lm.mean()))
    save("E4", out)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ms = [r["m"] for r in rows if False] or [r["m"] for r in out["lag"]]
    ax.plot(ms, [r["pred"] for r in out["lag"]], "k-", label="$(3m-1)/2$")
    ax.plot(ms, [r["lag"] for r in out["lag"]], "ro", label="enumeration")
    ax.set_xlabel("review period $m$")
    ax.set_ylabel("days to clean estimate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "e4.pdf")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- E5
def e5(seed=6):
    rng = np.random.default_rng(seed)
    beta, sigma = 1.0, 1.0
    a, b = np.array([10.0, 6.0]), np.array([11.0, 5.0])
    out = dict(static_opt=(static_opt_rev(a)[0], static_opt_rev(b)[0]), static_tau=(static_opt_rev(a)[1], static_opt_rev(b)[1]),
               dyn_opt=(float(dyn_opt_rev(a)), float(dyn_opt_rev(b))))

    def flow(alpha, tau):
        return (np.maximum(alpha[None, :] - beta * tau, 0)).sum(1)

    # (i) interior adaptive tolls: identical laws of the observation process
    N, n = 40000, 30
    res = []
    for alpha in (a, b):
        Y = np.empty((N, n))
        tau = np.full((N, 2), 3.0)
        for t in range(n):
            Y[:, t] = flow(alpha, tau) + rng.normal(0, sigma, N)
            # adaptive interior rule: toll depends on past observation, clipped to keep both periods active in both profiles
            tau = np.clip(3.0 + 0.1 * (Y[:, t:t + 1] - 11.0) * np.array([[1.0, -0.5]]), 0.5, 4.5)
        res.append(Y)
    ks = [stats.ks_2samp(res[0][:, t], res[1][:, t]).pvalue for t in range(n)]
    out["interior_ks_min_p"] = float(min(ks))
    out["interior_mean_gap"] = float(np.abs(res[0].mean() - res[1].mean()))
    # (ii) corner probe separates the profiles; accuracy of the likelihood-ratio test vs days
    rows = []
    tc = np.array([5.5, 5.5])
    mu = [float(np.maximum(x - beta * tc, 0).sum()) for x in (a, b)]
    for n in [1, 4, 16, 64, 144]:
        acc = []
        for mu_ in mu:
            ybar = mu_ + rng.normal(0, sigma / math.sqrt(n), 100000)
            thr = 0.5 * (mu[0] + mu[1])
            acc.append(np.mean((ybar < thr) if mu_ == mu[0] else (ybar >= thr)))
        rows.append(dict(n=n, acc=float(np.mean(acc)), pred=float(stats.norm.cdf(abs(mu[1] - mu[0]) * math.sqrt(n) / (2 * sigma)))))
    out["corner"] = dict(means=mu, rows=rows,
                         probe_rev=[float(rev(tc, x)) for x in (a, b)],
                         probe_cost=[float(static_opt_rev(x)[0] - rev(tc, x)) for x in (a, b)])
    # (iii) per-period sensors (O2): split recovered at sigma/sqrt(n)
    n = 25
    est = np.array([x - 0 for x in (a,)])
    y1 = a[0] - beta * 3.0 + rng.normal(0, sigma / math.sqrt(n), 100000)
    out["o2_split_se"] = dict(sim=float(y1.std()), pred=sigma / math.sqrt(n))
    save("E5", out)
    return out


# ---------------------------------------------------------------- A3, A5, A8
def a3(seed=7):
    rng = np.random.default_rng(seed)
    beta = 1.0
    out = {}
    nu, sigma = 0.5, 3.0
    ms = list(range(1, 25)) + [30, 40, 60]
    rows = []
    for phi in [1.0, 0.99, 0.95, 0.8]:
        L = []
        for m in ms:
            if phi == 1.0:
                v = exp_a.sim_batched(m, nu, sigma, beta, P=1200, days=2400, rng=rng)
            else:
                v = sim_ar1(m, nu, sigma, beta, phi, P=1200, days=2400, rng=rng)
            L.append(mean_ci(v)[0])
        rows.append(dict(phi=phi, best_m=ms[int(np.argmin(L))], best_loss=float(min(L)), loss_m1=L[0], loss_m=L))
    out["ar1"] = dict(nu=nu, sigma=sigma, rows=rows)
    # deterministic linear drift: cube-root law
    rows = []
    for delta in [0.05, 0.1, 0.2]:
        sigma = 2.0
        L, pr = [], []
        for m in range(1, 40):
            v = exp_a.sim_batched(m, 0, sigma, beta, P=800, days=1200, rng=rng, mode="drift", drift=delta)
            L.append(mean_ci(v)[0])
            pr.append((delta**2 * (13 * m**2 - 1) / 12 + sigma**2 / m) / (8 * beta))
        mstar = (6 * sigma**2 / (13 * delta**2)) ** (1 / 3)
        mi = int(np.argmin(pr)) + 1
        rows.append(dict(delta=delta, sigma=sigma, mstar=mstar, best_pred=mi, best_sim=int(np.argmin(L)) + 1,
                         max_rel_err=float(np.max(np.abs(np.array(L) - np.array(pr)) / np.array(pr)))))
    out["drift"] = rows
    # weekly seasonality: calendar-aligned weekday means vs cycle means
    sigma, A, P, T = 2.0, 3.0, 4000, 700
    t = np.arange(T)
    S = 1000 + A * np.sin(2 * np.pi * t / 7)
    obs = S[None, :] + rng.normal(0, sigma, (P, T))
    res = {}
    for m in [1, 2, 3, 7, 14]:
        ncyc = (T // m) - 1
        cm = obs[:, :(ncyc + 1) * m].reshape(P, ncyc + 1, m).mean(2)
        tau = cm[:, :-1, None] / 4
        Sc = S[:(ncyc + 1) * m].reshape(ncyc + 1, m)[1:][None]
        res[f"cycle_{m}"] = float(((Sc**2 / 8 - tau * (Sc - 2 * tau)).mean((1, 2))).mean())
    for w in [2, 4, 8]:
        ls = []
        for tt in range(7 * w, T):
            est = np.mean([obs[:, tt - 7 * j] for j in range(1, w + 1)], axis=0)
            tau = est / 4
            ls.append(S[tt] ** 2 / 8 - tau * (S[tt] - 2 * tau))
        res[f"weekday_{w}"] = float(np.mean(ls))
    res["weekday_pred_w4"] = sigma**2 / 4 / 8
    out["seasonal"] = res
    save("A3", out)
    return out


def sim_ar1(m, nu, sigma, beta, phi, P, days, rng, S0=1000.0):
    ncyc = max(days // m, 20)
    T = (ncyc + 1) * m
    x = rng.normal(0, nu / math.sqrt(1 - phi**2), P)
    S = np.empty((P, T))
    for t in range(T):
        x = phi * x + rng.normal(0, nu, P)
        S[:, t] = S0 + x
    obs = S + rng.normal(0, sigma, (P, T))
    cm = obs.reshape(P, ncyc + 1, m).mean(2)
    tau = cm[:, :-1, None] / (4 * beta)
    Sc = S.reshape(P, ncyc + 1, m)[:, 1:, :]
    return (Sc**2 / (8 * beta) - tau * (Sc - 2 * beta * tau)).mean((1, 2))


def a5(seed=8, N=400000):
    rng = np.random.default_rng(seed)
    out = {}
    beta, S, sigma, n = 1.0, 16.0, 1.0, 20
    rows = []
    for tbar in [2.0, 3.0, 4.0]:
        for delta in [0.5, 1.0, 2.0]:
            xm = S - 2 * beta * (tbar - delta)   # level "-"
            xp = S - 2 * beta * (tbar + delta)
            Ym = xm + rng.normal(0, sigma / math.sqrt(n), N)
            Yp = xp + rng.normal(0, sigma / math.sqrt(n), N)
            bh = (Ym - Yp) / (4 * delta)
            Sh = (Ym + Yp) / 2 + 2 * bh * tbar
            th = Sh / (4 * bh)
            xbar = S - 2 * beta * tbar
            rows.append(dict(tbar=tbar, delta=delta,
                             var_beta=float(bh.var()), var_beta_pred=sigma**2 / (8 * delta**2 * n),
                             var_S=float(Sh.var()), var_S_pred=sigma**2 * (delta**2 + tbar**2) / (2 * delta**2 * n),
                             var_tau=float(th.var()),
                             var_tau_pred=sigma**2 * (4 * beta**2 * delta**2 + xbar**2) / (128 * beta**4 * delta**2 * n),
                             bias_beta=float(bh.mean() - beta)))
    out["const_noise"] = rows
    # A8: noise proportional to flow (sd = kappa * mean flow at each level)
    kappa = 0.1
    rows = []
    for tbar in [1.0, 2.0, 3.0, 4.0, 5.0]:
        delta = 1.0
        xm = S - 2 * beta * (tbar - delta)
        xp = S - 2 * beta * (tbar + delta)
        sm, sp = kappa * xm, kappa * xp
        Ym = xm + rng.normal(0, sm / math.sqrt(n), N)
        Yp = xp + rng.normal(0, sp / math.sqrt(n), N)
        bh = (Ym - Yp) / (4 * delta)
        Sh = (Ym + Yp) / 2 + 2 * bh * tbar
        th = Sh / (4 * bh)
        xbar = S - 2 * beta * tbar
        # first-order delta method with heteroskedastic noise
        vY = (sm**2 + sp**2) / (4 * n)
        vb = (sm**2 + sp**2) / (16 * delta**2 * n)
        cov = (sm**2 - sp**2) / (8 * delta * n)
        pred = vY / (16 * beta**2) + xbar**2 * vb / (16 * beta**4) - 2 * xbar * cov / (16 * beta**3)
        const_pred = sigma**2 * (4 * beta**2 * delta**2 + xbar**2) / (128 * beta**4 * delta**2 * n)
        rows.append(dict(tbar=tbar, var_tau=float(th.var()), delta_method=float(pred), const_noise_formula=float(const_pred)))
    out["prop_noise"] = rows
    save("A5A8", out)
    return out


# ---------------------------------------------------------------- E7 shrinkage
def e7(seed=9, N=400000):
    rng = np.random.default_rng(seed)
    out = {"rows": []}
    s = 1.0
    for H in [2, 4, 8, 16]:
        for D in [0.0, 0.5, 2.0, 8.0, 32.0]:
            d = np.zeros(H)
            if D > 0:
                d = rng.normal(0, 1, H)
                d -= d.mean()
                d *= math.sqrt(D / (d**2).sum())
            u = rng.normal(0, s, (N, H))
            th = d[None, :] + u  # estimated toll deviation + common part handled separately
            ubar = u.mean(1, keepdims=True)
            dev = th - th.mean(1, keepdims=True)
            SS = (dev**2).sum(1, keepdims=True)

            def risk(w):
                est = th.mean(1, keepdims=True) + w * dev
                return float(((est - d[None, :] - 0)**2).sum(1).mean())

            # risk is measured against the dynamic oracle d (common component has true value 0)
            ws = {"static": 0.0, "dynamic": 1.0, "oracle": D / (D + (H - 1) * s**2),
                  "plugin_pp": np.maximum(0, 1 - (H - 1) * s**2 / SS)}
            if H >= 4:
                ws["james_stein"] = np.maximum(0, 1 - (H - 3) * s**2 / SS)
            row = dict(H=H, D=D)
            for k, w in ws.items():
                row[k] = risk(w)
            row["pred_static"] = s**2 + D
            row["pred_dynamic"] = H * s**2
            row["pred_oracle"] = s**2 + D * (H - 1) * s**2 / (D + (H - 1) * s**2)
            out["rows"].append(row)
    save("E7", out)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for H, mk in [(4, "o"), (8, "s")]:
        rr = [r for r in out["rows"] if r["H"] == H]
        ax.plot([r["D"] for r in rr], [r["dynamic"] for r in rr], "k" + mk + "-", label=f"dynamic, H={H}")
        ax.plot([r["D"] for r in rr], [r["james_stein"] for r in rr], "r" + mk + "--", label=f"James-Stein, H={H}")
        ax.plot([r["D"] for r in rr], [r["static"] for r in rr], "b" + mk + ":", label=f"static, H={H}")
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xlabel("heterogeneity $D=\\sum_i d_i^2$")
    ax.set_ylabel("toll-space risk")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "e7.pdf")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- E8 coupled periods
def e8(seed=10, N=100000):
    rng = np.random.default_rng(seed)
    out = {}
    H, beta = 6, 1.0
    rows = []
    for kappa in [0.0, 0.5, 2.0, 8.0]:
        Lap = np.zeros((H, H))
        for i in range(H - 1):
            Lap[i, i] += 1
            Lap[i + 1, i + 1] += 1
            Lap[i, i + 1] -= 1
            Lap[i + 1, i] -= 1
        B = beta * np.eye(H) + kappa * Lap
        Bi = np.linalg.inv(B)
        one = np.ones(H)
        alpha = 20 + 3 * rng.normal(0, 1, H)
        RD = alpha @ Bi @ alpha / 4
        c_or = one @ alpha / (2 * one @ B @ one)
        RS = (c_or * one) @ (alpha - B @ (c_or * one))

        def R(t, a):
            return t @ (a - B @ t)

        A = rng.normal(0, 1, (H, H))
        Sig = A @ A.T / H * 0.5
        eps = rng.multivariate_normal(np.zeros(H), Sig, N)
        LD = np.empty(N)
        LS = np.empty(N)
        for i in range(N):
            ah = alpha + eps[i]
            tD = Bi @ ah / 2
            c = one @ ah / (2 * one @ B @ one)
            LD[i] = RD - R(tD, alpha)
            LS[i] = RS - R(c * one, alpha)
        e = eps
        LDf = np.einsum("ni,ij,nj->n", e, Bi, e) / 4
        LSf = (e @ one) ** 2 / (4 * one @ B @ one)
        rows.append(dict(kappa=kappa, min_gap=float((LD - LS).min()), max_err_LD=float(np.abs(LD - LDf).max()),
                         max_err_LS=float(np.abs(LS - LSf).max()), frac_violations=float(np.mean(LD - LS < -1e-9)),
                         G=float(RD - RS), mean_gap=float((LD - LS).mean()),
                         mean_gap_formula=float(np.mean(LDf - LSf)), cond=float(np.linalg.cond(B))))
        # equality iff eps proportional to B 1
        e0 = 0.7 * (B @ one)
        ah = alpha + e0
        gap0 = (RD - R(Bi @ ah / 2, alpha)) - (RS - R((one @ ah / (2 * one @ B @ one)) * one, alpha))
        rows[-1]["gap_on_B1_direction"] = float(gap0)
    out["rows"] = rows
    save("E8", out)
    return out


# ---------------------------------------------------------------- E9 certainty equivalence / Bayes
def e9():
    out = {}
    beta = 1.0

    def exp_rev(tau, m, s):
        # E[tau * max(alpha - beta tau, 0)], alpha ~ N(m, s^2)
        d = (m - beta * tau) / s
        return tau * (s * stats.norm.pdf(d) + (m - beta * tau) * stats.norm.cdf(d))

    rows = []
    m = 10.0
    for s in [0.5, 2.0, 4.0, 6.0, 8.0, 12.0]:
        f = lambda t: -exp_rev(t, m, s)
        ts = np.linspace(0.01, 3 * m, 6001)
        t_b = float(ts[np.argmin([f(t) for t in ts])])
        t_ce = m / (2 * beta)
        rows.append(dict(s_over_m=s / m, tau_bayes=t_b, tau_plugin=t_ce, ratio=t_b / t_ce,
                         rev_bayes=float(-f(t_b)), rev_plugin=float(exp_rev(t_ce, m, s)),
                         gain_pct=float(100 * (-f(t_b) / exp_rev(t_ce, m, s) - 1))))
    out["rows"] = rows
    save("E9", out)
    return out


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["e4", "e5", "a3", "a5", "e7", "e8", "e9"]
    for w in which:
        print(w, flush=True)
        {"e4": e4, "e5": e5, "a3": a3, "a5": a5, "e7": e7, "e8": e8, "e9": e9}[w]()
