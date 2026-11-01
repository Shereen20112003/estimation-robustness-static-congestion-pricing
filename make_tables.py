"""Turn results.json into LaTeX tables so that no number in the paper is typed by hand."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)


def f(x, d=4):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if abs(x) < 10 ** (-d) and x != 0:
        return f"{x:.1e}"
    return f"{x:.{d}f}"


def pm(a, h, d=4):
    return f"{a:.{d}f}$\\pm${h:.{d}f}"


def tab(name, header, rows, colspec=None, caption=None):
    cs = colspec or "l" + "r" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{cs}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


# ---- E1
e1 = R["E1"]
tab("e1_ht", ["$H$", "$r=\\alpha_{\\max}/\\alpha_{\\min}$", "$\\rho(\\alpha^{(H)})$", "$4/H_H$", "$\\rho\\le 4/H_H$"],
    [[f"{x['H']:,}".replace(",", "{,}"), f(x["ratio_r"], 1), f(x["rho"], 4), f(x["bound"], 4), f(x["rho"] <= x["bound"] + 1e-12 or x["bound"] > 1)]
     for x in e1["heavy_tail"]])
tab("e1_main", ["Quantity", "Prediction", "Enumeration"],
    [["worst-case ratio $\\rho_{\\min}$", f(e1["pred_min"], 6), f(e1["min_rho"], 6)],
     ["location $r^\\ast$", f(e1["pred_argmin"], 5), f(e1["argmin_r"], 5)],
     ["max $|\\rho_{\\rm formula}-\\rho_{\\rm exact}|$ over grid", "0", f(e1["max_abs_err"], 2)],
     ["violations of $\\rho\\ge(\\sum\\alpha)^2/(H\\sum\\alpha^2)\\ge 4mM/(m+M)^2$ in 20{,}000 random profiles", "0", str(e1["bound_violations"])]],
    colspec="p{0.58\\linewidth}rr")

# ---- E2
rows = []
for x in R["E2"]["iid_revenue"]:
    rows.append([f(x["sigma"], 2), f(x["LD_pred"]), pm(x["LD"], x["LD_hw"]), f(x["LS_pred"]), pm(x["LS"], x["LS_hw"]),
                 f(x["diff_pred"]), pm(x["diff"], x["diff_hw"]), f(x["ok"])])
tab("e2_iid", ["$\\sigma_\\alpha$", "$L^D$ pred.", "$L^D$ sim.", "$L^S$ pred.", "$L^S$ sim.", "$L^D\\!-\\!L^S$ pred.", "$L^D\\!-\\!L^S$ sim.", "verified"], rows,
    colspec="rrrrrrrr")
c = R["E2"]["crossing"]
rows = [[f(x["fac"], 2), f(x["pred"], 3), pm(x["adv_quad"], x["adv_quad_hw"], 3), f(x["ok_quad"]), pm(x["adv_clipped"], x["adv_clipped_hw"], 3), f(x["ok_clipped"])]
        for x in c["rows"]]
tab("e2_cross", ["$\\sigma_\\alpha/\\sigma_{\\rm crit}$", "Corollary 4.4", "sim. (Lemma 3.1 loss)", "ok", "sim. (clipped revenue)", "ok"], rows, colspec="rrrcrc")
w = R["E2"]["welfare"]
rows = [[f(x["sigma"], 2), f(x["LD_pred"], 5), f(x["LD"], 5), f(x["LS_pred"], 5), f(x["LS"], 5), f(x["diff_pred"], 5), f(x["diff"], 5), f(x["ok"])] for x in w["rows"]]
tab("e2_welf", ["$\\sigma_a$", "$L^D$ pred.", "$L^D$ sim.", "$L^S$ pred.", "$L^S$ sim.", "diff pred.", "diff sim.", "verified"], rows, colspec="rrrrrrrr")

# ---- A2, A6, A7
a = R["A267"]
rows = [[f"correlation $\\rho={x['rho']}$", f(x["pred_closed"], 4), pm(x["diff"], x["hw"]), f(x["ok"])] for x in a["correlation"]]
rows.append(["Student-$t_3$ errors", f(a["student_t3"]["pred"], 4), pm(a["student_t3"]["diff"], a["student_t3"]["hw"]), f(a["student_t3"]["ok"])])
for x in a["bias"]:
    rows.append([f"bias $({x['b1']:+.1f},{x['b2']:+.1f})$", f(x["pred"], 4), pm(x["diff"], x["hw"]), f(x["ok"])])
tab("a2", ["Error structure", "Prediction of $E[L^D-L^S]$", "Simulation", "verified"], rows, colspec="lrrc")
rows = [[f(x["sigma"], 1), f(x["pred"], 4), pm(x["diff"], x["hw"]), f"{100 * x['rel_err']:+.1f}\\%"] for x in a["corner_errors"]]
tab("a6", ["$\\sigma_\\alpha$", "Lemma-1 prediction", "clipped revenue", "rel.\\ deviation"], rows, colspec="rrrr")
rows = [[x["H"], f(x["LD_pred"], 3), f(x["LD"], 3), f(x["LS_pred"], 3), f(x["LS"], 3), f(x["G"], 3), f(x["thr"], 3), f(x["G"] > x["thr"])] for x in a["periods"]]
tab("a7", ["$H$", "$E L^D$ pred.", "sim.", "$E L^S$ pred.", "sim.", "$G$", "threshold $(H\\!-\\!1)\\sigma^2/4\\beta$", "dynamic wins"], rows, colspec="rrrrrrrr")

# ---- E3
rows = []
for s in R["E3"]["settings"]:
    rows.append([f"{s['nu']}", f"{s['sigma']}", f(s["mstar_real"], 2), f"{s['mstar_int_pred']} / {s['mstar_int_sim']}", f(s["Lstar"], 4),
                 f(s["kalman_pred"], 4), pm(s["kalman_sim"], s["kalman_hw"]), f"{s['n_ok']}/30", f"{100 * s['max_rel_err']:.2f}\\%"])
tab("e3_main", ["$\\nu$", "$\\sigma$", "$m^\\ast$", "int.\\ opt.\\ pred./sim.", "$\\bar L(m^\\ast)$", "Kalman pred.", "Kalman sim.", "$m$ verified", "max rel.\\ err."], rows, colspec="rrrrrrrrr")
rows = [[f(x["ratio"], 2), f(x["m_star"], 3), x["best_int"], f(x["pred_m1"]), f((x["best_int"] == 1) == x["pred_m1"])] for x in R["E3"]["cor4"]]
tab("e3_cor4", ["$\\sigma^2/\\nu^2$", "$m^\\ast$ (relaxed)", "best integer $m$", "predicted $m{=}1$", "consistent"], rows, colspec="rrrcc")
rows = []
s0 = R["E3"]["settings"][0]
for m, v in s0["kalman_sched"].items():
    m = int(m)
    bp = (0.5**2 * (5 * m**2 + 1) / (6 * m) + 9 / m) / 8
    rows.append([m, f(bp, 4), f(v["pred"], 4), pm(v["sim"], v["hw"]), f(v["sim"] < bp)])
tab("e3_ksched", ["$m$", "cycle-mean (Thm.~\\ref{thm:review})", "Kalman-scheduled pred.", "Kalman-scheduled sim.", "dominates"], rows, colspec="rrrrc")
e = R["E3"]["ewma"]
rows = [[f(g, 2), f(p, 4), pm(s, h)] for g, p, s, h in zip(e["g"], e["pred"], e["sim"], e["hw"])]
tab("e3_ewma", ["gain $g$", "closed form", "simulation"], rows, colspec="rrr")

# ---- E4
rows = [[x["m"], f(x["pred"], 1), f(x["lag"], 1), f"{x['min']:.0f}--{x['max']:.0f}"] for x in R["E4"]["lag"]]
tab("e4_lag", ["$m$", "$(3m-1)/2$", "enumerated mean", "range"], rows, colspec="rrrr")
rows = [[f(x["delta"], 1), x["m"], f(x["loss_sched"], 4), f(x["loss_cusum"], 4), f(x["det_rate"], 3), f(x["mean_delay"], 2), f(x["false_alarm"], 3)] for x in R["E4"]["cusum"]["rows"]]
tab("e4_cusum", ["$\\Delta/\\sigma$", "$m$", "scheduled", "CUSUM+scheduled", "detect rate", "delay (days)", "pre-jump false alarm"], rows, colspec="rrrrrrr")

# ---- E5
e5 = R["E5"]
rows = [[x["n"], f(x["pred"], 4), f(x["acc"], 4)] for x in e5["corner"]["rows"]]
tab("e5_corner", ["probe days $n$", "predicted accuracy $\\Phi(\\Delta\\mu\\sqrt n/2\\sigma)$", "simulated accuracy"], rows, colspec="rrr")

# ---- A3
a3 = R["A3"]
rows = [[x["phi"], x["best_m"] if x["best_m"] < 60 else "$\\ge 60$", f(x["best_loss"], 4), f(x["loss_m1"], 4)] for x in a3["ar1"]["rows"]]
tab("a3_ar1", ["$\\phi$", "best $m$", "loss at best $m$", "loss at $m{=}1$"], rows, colspec="rrrr")
rows = [[f(x["delta"], 2), f(x["mstar"], 3), x["best_pred"], x["best_sim"], f"{100 * x['max_rel_err']:.2f}\\%"] for x in a3["drift"]]
tab("a3_drift", ["$\\delta$", "$m^\\ast=(6\\sigma^2/13\\delta^2)^{1/3}$", "best int.\\ pred.", "best int.\\ sim.", "max rel.\\ err."], rows, colspec="rrrrr")
sea = a3["seasonal"]
rows = [[f"cycle mean, $m={m}$", f(sea[f'cycle_{m}'], 4)] for m in [1, 2, 3, 7, 14]]
rows += [[f"weekday mean, $w={w}$ weeks", f(sea[f'weekday_{w}'], 4)] for w in [2, 4, 8]]
tab("a3_seas", ["Policy", "per-day loss"], rows, colspec="lr")

# ---- A5/A8
a5 = R["A5A8"]
rows = [[f(x["tbar"], 1), f(x["delta"], 1), f(x["var_beta_pred"], 5), f(x["var_beta"], 5), f(x["var_S_pred"], 4), f(x["var_S"], 4), f(x["var_tau_pred"], 4), f(x["var_tau"], 4)]
        for x in a5["const_noise"]]
tab("a5", ["$\\bar\\tau$", "$\\delta$", "Var$\\hat\\beta$ pred.", "sim.", "Var$\\hat S$ pred.", "sim.", "Var$\\hat\\tau$ 1st-order", "sim."], rows, colspec="rrrrrrrr")
rows = [[f(x["tbar"], 1), f(x["const_noise_formula"], 4), f(x["delta_method"], 4), f(x["var_tau"], 4)] for x in a5["prop_noise"]]
tab("a8", ["$\\bar\\tau$", "const.-noise formula", "delta method (prop.\\ noise)", "simulation"], rows, colspec="rrrr")

# ---- E7
rows = []
for x in R["E7"]["rows"]:
    if x["H"] in (2, 4, 8, 16) and x["D"] in (0.0, 2.0, 8.0, 32.0):
        rows.append([x["H"], f(x["D"], 1), f(x["static"], 3), f(x["dynamic"], 3), f(x["pred_oracle"], 3), f(x["oracle"], 3),
                     f(x["plugin_pp"], 3), f(x["james_stein"], 3) if "james_stein" in x else "--"])
tab("e7", ["$H$", "$D$", "static", "dynamic", "oracle $w^\\ast$ (pred.)", "oracle $w^\\ast$ (sim.)", "plug-in PP", "James--Stein"], rows, colspec="rrrrrrrr")

# ---- E8
rows = [[f(x["kappa"], 1), f(x["cond"], 1), f(x["max_err_LD"], 1), f(x["max_err_LS"], 1), f(x["min_gap"], 5), f(x["frac_violations"], 1), f(x["G"], 3), f(x["mean_gap"], 4), f(x["mean_gap_formula"], 4)]
        for x in R["E8"]["rows"]]
tab("e8", ["$\\kappa$", "cond$(B)$", "max err $L^D$", "max err $L^S$", "min $L^D\\!-\\!L^S$", "frac.\\ $<0$", "$G$", "mean gap", "formula"], rows, colspec="rrrrrrrrr")

# ---- E9
rows = [[f(x["s_over_m"], 2), f(x["tau_plugin"], 3), f(x["tau_bayes"], 3), f(x["rev_plugin"], 3), f(x["rev_bayes"], 3), f"{x['gain_pct']:.3f}\\%"] for x in R["E9"]["rows"] if x["s_over_m"] <= 0.61]
tab("e9", ["$s/\\mu$", "plug-in toll", "Bayes toll", "exp.\\ rev.\\ plug-in", "exp.\\ rev.\\ Bayes", "gain"], rows, colspec="rrrrrr")

# a few scalar macros
mac = []
mac.append(f"\\newcommand{{\\EoneMinRho}}{{{R['E1']['min_rho']:.4f}}}")
mac.append(f"\\newcommand{{\\EfiveIntKS}}{{{R['E5']['interior_ks_min_p']:.3f}}}")
mac.append(f"\\newcommand{{\\EfiveSE}}{{{R['E5']['o2_split_se']['sim']:.3f}}}")
eo = R["E4"]["estimate_once"]
mac.append(f"\\newcommand{{\\EfourOncePred}}{{{eo['pred']:.4f}}}")
mac.append(f"\\newcommand{{\\EfourOnceSim}}{{{eo['sim']:.4f}$\\pm${eo['hw']:.4f}}}")
mac.append(f"\\newcommand{{\\EfourFA}}{{{R['E4']['cusum']['false_alarm_rate_300d']:.3f}}}")
mac.append(f"\\newcommand{{\\EfourNJc}}{{{R['E4']['cusum']['no_jump_loss_cusum']:.4f}}}")
mac.append(f"\\newcommand{{\\EfourNJs}}{{{R['E4']['cusum']['no_jump_loss_sched']:.4f}}}")
mac.append(f"\\newcommand{{\\EfiveProbeA}}{{{R['E5']['corner']['probe_cost'][0]}}}")
mac.append(f"\\newcommand{{\\EfiveProbeB}}{{{R['E5']['corner']['probe_cost'][1]}}}")
(OUT / "macros.tex").write_text("\n".join(mac) + "\n")
print("tables written:", len(list(OUT.glob('*.tex'))))
