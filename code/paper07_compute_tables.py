#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paper07_compute_tables.py
=========================
Analytic companion of Paper VII. Reads the benchmark output (data/bench_*.csv, results/benchmark_report.txt),
recomputes the quantities quoted in the paper and writes the table bodies.

    Section 2   zero modes of the elastic network: the original loose random-walk geometry versus the compact
                four-domain geometry, and the count dim ker H = 12 - c for two rigid halves joined by c springs
    Section 3   Ledoit-Wolf against scikit-learn for T < D and T > D, positive definiteness, three-way CKI decomposition
    Section 4   bootstrap bias, percentile / bias-shifted / basic intervals, leakage of the resample-then-split scheme
    Section 5   sample and population shell curves CKI(r), distal exponents, DKI localisation

Outputs (results/): paper07_table{1,2,3,4}_rows.tex, paper07_tables.json, paper07_tables_summary.txt
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

import numpy as np
from scipy.linalg import null_space

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import kakutani_pharma as kp  # noqa: E402
import benchmark_allosteric_protein as bench  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA, RESULTS = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
STATE_NAME = {"S": "substrate", "E": "effector", "S+E": "both", "R": "apo replica"}


def read_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def m(x, d=3):
    """number in math mode (so that minus signs are typeset as minus signs)"""
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return r"$\infty$" if x > 0 else r"$-\infty$"
    if round(x, d) == 0:
        x = 0.0
    return f"${x:.{d}f}$"


# ----------------------------------------------------------------------------
# geometries
# ----------------------------------------------------------------------------
def loose_structure(rng):
    """The original construction: confined random walks (radius 8.5 A, spacing 4 A) that silently accept the last
    candidate when no valid step is found."""
    centers = np.array([[0.0, 0, 0], [15.0, 3, 0], [30.0, 0, 0], [45.0, 3, 0]])
    X, dom = [], []
    for k, c in enumerate(centers):
        pos = c + rng.normal(scale=2.0, size=3)
        n = 0
        while n < bench.N_PER_DOMAIN:
            for _ in range(2000):
                step = rng.normal(size=3)
                cand = pos + 3.8 * step / np.linalg.norm(step)
                if np.linalg.norm(cand - c) > 8.5:
                    continue
                if X and np.min(np.linalg.norm(np.array(X) - cand, axis=1)) < 4.0:
                    continue
                break
            X.append(cand); dom.append(k); pos = cand; n += 1
    return np.array(X), np.array(dom)


def zero_modes(H, rel=1e-6):
    ev = np.linalg.eigvalsh(H)
    return int(np.sum(ev < rel * ev[-1])), ev


def rigidity_row(name, X, dom):
    K = bench.spring_matrix(X)
    H = bench.anm_hessian(X, K)
    nz, ev = zero_modes(H)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    cont = {f"{a}|{b}": int(((dom[:, None] == a) & (dom[None, :] == b) & (D < bench.CUTOFF)).sum()) for a, b in ((0, 1), (1, 2), (2, 3), (0, 3))}
    left, right = np.nonzero(dom <= 1)[0], np.nonzero(dom >= 2)[0]
    c = int(np.count_nonzero(K[np.ix_(left, right)]))
    sub = []
    for part in (left, right):
        Kp = np.zeros_like(K); Kp[np.ix_(part, part)] = K[np.ix_(part, part)]
        Hp = bench.anm_hessian(X, Kp)
        idx = (3 * part[:, None] + np.arange(3)).ravel()
        sub.append(zero_modes(Hp[np.ix_(idx, idx)])[0])
    few = int(np.sum((K > 0).sum(axis=1) < 3))
    return {"geometry": name, "zero_modes": nz, "contacts": cont, "cross_springs_S1_S2": c, "ker_left": sub[0], "ker_right": sub[1],
            "predicted": 6 + 6 - c if c < 6 and sub == [6, 6] else 6, "first_nonzero": float(ev[nz]), "lambda_max": float(ev[-1]),
            "residues_few_springs": few, "spectrum_head": ev[:12].tolist()}


# ----------------------------------------------------------------------------
# model reconstruction (identical to benchmark_allosteric_protein.main)
# ----------------------------------------------------------------------------
def model():
    rng = np.random.default_rng(bench.SEED)
    X0, dom = bench.build_structure(rng)
    K0 = bench.spring_matrix(X0)
    s1, a12 = np.nonzero(dom == 1)[0], np.nonzero(dom == 3)[0]
    site_S = X0[s1].mean(0) + np.array([0.0, 9.0, 0.0])
    site_E = X0[a12].mean(0) + np.array([9.0, 0.0, 0.0])
    pocket = np.nonzero(np.linalg.norm(X0 - site_S, axis=1) < bench.POCKET_RADIUS)[0]
    eff_res = np.nonzero(np.linalg.norm(X0 - site_E, axis=1) < bench.POCKET_RADIUS)[0]
    D = np.linalg.norm(X0[:, None] - X0[None], axis=-1)
    iface = np.zeros_like(K0, dtype=bool)
    for a, b in ((1, 2), (2, 3)):
        iface |= (dom[:, None] == a) & (dom[None, :] == b) & (D < bench.CUTOFF)
    iface |= iface.T

    def hessian(state):
        K = K0.copy()
        if "E" in state:
            K[iface] *= bench.INTERFACE_FACTOR
        H = bench.anm_hessian(X0, K)
        if "S" in state:
            H = bench.add_ligand(H, X0, site_S, pocket, bench.K_LIGAND)
        if "E" in state:
            H = bench.add_ligand(H, X0, site_E, eff_res, bench.K_LIGAND)
        return H

    def shift(state, H):
        lam, V = bench.modes(H)
        f = np.zeros((len(X0), 3))
        for tag, site, res in (("S", site_S, pocket), ("E", site_E, eff_res)):
            if tag in state:
                u = site - X0[res]
                f[res] += u / np.linalg.norm(u, axis=1, keepdims=True)
        dx = (V @ ((V.T @ f.ravel()) / lam)).reshape(-1, 3)
        if np.any(f):
            dx *= 0.4 / np.sqrt(np.mean(np.sum(dx[np.any(f, axis=1)] ** 2, axis=1)))
        return dx, f

    return X0, dom, pocket, eff_res, iface, hessian, shift


def population_profile(X0, C_A, C_B, dmu, d, radii):
    """Exact Bhattacharyya distance of the model Gaussians restricted to S(r); the full set is evaluated in the
    (3N - 6)-dimensional internal subspace orthogonal to the rigid-body motions."""
    N = len(X0)
    Xc = X0 - X0.mean(0)
    T = np.zeros((3 * N, 6))
    for k in range(3):
        T[k::3, k] = 1.0
    for k, axis in enumerate(np.eye(3)):
        T[:, 3 + k] = np.cross(axis, Xc).ravel()
    W = null_space(T.T)
    out = []
    for r in radii:
        res = np.nonzero(d <= r + 1e-9)[0]
        if len(res) == N:
            A, B, mu = W.T @ C_A @ W, W.T @ C_B @ W, W.T @ dmu
        else:
            cols = kp.scaling.coord_index(res)
            A, B, mu = C_A[np.ix_(cols, cols)], C_B[np.ix_(cols, cols)], dmu[cols]
        c = kp.compute_cki(np.zeros(len(mu)), A, mu, B, decompose=False)
        out.append((len(res), c.total, c.mean_term, c.cov_term))
    return np.array(out)


def main() -> int:
    lines: list[str] = []

    def out(s=""):
        print(s)
        lines.append(s)

    summary = {}
    out("=" * 78)
    out("Paper VII companion: analytic constants and table bodies")
    out("=" * 78)

    # ---------------- Section 2 ----------------
    out("\n[Section 2] Zero modes of the elastic network (ANM cutoff %.0f A)" % bench.CUTOFF)
    loose = rigidity_row("loose random walk", *loose_structure(np.random.default_rng(bench.SEED)))
    compact = rigidity_row("compact globules", *bench.build_structure(np.random.default_rng(bench.SEED)))
    t1 = []
    for r in (loose, compact):
        out(f"  {r['geometry']:>18}: dim ker H = {r['zero_modes']}; contacts A1-1|S-1 {r['contacts']['0|1']}, S-1|S-2 {r['contacts']['1|2']}, "
            f"S-2|A1-2 {r['contacts']['2|3']}, A1-1|A1-2 {r['contacts']['0|3']}; springs across the S-1|S-2 cut c = {r['cross_springs_S1_S2']}; "
            f"dim ker of the two halves = {r['ker_left']}, {r['ker_right']}; first nonzero eigenvalue {r['first_nonzero']:.4f}")
        out(f"      eigenvalues 1..12: {', '.join(f'{x:.2e}' for x in r['spectrum_head'])}")
        t1.append(f"{r['geometry']} & {r['contacts']['0|1']} & {r['contacts']['1|2']} & {r['contacts']['2|3']} & {r['contacts']['0|3']} & "
                  f"{r['cross_springs_S1_S2']} & {r['ker_left']}, {r['ker_right']} & {r['zero_modes']} & {m(r['first_nonzero'], 4)} \\\\")
    pred_ok = loose["zero_modes"] == 12 - loose["cross_springs_S1_S2"] and compact["zero_modes"] == 6
    out(f"  loose geometry: two rigid halves joined by c = {loose['cross_springs_S1_S2']} spring(s) -> dim ker H = 12 - c = "
        f"{12 - loose['cross_springs_S1_S2']} (observed {loose['zero_modes']}): {pred_ok}")
    with open(os.path.join(RESULTS, "paper07_table1_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(t1) + "\n")
    summary["section2"] = {"loose": loose, "compact": compact, "prediction_holds": pred_ok}

    # ---------------- Section 3 ----------------
    out("\n[Section 3] Ledoit-Wolf shrinkage and the three-way decomposition")
    from sklearn.covariance import ledoit_wolf as sk_lw
    rng = np.random.default_rng(7)
    lw_rows = []
    for n, p in ((200, 360), (100, 360), (6000, 360)):
        Fm = rng.normal(size=(n, p)) @ (np.eye(p) + 0.3 * rng.normal(size=(p, p)) / np.sqrt(p))
        C, dl = kp.ledoit_wolf(Fm)
        Cs, ds = sk_lw(Fm)
        S = np.cov(Fm.T, bias=True)
        mu = np.trace(S) / p
        lmin_S, lmin = np.linalg.eigvalsh(S)[0], np.linalg.eigvalsh(C)[0]
        lw_rows.append((n, p, dl, abs(dl - ds), float(np.max(np.abs(C - Cs)) / np.max(np.abs(Cs))), lmin_S, lmin, dl * mu))
        out(f"  n = {n:>5}, p = {p}: delta = {dl:.4f} (|diff to sklearn| {abs(dl - ds):.1e}, rel. matrix diff {lw_rows[-1][4]:.1e}); "
            f"lambda_min(S) = {lmin_S:.2e}, lambda_min(Sigma*) = {lmin:.4f} >= delta mu = {dl * mu:.4f}")
    est = {r["state"]: r for r in read_csv("bench_estimators.csv")}
    rep = open(os.path.join(RESULTS, "benchmark_report.txt"), encoding="utf-8").read()
    mshort = re.search(r"short run: CKI = ([0-9.]+), shrinkage ([0-9.]+)", rep)
    short_cki, short_delta = float(mshort.group(1)), float(mshort.group(2))
    out(f"  benchmark, {bench.SHORT_FRAMES} frames < 3N = {3 * bench.N_RES}: sample covariance CKI = inf, Ledoit-Wolf CKI = {short_cki} (delta = {short_delta})")
    t2 = []
    for st in ("S", "E", "S+E", "R"):
        r = est[st]
        closure = r["mean"] + r["comm"] + r["rot"] - r["CKI"]
        out(f"  {STATE_NAME[st]:>12}: CKI {r['CKI']:.3f} = mean {r['mean']:.3f} + comm {r['comm']:.3f} + rot {r['rot']:.3f} (closure {closure:.1e}); "
            f"delta_LW {r['shrinkage_B']:.4f}; DKI {r['DKI']:.4f}")
        t2.append(f"{STATE_NAME[st]} & {m(r['DKI'], 4)} & {m(r['CKI'])} & {m(r['mean'])} & {m(r['comm'])} & {m(r['rot'])} & "
                  f"{m(r['shrinkage_A'], 4)} & {m(r['shrinkage_B'], 4)} \\\\")
    with open(os.path.join(RESULTS, "paper07_table2_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(t2) + "\n")
    t2b = []
    for n, p, dl, dd, rel, ls, lm, dm in lw_rows:
        e_dd = int(math.floor(math.log10(dd))) if dd > 0 else 0
        e_ls = int(math.floor(math.log10(abs(ls)))) if ls != 0 else 0
        dd_s = rf"${dd / 10 ** e_dd:.1f}\times10^{{{e_dd}}}$" if dd > 0 else "$0$"
        ls_s = rf"${ls / 10 ** e_ls:.1f}\times10^{{{e_ls}}}$" if abs(ls) < 1e-3 else m(ls, 4)
        e_rel = int(math.floor(math.log10(rel))) if rel > 0 else 0
        rel_s = rf"${rel / 10 ** e_rel:.1f}\times10^{{{e_rel}}}$" if rel > 0 else "$0$"
        t2b.append(f"{n} & {p} & {m(dl, 4)} & {dd_s} & {rel_s} & {ls_s} & {m(lm, 4)} & {m(dm, 4)} \\\\")
    with open(os.path.join(RESULTS, "paper07_table2b_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(t2b) + "\n")
    summary["section3"] = {"lw_checks": lw_rows, "short_run": [short_cki, short_delta], "estimators": est}

    # ---------------- Section 4 ----------------
    out("\n[Section 4] Split-trajectory null and the block bootstrap")
    reps = read_csv("bench_bootstrap_replicates.csv")
    block, half = int(reps[0]["block"]), int(reps[0]["half_frames"])
    mb = math.ceil(half / block)
    distinct = 1 - (1 - 1 / mb) ** mb
    leak = 1 - (1 - 1 / (2 * mb)) ** mb
    out(f"  half = {half} frames, block = {block} frames, m = {mb} blocks per half")
    out(f"  within-halves: expected fraction of distinct blocks in a half = 1 - (1 - 1/m)^m = {distinct:.3f}")
    out(f"  whole-trajectory resampling: a block of half 1 recurs in half 2 with probability ~ 1 - (1 - 1/2m)^m = {leak:.3f}")
    t3 = []
    for st in ("S", "E", "S+E", "R"):
        row = {}
        for scheme in ("within-halves", "whole"):
            x = np.array([r["excess_all"] for r in reps if r["state"] == st and r["scheme"] == scheme])
            est_all = [r["estimate_all"] for r in reps if r["state"] == st][0]
            lo, hi = np.quantile(x, [0.025, 0.975])
            bias = float(x.mean() - est_all)
            row[scheme] = (est_all, lo, hi, bias, lo <= est_all <= hi)
        e, lo, hi, bias, cov = row["within-halves"]
        _, wlo, whi, wbias, wcov = row["whole"]
        blo, bhi = lo - bias, hi - bias
        clo, chi = 2 * e - hi, 2 * e - lo
        out(f"  {STATE_NAME[st]:>12}: estimate {e:.3f}; within-halves percentile [{lo:.3f}, {hi:.3f}] (contains: {cov}), bias {bias:+.3f}, "
            f"bias-shifted [{blo:.3f}, {bhi:.3f}], basic [{clo:.3f}, {chi:.3f}]; whole-trajectory [{wlo:.3f}, {whi:.3f}] (contains: {wcov}, bias {wbias:+.3f})")
        sg = lambda x: f"${x:+.3f}$"  # noqa: E731  explicit sign for bias columns
        t3.append(f"{STATE_NAME[st]} & {m(e)} & [{m(lo)}, {m(hi)}] & {sg(bias)} & [{m(blo)}, {m(bhi)}] & [{m(clo)}, {m(chi)}] & "
                  f"[{m(wlo)}, {m(whi)}] & {sg(wbias)} \\\\")
        summary.setdefault("section4", {})[st] = {"estimate": e, "percentile": [lo, hi], "bias": bias, "shifted": [blo, bhi],
                                                  "basic": [clo, chi], "whole": [wlo, whi], "whole_bias": wbias,
                                                  "whole_contains": bool(wcov), "within_contains": bool(cov)}
    with open(os.path.join(RESULTS, "paper07_table3_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(t3) + "\n")
    summary["section4_mechanism"] = {"block": block, "half": half, "m": mb, "distinct": distinct, "leak": leak}

    # ---------------- Section 5 ----------------
    out("\n[Section 5] Shell scaling: sample excess curves and exact population curves")
    prof = read_csv("bench_cki_profile.csv")
    X0, dom, pocket, eff_res, iface, hessian, shift = model()
    H0 = hessian("")
    lam0, V0 = bench.modes(H0)
    C0 = bench.KT * (V0 / lam0) @ V0.T
    d0 = kp.pocket_distance(X0, pocket)
    radii = np.array(sorted({r["r"] for r in prof}))
    t4 = []
    for st in ("S", "E", "S+E", "R"):
        rows = sorted([r for r in prof if r["state"] == st], key=lambda r: r["r"])
        g_hat = [r["gamma_estimate"] for r in reps if r["state"] == st][0]
        gs = np.array([r["gamma"] for r in reps if r["state"] == st and r["scheme"] == "within-halves"])
        gs = gs[np.isfinite(gs)]
        defined = rows[-1]["excess_lo"] > 0
        if st == "R":
            pop = np.array([[len(np.nonzero(d0 <= r + 1e-9)[0]), 0.0, 0.0, 0.0] for r in radii])
            support = 0
        else:
            H = hessian(st)
            lam, V = bench.modes(H)
            C = bench.KT * (V / lam) @ V.T
            dx, f = shift(st, H)
            dH = np.abs(H - H0).reshape(bench.N_RES, 3, bench.N_RES, 3).sum(axis=(1, 3))
            support = int(np.sum(dH.sum(axis=1) > 1e-10))
            pop = population_profile(X0, C0, C, dx.ravel(), d0, radii)
        gp = kp.fit_scaling_exponent(pop[:, 0], pop[:, 1], radii, bench.R_MIN_DISTAL)[0] if st != "R" else float("nan")
        dki = est[st]
        hits = int(dki["top10_hits"]) if st != "R" else None
        gtxt = f"{g_hat:.3f} [{np.quantile(gs, 0.025):.3f}, {np.quantile(gs, 0.975):.3f}]" if defined else "n/a"
        out(f"  {STATE_NAME[st]:>12}: Hessian change supported on {support} residues; sample excess pocket {rows[0]['excess']:.3f} -> all {rows[-1]['excess']:.3f}, "
            f"gamma {gtxt}; population CKI pocket {pop[0, 1]:.3f} -> all {pop[-1, 1]:.3f} (ratio {pop[-1, 1] / pop[0, 1] if pop[0, 1] else float('nan'):.3f}), "
            f"population gamma {gp:.3f}; DKI {dki['DKI']:.4f}, top-10 hits {hits}")
        gcell = (f"{m(g_hat, 2)} [{m(float(np.quantile(gs, 0.025)), 2)}, {m(float(np.quantile(gs, 0.975)), 2)}]" if defined else "n/a")
        t4.append(f"{STATE_NAME[st]} & {support if st != 'R' else '--'} & {m(rows[0]['excess'])} & {m(rows[-1]['excess'])} & "
                  f"[{m(rows[-1]['excess_lo'])}, {m(rows[-1]['excess_hi'])}] & {gcell} & "
                  f"{m(pop[0, 1]) if st != 'R' else '$0$'} & {m(pop[-1, 1]) if st != 'R' else '$0$'} & {m(gp, 2) if st != 'R' else 'n/a'} & "
                  f"{m(dki['DKI'], 3)} & {str(hits) + '/10' if hits is not None else '--'} \\\\")
        summary.setdefault("section5", {})[st] = {"support": support, "excess_pocket": rows[0]["excess"], "excess_all": rows[-1]["excess"],
                                                  "gamma": g_hat if defined else None, "pop_pocket": float(pop[0, 1]),
                                                  "pop_all": float(pop[-1, 1]), "pop_gamma": gp, "pop_curve": pop.tolist()}
    with open(os.path.join(RESULTS, "paper07_table4_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(t4) + "\n")
    t4b = []
    curves = {st: sorted([r for r in prof if r["state"] == st], key=lambda r: r["r"]) for st in ("S", "E", "S+E")}
    for i, r0 in enumerate(radii):
        cells = [m(r0, 1), str(int(curves["S"][i]["n_res"]))]
        for st in ("S", "E", "S+E"):
            cells += [m(curves[st][i]["excess"]), m(summary["section5"][st]["pop_curve"][i][1])]
        t4b.append(" & ".join(cells) + " \\\\")
    with open(os.path.join(RESULTS, "paper07_table4b_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(t4b) + "\n")

    with open(os.path.join(RESULTS, "paper07_tables.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(RESULTS, "paper07_tables_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwritten: results/paper07_table{1,2,3,4}_rows.tex, paper07_tables.json, paper07_tables_summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
