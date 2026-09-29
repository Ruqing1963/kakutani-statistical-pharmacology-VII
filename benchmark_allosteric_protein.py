#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_allosteric_protein.py
===============================
End-to-end validation of kakutani_pharma on synthetic trajectories of a four-domain allosteric protein
(domains A1-1 | S-1 | S-2 | A1-2, 30 residues each) with known ground truth.

Model
    structure       coarse-grained C-alpha chain, four compact domains in a row
    dynamics        anisotropic network model (cutoff 12 A); frames are drawn from an Ornstein-Uhlenbeck process in the
                    normal-mode basis (relaxation time proportional to 1/lambda_k), so consecutive frames are correlated
                    as in molecular dynamics; every frame receives a random rigid-body rotation and translation
    substrate (S)   a ligand node bound by springs to the pocket of S-1 (Schur complement), plus a small linear-response
                    displacement of the pocket: a local perturbation
    effector (E)    a ligand node bound at the far end of A1-2, and stiffening of the S-1|S-2 and S-2|A1-2 interfaces:
                    a long-range allosteric perturbation
    S+E             both ligands
    control (R)     an independent apo replica (different random seed): no perturbation
    dihedrals       per-residue metastable (phi, psi, chi1) Markov chains; ligands lower the free energy of one rotamer and
                    one backbone basin near their sites (and along the interface for the effector)

Diagnostics (per ligand state L versus apo)
    DKI per residue and total (Laplace-smoothed), CKI with mean / covariance / commuting / rotation parts (Ledoit-Wolf),
    spectral fingerprint F_L, CKI(r) around the substrate pocket with split-trajectory null baseline, block-bootstrap
    confidence intervals and the distal exponent gamma, dynamic cross-correlation and contact-network maps.

Outputs: data/bench_*.csv, figures/fig07_*.{png,pdf}, results/benchmark_report.txt
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kakutani_pharma as kp  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 20260929
DOMAINS = ("A1-1", "S-1", "S-2", "A1-2")
N_PER_DOMAIN = 30
N_RES = N_PER_DOMAIN * len(DOMAINS)
CUTOFF = 13.0
K_SEQ = 3.0
KT = 0.6
N_FRAMES = 6000
TAU_SLOW = 60.0           # relaxation time of the slowest apo mode, in frames
K_LIGAND = 1.5
INTERFACE_FACTOR = 3.0
POCKET_RADIUS = 9.0
R_MIN_DISTAL = 15.0
N_BOOT = 40
N_SHELLS = 10
SHORT_FRAMES = 200        # short-trajectory demonstration (fewer frames than 3N coordinates)

DATA, FIGS, RES = (os.path.join(HERE, d) for d in ("data", "figures", "results"))
for d in (DATA, FIGS, RES):
    os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
# Structure and elastic network
# ----------------------------------------------------------------------------
def build_structure(rng) -> tuple[np.ndarray, np.ndarray]:
    centers = np.array([[0.0, 0, 0], [15.0, 3, 0], [30.0, 0, 0], [45.0, 3, 0]])
    X, dom = [], []
    """Compact globular domains: random sequential packing in a sphere of radius 9.5 A (minimum C-alpha spacing
    3.8 A, about protein density), residues threaded into a chain by a nearest-neighbour tour that starts next to
    the end of the previous domain."""
    last = None
    for k, c in enumerate(centers):
        pts = []
        while len(pts) < N_PER_DOMAIN:
            cand = c + rng.uniform(-9.5, 9.5, size=3)
            if np.linalg.norm(cand - c) > 9.5:
                continue
            if pts and np.min(np.linalg.norm(np.array(pts) - cand, axis=1)) < 3.8:
                continue
            if X and np.min(np.linalg.norm(np.array(X) - cand, axis=1)) < 3.8:
                continue
            pts.append(cand)
        pts = np.array(pts)
        start = 0 if last is None else int(np.argmin(np.linalg.norm(pts - last, axis=1)))
        order, left = [start], set(range(N_PER_DOMAIN)) - {start}
        while left:
            cur = pts[order[-1]]
            nxt = min(left, key=lambda j: np.linalg.norm(pts[j] - cur))
            order.append(nxt); left.remove(nxt)
        for j in order:
            X.append(pts[j]); dom.append(k)
        last = pts[order[-1]]
    return np.array(X), np.array(dom)


def anm_hessian(X: np.ndarray, kmat: np.ndarray) -> np.ndarray:
    N = len(X)
    H = np.zeros((3 * N, 3 * N))
    for i in range(N):
        for j in range(i + 1, N):
            if kmat[i, j] == 0:
                continue
            d = X[j] - X[i]
            blk = -kmat[i, j] * np.outer(d, d) / (d @ d)
            H[3*i:3*i+3, 3*j:3*j+3] = blk; H[3*j:3*j+3, 3*i:3*i+3] = blk
            H[3*i:3*i+3, 3*i:3*i+3] -= blk; H[3*j:3*j+3, 3*j:3*j+3] -= blk
    return H


def spring_matrix(X: np.ndarray) -> np.ndarray:
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    K = (D < CUTOFF).astype(float)
    np.fill_diagonal(K, 0)
    idx = np.arange(len(X) - 1)
    K[idx, idx + 1] = K[idx + 1, idx] = K_SEQ
    return K


def add_ligand(H: np.ndarray, X: np.ndarray, site: np.ndarray, residues: np.ndarray, k: float) -> np.ndarray:
    """Bind a free ligand node to `residues` and eliminate it (Schur complement)."""
    N = len(X)
    Xl = np.vstack([X, site])
    Kl = np.zeros((N + 1, N + 1))
    Kl[residues, N] = Kl[N, residues] = k
    Hl = anm_hessian(Xl, Kl)
    Hfull = np.zeros((3 * N + 3, 3 * N + 3))
    Hfull[:3 * N, :3 * N] = H
    Hfull += Hl
    A, B, C = Hfull[:3 * N, :3 * N], Hfull[:3 * N, 3 * N:], Hfull[3 * N:, 3 * N:]
    return A - B @ np.linalg.solve(C, B.T)


def modes(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam, V = np.linalg.eigh(H)
    return lam[6:], V[:, 6:]          # drop the six rigid-body zero modes


# ----------------------------------------------------------------------------
# Trajectories
# ----------------------------------------------------------------------------
def random_rotations(rng, n):
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    a, b, c, d = q.T
    return np.stack([np.stack([a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], -1),
                     np.stack([2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)], -1),
                     np.stack([2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d], -1)], 1)


def simulate_coordinates(rng, X0: np.ndarray, H: np.ndarray, shift: np.ndarray, zeta: float, T: int) -> np.ndarray:
    lam, V = modes(H)
    sig = np.sqrt(KT / lam)
    rho = np.exp(-lam / zeta)
    q = rng.normal(size=len(lam)) * sig
    out = np.empty((T, len(X0), 3))
    base = X0 + shift
    noise_scale = np.sqrt(1 - rho ** 2) * sig
    for t in range(T):
        q = rho * q + noise_scale * rng.normal(size=len(lam))
        out[t] = base + (V @ q).reshape(-1, 3)
    R = random_rotations(rng, T)
    return np.einsum("tkj,taj->tak", R, out) + rng.normal(scale=15.0, size=(T, 1, 3))


def simulate_dihedrals(rng, eps_rot: np.ndarray, eps_rama: np.ndarray, T: int, hop: float = 0.3) -> dict:
    """Metastable Markov chains (Metropolis between wells) for chi1 (3 wells) and phi/psi (2 basins) per residue."""
    R = eps_rot.shape[0]

    def chain(eps):
        K = eps.shape[1]
        s = np.array([rng.choice(K, p=np.exp(-e) / np.exp(-e).sum()) for e in eps])
        out = np.empty((T, R), dtype=np.int8)
        for t in range(T):
            prop = (s + rng.integers(1, K, size=R)) % K
            acc = rng.random(R) < hop * np.minimum(1.0, np.exp(eps[np.arange(R), s] - eps[np.arange(R), prop]))
            s = np.where(acc, prop, s)
            out[t] = s
        return out

    rot, rama = chain(eps_rot), chain(eps_rama)
    chi_c = np.array([60.0, 180.0, 300.0])
    phi_c, psi_c = np.array([-63.0, -120.0]), np.array([-43.0, 130.0])
    chi1 = (chi_c[rot] + rng.normal(scale=12.0, size=rot.shape)) % 360
    phi = (phi_c[rama] + rng.normal(scale=12.0, size=rama.shape)) % 360
    psi = (psi_c[rama] + rng.normal(scale=12.0, size=rama.shape)) % 360
    return {"phi": phi, "psi": psi, "chi1": chi1, "rot": rot, "rama": rama}


def equilibrium(eps):
    p = np.exp(-eps)
    return p / p.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Figure style
# ----------------------------------------------------------------------------
SURFACE, INK, INK_2, MUTED, GRID_C, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
STATE_COLOR = {"S": ORANGE, "E": BLUE, "S+E": VIOLET, "R": MUTED}
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 9, "axes.unicode_minus": False,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlelocation": "left", "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID_C, "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 7.5, "legend.labelcolor": INK_2, "mathtext.fontset": "dejavusans",
})


def tidy(ax, grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.6)
    if not grid:
        ax.grid(False)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    lines: list[str] = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    ok_all = True
    out("=" * 78)
    out("kakutani_pharma benchmark - four-domain allosteric protein (A1-1 | S-1 | S-2 | A1-2), synthetic trajectories")
    out(f"{N_RES} residues, {N_FRAMES} frames per state, ANM cutoff {CUTOFF} A, seed {SEED}")
    out("=" * 78)

    # ---------- model ----------
    X0, dom = build_structure(rng)
    K0 = spring_matrix(X0)
    H_apo = anm_hessian(X0, K0)
    n_zero = int(np.sum(np.linalg.eigvalsh(H_apo) < 1e-6 * np.linalg.eigvalsh(H_apo)[-1]))
    if n_zero != 6:
        raise RuntimeError(f"elastic network has {n_zero} zero modes (expected 6): not rigid")
    lam_apo, _ = modes(H_apo)
    zeta = TAU_SLOW * lam_apo[0]
    s1 = np.nonzero(dom == 1)[0]
    a12 = np.nonzero(dom == 3)[0]
    # substrate site: outer face of S-1; effector site: far end of A1-2
    site_S = X0[s1].mean(0) + np.array([0.0, 9.0, 0.0])
    site_E = X0[a12].mean(0) + np.array([9.0, 0.0, 0.0])
    pocket = np.nonzero(np.linalg.norm(X0 - site_S, axis=1) < POCKET_RADIUS)[0]
    eff_res = np.nonzero(np.linalg.norm(X0 - site_E, axis=1) < POCKET_RADIUS)[0]
    D = np.linalg.norm(X0[:, None] - X0[None], axis=-1)
    iface = np.zeros_like(K0, dtype=bool)
    for a, b in ((1, 2), (2, 3)):
        iface |= (dom[:, None] == a) & (dom[None, :] == b) & (D < CUTOFF)
    iface |= iface.T
    iface_res = np.nonzero(iface.any(axis=1))[0]
    out(f"\n[model] pocket (S-1) residues: {len(pocket)}, effector-site (A1-2) residues: {len(eff_res)}, "
        f"interface residues S-1|S-2|A1-2: {len(iface_res)}; slowest apo mode lambda = {lam_apo[0]:.4f}")

    def hessian(state):
        K = K0.copy()
        if "E" in state:
            K[iface] *= INTERFACE_FACTOR
        H = anm_hessian(X0, K)
        if "S" in state:
            H = add_ligand(H, X0, site_S, pocket, K_LIGAND)
        if "E" in state:
            H = add_ligand(H, X0, site_E, eff_res, K_LIGAND)
        return H

    def shift(state, H):
        lam, V = modes(H)
        f = np.zeros((N_RES, 3))
        for tag, site, res in (("S", site_S, pocket), ("E", site_E, eff_res)):
            if tag in state:
                u = site - X0[res]
                f[res] += u / np.linalg.norm(u, axis=1, keepdims=True)
        dx = (V @ ((V.T @ f.ravel()) / lam)).reshape(-1, 3)
        if np.any(f):
            dx *= 0.4 / np.sqrt(np.mean(np.sum(dx[np.any(f, axis=1)] ** 2, axis=1)))
        return dx

    base_rot = rng.normal(scale=0.8, size=(N_RES, 3)); base_rama = rng.normal(scale=0.8, size=(N_RES, 2))
    dS = np.linalg.norm(X0 - site_S, axis=1); dE = np.linalg.norm(X0 - site_E, axis=1)

    def dihedral_energies(state):
        er, ea = base_rot.copy(), base_rama.copy()
        if "S" in state:
            er[:, 0] -= 2.0 * np.exp(-dS / 4.0); ea[:, 0] -= 1.0 * np.exp(-dS / 4.0)
        if "E" in state:
            er[:, 0] -= 2.0 * np.exp(-dE / 4.0); ea[:, 0] -= 1.0 * np.exp(-dE / 4.0)
            er[iface_res, 1] -= 0.8
        return er, ea

    states = ("apo", "S", "E", "S+E", "R")
    traj, dih, eq = {}, {}, {}
    for st in states:
        H = hessian("" if st in ("apo", "R") else st)
        dx = shift("" if st in ("apo", "R") else st, H)
        srng = np.random.default_rng(SEED + 17 * (states.index(st) + 1))
        traj[st] = simulate_coordinates(srng, X0, H, dx, zeta, N_FRAMES)
        er, ea = dihedral_energies("" if st in ("apo", "R") else st)
        dih[st] = simulate_dihedrals(srng, er, ea, N_FRAMES)
        eq[st] = (equilibrium(er), equilibrium(ea))
    out(f"  trajectories simulated in {time.time() - t_start:.0f} s")

    # ---------- featurizer checks ----------
    out("\n[1] Featurizer")
    occ_err, agree_min = 0.0, 1.0
    for st in states:
        st_codes = kp.residue_states({k: dih[st][k] for k in ("phi", "psi", "chi1")})
        rot_rec = st_codes % 3
        rama_rec = st_codes // 3
        agree_min = min(agree_min, float(np.mean(rot_rec == dih[st]["rot"])), float(np.mean(rama_rec == dih[st]["rama"])))
        emp = np.stack([np.mean(dih[st]["rot"] == s, axis=0) for s in range(3)], 1)
        occ_err = max(occ_err, float(np.max(np.abs(emp - eq[st][0]))))
    bin_ok = agree_min > 0.9999
    ok_all &= bin_ok
    out(f"  dihedral binning agrees with the generating metastable states in a fraction {agree_min:.6f} of all residue-frames"
        f" (well noise 12 deg, basin boundaries >= 60 deg away)  {'PASS' if bin_ok else 'FAIL'}")
    out(f"  max |empirical rotamer occupancy - equilibrium| = {occ_err:.3f} (finite, correlated sampling)")
    aligned, ref = kp.align_trajectories([traj[s] for s in states])
    AL = dict(zip(states, aligned))
    ref_err = np.max(np.abs(np.linalg.norm(ref[:, None] - ref[None], axis=-1) - D))
    rmsf = np.sqrt(np.mean(np.sum((AL["apo"] - AL["apo"].mean(0)) ** 2, axis=2), axis=0))
    lam, V = modes(H_apo)
    rmsf_true = np.sqrt(KT * np.sum((V.reshape(N_RES, 3, -1) ** 2 / lam).sum(axis=1), axis=1))
    rmsf_corr = float(np.corrcoef(rmsf, rmsf_true)[0, 1])
    align_ok = ref_err < 1.0 and rmsf_corr > 0.95
    ok_all &= align_ok
    out(f"  joint alignment of 5 x {N_FRAMES} frames with random rigid motions: max |d_ij(ref) - d_ij(model)| = {ref_err:.3f} A, "
        f"RMSF correlation with the model = {rmsf_corr:.4f}  {'PASS' if align_ok else 'FAIL'}")
    F = {s: kp.frames_matrix(AL[s]) for s in states}
    tau = kp.integrated_autocorrelation_time((F["apo"] - F["apo"].mean(0)) @ np.linalg.svd(F["apo"] - F["apo"].mean(0), full_matrices=False)[2][0])
    block = int(math.ceil(2 * tau))
    out(f"  integrated autocorrelation time of the leading principal component: {tau:.1f} frames -> bootstrap block {block}")

    # ---------- estimators ----------
    out("\n[2] Estimators (ligand state versus apo)")
    counts = {s: kp.occupancy_counts(kp.residue_states({k: dih[s][k] for k in ("phi", "psi", "chi1")})) for s in states}
    hA, hB = N_FRAMES // 2, N_FRAMES // 2
    dki_null = kp.compute_dki(kp.occupancy_counts(kp.residue_states({k: dih["apo"][k][:hA] for k in ("phi", "psi", "chi1")})),
                              kp.occupancy_counts(kp.residue_states({k: dih["apo"][k][hA:] for k in ("phi", "psi", "chi1")}))).total
    out(f"  split-trajectory DKI null (apo first half vs second half): {dki_null:.4f}")
    raw_short = kp.cki_from_frames(F["apo"][:SHORT_FRAMES], F["S"][:SHORT_FRAMES], shrinkage=None).total
    lw_short = kp.cki_from_frames(F["apo"][:SHORT_FRAMES], F["S"][:SHORT_FRAMES])
    raw_full = kp.cki_from_frames(F["apo"], F["S"], shrinkage=None).total
    out(f"  sample covariance: CKI = {raw_full} with {N_FRAMES} frames (six rigid-body null directions after alignment), "
        f"{raw_short} with {SHORT_FRAMES} frames < 3N = {3 * N_RES}")
    out(f"  Ledoit-Wolf: finite in both cases (short run: CKI = {lw_short.total:.2f}, shrinkage {lw_short.shrinkage[0]:.3f})")
    ok_all &= np.isinf(raw_full) and np.isinf(raw_short) and np.isfinite(lw_short.total)

    rows_est, dki_rows, fps, dkis = [], [], {}, {}
    out(f"  {'state':>5} {'DKI':>8} {'CKI':>9} {'mean':>8} {'cov':>8} {'comm':>8} {'rot':>8} {'delta_LW':>8} {'top-10 DKI residues in pocket/site/interface':>44}")
    for st in ("S", "E", "S+E", "R"):
        dk = kp.compute_dki(counts["apo"], counts[st])
        dkis[st] = dk
        ck = kp.cki_from_frames(F["apo"], F[st])
        top = np.argsort(dk.per_residue)[::-1][:10]
        truth_S, truth_E = set(np.nonzero(dS < 12.0)[0]), set(np.nonzero(dE < 12.0)[0]) | set(iface_res)
        near = truth_S if st == "S" else truth_E if st == "E" else truth_S | truth_E
        hits = sum(int(i in near) for i in top)
        rows_est.append({"state": st, "DKI": dk.total, "log_affinity": dk.log_affinity, "CKI": ck.total, "mean": ck.mean_term,
                         "cov": ck.cov_term, "comm": ck.comm_term, "rot": ck.rot_term, "shrinkage_A": ck.shrinkage[0],
                         "shrinkage_B": ck.shrinkage[1], "top10_hits": hits if st != "R" else -1})
        fps[st] = kp.spectral_fingerprint(*[kp.estimators.mean_cov(F[s])[1] for s in ("apo", st)], k=20)
        out(f"  {st:>5} {dk.total:>8.4f} {ck.total:>9.3f} {ck.mean_term:>8.3f} {ck.cov_term:>8.3f} {ck.comm_term:>8.3f} {ck.rot_term:>8.3f}"
            f" {ck.shrinkage[1]:>8.4f} {(str(hits) + '/10') if st != 'R' else '(control)':>44}")
        for i in range(N_RES):
            dki_rows.append({"state": st, "residue": i, "domain": DOMAINS[dom[i]], "h_i": dk.per_residue[i],
                             "d_substrate": dS[i], "d_effector": dE[i]})
    write_csv(os.path.join(DATA, "bench_estimators.csv"), rows_est)
    write_csv(os.path.join(DATA, "bench_dki_per_residue.csv"), dki_rows)
    est = {r["state"]: r for r in rows_est}
    dki_ok = est["S"]["top10_hits"] >= 8 and est["E"]["top10_hits"] >= 8 and est["R"]["DKI"] < 3 * dki_null
    ok_all &= dki_ok
    out(f"  Acceptance: DKI top-10 residues lie in the perturbed set (energy shift >= 0.1 kT) for S and E (>= 8/10), "
        f"control DKI {est['R']['DKI']:.4f} < 3 x split null  {'PASS' if dki_ok else 'FAIL'}")
    rot_ok = all(r["rot"] >= -1e-8 for r in rows_est)
    ok_all &= rot_ok
    out(f"  Acceptance: rotation part D_rot >= 0 in every comparison (Fiedler)  {'PASS' if rot_ok else 'FAIL'}")

    # ---------- scaling ----------
    out("\n[3] Pocket-centred shell scaling with split-trajectory null and block bootstrap")
    d_pocket = kp.pocket_distance(ref, pocket)
    radii = kp.shell_radii(d_pocket, N_SHELLS, r_first=0.0)
    prof_rows, boots, splits = [], {}, {}
    out(f"  radii (A): {', '.join(f'{r:.1f}' for r in radii)}; distal fit r >= {R_MIN_DISTAL} A; block {block} frames; {N_BOOT} replicates")
    for st in ("S", "E", "S+E", "R"):
        t0 = time.time()
        sp = kp.split_null(F["apo"], F[st], d_pocket, radii)
        bs = kp.block_bootstrap(F["apo"], F[st], d_pocket, radii, R_MIN_DISTAL, block, n_boot=N_BOOT, seed=SEED + 5)
        g, _ = kp.fit_scaling_exponent(sp["n_res"], sp["excess"], radii, R_MIN_DISTAL)
        splits[st], boots[st] = sp, bs
        for i, r in enumerate(radii):
            prof_rows.append({"state": st, "r": r, "n_res": int(sp["n_res"][i]), "signal": sp["signal"][i], "null": sp["null"][i],
                              "excess": sp["excess"][i], "excess_lo": bs.excess_lo[i], "excess_hi": bs.excess_hi[i]})
        defined = bs.excess_lo[-1] > 0            # gamma is a log-log slope: undefined when the excess is consistent with 0
        gtxt = f"{g:.3f} [{bs.gamma_ci[0]:.3f}, {bs.gamma_ci[1]:.3f}]" if defined else "n/a (excess consistent with 0)"
        out(f"  {st:>5}: excess CKI at r = {radii[-1]:.1f} A (all residues) = {sp['excess'][-1]:.3f} [{bs.excess_lo[-1]:.3f}, {bs.excess_hi[-1]:.3f}]; "
            f"pocket shell {sp['excess'][0]:.3f}; distal gamma = {gtxt}  ({time.time() - t0:.0f} s)")
        sp["gamma_defined"] = defined
        splits[st]["gamma"] = g
    write_csv(os.path.join(DATA, "bench_cki_profile.csv"), prof_rows)
    rep_rows = []
    for st in ("S", "E", "S+E", "R"):
        whole = kp.block_bootstrap(F["apo"], F[st], d_pocket, radii, R_MIN_DISTAL, block, n_boot=N_BOOT, seed=SEED + 5,
                                   scheme="whole")
        out(f"  {st:>5}: resample-then-split (leaky) scheme: excess at r = {radii[-1]:.1f} A in "
            f"[{whole.excess_lo[-1]:.3f}, {whole.excess_hi[-1]:.3f}] vs estimate {splits[st]['excess'][-1]:.3f}")
        for scheme, bs in (("within-halves", boots[st]), ("whole", whole)):
            for k in range(N_BOOT):
                rep_rows.append({"state": st, "scheme": scheme, "replicate": k, "excess_pocket": bs.excess_reps[k, 0],
                                 "excess_all": bs.excess_reps[k, -1], "gamma": bs.gamma[k],
                                 "estimate_pocket": splits[st]["excess"][0], "estimate_all": splits[st]["excess"][-1],
                                 "gamma_estimate": splits[st]["gamma"], "block": block,
                                 "half_frames": N_FRAMES // 2})
    write_csv(os.path.join(DATA, "bench_bootstrap_replicates.csv"), rep_rows)
    ctrl_ok = boots["R"].excess_lo[-1] <= 0 <= boots["R"].excess_hi[-1]
    sig_ok = all(boots[s].excess_lo[-1] > 0 for s in ("S", "E", "S+E"))
    gam_ok = boots["E"].gamma_ci[0] > boots["S"].gamma_ci[1]
    ok_all &= ctrl_ok and sig_ok and gam_ok
    out(f"  Acceptance: control excess CI contains 0  {'PASS' if ctrl_ok else 'FAIL'}; ligand excess CI excludes 0 (S, E, S+E)  {'PASS' if sig_ok else 'FAIL'}")
    out(f"  Acceptance: distal exponent separates the allosteric effector from the local substrate (CI_E above CI_S)  {'PASS' if gam_ok else 'FAIL'}")

    # ---------- maps ----------
    dccm = {s: kp.dynamic_cross_correlation(AL[s]) for s in ("apo", "E", "S")}
    cn = {s: kp.contact_network(AL[s], stride=20) for s in ("apo", "E")}

    # ---------- figures ----------
    out("\n[4] Figures")
    bounds = [N_PER_DOMAIN * k - 0.5 for k in range(1, len(DOMAINS))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), dpi=200)
    for ax, M, title, cmap, lim in ((axes[0], dccm["apo"], "A  Dynamic cross-correlation, apo", "RdBu_r", 1.0),
                                    (axes[1], dccm["E"] - dccm["apo"], "B  Effector minus apo", "RdBu_r", None),
                                    (axes[2], cn["E"] - cn["apo"], "C  Contact network, effector minus apo", "PuOr_r", None)):
        v = lim if lim else float(np.max(np.abs(M)))
        im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v, origin="lower", interpolation="nearest")
        for b in bounds:
            ax.axhline(b, color=INK_2, lw=0.5); ax.axvline(b, color=INK_2, lw=0.5)
        ax.set_xticks([N_PER_DOMAIN * (k + 0.5) for k in range(4)]); ax.set_xticklabels(DOMAINS)
        ax.set_yticks([N_PER_DOMAIN * (k + 0.5) for k in range(4)]); ax.set_yticklabels(DOMAINS)
        ax.set_title(title); tidy(ax, grid=False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"fig07_correlation_maps.{ext}"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), dpi=200)
    (a1, a2), (a3, a4) = axes
    for st, mk in (("S", "s"), ("E", "o"), ("S+E", "D"), ("R", "^")):
        dk = dkis[st].per_residue
        a1.plot(np.arange(N_RES), dk, color=STATE_COLOR[st], lw=1.0, marker=mk, ms=2.5, label=f"{st} (DKI = {dk.sum():.3f})")
    for b in bounds:
        a1.axvline(b, color=AXIS, lw=0.8)
    for k, nm in enumerate(DOMAINS):
        a1.text(N_PER_DOMAIN * (k + 0.5), 1.02, nm, transform=a1.get_xaxis_transform(), ha="center", fontsize=8, color=INK_2)
    a1.set_xlabel("residue"); a1.set_ylabel(r"local index $h_i$ (Laplace-smoothed)")
    a1.set_title("A  First-order DKI per residue", pad=14)
    a1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.97), frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1.0); tidy(a1)
    for st, mk in (("S", "s"), ("E", "o"), ("S+E", "D"), ("R", "^")):
        sp, bs = splits[st], boots[st]
        a2.fill_between(sp["n_res"], bs.excess_lo, bs.excess_hi, color=STATE_COLOR[st], alpha=0.18, lw=0)
        lab = (rf"{st}: $\gamma$ = {sp['gamma']:.2f} [{bs.gamma_ci[0]:.2f}, {bs.gamma_ci[1]:.2f}]" if sp["gamma_defined"]
               else rf"{st}: excess $\approx 0$, $\gamma$ n/a")
        a2.plot(sp["n_res"], sp["excess"], color=STATE_COLOR[st], marker=mk, ms=4, lw=1.3, label=lab)
    a2.axhline(0, color=INK_2, lw=0.8)
    a2.set_xlabel(r"residues in shell $|S(r)|$ (ordered by distance from the substrate pocket)")
    a2.set_ylabel(r"excess CKI$(r)$ = signal $-$ split null")
    a2.set_title("B  Shell scaling with 95% block-bootstrap bands"); a2.legend(loc="upper left"); tidy(a2)
    sp = splits["E"]
    a3.plot(sp["n_res"], sp["signal"], color=BLUE, marker="o", ms=4, lw=1.3, label="effector vs apo (half trajectories)")
    a3.plot(sp["n_res"], sp["null_A"], color=MUTED, marker="^", ms=4, lw=1.0, ls="--", label="split null, apo halves")
    a3.plot(sp["n_res"], sp["null_B"], color=MUTED, marker="v", ms=4, lw=1.0, ls=":", label="split null, effector halves")
    a3.set_xlabel(r"residues in shell $|S(r)|$"); a3.set_ylabel(r"CKI$(r)$")
    a3.set_title("C  Signal and split-trajectory null (effector)"); a3.legend(loc="upper left"); tidy(a3)
    for st, mk in (("S", "s"), ("E", "o"), ("R", "^")):
        fp = fps[st]
        a4.plot(np.arange(1, len(fp.dlnlam) + 1), fp.dlnlam, color=STATE_COLOR[st], marker=mk, ms=4, lw=1.2,
                label=rf"{st}: $\Delta\ln\lambda_i$")
    a4.axhline(0, color=INK_2, lw=0.8)
    a4b = a4.twinx()
    a4b.plot(np.arange(1, 21), fps["E"].theta, color=BLUE, lw=0.8, ls=":", marker=".", label=r"E: $\theta_i$ (deg)")
    a4b.set_ylabel(r"eigenvector rotation $\theta_i$ (deg), effector", color=INK_2); a4b.grid(False)
    a4.set_xticks(range(1, 21, 2))
    a4.set_xlabel("mode index $i$ (covariance eigenvalues, descending)"); a4.set_ylabel(r"$\Delta\ln\lambda_i$ versus apo")
    a4.set_title(r"D  Spectral fingerprint $\mathcal{F}_L=\{\Delta\ln\lambda_i,\theta_i\}$")
    h1, l1 = a4.get_legend_handles_labels(); h2, l2 = a4b.get_legend_handles_labels()
    a4.legend(h1 + h2, l1 + l2, loc="lower right"); tidy(a4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"fig07_kakutani_diagnostics.{ext}"))
    plt.close(fig)
    out("  figures/fig07_correlation_maps.{png,pdf}, figures/fig07_kakutani_diagnostics.{png,pdf}")

    out(f"\nruntime {time.time() - t_start:.0f} s")
    out("OVERALL ACCEPTANCE: " + ("PASS" if ok_all else "FAIL"))
    with open(os.path.join(RES, "benchmark_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
