"""
scaling.py
==========
Spatially ordered finite-size scaling of the correlated Kakutani index.

Residues are ordered by their distance d(i, pocket) from the binding pocket; the degree-of-freedom set grows shell by
shell, S(r) = {i : d(i, pocket) <= r}, and CKI(r) is the index restricted to the Cartesian coordinates of S(r).
The distal scaling exponent gamma is the log-log slope of CKI(r) against |S(r)| beyond a radius r_min.

Finite sampling makes CKI positive even for two samples of the same ensemble. Two corrections are provided:
    split_null        contiguous halves of each trajectory: CKI(A1, A2), CKI(B1, B2) give the null baseline, and
                      CKI(A1, B1), CKI(A2, B2) give the signal at the same sample size; excess = signal - null
    block_bootstrap   moving-block resampling inside each half of both trajectories (blocks longer than the autocorrelation time)
                      gives percentile confidence intervals for the excess curve and for gamma.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimators import cki_from_frames


def pocket_distance(ref_ca: np.ndarray, pocket: list[int] | np.ndarray) -> np.ndarray:
    """d(i, pocket) = min_{j in pocket} |r_i - r_j| on the reference structure (Angstrom); 0 for pocket residues."""
    d = np.linalg.norm(ref_ca[:, None, :] - ref_ca[None, np.asarray(pocket), :], axis=-1)
    return d.min(axis=1)


def shell_radii(d: np.ndarray, n_shells: int = 12, r_first: float | None = None) -> np.ndarray:
    """Radii from the pocket shell to the farthest residue, spaced so that shells hold similar residue counts."""
    r0 = r_first if r_first is not None else max(np.sort(d)[min(len(d) - 1, 5)], 1e-6)
    qs = np.quantile(d[d >= r0], np.linspace(0, 1, n_shells))
    return np.unique(np.r_[r0, qs[1:]])


def coord_index(res: np.ndarray) -> np.ndarray:
    res = np.asarray(res)
    return (3 * res[:, None] + np.arange(3)[None, :]).ravel()


def cki_profile(F_A: np.ndarray, F_B: np.ndarray, d: np.ndarray, radii: np.ndarray,
                shrinkage: str | None = "ledoit-wolf") -> dict[str, np.ndarray]:
    """CKI(r) and its mean / covariance parts for every radius; F are (n_frames, 3N) aligned frame matrices."""
    out = {k: np.zeros(len(radii)) for k in ("cki", "mean", "cov", "n_res")}
    for i, r in enumerate(radii):
        res = np.nonzero(d <= r + 1e-9)[0]
        cols = coord_index(res)
        c = cki_from_frames(F_A[:, cols], F_B[:, cols], shrinkage=shrinkage, decompose=False)
        out["cki"][i], out["mean"][i], out["cov"][i], out["n_res"][i] = c.total, c.mean_term, c.cov_term, len(res)
    out["radii"] = np.asarray(radii, float)
    return out


def fit_scaling_exponent(n_res: np.ndarray, cki: np.ndarray, radii: np.ndarray, r_min: float) -> tuple[float, float]:
    """Least-squares slope gamma of ln CKI(r) against ln |S(r)| over the distal radii r >= r_min."""
    m = (radii >= r_min) & (cki > 0) & np.isfinite(cki)
    if m.sum() < 2:
        return np.nan, np.nan
    g, c = np.polyfit(np.log(n_res[m]), np.log(cki[m]), 1)
    return float(g), float(c)


def split_null(F_A: np.ndarray, F_B: np.ndarray, d: np.ndarray, radii: np.ndarray,
               shrinkage: str | None = "ledoit-wolf") -> dict[str, np.ndarray]:
    """Split-trajectory null baseline and the equal-sample-size signal; excess = signal - null."""
    hA, hB = len(F_A) // 2, len(F_B) // 2
    A1, A2, B1, B2 = F_A[:hA], F_A[hA:2 * hA], F_B[:hB], F_B[hB:2 * hB]
    nullA = cki_profile(A1, A2, d, radii, shrinkage)["cki"]
    nullB = cki_profile(B1, B2, d, radii, shrinkage)["cki"]
    s1 = cki_profile(A1, B1, d, radii, shrinkage)
    s2 = cki_profile(A2, B2, d, radii, shrinkage)["cki"]
    null = 0.5 * (nullA + nullB)
    signal = 0.5 * (s1["cki"] + s2)
    return {"radii": np.asarray(radii, float), "n_res": s1["n_res"], "null": null, "null_A": nullA, "null_B": nullB,
            "signal": signal, "excess": signal - null}


def integrated_autocorrelation_time(x: np.ndarray, max_lag: int | None = None) -> float:
    """Integrated autocorrelation time (frames) of a scalar series with Sokal's automatic window (c = 5)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for m in range(1, max_lag or n):
        tau = 1 + 2 * np.sum(acf[1:m + 1])
        if m >= 5 * tau:
            break
    return float(max(tau, 1.0))


def block_resample(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap index of length n (blocks of `block` consecutive frames)."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nb)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


@dataclass
class BootstrapResult:
    radii: np.ndarray
    excess_lo: np.ndarray
    excess_hi: np.ndarray
    excess_median: np.ndarray
    gamma: np.ndarray            # bootstrap distribution of the distal exponent of the excess curve
    gamma_ci: tuple[float, float]
    block: int
    excess_reps: np.ndarray      # (n_boot, n_radii) replicate excess curves
    scheme: str


def block_bootstrap(F_A: np.ndarray, F_B: np.ndarray, d: np.ndarray, radii: np.ndarray, r_min: float,
                    block: int, n_boot: int = 100, level: float = 0.95, seed: int = 0,
                    shrinkage: str | None = "ledoit-wolf", scheme: str = "within-halves") -> BootstrapResult:
    """Percentile confidence intervals of the split-null excess curve and of its distal exponent.

    scheme = "within-halves" (default) resamples blocks separately inside each contiguous half, so that no frame can
    enter both halves of a replicate. scheme = "whole" resamples the whole trajectory before splitting; it is kept
    only to demonstrate the cross-half leakage that deflates the null (see Paper VII, Theorem 4.1)."""
    if scheme not in ("within-halves", "whole"):
        raise ValueError(scheme)
    rng = np.random.default_rng(seed)
    ex, gam = [], []

    def resample(n):
        if scheme == "whole":
            return block_resample(n, block, rng)
        h = n // 2
        return np.r_[block_resample(h, block, rng), h + block_resample(h, block, rng)]

    for _ in range(n_boot):
        ia, ib = resample(len(F_A)), resample(len(F_B))
        s = split_null(F_A[ia], F_B[ib], d, radii, shrinkage)
        ex.append(s["excess"])
        gam.append(fit_scaling_exponent(s["n_res"], s["excess"], s["radii"], r_min)[0])
    ex, gam = np.array(ex), np.array(gam)
    a = (1 - level) / 2
    g = gam[np.isfinite(gam)]
    gci = (float(np.quantile(g, a)), float(np.quantile(g, 1 - a))) if g.size else (np.nan, np.nan)
    return BootstrapResult(np.asarray(radii, float), np.quantile(ex, a, axis=0), np.quantile(ex, 1 - a, axis=0),
                           np.median(ex, axis=0), gam, gci, block, ex, scheme)
