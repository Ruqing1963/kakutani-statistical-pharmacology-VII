"""
estimators.py
=============
Kakutani indices between two conformational ensembles A and B.

    compute_dki     first-order (marginal) drug Kakutani index of Paper I, generalised to categorical
                    metastable states: h_i = 2 (1 - sum_s sqrt(p_is q_is)), DKI = sum_i h_i,
                    Kakutani affinity Pi = prod_i (1 - h_i / 2); Laplace (add-alpha) smoothing of the counts.
    ledoit_wolf     analytical Ledoit-Wolf (2004) shrinkage towards a scaled identity.
    compute_cki     second-order (correlated) Kakutani index of Paper II: the Bhattacharyya distance between
                    Gaussian ensembles, D = D_mean + D_cov, with D_cov = D_comm + D_rot (commuting spectral part and
                    eigenvector-rotation part, D_rot >= 0).
    spectral_fingerprint   F_L = {Delta ln lambda_i, theta_i}: log-eigenvalue shifts and eigenvector rotation angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve, subspace_angles


# ----------------------------------------------------------------------------
# First order
# ----------------------------------------------------------------------------
@dataclass
class DKIResult:
    total: float                 # DKI = sum_i h_i
    per_residue: np.ndarray      # h_i
    log_affinity: float          # ln Pi = sum_i ln(1 - h_i / 2)
    p_A: np.ndarray
    p_B: np.ndarray


def laplace_smooth(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """(n_is + alpha) / (n_i + K alpha) for a (n_residues, K) count matrix."""
    counts = np.asarray(counts, float)
    return (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * counts.shape[1])


def compute_dki(p_A: np.ndarray, p_B: np.ndarray, n_A: int | None = None, n_B: int | None = None,
                alpha: float = 1.0) -> DKIResult:
    """First-order DKI from occupancies p (n_residues, n_states).

    If n_A / n_B (numbers of frames) are given, p is interpreted as observed frequencies and Laplace-smoothed
    with pseudo-count alpha; if p already holds integer counts, pass n_A = n_B = None and counts are detected
    (rows summing to more than one). Unsmoothed probabilities are used only if both n are None and rows sum to 1.
    """
    p_A, p_B = np.asarray(p_A, float), np.asarray(p_B, float)
    if p_A.ndim == 1:                      # Bernoulli occupancies of a two-state switch per residue
        p_A, p_B = p_A[:, None], p_B[:, None]
    if p_A.shape[1] == 1:
        p_A, p_B = np.hstack([p_A, 1 - p_A]), np.hstack([p_B, 1 - p_B])

    def prep(p, n):
        if n is not None:
            return laplace_smooth(p * n, alpha)
        if np.any(p.sum(axis=1) > 1 + 1e-9):
            return laplace_smooth(p, alpha)
        return p / p.sum(axis=1, keepdims=True)

    a, b = prep(p_A, n_A), prep(p_B, n_B)
    bc = np.clip(np.sum(np.sqrt(a * b), axis=1), 0.0, 1.0)
    h = 2 * (1 - bc)
    with np.errstate(divide="ignore"):
        log_aff = float(np.sum(np.log(bc)))
    return DKIResult(float(h.sum()), h, log_aff, a, b)


# ----------------------------------------------------------------------------
# Shrinkage
# ----------------------------------------------------------------------------
def ledoit_wolf(F: np.ndarray, assume_centered: bool = False) -> tuple[np.ndarray, float]:
    """Analytical Ledoit-Wolf shrinkage estimator for frames F (n_samples, p).

    Sigma* = (1 - delta) S + delta * mu I, mu = tr(S) / p,
    delta = min(1, b^2 / d^2), d^2 = ||S - mu I||_F^2, b^2 = (1 / n^2) sum_k ||x_k x_k^T - S||_F^2,
    evaluated in O(n p + p^2) via sum_k ||x_k x_k^T - S||_F^2 = sum_k ||x_k||^4 - n ||S||_F^2.
    Identical to sklearn.covariance.ledoit_wolf.
    """
    X = np.asarray(F, float)
    if not assume_centered:
        X = X - X.mean(axis=0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    d2 = np.sum(S ** 2) - 2 * mu * np.trace(S) + p * mu ** 2
    x2 = np.sum(X ** 2, axis=1)
    b2 = (np.sum(x2 ** 2) - n * np.sum(S ** 2)) / n ** 2
    delta = 0.0 if d2 <= 0 else float(min(1.0, max(b2, 0.0) / d2))
    Sig = (1 - delta) * S
    Sig[np.diag_indices(p)] += delta * mu
    return Sig, delta


def mean_cov(F: np.ndarray, shrinkage: str | None = "ledoit-wolf") -> tuple[np.ndarray, np.ndarray, float]:
    mu = F.mean(axis=0)
    if shrinkage in (None, "none"):
        Fc = F - mu
        return mu, Fc.T @ Fc / len(F), 0.0
    if shrinkage == "ledoit-wolf":
        C, d = ledoit_wolf(F)
        return mu, C, d
    raise ValueError(shrinkage)


# ----------------------------------------------------------------------------
# Second order
# ----------------------------------------------------------------------------
@dataclass
class CKIResult:
    total: float          # D_B = D_mean + D_cov
    mean_term: float      # (1/8) dmu^T Sbar^-1 dmu
    cov_term: float       # (1/2) [ln det Sbar - (ln det C_A + ln det C_B) / 2]
    comm_term: float      # sum_i (1/2) ln cosh(eta_i / 2), eta_i = ln(lambda_i^B / lambda_i^A), sorted spectra
    rot_term: float       # cov_term - comm_term >= 0 (Fiedler)
    shrinkage: tuple = field(default=(0.0, 0.0))


def _logdet(C: np.ndarray) -> float:
    try:
        c, low = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf
    d = np.diag(c)
    if np.any(d <= 0):
        return -np.inf
    return float(2 * np.sum(np.log(d)))


def compute_cki(mu_A: np.ndarray, C_A: np.ndarray, mu_B: np.ndarray, C_B: np.ndarray,
                decompose: bool = True) -> CKIResult:
    """Correlated Kakutani index (Bhattacharyya distance) between N(mu_A, C_A) and N(mu_B, C_B).

    The covariances must be positive definite; sample covariances of aligned Cartesian coordinates are not
    (six rigid-body directions and, for n_frames <= 3N, many more are exactly singular), so pass Ledoit-Wolf
    estimates (see `cki_from_frames`). A singular input returns +inf for the covariance term.
    """
    Sbar = 0.5 * (C_A + C_B)
    dmu = np.asarray(mu_B, float) - np.asarray(mu_A, float)
    lA, lB, lS = _logdet(C_A), _logdet(C_B), _logdet(Sbar)
    if not np.isfinite(lA) or not np.isfinite(lB) or not np.isfinite(lS):
        return CKIResult(np.inf, np.nan, np.inf, np.nan, np.nan)
    c = cho_factor(Sbar, lower=True, check_finite=False)
    mean_term = float(dmu @ cho_solve(c, dmu, check_finite=False)) / 8
    cov_term = 0.5 * (lS - 0.5 * (lA + lB))
    comm = rot = np.nan
    if decompose:
        a = np.linalg.eigvalsh(C_A)[::-1]
        b = np.linalg.eigvalsh(C_B)[::-1]
        eta = np.log(b / a)
        comm = float(np.sum(0.5 * np.log(np.cosh(eta / 2))))
        rot = cov_term - comm
    return CKIResult(mean_term + cov_term, mean_term, cov_term, comm, rot)


def cki_from_frames(F_A: np.ndarray, F_B: np.ndarray, shrinkage: str | None = "ledoit-wolf",
                    decompose: bool = True) -> CKIResult:
    """CKI from two frame matrices (n_frames, p) of jointly aligned coordinates."""
    mA, CA, dA = mean_cov(F_A, shrinkage)
    mB, CB, dB = mean_cov(F_B, shrinkage)
    r = compute_cki(mA, CA, mB, CB, decompose=decompose)
    r.shrinkage = (dA, dB)
    return r


# ----------------------------------------------------------------------------
# Spectral fingerprint
# ----------------------------------------------------------------------------
@dataclass
class Fingerprint:
    lam_A: np.ndarray
    lam_B: np.ndarray
    dlnlam: np.ndarray       # Delta ln lambda_i = ln lambda_i^B - ln lambda_i^A (spectra sorted descending)
    theta: np.ndarray        # theta_i = arccos |u_i . v_i| in degrees (eigenvectors matched by rank)
    principal_angles: np.ndarray   # principal angles between the leading k-dimensional subspaces, degrees


def spectral_fingerprint(C_A: np.ndarray, C_B: np.ndarray, k: int = 20) -> Fingerprint:
    """F_L = {Delta ln lambda_i, theta_i} for the leading k modes. Matching by rank is unstable for nearly
    degenerate eigenvalues; the principal angles of the leading subspaces are basis-independent."""
    la, U = np.linalg.eigh(C_A)
    lb, V = np.linalg.eigh(C_B)
    la, U, lb, V = la[::-1], U[:, ::-1], lb[::-1], V[:, ::-1]
    k = min(k, len(la))
    cosang = np.clip(np.abs(np.sum(U[:, :k] * V[:, :k], axis=0)), 0, 1)
    pa = np.degrees(np.sort(subspace_angles(U[:, :k], V[:, :k])))
    return Fingerprint(la[:k], lb[:k], np.log(lb[:k] / la[:k]), np.degrees(np.arccos(cosang)), pa)
