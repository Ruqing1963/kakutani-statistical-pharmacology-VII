"""
featurizer.py
=============
Features of a molecular-dynamics trajectory used by the Kakutani estimators.

    first order   metastable dihedral states (phi, psi, chi1) per residue and their occupancies p_i
    network       frame-averaged weighted residue contact map
    second order  mass-weighted aligned C-alpha mean structure mu_L and covariance C_L

Input is plain NumPy: coordinates of shape (n_frames, n_atoms, 3) in Angstrom plus a list of atom records.
A small multi-model PDB reader is included; if MDTraj is installed, `load_trajectory` uses it for DCD/XTC/etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHI1_GAMMA = {"CG", "SG", "OG", "OG1", "CG1"}


@dataclass
class Topology:
    """Atom records: residue sequence index, residue name, atom name, element mass."""
    resid: np.ndarray
    resname: np.ndarray
    name: np.ndarray
    mass: np.ndarray

    def select(self, name: str) -> np.ndarray:
        return np.nonzero(self.name == name)[0]

    @property
    def n_residues(self) -> int:
        return int(np.unique(self.resid).size)


_MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "H": 1.008}


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def read_pdb_models(path: str) -> tuple[np.ndarray, Topology]:
    """Read a multi-model PDB (MODEL/ENDMDL). Returns coordinates (n_frames, n_atoms, 3) and the topology of
    the first model. Hydrogens are kept; alternate locations other than ' ' and 'A' are skipped."""
    frames, cur = [], []
    recs = None
    first_recs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                if line[16] not in (" ", "A"):
                    continue
                cur.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                if recs is None:
                    elem = line[76:78].strip() or line[12:16].strip()[0]
                    first_recs.append((line[22:27], line[17:20].strip(), line[12:16].strip(), _MASS.get(elem.upper()[:1], 12.0)))
            elif rec == "ENDMDL":
                frames.append(cur)
                cur = []
                recs = first_recs
    if cur:
        frames.append(cur)
    if not frames:
        raise ValueError(f"no atoms found in {path}")
    n = len(frames[0])
    if any(len(fr) != n for fr in frames):
        raise ValueError("models have different atom counts")
    keys, resid = {}, []
    for r in first_recs:
        resid.append(keys.setdefault(r[0], len(keys)))
    top = Topology(np.array(resid), np.array([r[1] for r in first_recs]), np.array([r[2] for r in first_recs]),
                   np.array([r[3] for r in first_recs]))
    return np.array(frames, dtype=float), top


def write_pdb_models(path: str, coords: np.ndarray, top: Topology) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for k, fr in enumerate(coords):
            f.write(f"MODEL     {k + 1:>4}\n")
            for a, (x, y, z) in enumerate(fr):
                el = top.name[a][0]
                f.write(f"ATOM  {a + 1:>5} {top.name[a]:<4} {top.resname[a]:>3} A{top.resid[a] + 1:>4}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}\n")
            f.write("ENDMDL\n")


def load_trajectory(traj: str, top: str | None = None) -> tuple[np.ndarray, Topology]:
    """Load any trajectory format through MDTraj if available (coordinates converted to Angstrom);
    multi-model PDB files are read natively."""
    if traj.lower().endswith(".pdb") and top is None:
        return read_pdb_models(traj)
    try:
        import mdtraj as md
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("reading this format requires MDTraj (pip install mdtraj); "
                          "multi-model PDB files are supported without it") from exc
    t = md.load(traj, top=top)
    atoms = list(t.topology.atoms)
    topo = Topology(np.array([a.residue.index for a in atoms]), np.array([a.residue.name for a in atoms]),
                    np.array([a.name for a in atoms]),
                    np.array([a.element.mass if a.element is not None else 12.0 for a in atoms]))
    return t.xyz * 10.0, topo


# ----------------------------------------------------------------------------
# Dihedral angles and metastable states
# ----------------------------------------------------------------------------
def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Dihedral angle in degrees in [0, 360), vectorised over leading axes (IUPAC sign convention)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return np.degrees(np.arctan2(y, x)) % 360.0


def backbone_dihedrals(coords: np.ndarray, top: Topology) -> dict[str, np.ndarray]:
    """phi, psi, chi1 per residue, shape (n_frames, n_residues), NaN where undefined (termini, Gly/Ala chi1)."""
    T = coords.shape[0]
    R = top.n_residues
    out = {k: np.full((T, R), np.nan) for k in ("phi", "psi", "chi1")}
    idx = {}
    for a, (r, nm) in enumerate(zip(top.resid, top.name)):
        idx.setdefault(int(r), {})[nm] = a
    for r in range(R):
        at = idx.get(r, {})
        prev, nxt = idx.get(r - 1, {}), idx.get(r + 1, {})
        if all(k in at for k in ("N", "CA", "C")):
            if "C" in prev:
                out["phi"][:, r] = dihedral(coords[:, prev["C"]], coords[:, at["N"]], coords[:, at["CA"]], coords[:, at["C"]])
            if "N" in nxt:
                out["psi"][:, r] = dihedral(coords[:, at["N"]], coords[:, at["CA"]], coords[:, at["C"]], coords[:, nxt["N"]])
            g = next((k for k in CHI1_GAMMA if k in at), None)
            if "CB" in at and g is not None:
                out["chi1"][:, r] = dihedral(coords[:, at["N"]], coords[:, at["CA"]], coords[:, at["CB"]], coords[:, at[g]])
    return out


def ramachandran_state(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """0 = alpha-helical basin, 1 = beta/extended basin, 2 = left-handed (phi > 0). Angles in degrees [0, 360)."""
    ph = (phi + 180.0) % 360.0 - 180.0
    ps = (psi + 180.0) % 360.0 - 180.0
    state = np.where((ps > -120.0) & (ps < 50.0), 0, 1)
    state = np.where(ph > 0.0, 2, state)
    return np.where(np.isnan(phi) | np.isnan(psi), 0, state).astype(np.int8)


def rotamer_state(chi1: np.ndarray) -> np.ndarray:
    """0 = g+ (0-120), 1 = t (120-240), 2 = g- (240-360); residues without chi1 are assigned state 0."""
    s = np.floor(np.nan_to_num(chi1, nan=60.0) / 120.0).astype(np.int8)
    return np.clip(s, 0, 2)


def residue_states(dihedrals: dict[str, np.ndarray]) -> np.ndarray:
    """Composite metastable state per residue and frame: 3 x Ramachandran basin + rotamer (9 states)."""
    return (3 * ramachandran_state(dihedrals["phi"], dihedrals["psi"]) + rotamer_state(dihedrals["chi1"])).astype(np.int8)


def occupancy_counts(states: np.ndarray, n_states: int = 9) -> np.ndarray:
    """Counts of each metastable state per residue, shape (n_residues, n_states)."""
    T, R = states.shape
    counts = np.zeros((R, n_states))
    for s in range(n_states):
        counts[:, s] = np.sum(states == s, axis=0)
    return counts


# ----------------------------------------------------------------------------
# Contact network
# ----------------------------------------------------------------------------
def contact_network(ca: np.ndarray, r0: float = 8.0, n: int = 6, m: int = 12, min_sep: int = 3,
                    stride: int = 1) -> np.ndarray:
    """Frame-averaged rational switching function s(r) = (1 - (r/r0)^n) / (1 - (r/r0)^m) between C-alpha atoms,
    excluding pairs closer than `min_sep` in sequence. Returns a symmetric (N, N) weight matrix in [0, 1]."""
    X = ca[::stride]
    N = X.shape[1]
    W = np.zeros((N, N))
    for fr in X:
        d = np.linalg.norm(fr[:, None, :] - fr[None, :, :], axis=-1) / r0
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (1 - d ** n) / (1 - d ** m)
        s[np.isclose(d, 1.0)] = n / m
        W += s
    W /= len(X)
    ii, jj = np.indices((N, N))
    W[np.abs(ii - jj) < min_sep] = 0.0
    return W


# ----------------------------------------------------------------------------
# Alignment, mean structure and covariance
# ----------------------------------------------------------------------------
def kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Optimal rotation R minimising sum_a w_a |R P_a - Q_a|^2 for centred P, Q (both (n, 3))."""
    H = (P * w[:, None]).T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def align_trajectories(trajs: list[np.ndarray], masses: np.ndarray | None = None, n_iter: int = 8,
                       tol: float = 1e-6) -> tuple[list[np.ndarray], np.ndarray]:
    """Iterative mass-weighted Kabsch superposition of one or several trajectories onto their common mean
    structure. Returns the aligned trajectories and the reference (mean) structure."""
    n = trajs[0].shape[1]
    w = np.ones(n) if masses is None else np.asarray(masses, float)
    w = w / w.sum()
    cen = [X - np.einsum("a,tak->tk", w, X)[:, None, :] for X in trajs]
    ref = cen[0][0].copy()
    for _ in range(n_iter):
        out = []
        for X in cen:
            H = np.einsum("tak,a,al->tkl", X, w, ref)
            U, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(np.einsum("tlk,tjl->tkj", Vt, U)))
            D = np.zeros((len(X), 3, 3)); D[:, 0, 0] = 1; D[:, 1, 1] = 1; D[:, 2, 2] = d
            Rm = np.einsum("tlk,tlm,tjm->tkj", Vt, D, U)
            out.append(np.einsum("tkj,taj->tak", Rm, X))
        new_ref = np.concatenate(out).mean(axis=0)
        new_ref -= w @ new_ref
        shift = np.sqrt(np.sum(w * np.sum((new_ref - ref) ** 2, axis=1)))
        ref = new_ref
        if shift < tol:
            break
    return out, ref


def mean_and_covariance(X: np.ndarray, masses: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean structure (3N,) and sample covariance (3N, 3N) of aligned coordinates. With masses, coordinates are
    scaled by sqrt(m_a / mean(m)) first (mass-weighted covariance)."""
    F = frames_matrix(X, masses)
    mu = F.mean(axis=0)
    Fc = F - mu
    return mu, Fc.T @ Fc / len(F)


def frames_matrix(X: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """(n_frames, 3N) matrix of (optionally mass-weighted) Cartesian coordinates."""
    if masses is not None:
        s = np.sqrt(np.asarray(masses, float) / np.mean(masses))
        X = X * s[None, :, None]
    return X.reshape(len(X), -1)


def dynamic_cross_correlation(X: np.ndarray) -> np.ndarray:
    """Normalised dynamic cross-correlation C_ij = <dr_i . dr_j> / sqrt(<dr_i^2> <dr_j^2>) of aligned coordinates."""
    D = X - X.mean(axis=0)
    G = np.einsum("tik,tjk->ij", D, D) / len(X)
    s = np.sqrt(np.diag(G))
    return G / np.outer(s, s)
