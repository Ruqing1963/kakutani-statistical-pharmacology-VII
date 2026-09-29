"""Unit tests for kakutani_pharma. Run with `python -m pytest tests` or `python tests/test_kakutani_pharma.py`."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kakutani_pharma as kp  # noqa: E402
from kakutani_pharma.featurizer import Topology  # noqa: E402


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def test_dihedral_known_values():
    p0, p1, p2 = np.array([1.0, 0, 0]), np.array([0.0, 0, 0]), np.array([0.0, 0, 1])
    for ang in (60.0, 180.0, 300.0):
        t = np.radians(ang)
        p3 = np.array([np.cos(t), np.sin(t), 1.0])
        assert abs(kp.dihedral(p0, p1, p2, p3) - ang) < 1e-9


def test_pdb_roundtrip_and_backbone_dihedrals():
    rng = np.random.default_rng(1)
    names = ["N", "CA", "C", "CB", "CG"] * 3
    resid = np.repeat(np.arange(3), 5)
    top = Topology(resid, np.array(["LEU"] * 15), np.array(names), np.full(15, 12.0))
    X = rng.normal(scale=3.0, size=(4, 15, 3))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.pdb")
        kp.write_pdb_models(path, X, top)
        Y, top2 = kp.read_pdb_models(path)
    assert Y.shape == X.shape and np.max(np.abs(X - Y)) < 5e-4
    assert list(top2.name) == names and list(top2.resid) == list(resid)
    dh = kp.backbone_dihedrals(Y, top2)
    at = lambda r, nm: 5 * r + names.index(nm)  # noqa: E731
    phi1 = kp.dihedral(Y[:, at(0, "C")], Y[:, at(1, "N")], Y[:, at(1, "CA")], Y[:, at(1, "C")])
    chi0 = kp.dihedral(Y[:, at(0, "N")], Y[:, at(0, "CA")], Y[:, at(0, "CB")], Y[:, at(0, "CG")])
    assert np.allclose(dh["phi"][:, 1], phi1) and np.allclose(dh["chi1"][:, 0], chi0)
    assert np.all(np.isnan(dh["phi"][:, 0])) and np.all(np.isnan(dh["psi"][:, 2]))


def test_alignment_removes_rigid_motion():
    rng = np.random.default_rng(2)
    ref = rng.normal(scale=8.0, size=(40, 3))
    noise = 0.3
    X = np.array([(ref + rng.normal(scale=noise, size=ref.shape)) @ random_rotation(rng).T + rng.normal(scale=20, size=3)
                  for _ in range(200)])
    (Xa,), mean = kp.align_trajectories([X])
    dm = lambda P: np.linalg.norm(P[:, None] - P[None], axis=-1)  # noqa: E731
    assert np.max(np.abs(dm(mean) - dm(ref))) < 0.2
    rmsd = np.sqrt(np.mean(np.sum((Xa - mean) ** 2, axis=2)))
    assert abs(rmsd - noise * np.sqrt(3)) < 0.1


def test_ledoit_wolf_matches_sklearn():
    from sklearn.covariance import ledoit_wolf as sk_lw
    rng = np.random.default_rng(3)
    for n, p in ((50, 120), (500, 40), (30, 30)):
        F = rng.normal(size=(n, p)) @ rng.normal(size=(p, p))
        C, d = kp.ledoit_wolf(F)
        Cs, ds = sk_lw(F)
        assert abs(d - ds) < 1e-10 and np.max(np.abs(C - Cs)) < 1e-8 * np.max(np.abs(Cs))


def test_cki_identities():
    rng = np.random.default_rng(4)
    p = 12
    A = rng.normal(size=(p, p)); CA = A @ A.T + 0.1 * np.eye(p)
    B = rng.normal(size=(p, p)); CB = B @ B.T + 0.1 * np.eye(p)
    mu = rng.normal(size=p)
    assert abs(kp.compute_cki(mu, CA, mu, CA).total) < 1e-10
    r1, r2 = kp.compute_cki(mu, CA, 0 * mu, CB), kp.compute_cki(0 * mu, CB, mu, CA)
    assert abs(r1.total - r2.total) < 1e-10 and r1.rot_term >= -1e-10
    lam, U = np.linalg.eigh(CA)
    CC = U @ np.diag(np.sort(lam * np.exp(rng.normal(size=p)))) @ U.T    # same eigenvectors, same ordering
    rc = kp.compute_cki(mu, CA, mu, CC)
    assert abs(rc.rot_term) < 1e-9
    one = kp.compute_cki(np.array([0.0]), np.array([[1.0]]), np.array([2.0]), np.array([[4.0]]))
    sb = 2.5
    assert abs(one.total - (4 / (8 * sb) + 0.5 * np.log(sb / 2.0))) < 1e-12


def test_shrinkage_rescues_singular_covariance():
    rng = np.random.default_rng(5)
    F_A, F_B = rng.normal(size=(40, 90)), rng.normal(size=(40, 90))
    assert np.isinf(kp.cki_from_frames(F_A, F_B, shrinkage=None).total)
    assert np.isfinite(kp.cki_from_frames(F_A, F_B).total)


def test_dki_bernoulli_matches_paper_one():
    p, q = np.array([0.3, 0.5, 0.9]), np.array([0.4, 0.5, 0.2])
    r = kp.compute_dki(p, q)
    h = 2 * (1 - (np.sqrt(p * q) + np.sqrt((1 - p) * (1 - q))))
    assert np.allclose(r.per_residue, h) and abs(r.log_affinity - np.sum(np.log(1 - h / 2))) < 1e-12
    counts = np.array([[10, 0, 0], [3, 3, 4]])
    rs = kp.compute_dki(counts, counts)
    assert abs(rs.total) < 1e-12 and np.allclose(rs.p_A[0], [11 / 13, 1 / 13, 1 / 13])


def test_autocorrelation_time_ar1():
    rng = np.random.default_rng(6)
    rho, n = 0.9, 200_000
    x = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    tau = kp.integrated_autocorrelation_time(x)
    assert abs(tau / ((1 + rho) / (1 - rho)) - 1) < 0.1


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL  {name}: {exc!r}")
    print(f"{'ALL TESTS PASSED' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)
