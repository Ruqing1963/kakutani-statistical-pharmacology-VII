# Statistical Pharmacology via Kakutani Dichotomy VII — `kakutani_pharma`

**The kakutani_pharma Pipeline, Ledoit–Wolf Regularized CKI, Split-Trajectory Null Subtraction, and Shell-Scaling Exponents in Multi-Domain Allostery**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23039869.svg)](https://doi.org/10.5281/zenodo.23039869)

Zhengyi Chen<sup>1</sup>, Ruqing Chen<sup>2</sup>

<sup>1</sup> Guangxi Key Laboratory of Drug Discovery and Optimization, School of Pharmacy, Guilin Medical University, Guilin 541199, P. R. China — chenzhengyi@glmc.edu.cn
<sup>2</sup> GUT Geoservice Inc., Montreal, Québec, Canada — ruqing@hotmail.com

Part VII of the series *Statistical Pharmacology via Kakutani Dichotomy* (Paper I: DOI 10.5281/zenodo.23005921; II: 10.5281/zenodo.23012217; III: 10.5281/zenodo.23015709; IV: 10.5281/zenodo.23018236; V: 10.5281/zenodo.23027704; VI: 10.5281/zenodo.23029360). Papers I–VI derived the indices on model ensembles. This repository packages them as a Python library that takes trajectory coordinates as input.

## Status

The package, its end-to-end benchmark and the paper (`paper/paper07_kakutani_pharma_pipeline.pdf`, 16 pages) are complete. The 8 unit tests and all acceptance tests of the benchmark pass (`results/benchmark_report.txt`). The analytic companion `code/paper07_compute_tables.py` writes `results/paper07_tables_summary.txt` and the table bodies `results/paper07_table{1,2,2b,3,4,4b}_rows.tex`.

Main results of the paper:

- **Rigidity (Proposition 2.2).** `dim ker H = 6` if and only if the elastic network is infinitesimally rigid. Two rigid halves joined by `c` springs give `12 - c`, which explains the 11 zero modes of a first benchmark geometry: S-1 and S-2 had no contacts and were joined by one backbone spring.
- **Shrinkage (Theorem 3.1).** Ledoit–Wolf is positive definite for any number of frames, and `CKI = D_mean + D_comm + D_rot` with all three terms nonnegative (Fiedler).
- **Null subtraction (Theorem 4.1).** The split-trajectory excess is unbiased under the null up to `O(tau/h)`; the replica falls from 3.10 to 0.010 [-0.101, 0.118]. Resampling the whole trajectory before splitting leaks about 40 % of the blocks across halves and inflates the excess by about +3.0, so none of the four leaky intervals contains its estimate.
- **Duplicate-frame bias (Proposition 4.2).** Within-half replicates keep about 64.5 % distinct blocks. The bootstrap bias is +0.126 for the substrate, and bias-shifted and basic intervals are reported next to the percentile interval.
- **Shell scaling (Theorem 5.1, Corollary 5.2).** The population curve is nondecreasing and has a plateau for perturbations confined to the pocket. Corollary 5.2 proves that the plateau is exact for the substrate model, even in the rigid-body gauge: the likelihood ratio depends on the pocket coordinates only. The computed curve is 0.776914237794 at every shell, so the population exponent is 0. The sample exponents are 0.12 for the substrate, 1.36 for the effector and 0.88 for both.


**The benchmark uses synthetic trajectories with known ground truth. The package has not yet been applied to a real protein–ligand MD trajectory.** MDTraj was not installed in the development environment, so the MDTraj loader path is untested. The native multi-model PDB reader is tested.

## Modules

| module | contents |
|---|---|
| `featurizer.py` | multi-model PDB reader and writer; optional MDTraj loader (DCD, XTC, …); φ, ψ, χ1 from backbone and side-chain atoms; 9 metastable states per residue (3 Ramachandran basins × 3 rotamers) and their occupancies; rational-switching contact network; iterative mass-weighted Kabsch alignment of several trajectories onto a common mean; mean structure μ_L and covariance C_L; dynamic cross-correlation |
| `estimators.py` | `compute_dki`: first-order index `h_i = 2(1 − Σ_s sqrt(p_is q_is))`, `DKI = Σ h_i`, affinity `Π (1 − h_i/2)`, with Laplace (add-α) smoothing; `ledoit_wolf`: analytical shrinkage in O(np + p²), identical to scikit-learn; `compute_cki`: Bhattacharyya distance `D = D_mean + D_cov`, `D_cov = D_comm + D_rot` with `D_rot ≥ 0`; `spectral_fingerprint`: `F_L = {Δ ln λ_i, θ_i}` and principal angles |
| `scaling.py` | pocket distances, shells `S(r) = {i : d(i, pocket) ≤ r}`, profile `CKI(r)`, distal exponent γ (log-log slope against `|S(r)|`), split-trajectory null baseline, moving-block bootstrap inside each half (the leaky `scheme="whole"` is kept only for demonstration), integrated autocorrelation time |

The sample covariance of aligned Cartesian coordinates is always singular: alignment leaves six rigid-body null directions, and with fewer frames than `3N` many more appear. The raw `ln det C` is then `−∞`, and the index is infinite. Ledoit–Wolf shrinkage makes every covariance positive definite; the shrinkage weight is reported with every result.

## Quick start on a real trajectory

```python
import kakutani_pharma as kp

X_apo, top = kp.load_trajectory("apo.xtc", top="apo.pdb")        # MDTraj; or kp.read_pdb_models("apo_multimodel.pdb")
X_lig, _   = kp.load_trajectory("ligand.xtc", top="apo.pdb")

ca = top.select("CA")
(A, B), ref = kp.align_trajectories([X_apo[:, ca], X_lig[:, ca]], masses=top.mass[ca])
F_A, F_B = kp.frames_matrix(A), kp.frames_matrix(B)

cki = kp.cki_from_frames(F_A, F_B)                                # Ledoit-Wolf by default
print(cki.total, cki.mean_term, cki.cov_term, cki.comm_term, cki.rot_term, cki.shrinkage)

states_A = kp.residue_states(kp.backbone_dihedrals(X_apo, top))
states_B = kp.residue_states(kp.backbone_dihedrals(X_lig, top))
dki = kp.compute_dki(kp.occupancy_counts(states_A), kp.occupancy_counts(states_B))

d = kp.pocket_distance(ref, pocket=[41, 42, 45, 88])             # residue indices of the binding pocket
radii = kp.shell_radii(d, 10, r_first=0.0)
split = kp.split_null(F_A, F_B, d, radii)                         # signal, null and excess at equal sample size
tau = kp.integrated_autocorrelation_time(F_A @ F_A.std(0))        # choose a block of about 2 tau frames
boot = kp.block_bootstrap(F_A, F_B, d, radii, r_min=15.0, block=int(2 * tau) + 1, n_boot=100)
```

Both trajectories must hold the same residues in the same order. Frames should be spaced so that each trajectory contains many autocorrelation times.

## Benchmark: four-domain allosteric protein (synthetic)

`benchmark_allosteric_protein.py` builds a coarse-grained Cα model of a four-domain protein. The domains A1-1 | S-1 | S-2 | A1-2 have 30 residues each, 120 in all. Each domain is a compact globule, and neighbouring domains share 150–171 contacts.

The trajectories are generated as follows:

- The dynamics come from an anisotropic network model with a 13 Å cutoff. Exactly six zero modes are checked.
- Each normal mode follows an Ornstein–Uhlenbeck process with relaxation time proportional to `1/λ_k`, so frames are time-correlated as in MD. Every frame also receives a random rigid-body motion.
- The dihedrals are metastable Markov chains.
- There are five states, each with 6000 frames:
  - **apo;**
  - **substrate:** a ligand node bound to the S-1 pocket, which is a local perturbation;
  - **effector:** a ligand node at the far end of A1-2, plus stiffened S-1|S-2 and S-2|A1-2 interfaces, which is a long-range perturbation;
  - **both** ligands;
  - **an independent apo replica** as a negative control.

| state | DKI | CKI | mean | cov | comm | rot | top-10 DKI residues in the perturbed set | excess CKI (all residues), 95 % CI | distal γ, 95 % CI |
|---|---|---|---|---|---|---|---|---|---|
| substrate | 0.421 | 3.848 | 0.774 | 3.074 | 0.005 | 3.068 | 9/10 | 0.813 [0.770, 1.202] | 0.118 [0.052, 0.299] |
| effector | 2.701 | 7.210 | 1.430 | 5.780 | 1.307 | 4.473 | 10/10 | 4.112 [3.828, 4.389] | 1.361 [1.302, 1.463] |
| both | 2.683 | 7.573 | 1.761 | 5.812 | 1.326 | 4.486 | 10/10 | 4.492 [4.287, 4.857] | 0.881 [0.804, 0.915] |
| apo replica (control) | 0.186 | 3.099 | 0.051 | 3.048 | 0.012 | 3.036 | — | 0.010 [−0.101, 0.118] | n/a (excess ≈ 0) |

The table reads as follows:

- **The raw CKI is not a signal by itself.** The control replica already scores 3.10, almost all of it the finite-sample bias of the covariance term with 360 coordinates. After the split-trajectory null is subtracted, the control's excess is 0.010, and its 95 % interval contains zero. All three ligand states have intervals that exclude zero.
- **The distal exponent separates the two mechanisms.** The substrate's excess is already reached at the pocket shell (0.72) and barely grows outward, so γ is close to 0. The effector's excess grows across the whole protein, with γ = 1.36, and its interval lies entirely above the substrate's.
- **The DKI localises the perturbation.** Here the perturbed set is the residues whose generating energy shift is at least 0.1 kT. The control's DKI (0.19) is below the split-trajectory DKI null (0.40).
- **The spectral fingerprint is informative for the effector:** Δ ln λ₁ = −0.65 on the softest mode. For the local substrate, the top-mode shifts are the same size as those of the control replica, about ±0.17, so they should not be interpreted.
- **Checks on the featurizer and alignment:**
  - Dihedral binning agrees with the generating states in 99.9997 % of residue-frames.
  - Joint alignment of 5 × 6000 randomly rotated frames recovers the model's distances to 0.31 Å.
  - The aligned RMSF correlates with the model at 0.996.
  - With 200 frames (fewer than the 360 coordinates) the sample covariance gives an infinite CKI, while Ledoit–Wolf gives 66.2 with shrinkage 0.157.

Figures: `figures/fig07_correlation_maps.{png,pdf}` shows cross-correlation and contact-network maps. `figures/fig07_kakutani_diagnostics.{png,pdf}` shows the DKI per residue, the shell scaling with bootstrap bands, the signal against the split null, and the spectral fingerprint.

## Limitations

- **The model is synthetic and coarse-grained.** It has a Cα elastic network, so some consecutive virtual bonds are long (up to 16 Å). It has harmonic dynamics and independent dihedral chains. Real trajectories add anharmonic basins, side-chain coupling and slow drift. The split null and the block bootstrap exist for exactly that, but their calibration on real MD is still to be done.
- **γ applies only to a clear signal.** It is a log-log slope, so it is reported only when the excess interval excludes zero.
- **The bootstrap intervals are skewed upward.** Duplicated frames in a resample slightly inflate all four CKI terms. The within-half scheme keeps the split structure intact, but the intervals remain conservative on the upper side.
- **The spectral fingerprint depends on eigenvector matching.** It matches eigenvectors by rank, which is unstable for nearly degenerate modes. The principal angles are basis-independent and are also returned.

## Reproducing

```bash
pip install -r requirements.txt
python tests/test_kakutani_pharma.py          # or: python -m pytest tests
python benchmark_allosteric_protein.py        # about 15 minutes; prints OVERALL ACCEPTANCE: PASS
python code/paper07_compute_tables.py         # analytic companion, writes results/paper07_*
cd paper && pdflatex paper07_kakutani_pharma_pipeline.tex && pdflatex paper07_kakutani_pharma_pipeline.tex && pdflatex paper07_kakutani_pharma_pipeline.tex
```

## Citation

```bibtex
@article{ChenChen2026KakutaniVII,
  author  = {Chen, Zhengyi and Chen, Ruqing},
  title   = {Statistical Pharmacology via Kakutani Dichotomy VII: The kakutani\_pharma Pipeline, Ledoit--Wolf Regularized CKI,
             Split-Trajectory Null Subtraction, and Shell-Scaling Exponents in Multi-Domain Allostery},
  year    = {2026},
  doi     = {10.5281/zenodo.23039869},
  url     = {https://doi.org/10.5281/zenodo.23039869}
}
```

## Repository

https://github.com/Ruqing1963/kakutani-statistical-pharmacology-VII

## Funding

This project was supported by the National Natural Science Foundation of China (No. 22464010), the Guangxi Natural Science Foundation of China (No. 2025GXNSFAA069294), and the 2025 Bagui Youth Top Talent Project.

## License

Code under the MIT License; text, figures and data under CC BY 4.0 (see `LICENSE`).
