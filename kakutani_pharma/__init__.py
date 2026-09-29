"""kakutani_pharma: Kakutani indices of conformational ensembles from molecular-dynamics trajectories.

Modules
    featurizer   dihedral metastable states, contact networks, aligned mean structure and covariance
    estimators   first-order DKI, Ledoit-Wolf shrinkage, second-order CKI, spectral fingerprint F_L
    scaling      pocket-centred shell profile CKI(r), distal exponent, split-trajectory null, block bootstrap
"""

from .estimators import (CKIResult, DKIResult, Fingerprint, cki_from_frames, compute_cki, compute_dki,
                         laplace_smooth, ledoit_wolf, spectral_fingerprint)
from .featurizer import (Topology, align_trajectories, backbone_dihedrals, contact_network, dihedral,
                         dynamic_cross_correlation, frames_matrix, load_trajectory, mean_and_covariance,
                         occupancy_counts, read_pdb_models, residue_states, write_pdb_models)
from .scaling import (block_bootstrap, cki_profile, fit_scaling_exponent, integrated_autocorrelation_time,
                      pocket_distance, shell_radii, split_null)

__version__ = "0.1.0"
