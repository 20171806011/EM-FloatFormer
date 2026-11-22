"""Receiver implementations for research/realdata_sdr_ofdm."""

from research.realdata_sdr_ofdm.receivers.classical import (
    CLASSICAL_METHODS,
    ClassicalConfig,
    ClassicalReceiver,
    ReceiverOutput,
    calibrate_classical,
    calibrate_on_dev,
    normalize_method_name,
    run_frame,
)
from research.realdata_sdr_ofdm.receivers.group_a_reference import (
    GROUP_A_MODES,
    KEY_PILOT_INDICES,
    VAL_PILOT_INDICES,
    GroupAConfig,
    GroupAReferenceReceiver,
    build_reference_pool,
    calibrate_group_a_on_dev,
    calibrate_group_a_temperature,
    normalize_group_a_mode,
)
from research.realdata_sdr_ofdm.receivers.group_a_reference import (
    run_frame as run_group_a_frame,
)
from research.realdata_sdr_ofdm.receivers.group_b_fusion import (
    GROUP_B_MODES,
    CandidateBranch,
    GroupBConfig,
    GroupBFusionReceiver,
    build_candidate_bank,
    build_candidate_branches,
    calibrate_group_b_on_dev,
    fuse_candidate_branches,
    normalize_group_b_mode,
)
from research.realdata_sdr_ofdm.receivers.group_b_fusion import (
    run_frame as run_group_b_frame,
)
from research.realdata_sdr_ofdm.receivers.group_c_trainable import (
    GROUP_C_MODES,
    BranchWeightAttentionNet,
    GroupCConfig,
    GroupCReceiver,
    GroupCTrainableReceiver,
    differentiable_mixture_llrs_torch,
)
from research.realdata_sdr_ofdm.receivers.group_c_trainable import (
    run_frame as run_group_c_frame,
)

__all__ = [
    "CLASSICAL_METHODS",
    "ClassicalConfig",
    "ClassicalReceiver",
    "ReceiverOutput",
    "calibrate_classical",
    "calibrate_on_dev",
    "normalize_method_name",
    "run_frame",
    "GROUP_A_MODES",
    "KEY_PILOT_INDICES",
    "VAL_PILOT_INDICES",
    "GroupAConfig",
    "GroupAReferenceReceiver",
    "build_reference_pool",
    "calibrate_group_a_on_dev",
    "calibrate_group_a_temperature",
    "normalize_group_a_mode",
    "run_group_a_frame",
    "GROUP_B_MODES",
    "CandidateBranch",
    "GroupBConfig",
    "GroupBFusionReceiver",
    "build_candidate_bank",
    "build_candidate_branches",
    "calibrate_group_b_on_dev",
    "fuse_candidate_branches",
    "normalize_group_b_mode",
    "run_group_b_frame",
    "GROUP_C_MODES",
    "BranchWeightAttentionNet",
    "GroupCConfig",
    "GroupCReceiver",
    "GroupCTrainableReceiver",
    "differentiable_mixture_llrs_torch",
    "run_group_c_frame",
]
