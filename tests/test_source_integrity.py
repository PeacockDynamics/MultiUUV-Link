import hashlib
from pathlib import Path

from multiuuv_link.config import (
    BLACKOUT_START_PROBABILITY,
    COMMUNICATION_RANGE,
    FOV_HEIGHT,
    FOV_WIDTH,
    HYBRID_CLASSICAL_REGRET_MAX_43,
    HYBRID_LEARNED_MARGIN_MIN_43,
    MAX_COMM_DELAY,
    MIN_COMM_DELAY,
    PACKET_LOSS_PROBABILITY,
    ROLLOUT_STEP_SIZE,
    SAFE_SEPARATION,
)


PROTECTED_HASHES = {
    "notebooks/MultiUUV_Link_Prototype.ipynb": "ed4eb288989a989f7303973e5f209f038b2c6e0f66a9ab6e1ed07430010ef0bc",
    "notebooks/MultiUUV_Link_Prototype_Author_Implementation.pdf": "4f2b928f3342d5ccb1b22d6022c71791ae6fdd900ac67cd6cbe8159bc4d32897",
    "assets/sample_map.png": "aba92dea6e6d597f14e4c83f5097e0d00ef9576905fd621d1ad134eeda4f2193",
    "artifacts/frontier_value_model_prototype.pt": "0c251d009c3bdacee3ccbd71fc1cc9a56c05a1a9866f02b23396d67dd5a470d0",
    "artifacts/multiuuv_fault_recovery_showcase_prototype.gif": "423ac1f43bbf652725b9dc27b0ae9497ed697ffa3d2ee854f233770edacd2420",
    "artifacts/multiuuv_final_metrics_prototype.csv": "6d51e798ad13c6afade3b902fe24dadfcebdc500eb769b78678400306878c3b8",
    "artifacts/multiuuv_final_summary_prototype.json": "68c4d3303d8a62b3a08d8c4c08c9a7a1c37039ea0e975eaf840fc3b32f2cd064",
    "artifacts/multiuuv_hybrid_mission_prototype.gif": "52bf54cad018f566a0e593ef3e21681ae202ff46b9e772df4fdbd9fa533ea25b",
}


def test_authoritative_sources_and_validated_artifacts_are_unchanged():
    for relative_path, expected in PROTECTED_HASHES.items():
        digest = hashlib.sha256(Path(relative_path).read_bytes()).hexdigest()
        assert digest == expected, relative_path


def test_final_validated_numerical_configuration():
    assert (FOV_WIDTH, FOV_HEIGHT) == (128, 128)
    assert ROLLOUT_STEP_SIZE == 32
    assert SAFE_SEPARATION == 128.0
    assert COMMUNICATION_RANGE == 400.0
    assert PACKET_LOSS_PROBABILITY == 0.25
    assert (MIN_COMM_DELAY, MAX_COMM_DELAY) == (1, 5)
    assert BLACKOUT_START_PROBABILITY == 0.02
    assert HYBRID_LEARNED_MARGIN_MIN_43 == 0.03
    assert HYBRID_CLASSICAL_REGRET_MAX_43 == 0.02
