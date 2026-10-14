import json
import hashlib
import numpy as np
from pathlib import Path
from experiments.audit_passive_transport_balance_components import ROOT,OUT,coefficients


def test_known_gas_gains_and_no_refit_diagnostic():
    np.testing.assert_allclose(coefficients("pO2"),[3e-7,5.1e-6,44000.0],rtol=1e-14)
    result=json.loads((OUT/"audit.json").read_text())
    assert result["complete"]
    assert result["new_simulations"]==result["new_random_realizations"]==result["new_model_fits"]==0
    for relative,digest in result["input_hashes"].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    for row in result["records"].values():
        for gas in row["conditions"]["clean_full_field_control"].values():
            assert gas["existing_finite_volume_operator"]["residual_relative_to_time_target"]<.001


def test_depth_simplification_is_not_promoted_despite_correct_rank():
    result=json.loads((ROOT/"artifacts/_audits/sensor_depth_transport_development/audit.json").read_text())
    assert result["complete"] and not result["hard_pruning_authorized"]
    assert not result["known_axial_control_fed_to_estimator"]
    for row in result["records"].values():
        for gas in ("pH2O","pProducts"):
            control=row["clean"][gas]
            assert control["omitted_axial_relative_to_time_target"]>.3
            assert control["with_known_axial_control"]["residual_relative_to_time_target"]<.001
    assert len(result["decision"]["ranking"])==6
    assert result["decision"]["fit_diagnostics"]["attempt_count"]==54
    assert not result["joint_coverage_calibrated"]
