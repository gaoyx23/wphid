import json
from pathlib import Path

from experiments import paper_h4ag_local_elasticity_design as design


def test_elasticity_noise_formula_improves_with_measurement_precision():
    assert design._elasticity_noise_std(0.005, 0.20) < design._elasticity_noise_std(
        0.01, 0.20
    )


def test_retained_elasticity_design_matches_protocol_and_does_not_open_discovery():
    payload = json.loads(Path(design.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == design.PROTOCOL_HASH
    assert payload["aggregate"]["full_discovery_authorized"] is False
