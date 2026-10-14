import json
from pathlib import Path

from experiments import paper_h4c_noise1_hardest_coverage as coverage


def test_retained_hardest_role_development_coverage_passes_without_promoting_h4():
    payload = json.loads(Path(coverage.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == coverage.PROTOCOL_HASH
    assert payload["aggregate"]["correct_usable_outputs"] == 4
    assert payload["aggregate"]["wrong_outputs"] == 0
    assert payload["aggregate"]["minimum_rescue_bic_gain"] >= 200.0
    assert payload["aggregate"]["decision"] == "DEVELOPMENT-GO"
    assert payload["aggregate"]["overall_h4_decision"] == "NO-GO"
