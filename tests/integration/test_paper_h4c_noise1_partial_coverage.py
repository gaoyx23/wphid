import json
from pathlib import Path

from experiments import paper_h4c_noise1_partial_coverage as coverage


def test_retained_partial_role_is_safe_but_below_frozen_coverage_gate():
    payload = json.loads(Path(coverage.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == coverage.PROTOCOL_HASH
    assert payload["aggregate"]["correct_usable_outputs"] == 3
    assert payload["aggregate"]["abstentions"] == 1
    assert payload["aggregate"]["wrong_outputs"] == 0
    assert payload["aggregate"]["correct_usable_output_rate"] == 0.75
    assert payload["aggregate"]["identified_set_truth_coverage"] == 4
    assert payload["aggregate"]["identified_set_wrong_exclusions"] == 0
    assert payload["aggregate"]["decision"] == "DEVELOPMENT-NO-GO"
    assert payload["aggregate"]["overall_h4_decision"] == "NO-GO"
