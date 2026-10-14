import json
from pathlib import Path

from experiments import paper_h4an_spatial_response_error_budget as budget


def test_spatial_error_budget_retains_every_predeclared_scenario():
    payload = json.loads(Path(budget.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == budget.PROTOCOL_HASH
    assert {row["scenario"] for row in payload["scenarios"]} == set(
        budget.SCENARIOS
    )
    assert payload["aggregate"]["overall_passive_h4_decision"] == "NO-GO"
