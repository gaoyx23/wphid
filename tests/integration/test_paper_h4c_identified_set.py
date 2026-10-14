import json
from pathlib import Path

from experiments import paper_h4c_identified_set as experiment


def test_partition_coarsening_is_directional():
    assert experiment._is_coarsening("TOD", "T|O|D")
    assert experiment._is_coarsening("TO|D", "T|O|D")
    assert not experiment._is_coarsening("T|O|D", "TOD")
    assert not experiment._is_coarsening("TO|D", "T|OD")


def test_identified_set_v2_combines_retained_results_without_rerunning_simulation():
    payload = json.loads(Path(experiment.DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert payload["protocol_sha256"] == experiment.PROTOCOL_HASH
    assert payload["aggregate"]["truth_covered"] == 3
    assert payload["aggregate"]["count"] == 3
    assert payload["aggregate"]["unique_partition_outputs"] == 1
    assert payload["aggregate"]["decision"] == "DEVELOPMENT-GO"
    independent = next(
        row for row in payload["records"] if row["truth_partition"] == "T|O|D"
    )
    assert independent["source_version"] == "v12_directed_rescue"
    assert independent["unique_partition"] == "T|O|D"
