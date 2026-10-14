import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paper_assets_are_complete_and_hash_validated():
    manifest_path = ROOT / "docs" / "paper" / "paper_asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy"].startswith("read-only transformation")
    assert {"H4-AL", "H4-AO", "H4-AQ", "H4-AR", "H4-AT"}.issubset(
        manifest["sources"]
    )
    assert {"R94", "R95", "R97", "R98", "R99", "R106", "R107", "R108"}.issubset(
        manifest["sources"]
    )
    assert len(manifest["outputs"]) == 42
    assert len(manifest["auxiliary_generators"]) == 8
    assert "docs/paper/figures/fig15_domain_shift_evidence.png" in manifest["outputs"]
    assert "docs/paper/figures/fig16_thermoelastic_audit.png" in manifest["outputs"]

    for row in manifest["sources"].values():
        path = ROOT / row["path"]
        assert path.exists()
        assert digest(path) == row["sha256"]

    for relative, expected in manifest["outputs"].items():
        path = ROOT / relative
        assert path.exists()
        assert path.stat().st_size > 0
        assert digest(path) == expected

    for relative, expected in manifest["auxiliary_generators"].items():
        path = ROOT / relative
        assert path.exists()
        assert digest(path) == expected


def test_paper_assets_cover_go_and_no_go_evidence():
    table = (ROOT / "docs" / "paper" / "tables" / "table_no_go_boundaries.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "Stage-1 measurement noise" in table
    assert "cross-grid model mismatch" in table
    assert "spatial BIC-only equivalence safety" in table
    assert "19/20 exact-equivalence abstentions" in table
    assert "candidate-out response transfer" in table
    assert "13/24 candidate-out cases falsely admitted" in table

    discovery = (ROOT / "docs" / "paper" / "tables" / "table_discovery_robustness.csv").read_text(
        encoding="utf-8-sig"
    )
    for experiment in ("M4", "M5", "M6", "O1"):
        assert experiment in discovery

    h4 = (ROOT / "docs" / "paper" / "tables" / "table_h4_identifiability_chain.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "H4-C passive partial-sharing partition" in h4
    assert "3/4 unique correct" in h4
    assert "4/4 set coverage; 0 wrong" in h4
    assert "H4-AL independent response bridge" in h4
    assert "40/40 positive correct" in h4
    assert "20/20 equivalent abstain" in h4
    assert "H4-AO spatial BIC-only response" in h4
    assert "19/20 exact-equivalent abstain" in h4
    assert "H4-AQ equivalence-guarded spatial response" in h4
    assert "39/40 positive correct" in h4
    assert "60/60 near/exact-equivalent abstain" in h4
    assert "H4-AR candidate-out finite-volume audit" in h4
    assert "13/24 candidate-out false admissions" in h4
    assert "H4-AT adequacy-guarded cross-solver response" in h4
    assert "57/60 in-grammar correct" in h4
    assert "60/60 candidate-out abstain" in h4
