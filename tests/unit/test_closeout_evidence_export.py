import csv
import json
import xml.etree.ElementTree as ET
import pytest
from reproducibility.assemble_closeout_evidence import ROOT,OUT,pointer,digest


def test_pointer_handles_decimal_keys_lists_and_escaped_tokens():
    data={"0.005":[{"a/b":{"~":3}}]}
    assert pointer(data,"/0.005/0/a~1b/~0")==3
    with pytest.raises(KeyError):pointer(data,"/missing")


def test_all_exported_values_match_source_fields_and_hashes():
    ledger=json.loads((OUT/"claims.json").read_text(encoding="utf-8"));sources={}
    assert ledger["new_simulations"]==ledger["new_model_fits"]==ledger["new_projections"]==0
    assert len(ledger["claims"])==192
    for row in ledger["claims"]:
        path=ROOT/row["source"]
        assert digest(path)==row["source_sha256"]
        if row["source"] not in sources:sources[row["source"]]=json.loads(path.read_text(encoding="utf-8"))
        assert pointer(sources[row["source"]],row["json_pointer"])==row["value"]


def test_grammar_and_optimizer_differences_not_erased():
    with (OUT/"optimizer-supports.csv").open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==12
    assert all(row["grammar_same"]=="True" for row in rows)
    assert sum(row["official_same"]=="True" for row in rows)==11


def test_nested_plot_does_not_hide_truth_omissions_or_mix_counts():
    with (OUT/"nested-data.csv").open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    assert [row["all_five_truth_covered"] for row in rows]==["False","False","True"]
    assert [row["crack_supports"] for row in rows]==["5","8","9"]
    svg=(OUT/"fixed-rule-data-addition.svg").read_text(encoding="utf-8")
    ET.fromstring(svg)
    assert "Different candidate units" in svg and "not one fitted joint support" in svg


def test_old_figures_preserved_and_scope_checked_separately():
    report=json.loads((OUT/"existing-figure-review.json").read_text(encoding="utf-8"))
    assert not report["original_assets_modified"]
    assert len(report["records"])==16
    for row in report["records"]:
        assert digest(ROOT/row["path"])==row["observed_sha256"]==row["manifest_sha256"]
        assert not row["scientific_caption_certified"]


def test_pack_manifest_binds_outputs_and_generator():
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["new_scientific_experiments"]==0
    assert digest(ROOT/manifest["generator"])==manifest["generator_sha256"]
    for name,expected in manifest["outputs"].items():assert digest(OUT/name)==expected


def test_prior_forward_results_retained_with_scope_and_failed_group():
    with (OUT/"conditional-forward-groups.csv").open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==30
    old=[row for row in rows if row["source_id"]=="r155"]
    darcy=[row for row in rows if row["source_id"]=="r161"]
    assert sum(row["criterion_satisfied"]=="True" for row in old)==14
    assert all(row["criterion_satisfied"]=="True" for row in darcy)


def test_actual_manuscript_figures_bound_without_rewriting_paper():
    report=json.loads((OUT/"manuscript-figure-review.json").read_text(encoding="utf-8"))
    assert report["manuscript_matches_frozen"]
    assert digest(ROOT/report["manuscript"])==report["frozen_manuscript_sha256"]
    assert not report["manuscript_body_modified"] and not report["figures_modified"]
    assert len(report["records"])==6
    assert sum(row["stale_result_status"] for row in report["records"])==1
    for row in report["records"]:
        assert digest(ROOT/row["path"])==row["observed_sha256"]
        assert not row["caption_rewritten"]
