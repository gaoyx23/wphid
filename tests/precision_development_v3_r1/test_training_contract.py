"""The fixed-point diagnostic must restore exactly the saved Q1 split."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def test_training_and_heldout_order_match_the_frozen_Q1_protocol():
    current = read('reproducibility/precision_development_v3_r1/preflight-protocol.json')
    prior = read('reproducibility/final_candidate_v1/scientific-protocol.json')
    assert current['train_conditions'] == prior['train_conditions']
    assert current['heldout_conditions'] == prior['heldout_conditions']
    assert not set(current['train_conditions']) & set(current['heldout_conditions'])


def test_failed_configuration_attempt_is_preserved_with_no_computed_moments():
    failed = read('artifacts/paper_precision_development_v3/moment_preflight_20261011/assessment.json')
    current = read('reproducibility/precision_development_v3_r1/preflight-protocol.json')
    assert failed['saved_points'] == 18
    assert failed['assessable_points'] == 0
    assert failed['new_nonlinear_fits'] == 0
    assert all(r['error'] == 'Training scale differs from saved point' for r in failed['records'])
    assert current['configuration_revision']['mathematics_changed'] is False
    assert current['saved_points'] == len(current['roles']) * len(current['fixed_seeds']) == 18
