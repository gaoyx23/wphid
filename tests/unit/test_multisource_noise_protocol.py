"""Noise and checkpoint primitives; no new physical simulations."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import numpy as np

spec = spec_from_file_location("multisource_audit", Path(__file__).resolve().parents[2]/"experiments/audit_multisource_finite_noise.py")
module = module_from_spec(spec); spec.loader.exec_module(module)


def clean_fields():
    return {n:np.ones((3, 3, 3))*(500. if n == "T" else .2 if n == "crack" else 1.) for n in module.FIELDS}


def test_repeated_identity_is_reproducible_without_clipping():
    a, c = module.draw_observations(clean_fields(), 0, 1)
    b, d = module.draw_observations(clean_fields(), 0, 1)
    for name in a: np.testing.assert_array_equal(a[name], b[name])
    np.testing.assert_array_equal(c, d)
    assert module.domain_violations(a) == 0


def test_source_and_draw_seeds_are_distinct():
    a, _ = module.draw_observations(clean_fields(), 0, 1)
    b, _ = module.draw_observations(clean_fields(), 1, 1)
    c, _ = module.draw_observations(clean_fields(), 0, 2)
    assert not np.array_equal(a["pO2"], b["pO2"])
    assert not np.array_equal(a["pO2"], c["pO2"])


def test_nonpressure_noise_matches_declared_modes():
    clean = clean_fields(); observed, coefficients = module.draw_observations(clean, 2, 3)
    np.testing.assert_allclose(observed["T"]-clean["T"], 2.*coefficients[2])
    np.testing.assert_allclose(observed["crack"]-clean["crack"], .02*clean["crack"]*coefficients[3])
    np.testing.assert_array_equal(observed["stress"], clean["stress"])
    np.testing.assert_array_equal(observed["carbon"], clean["carbon"])


def test_partial_candidate_checkpoint_is_preserved(tmp_path):
    path = tmp_path/"draw.json"
    module.write_json(path, {"residuals":{"a":[1.]}, "complete":False})
    saved = json.loads(path.read_text())
    saved["residuals"]["b"] = [2.]
    module.write_json(path, saved)
    assert json.loads(path.read_text())["residuals"] == {"a":[1.], "b":[2.]}
