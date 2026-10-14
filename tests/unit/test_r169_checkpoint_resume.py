"""Check that interrupted calibration retains the exact random stream."""

import json

import numpy as np
import pytest

from experiments import paper_r169_96repeat_geometry_profile_budget as experiment


def test_calibration_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "CHECKPOINT_DIR", tmp_path / "resumed")
    monkeypatch.setattr(experiment, "CALIBRATION_DRAWS", 240)
    monkeypatch.setattr(experiment, "REPEAT_COUNT", 3)
    monkeypatch.setattr(experiment, "K_THROUGH_NODES", (9.0,))
    monkeypatch.setattr(experiment, "CANDIDATE_FRACTIONS", (0.0, 1.0))
    monkeypatch.setattr(experiment, "_candidate_truth", lambda *args: args[1])
    monkeypatch.setattr(
        experiment, "_observe", lambda truth, model, rng, blocks: rng.normal()
    )
    calls = 0

    def scores(observations, truth, *args):
        nonlocal calls
        calls += 1
        if calls == 111:
            raise RuntimeError("simulated process interruption")
        value = float(np.mean(observations))
        return {
            structure: (value - index) ** 2
            for index, structure in enumerate(experiment.FORCHHEIMER_STRUCTURES)
        }, {}

    monkeypatch.setattr(experiment, "_profile_geometry_and_solver", scores)
    model = experiment.NOISE_MODELS[0]
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        experiment._calibrate(model, (), None, None)
    partial_path = experiment._progress_path(model)
    saved = json.loads(partial_path.read_text(encoding="utf-8"))
    assert len(saved["calibration_gaps"]) == 100
    resumed = experiment._calibrate(model, (), None, None)
    resumed_gaps = json.loads(partial_path.read_text(encoding="utf-8"))[
        "calibration_gaps"
    ]

    monkeypatch.setattr(experiment, "CHECKPOINT_DIR", tmp_path / "uninterrupted")
    uninterrupted = experiment._calibrate(model, (), None, None)
    uninterrupted_gaps = json.loads(
        experiment._progress_path(model).read_text(encoding="utf-8")
    )["calibration_gaps"]
    assert resumed == uninterrupted
    assert resumed_gaps == uninterrupted_gaps


def test_resume_rejects_changed_repeat_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "CHECKPOINT_DIR", tmp_path)
    model = experiment.NOISE_MODELS[0]
    experiment._atomic_checkpoint(
        experiment._progress_path(model), experiment._load_progress(model)
    )
    monkeypatch.setattr(experiment, "REPEAT_COUNT", experiment.REPEAT_COUNT + 1)
    with pytest.raises(RuntimeError, match="incompatible partial checkpoint"):
        experiment._load_progress(model)
