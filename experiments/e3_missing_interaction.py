"""Command entry point for the E3 missing-interaction gate."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from experiments.e3_missing_interaction_core import (
    DEFAULT_OUTPUT_DIR,
    EXPECTED_MISSING_INTERACTION,
    NOISE_FRACTION,
    SEEDS,
    TRAIN_IDS,
    VALIDATION_IDS,
    run_experiment,
    run_seed,
)


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "EXPECTED_MISSING_INTERACTION",
    "NOISE_FRACTION",
    "SEEDS",
    "TRAIN_IDS",
    "VALIDATION_IDS",
    "run_experiment",
    "run_seed",
]


if __name__ == "__main__":
    experiment_metrics = run_experiment()
    aggregate = experiment_metrics["aggregate"]
    print(
        "E3 seeds passing:",
        f"{aggregate['passing_seed_count']}/{len(aggregate['seeds'])}",
        "detection rate:",
        f"{aggregate['missing_interaction_detection_rate']:.3f}",
        "max false interactions:",
        aggregate["maximum_false_interactions"],
    )
