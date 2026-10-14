"""Check saved predictions and units once; never optimize or profile a model."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT))
import numpy as np
from reproducibility.matched_completion_v1 import workflow
from reproducibility.final_candidate_v1.scientific_io import load_conditions
from wphid.final_candidate_v1.pipeline import prepare_views, frozen_residual_report
from wphid.final_candidate_v1.problem import paper_families
from wphid.final_candidate_v1.solver import Candidate, Mechanism, FittedCandidate, design


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    stage = sys.argv[1]
    if stage not in ('development', 'fresh'):
        raise ValueError('Unknown comparison stage')
    workflow.verify_freeze()
    output = ROOT / 'artifacts/paper_comparison_completion_20261013'
    destination = output / (stage + '-saved-adam-replay.json')
    if destination.exists():
        raise FileExistsError('Completed replay exists; inspect receipt instead')
    complete = read(workflow.OUTPUT / (stage + '-execution-complete.json'))
    original = workflow.ORIGINAL
    family = next(f for f in paper_families() if f.name == original['Q1']['known_family'])
    rows = []
    for relative, expected in complete['fit_sha256'].items():
        record = read(ROOT / relative)
        if sha(ROOT / relative) != expected:
            raise ValueError('Saved estimate changed')
        if record['job']['program'] != 'adam':
            continue
        if 'fit' not in record:
            rows.append({'file': relative, 'fit_sha256': expected, 'passed': False, 'reason': 'Failed estimate retained'})
            continue
        packet = record['job']['input']
        conditions = load_conditions(ROOT / packet['file'], packet['sha256'])
        training, heldout = prepare_views(conditions, (family,), tuple(original['train_conditions']), tuple(original['heldout_conditions']))
        saved = record['fit']
        hypothesis = saved['hypothesis'][0]
        candidate = Candidate((Mechanism(hypothesis['mechanism'], family, tuple(hypothesis['groups'])),))
        parameters = np.asarray(record['flat_parameters'])
        coefficients = tuple(np.asarray(c) for c in saved['scaled_coefficients'])
        if saved['training_scale_sha256'] != training.scale.fingerprint:
            raise ValueError('Saved training scale changed')
        rss = tuple(float((y - matrix @ c) @ (y - matrix @ c)) for y, matrix, c in zip(training.targets, design(training, candidate, parameters), coefficients))
        np.testing.assert_allclose(rss, saved['equation_training_rss'], rtol=2e-11, atol=1e-13)
        objective = sum(r / len(y) for r, y in zip(rss, training.targets))
        np.testing.assert_allclose(objective, saved['training_objective'], rtol=2e-11, atol=1e-13)
        for e, scaled in enumerate(coefficients):
            norms = np.r_[training.scale.base_norms[e], training.scale.norms_for(family)[e]]
            np.testing.assert_allclose(scaled / norms, saved['physical_coefficients'][e], rtol=2e-13, atol=1e-13)
        fit = FittedCandidate(candidate, parameters, coefficients, rss, saved['training_objective'], saved['common_final_score'],
            saved['nominal_parameter_count'], saved['training_scale_sha256'], saved['optimizer_status'], saved['objective_calls'],
            saved['cpu_seconds'], saved['wall_seconds'], tuple(saved['restarts']))
        current = frozen_residual_report(heldout, fit)
        np.testing.assert_allclose(current['equation_weak_target_relative_residual'], record['frozen_heldout']['equation_weak_target_relative_residual'], rtol=2e-13, atol=1e-13)
        rows.append({'file': relative, 'fit_sha256': expected, 'passed': True, 'new_parameter_or_coefficient_fits': 0})
    output.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump({'stage': stage, 'execution_receipt_sha256': sha(workflow.OUTPUT / (stage + '-execution-complete.json')),
            'source_freeze_sha256': sha(workflow.OUTPUT / 'source-freeze.json'), 'verification_source_sha256': sha(__file__),
            'rows': rows, 'all_saved_finite_points_passed': all(r['passed'] for r in rows),
            'new_optimization_or_profiles': 0, 'truth_imports': 0, 'not_a_scientific_precision_pass': True}, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'stage': stage, 'records': len(rows), 'passed': sum(r['passed'] for r in rows)}))


if __name__ == '__main__':
    main()
