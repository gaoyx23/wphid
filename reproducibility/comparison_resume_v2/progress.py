"""Read persisted completion and runtime records without loading scientific truth."""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'artifacts/paper_comparison_resume_v2/gaussian_fresh_confirmation'


def main():
    fits = len(list((OUTPUT / 'fits').glob('*.json')))
    journal = OUTPUT / 'driver-cpu-checkpoints.jsonl'
    state = None
    if journal.exists():
        with journal.open('rb') as stream:
            stream.seek(max(0, journal.stat().st_size - 8192))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                state = json.loads(line)
                break
            except (ValueError, UnicodeError):
                continue
    result = {'saved': fits, 'declared': 54, 'all_stages_complete': (OUTPUT / 'all-stages-complete.json').exists(),
              'scientific_execution_complete': (OUTPUT / 'execution-complete.json').exists(),
              'checkpoint_age_seconds': None if state is None else round(time.time() - state['utc_epoch'], 1),
              'active_recorded_CPU_seconds': [] if state is None else [round(s['job_CPU_seconds_observed'], 1) for s in state['states']],
              'truth_or_parameter_scores_read': False}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
