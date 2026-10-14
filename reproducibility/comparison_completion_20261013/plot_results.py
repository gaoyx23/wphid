"""Plot saved comparative endpoints, with failed estimates visible."""

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'artifacts/paper_comparison_completion_20261013'


def main():
    report = json.loads((OUTPUT / 'completion-assessment.json').read_text(encoding='utf-8'))
    path = OUTPUT / 'comparison-cases.csv'
    expected = report['bound_receipts'][path.relative_to(ROOT).as_posix()]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Plot input differs from final comparative receipt')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    generated = {}
    for stage in sorted({r['stage'] for r in rows}):
        selected = [r for r in rows if r['stage'] == stage]
        programs = [p for p in ('wphid', 'adam', 'gaussian_quasi_likelihood') if any(r['program'] == p for r in selected)]
        labels = {'wphid': 'WPHID', 'adam': 'ADAM adapted', 'gaussian_quasi_likelihood': 'Gaussian QL'}
        colors = {'wphid': '#0072B2', 'adam': '#D55E00', 'gaussian_quasi_likelihood': '#009E73'}
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True)
        for column, role in enumerate(('shared', 'partial', 'independent')):
            subset = [r for r in selected if r['role'] == role]
            finite = [float(r['case_score']) for r in subset if r['case_score'] and np.isfinite(float(r['case_score']))]
            upper = max(10., max(finite, default=1.) * 2.)
            seeds = sorted({int(r['seed']) for r in subset})
            for seed in seeds:
                linked = [next((r for r in subset if int(r['seed']) == seed and r['program'] == p), None) for p in programs]
                y = [max(float(r['case_score']), 1e-4) if r and r['case_score'] else np.nan for r in linked]
                axes[0, column].plot(range(len(programs)), y, color='#BBBBBB', alpha=.45, linewidth=.6, zorder=1)
            for index, program in enumerate(programs):
                cases = [r for r in subset if r['program'] == program]
                passed = sum(r['precision_pass'] == 'True' for r in cases)
                scores = [max(float(r['case_score']), 1e-4) for r in cases if r['case_score']]
                axes[0, column].scatter(np.full(len(scores), index), scores, color=colors[program], s=23, alpha=.8, zorder=2)
                missing = len(cases) - len(scores)
                if missing:
                    axes[0, column].scatter([index], [upper], marker='x', color=colors[program], s=65)
                axes[0, column].text(index, .98, f'{passed}/{len(cases)} pass' + (f'\n{missing} invalid' if missing else ''),
                    transform=axes[0, column].get_xaxis_transform(), ha='center', va='top', fontsize=8)
                cpu = [float(r['cpu_seconds']) for r in cases if r['cpu_seconds'] and float(r['cpu_seconds']) > 0]
                axes[1, column].scatter(np.full(len(cpu), index), cpu, color=colors[program], s=23, alpha=.7)
                if cpu:
                    axes[1, column].plot([index-.18, index+.18], [np.median(cpu)]*2, color='black', linewidth=2)
            axes[0, column].axhline(1., color='black', linestyle='--', linewidth=1)
            axes[0, column].set_yscale('log')
            axes[0, column].set_ylim(top=upper*2.)
            axes[0, column].set_title(role.capitalize() + ' parameters')
            axes[1, column].set_yscale('log')
            for axis in axes[:, column]:
                axis.set_xticks(range(len(programs)), [labels[p] for p in programs], rotation=12)
                axis.grid(axis='y', alpha=.2)
        axes[0, 0].set_ylabel('Maximum gate-normalized parameter error')
        axes[1, 0].set_ylabel('Solver CPU seconds (black line: median)')
        title = 'Fresh conditional comparison' if stage.startswith('fresh') else 'Development diagnostic: archived observations'
        fig.suptitle(title + '\nKnown family, active support and partition; error score <= 1 passes the reaction parameter gate', fontsize=12)
        for extension in ('svg', 'png'):
            destination = OUTPUT / (stage + '-comparison.' + extension)
            if destination.exists():
                raise FileExistsError('Scientific plot already generated')
            fig.savefig(destination, dpi=160)
            generated[destination.relative_to(ROOT).as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
        plt.close(fig)
    receipt = OUTPUT / 'plot-receipt.json'
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump({'input_csv_sha256': expected, 'plot_sha256': generated, 'new_fits': 0}, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'plots': list(generated), 'new_fits': 0}))


if __name__ == '__main__':
    main()
