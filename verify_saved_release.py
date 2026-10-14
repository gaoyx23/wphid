"""Verify published saved-file hashes and tabular summaries without model execution."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root=Path(__file__).resolve().parent
manifest=root/'reproduction-manifest.json'
if not manifest.is_file():raise FileNotFoundError('Extract the data-code release archive before running this utility.')
records=json.loads(manifest.read_text(encoding='utf-8'))['files']
for name,entry in records.items():
    p=(root/name).resolve()
    if not p.is_relative_to(root):raise ValueError('Path outside extracted release')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    if h.hexdigest()!=entry['sha256']:raise ValueError('Saved-file identity mismatch: '+name)
base=root/'artifacts/paper_comparison_resume_v4/final_evidence'
rows=list(csv.DictReader((base/'comparison-cases.csv').open(encoding='utf-8-sig',newline='')))
assert len(rows)==216
assert len({tuple(r[x] for x in ['stage','seed','role','program']) for r in rows})==216
fresh=[r for r in rows if r['stage']=='fresh_conditional_comparison'];assert len(fresh)==162
summary=json.loads((base/'three-program-role-summary.json').read_text(encoding='utf-8'))
for role,programs in summary['roles'].items():
 for program,s in programs.items():
  subset=[r for r in fresh if r['role']==role and r['program']==program]
  assert len(subset)==18
  assert sum(r['precision_pass']=='True' for r in subset)==s['reaction_parameters_and_amplitudes_precision_passes']
  assert abs(statistics.median(float(r['case_score']) for r in subset)-s['median_case_score_among_assessable'])<1e-12
print(json.dumps({'verified_saved_files':len(records),'unique_scored_rows':216,'fresh_rows':162,'new_scientific_runs':0}))
