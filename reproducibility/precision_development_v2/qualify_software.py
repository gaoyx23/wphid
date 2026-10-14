"""Record independent component-solver software checks, never scientific fits."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path.cwd().resolve()!=ROOT:
        raise RuntimeError("Run only from the independent algorithm worktree")
    sys.path.insert(0,str(ROOT/'src'))
    import wphid
    if Path(wphid.__file__).resolve()!=ROOT/'src/wphid/__init__.py':
        raise RuntimeError("Unexpected wphid import outside this worktree")
    prior=ROOT/'artifacts/paper_final_candidate_v1/scientific_confirmation_20261011/source-freeze.json'
    locked=json.loads(prior.read_text(encoding='utf-8'))
    identities={**locked['source_sha256'],**locked['frozen_reference_sha256']}
    for name,expected in identities.items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Frozen reference changed: '+name)
    parent=ROOT/'artifacts/paper_precision_development_v2/software_qualification'
    attempt=1
    while (parent/f'attempt-{attempt:02d}').exists():attempt+=1
    output=parent/f'attempt-{attempt:02d}'
    output.mkdir(parents=True)
    sources=[*sorted((ROOT/'src/wphid/precision_development_v2').glob('*.py')),
             *sorted((ROOT/'tests/precision_development_v2').glob('*.py')),
             *sorted(Path(__file__).parent.glob('*.py')),
             Path(__file__).parent/'development-protocol.json']
    source_ids={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    env=os.environ.copy()
    env.update({'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1',
                'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                'NUMEXPR_NUM_THREADS':'1','PYTHONIOENCODING':'utf-8'})
    command=[sys.executable,'-B','-m','pytest','tests/precision_development_v2','-q',
             '-p','no:cacheprovider','--junitxml',str(output/'junit.xml')]
    completed=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
    (output/'pytest.stdout.log').write_text(completed.stdout,encoding='utf-8')
    (output/'pytest.stderr.log').write_text(completed.stderr,encoding='utf-8')
    suites=ET.parse(output/'junit.xml').getroot()
    cases=list(suites.iter('testcase'))
    failed=sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped=sum(c.find('skipped') is not None for c in cases)
    for name,expected in {**identities,**source_ids}.items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Source changed during qualification: '+name)
    record={'version':'WPHID_COMPONENT_PROFILE_DEVELOPMENT_V2','date':'2026-10-11',
            'role':'artificial software fixtures; not identification performance',
            'tests':len(cases),'failed':failed,'skipped':skipped,'all_passed':completed.returncode==0 and failed==0 and skipped==0,
            'command':command,'python_executable':sys.executable,'wphid_import':str(Path(wphid.__file__).resolve()),
            'source_sha256':source_ids,'frozen_V1_reference_sha256':identities,
            'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','pytest']},
            'device':'CPU','dtype':'float64','threads_per_process':1,
            'new_scientific_estimates':0,'new_physical_trajectories':0,'new_scientific_noise':0,
            'pytest_stdout_sha256':sha(output/'pytest.stdout.log'),'junit_sha256':sha(output/'junit.xml')}
    (output/'qualification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'qualification':output.relative_to(ROOT).as_posix(),'tests':len(cases),
                      'failed':failed,'skipped':skipped,'all_passed':record['all_passed'],'new_scientific_estimates':0}))
    if not record['all_passed']:
        print(completed.stdout)
        raise SystemExit(completed.returncode or 1)


if __name__=='__main__':main()
