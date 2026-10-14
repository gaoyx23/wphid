"""Freeze and test only independent replay unit accounting; no fitting."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(1,str(ROOT))
from reproducibility.matched_completion_resume_v2 import complete as c
def main():
    c.verify();output=c.OUTPUT/'replay-r1-software';output.mkdir(parents=True,exist_ok=True)
    if (output/'qualification.json').exists():raise FileExistsError('Already qualified')
    files=[Path(__file__).resolve(),Path(__file__).with_name('replay_r1.py'),ROOT/'tests/matched_completion_resume_v2/test_replay_units_r1.py']
    sources={p.relative_to(ROOT).as_posix():c.w.sha(p) for p in files}
    env=os.environ.copy();env.update({'PYTHONPATH':os.pathsep.join((str(ROOT/'src'),str(ROOT))),'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'})
    command=[sys.executable,'-B','-m','pytest','tests/matched_completion_resume_v2/test_replay_units_r1.py','-q','-p','no:cacheprovider','--junitxml',str(output/'junit.xml')]
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
    (output/'stdout.log').write_text(result.stdout,encoding='utf-8');(output/'stderr.log').write_text(result.stderr,encoding='utf-8')
    cases=list(ET.parse(output/'junit.xml').getroot().iter('testcase'));failed=sum(t.find('failure') is not None or t.find('error') is not None for t in cases)
    c.w.verify_hashes(sources)
    with (output/'qualification.json').open('x',encoding='utf-8') as f:json.dump({'tests':len(cases),'failures':failed,'all_passed':result.returncode==0 and failed==0,'source_sha256':sources,'junit_sha256':c.w.sha(output/'junit.xml'),'new_scientific_fits_noise_or_data':0,'frozen_before_independent_postfit_replay':True},f,indent=2)
    print(result.stdout,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
