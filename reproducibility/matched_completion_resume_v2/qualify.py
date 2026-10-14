"""Qualify only V2 administrative orchestration, never frozen old tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(1,str(ROOT))
from reproducibility.matched_completion_resume_v2 import complete as c
def main():
    c.r1.OUTPUT=c.OLD;c.r1.bind_original();c.r1.verify()
    output=ROOT/'artifacts/paper_matched_completion_resume_v2/software';output.mkdir(parents=True,exist_ok=True)
    if (output/'qualification.json').exists():raise FileExistsError('Already qualified')
    files=[*sorted(Path(__file__).parent.glob('*.py')),c.PROTOCOL_PATH,*sorted((ROOT/'tests/matched_completion_resume_v2').glob('*.py'))]
    sources={p.relative_to(ROOT).as_posix():c.w.sha(p) for p in files}
    env=os.environ.copy();env.update({'PYTHONPATH':os.pathsep.join((str(ROOT/'src'),str(ROOT))),'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'})
    command=[sys.executable,'-B','-m','pytest','tests/matched_completion_resume_v2','-q','-p','no:cacheprovider','--junitxml',str(output/'junit.xml')]
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
    (output/'stdout.log').write_text(result.stdout,encoding='utf-8');(output/'stderr.log').write_text(result.stderr,encoding='utf-8')
    cases=list(ET.parse(output/'junit.xml').getroot().iter('testcase'));failed=sum(c.find('failure') is not None or c.find('error') is not None for c in cases);skipped=sum(c.find('skipped') is not None for c in cases)
    c.w.verify_hashes(sources)
    with (output/'qualification.json').open('x',encoding='utf-8') as f:json.dump({'tests':len(cases),'failed':failed,'skipped':skipped,'all_passed':result.returncode==0 and failed==0 and skipped==0,'source_sha256':sources,'junit_sha256':c.w.sha(output/'junit.xml'),'new_scientific_fits':0},f,indent=2)
    print(result.stdout,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
