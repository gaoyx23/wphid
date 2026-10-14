"""Qualify only the new orchestration; prior algorithm tests are not rerun."""
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reproducibility.matched_completion_v1.workflow import OUTPUT,save,sha,verify_runtime,old_receipts

def main():
    verify_runtime()
    old_receipts()
    path=OUTPUT/'software/qualification.json'
    if path.exists():raise FileExistsError('Already qualified; use saved receipt')
    directory=path.parent
    directory.mkdir(parents=True,exist_ok=True)
    files=[*sorted(Path(__file__).parent.glob('*.py')),Path(__file__).with_name('protocol.json'),*sorted((ROOT/'tests/matched_completion_v1').glob('*.py'))]
    identities={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    env=os.environ.copy()
    env.update({'PYTHONPATH':os.pathsep.join((str(ROOT/'src'),str(ROOT))),'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'})
    command=[sys.executable,'-B','-m','pytest','tests/matched_completion_v1','-q','-p','no:cacheprovider','--junitxml',str(directory/'junit.xml')]
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
    (directory/'stdout.log').write_text(result.stdout,encoding='utf-8')
    (directory/'stderr.log').write_text(result.stderr,encoding='utf-8')
    cases=list(ET.parse(directory/'junit.xml').getroot().iter('testcase'))
    failed=sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped=sum(c.find('skipped') is not None for c in cases)
    save(path,{'tests':len(cases),'failures':failed,'skipped':skipped,'all_passed':result.returncode==0 and failed==0 and skipped==0,
               'source_sha256':identities,'junit_sha256':sha(directory/'junit.xml'),'new_scientific_fits':0,'command':command})
    print(result.stdout,flush=True)
    if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
