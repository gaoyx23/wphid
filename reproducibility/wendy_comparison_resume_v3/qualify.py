"""Qualify administrative changes without fitting or generating observations."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reproducibility.wendy_comparison_resume_v3 import complete as c
def main():
    target=Path(__file__).with_name('qualification.json')
    if target.exists():raise FileExistsError('Qualification already saved')
    result=subprocess.run([sys.executable,'-B','-m','pytest','-q','tests/wendy_comparison_resume_v3'],cwd=ROOT,text=True,capture_output=True)
    stdout=Path(__file__).with_name('qualification-stdout.txt')
    if stdout.exists():raise FileExistsError('Qualification log already exists')
    stdout.write_text(result.stdout+result.stderr,encoding='utf-8')
    files=[p for p in Path(__file__).parent.glob('*') if p.is_file() and p.name!='qualification.json']
    files.extend((ROOT/'tests/wendy_comparison_resume_v3').glob('*.py'))
    c.save(target,{'all_passed':result.returncode==0,'returncode':result.returncode,'source_sha256':{p.relative_to(ROOT).as_posix():c.f.sha(p) for p in files},
        'new_parameter_fits':0,'new_noise_or_integrations':0,'actual_cwd':str(Path.cwd()),'intended_import':str(ROOT/'src/wphid/__init__.py')})
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
