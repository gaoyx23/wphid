"""Assemble explicitly scoped local data/code candidate; no external publish."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/paper/manuscript/ress-20261010"
NEW = ROOT / "artifacts/paper_final_candidate_v1"
OUTPUT = NEW / "release_candidate_20261011"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    scientific = NEW / "scientific_confirmation_20261011"
    if not (scientific / "assessment.json").exists() or not (scientific / "read-only-diagnostics.json").exists():
        raise ValueError("Complete scientific evidence before packaging")
    verification = scientific / "saved-estimate-verification.json"
    if not verification.exists():
        subprocess.run([sys.executable,"-B",str(Path(__file__).parent/"verify_saved_estimates.py")],cwd=ROOT,check=True)
    if not json.loads(verification.read_text(encoding="utf-8"))["all_789_record_identities_checked"]:
        raise ValueError("Saved model outputs have not all been verified")
    adaptation = scientific / "adaptation-scale-diagnostic.json"
    if not adaptation.exists():
        subprocess.run([sys.executable,"-B",str(Path(__file__).parent/"diagnose_adaptation_scale.py")],cwd=ROOT,check=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    archive = OUTPUT / "wphid-data-code-release-candidate-20261011.zip"
    if archive.exists():
        raise FileExistsError("Release candidate exists; inspect, do not overwrite")
    paths = {}

    def include(path,target=None,role="new_qualification_source_or_result"):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT) and not path.is_relative_to(ROOT.parents[1]):
            raise ValueError("Read escaped named research roots")
        target = target or path.relative_to(ROOT).as_posix()
        if target.startswith(("/","..")) or any(part in {".git",".codex",".agents",".aws","__pycache__"} for part in Path(target).parts):
            raise ValueError("Invalid or unrelated archive content")
        if target in paths and sha(path) != paths[target]["sha256"]:
            raise ValueError("Conflicting historical/current file")
        paths[target] = {"path":path,"sha256":sha(path),"bytes":path.stat().st_size,"role":role}

    lock = json.loads((scientific / "source-freeze.json").read_text(encoding="utf-8"))
    for relative,expected in {**lock["source_sha256"],**lock["frozen_reference_sha256"]}.items():
        if sha(ROOT/relative) != expected:
            raise ValueError("Frozen scientific source changed: "+relative)
        include(ROOT/relative)
    implementation = json.loads((ROOT / "reproducibility/final_candidate_v1/implementation-manifest.json").read_text(encoding="utf-8"))
    for relative,expected in implementation["source_sha256"].items():
        if sha(ROOT/relative) != expected:
            raise ValueError("Original software source changed: "+relative)
        include(ROOT/relative)
    for relative in ("tests/final_candidate_v1/test_exact_gradient.py",
                     "src/wphid/weak/__init__.py","src/wphid/discovery/__init__.py"):
        if (ROOT/relative).exists():
            include(ROOT/relative)
    for path in (ROOT/"reproducibility/final_candidate_v1").glob("*"):
        if path.is_file():
            include(path)
    for folder in (scientific,NEW/"cost_development_20261011",NEW/"exact_gradient_cost_development_20261011",
                   ROOT/"artifacts/_software_qualification/final_candidate_v1"):
        for path in folder.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                include(path)
    resume = NEW/"execution_resume_20261011_01/resume-audit.json"
    if resume.exists():
        resume_record = json.loads(resume.read_text(encoding="utf-8"))
        for relative,expected in resume_record["completed_fit_sha256"].items():
            if sha(ROOT/relative) != expected:
                raise ValueError("A saved estimate changed across checkpoint continuation: "+relative)
        include(resume,role="checkpoint_continuation_without_completed_fit_reruns")
    catalogue = PAPER / "assets/source-catalogue.csv"
    with catalogue.open(encoding="utf-8-sig",newline="") as stream:
        historical = list(csv.DictReader(stream))
    for record in historical:
        source_root = ROOT if record["location"] == "worktree" else ROOT.parents[1]
        path = source_root/record["source"]
        if sha(path) != record["sha256"]:
            raise ValueError("Historical source changed: "+record["source"])
        include(path,record["source"],"historical_source_record_original_bytes")
    include(catalogue)
    for name in ("final-candidate-contract.json","adam-comparison-status.json","journal-scope-review.json",
                 "identification-claims.json","precision-eight-packets.json","comparison-effects.json"):
        include(PAPER/"assets"/name)
    record = {"date":"2026-10-11","status":"local_author_review_candidate_not_public_release",
              "public_repository_or_identifier":None,"code_license":None,"author_data_publication_permission":True,
              "archive_name":archive.name,"historical_original_source_records":len(historical),
              "historical_large_array_completeness_certified":False,
              "historical_execution_environment_reconstructed":False,
              "source_freeze_sha256":sha(scientific/"source-freeze.json"),
              "input_manifest_sha256":sha(scientific/"input-manifest.json"),
              "assessment_sha256":sha(scientific/"assessment.json"),
              "saved_estimate_verification_sha256":sha(verification),
              "adaptation_scale_diagnostic_sha256":sha(adaptation),
              "checkpoint_resume_audit_sha256":sha(resume) if resume.exists() else None,
              "saved_job_CPU_totals_include_interrupted_unsaved_work":False,
              "files":{target:{k:v for k,v in value.items() if k != "path"} for target,value in sorted(paths.items())}}
    manifest = OUTPUT/"release-manifest.json"
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for target,value in sorted(paths.items()):
            stream.write(value["path"],arcname=target)
        stream.write(manifest,arcname="release-manifest.json")
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError("Release archive failed integrity check")
        for target,value in paths.items():
            if hashlib.sha256(stream.read(target)).hexdigest() != value["sha256"]:
                raise ValueError("Archive bytes differ from bound file")
    identity = {"archive":archive.name,"bytes":archive.stat().st_size,"sha256":sha(archive),
                "manifest_sha256":sha(manifest),"files_checked":len(paths),
                "historical_records_embedded":len(historical),"new_scientific_runs_in_packaging":0,
                "public_release_completed":False}
    (OUTPUT/"archive-identity.json").write_text(json.dumps(identity,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(identity,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
