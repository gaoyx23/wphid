"""Qualify isolated executable behavior on hand-written software fixtures.

No scientific observations, generator, experiment entry point, or historical
fit is loaded. Output is software evidence, never recovery-performance data.
"""

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/_software_qualification/final_candidate_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    resolved = path.resolve()
    if not (resolved.is_relative_to(OUTPUT) or resolved.is_relative_to(HERE)):
        raise ValueError("Software output escaped its isolated roots")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Software qualification must run from the algorithm worktree")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    import wphid
    import numpy
    import scipy
    import torch
    from wphid.final_candidate_v1 import VERSION
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not Path(wphid.__file__).resolve().is_relative_to(ROOT / "src"):
        raise ValueError("WPHID imported outside the isolated source tree")
    contract = json.loads((ROOT / "docs/paper/manuscript/ress-20261010/assets/final-candidate-contract.json").read_text())
    frozen = dict(contract["reference_source_sha256"])
    for relative in ("src/wphid/weak/axisymmetric.py", "src/wphid/weak/axisymmetric_library.py", ".gitattributes"):
        frozen[relative] = sha(ROOT / relative)
    for name, expected in frozen.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen reference changed: " + name)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    test_path = ROOT / "tests/final_candidate_v1/test_pipeline.py"
    xml = OUTPUT / "pytest.xml"
    checked = subprocess.run([sys.executable, "-B", "-m", "pytest", str(test_path),
                              "-p", "no:cacheprovider", "--junitxml="+str(xml), "-q"],
                             cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    suites = ET.parse(xml).getroot().findall(".//testsuite") if xml.is_file() else []
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0))+int(s.get("errors", 0)) for s in suites)
    if checked.returncode or failures or not tests:
        print(checked.stdout)
        print(checked.stderr)
        raise RuntimeError("Candidate software fixtures failed")
    for name, expected in frozen.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen reference changed during software check: " + name)
    environment = default_environment()
    environment["extra"] = ""
    closure, queue = {}, ["numpy", "scipy", "torch", "pytest", "packaging"]
    while queue:
        name = canonicalize_name(queue.pop())
        if name in closure:
            continue
        distribution = metadata.distribution(name)
        closure[name] = distribution.version
        for spec in distribution.requires or []:
            requirement = Requirement(spec)
            if requirement.marker is None or requirement.marker.evaluate(environment):
                if not requirement.specifier.contains(metadata.version(requirement.name), prereleases=True):
                    raise ValueError("Installed dependency violates requirement: " + spec)
                queue.append(requirement.name)
    installed = {canonicalize_name(d.metadata["Name"]): d.version for d in metadata.distributions() if d.metadata["Name"]}
    runtime = {"version": "CANDIDATE_CURRENT_SOFTWARE_RUNTIME_LOCK_V1", "python": sys.version,
               "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine(),
               "device": "CPU", "dtype": "float64", "torch_version": torch.__version__,
               "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
               "cpu_model_environment": os.environ.get("PROCESSOR_IDENTIFIER"),
               "logical_processors": os.cpu_count(), "numerical_test_dependency_closure": dict(sorted(closure.items())),
               "installed_distribution_snapshot": dict(sorted(installed.items())),
               "import_paths": {m.__name__: str(Path(m.__file__).resolve()) for m in (wphid, numpy, scipy, torch)},
               "historical_environment_reconstruction": False, "scientific_runtime_qualified": False}
    write(HERE / "runtime-lock.json", runtime)
    # Platform-specific exact versions for this software fixture environment.
    (HERE / "requirements-software-lock.txt").write_text(
        "# Current CPU software qualification only; not a historical scientific lock\n"+
        "\n".join(f"{n}=={v}" for n, v in sorted(closure.items()))+"\n", encoding="utf-8")
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "src/wphid/final_candidate_v1").glob("*.py"))}
    sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (test_path, Path(__file__),
                    HERE / "official-source-lock.json", HERE / "execution-plan.json")})
    report = {"version": VERSION, "date": "2026-10-11", "status": "hand_written_software_fixtures_passed",
              "software_tests_passed": tests, "failures": failures, "research_runs_started": 0,
              "scientific_recovery_cases": 0, "historical_results_rescored": False,
              "default_estimator_changed": False, "source_sha256": sources,
              "frozen_reference_sha256": frozen, "runtime_lock_sha256": sha(HERE / "runtime-lock.json"),
              "requirements_lock_sha256": sha(HERE / "requirements-software-lock.txt"),
              "pytest_xml_sha256": sha(xml),
              "scope": ["training/heldout isolation", "common null/nonempty final score",
                        "active partition counting and separate mechanism bindings", "signed physical coefficient mapping",
                        "fixed prediction without heldout coefficient refits", "same differentiable weak atoms",
                        "NumPy/torch gradient consistency", "legacy weak projector interoperability",
                        "objective/backward budget accounting and retained failures", "own-objective Adam terminal selection"],
              "scientific_precision_qualification_established": False,
              "matched_ADAM_performance_established": False,
              "complete_final_estimator_qualified": False,
              "candidate_reporting_gates_calibrated": False}
    write(HERE / "implementation-manifest.json", report)
    write(OUTPUT / "qualification.json", report)
    print(json.dumps({"software_tests_passed": tests, "research_runs_started": 0,
                      "source_files_bound": len(sources), "wphid_import": str(Path(wphid.__file__).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
