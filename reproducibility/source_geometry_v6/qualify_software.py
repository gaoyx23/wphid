"""Bounded, source-frozen artificial qualification with independent outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "reproducibility/source_geometry_v6/software-protocol.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    # Each output directory must be new; exclusive mode also prevents overwrite.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]+", args.attempt):
        raise SystemExit("attempt must be a simple independent output name")
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from the actual algorithm worktree")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    output_parent = (ROOT / protocol["output_parent"]).resolve()
    output = (output_parent / args.attempt).resolve()
    if not output_parent.is_relative_to(ROOT / "artifacts") or output.parent != output_parent:
        raise SystemExit("output must stay in the independent local artifact root")
    if output.exists():
        raise SystemExit("existing attempt retained; inspect completion and hashes before another run")
    sys.path.insert(0, str(ROOT / "src"))
    import wphid
    import wphid.source_geometry_v6.geometry as geometry
    package = Path(wphid.__file__).resolve()
    module = Path(geometry.__file__).resolve()
    if not package.is_relative_to(ROOT / "src") or not module.is_relative_to(ROOT / "src"):
        raise SystemExit("unexpected imported package: main editable install is forbidden")
    identities = [{"path": p, "sha256": digest(ROOT / p)} for p in protocol["source_files"]]
    output.mkdir(parents=True, exist_ok=False)
    write(output / "source-freeze.json", {
        "protocol": protocol["version"], "cwd": str(Path.cwd().resolve()),
        "imported_wphid": str(package), "imported_geometry": str(module),
        "output_root": str(output), "sources": identities,
        "scientific_inputs": [], "frozen_versions_modified": False,
    })
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    timed_out = False
    try:
        process = subprocess.run(
            [sys.executable, "-m", "pytest", protocol["test_path"], "-q", "-p", "no:cacheprovider"],
            cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8",
            timeout=protocol["wall_cap_seconds"], check=False,
        )
        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        stdout = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
        returncode = None
    with (output / "pytest-output.txt").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(stdout + "\n" + stderr)
    unchanged = all(digest(ROOT / item["path"]) == item["sha256"] for item in identities)
    count = re.search(r"(\d+) passed", stdout)
    passed = int(count.group(1)) if count else 0
    assessment = {
        "protocol": protocol["version"], "status": "completed",
        "software_passed": returncode == 0 and unchanged and not timed_out and passed > 0,
        "artificial_checks_passed": passed, "pytest_returncode": returncode,
        "timed_out": timed_out, "wall_seconds": time.perf_counter() - started,
        "source_identities_unchanged": unchanged,
        "scientific_precision_established": False, "adoption_gate_passed": False,
        "new_fits": 0, "new_trajectories": 0, "new_noise_draws": 0,
        "original_eight_package_count": "6/8",
    }
    write(output / "assessment.json", assessment)
    write(output / "execution-complete.json", {
        "status": "completed", "assessment_sha256": digest(output / "assessment.json"),
        "source_freeze_sha256": digest(output / "source-freeze.json"),
        "software_passed": assessment["software_passed"], "scientific_run": False,
    })
    print(json.dumps(assessment, ensure_ascii=False, indent=2))
    raise SystemExit(0 if assessment["software_passed"] else 1)


if __name__ == "__main__":
    main()
