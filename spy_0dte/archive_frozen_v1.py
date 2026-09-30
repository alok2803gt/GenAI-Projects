"""
Assemble the immutable frozen_v1 research package (reproducible negative result).

    ./.venv/bin/python archive_frozen_v1.py --sensitivity-log PATH --report-log PATH

Writes archive/frozen_v1/ with the source code, candidate set, OOS predictions,
fold log, report outputs, environment versions, SHA-256 hashes of every package
file and of every INPUT dataset file, then makes the package read-only.
Verify later with:   cd archive/frozen_v1 && shasum -a 256 -c SHA256SUMS
                     shasum -a 256 -c INPUT_SHA256SUMS   (run from spy_0dte/)
The holdout is not read: input files are hashed as bytes only.
"""
import argparse
import hashlib
import json
import platform
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "archive" / "frozen_v1"
SOURCES = ["spy0dte_framework.py", "option_returns.py", "execution_sensitivity.py", "run_real.py",
           "run_holdout.py", "test_core.py", "fetch_alpaca_bars.py", "fetch_alpaca_options.py",
           "requirements.txt", "archive_frozen_v1.py"]
ARTIFACTS = {"data/optret_candidates_frozen_v1.parquet": "optret_candidates.parquet",
             "data/optret_oos_predictions_frozen_v1.parquet": "optret_oos_predictions.parquet",
             "data/optret_folds.csv": "optret_folds.csv"}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow", "requests"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity-log", required=True)
    ap.add_argument("--report-log", required=True)
    args = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists -- frozen packages are never overwritten")
    (OUT / "src").mkdir(parents=True)

    for s in SOURCES:
        shutil.copy2(HERE / s, OUT / "src" / s)
    for src, dst in ARTIFACTS.items():
        shutil.copy2(HERE / src, OUT / dst)
    shutil.copy2(args.sensitivity_log, OUT / "execution_sensitivity_output.txt")
    shutil.copy2(args.report_log, OUT / "option_returns_output.txt")
    shutil.copy2(HERE / "RESEARCH_CONCLUSION.md", OUT / "RESEARCH_CONCLUSION.md")

    inputs = [HERE / "data" / "SPY_1min_sip.parquet"] + sorted((HERE / "data" / "SPY_0dte_option_bars").glob("*.parquet"))
    input_lines = [f"{sha256(p)}  {p.relative_to(HERE)}" for p in inputs]
    (OUT / "INPUT_SHA256SUMS").write_text("\n".join(input_lines) + "\n")
    combined_inputs = hashlib.sha256("\n".join(input_lines).encode()).hexdigest()

    try:
        repo = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        repo = ""
    manifest = dict(
        package="spy_0dte frozen_v1", created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status="ARCHIVED -- stopped under pre-registered rule; holdout unexamined",
        holdout_start="2025-09-22", development_sessions="2024-02-01 .. 2025-09-19 (option data); OOS 2024-08-01 .. 2025-09-19",
        repo_head=repo or None,
        repo_note="spy_0dte/ was untracked in git when frozen; source identity is the SHA-256 of src/* below",
        python=platform.python_version(), platform=platform.platform(),
        packages={p: version(p) for p in PACKAGES},
        inputs=dict(files=len(inputs), combined_sha256=combined_inputs,
                    spy_bars_sha256=input_lines[0].split()[0], list="INPUT_SHA256SUMS"),
        reproduce=["./.venv/bin/python option_returns.py --rebuild",
                   "./.venv/bin/python execution_sensitivity.py   # asserts it reproduces the frozen predictions",
                   "./.venv/bin/python test_core.py"],
        future_holdout_test=("exactly one frozen specification: short 1.0 expected-move call spread, "
                             "$3 wide, every decision bar 10:00-14:30, exit 15:45, lambda 0.25 -- nothing changed after viewing"),
    )
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(OUT)}\n" for p in files))

    for p in list(OUT.rglob("*")) + [OUT]:
        mode = p.stat().st_mode
        p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(f"frozen package written (read-only): {OUT}\n{len(files) + 1} files; inputs combined sha256 {combined_inputs}")


if __name__ == "__main__":
    main()
