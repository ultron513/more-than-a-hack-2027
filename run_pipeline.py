"""Reproduce and validate the possession submission."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SUBMISSION = "evaluation/results/candidate_temporal_v4.csv"

PIPELINE = [
    "src/data/validate_assets.py",
    "src/features/extract_tracking_features.py",
    "src/features/extract_tagging_features.py",
    "src/features/merge_features.py",
    "src/models/temporal_possession.py",
    "src/models/postprocess_possession.py",
]


def run_script(script, *args):
    command = [sys.executable, str(ROOT / script), *args]
    print(f"\n===== Running {script} =====", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Also score the regenerated CSV against the public benchmark.",
    )
    args = parser.parse_args()

    for script in PIPELINE:
        run_script(script)

    run_script(
        "evaluation/validate_submission.py",
        SUBMISSION,
    )

    if args.evaluate:
        run_script("evaluation/run_evaluation.py")

    print("\nPIPELINE COMPLETE")
    print(f"Validated submission: {SUBMISSION}")


if __name__ == "__main__":
    main()
