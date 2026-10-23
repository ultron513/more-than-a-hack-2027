"""Reproduce and validate the possession submission."""
import argparse
import hashlib
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


def sha256(path):
    """Return the SHA-256 digest of a generated file, read in binary mode."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument(
        "--check-reproducibility",
        action="store_true",
        help="Require the regenerated CSV to match the existing file byte-for-byte.",
    )
    args = parser.parse_args()

    output_path = ROOT / SUBMISSION
    if args.check_reproducibility and not output_path.is_file():
        parser.error(
            f"Cannot compare output: {SUBMISSION} does not exist yet. "
            "Run once without --check-reproducibility first."
        )

    previous_hash = sha256(output_path) if output_path.is_file() else None

    for script in PIPELINE:
        run_script(script)

    run_script("evaluation/validate_submission.py", SUBMISSION)

    current_hash = sha256(output_path)
    print(f"\nSubmission SHA-256: {current_hash}")

    if args.check_reproducibility:
        if current_hash != previous_hash:
            raise SystemExit(
                "REPRODUCIBILITY CHECK FAILED: regenerated CSV differs "
                "from the existing file byte-for-byte."
            )
        print("REPRODUCIBILITY CHECK PASSED: output matches byte-for-byte.")

    if args.evaluate:
        run_script("evaluation/run_evaluation.py")

    print("\nPIPELINE COMPLETE")
    print(f"Validated submission: {SUBMISSION}")


if __name__ == "__main__":
    main()
