import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "evaluation" / "results" /
    "candidate_temporal_v3.csv"
)

OUTPUT_FILE = (
    ROOT / "evaluation" / "results" /
    "candidate_temporal_v4.csv"
)

EXPECTED_FRAMES = 199_319
MIN_RUN_FRAMES = 50  # 2 seconds at 25 fps

HALVES = [
    (11_185, 90_652),
    (110_808, 190_198),
]


def merge_short_runs(labels, min_frames):
    """Merge short interior runs only when both neighbours agree."""
    labels = labels.copy()
    total_merged = 0

    while True:
        boundaries = np.r_[
            0,
            np.flatnonzero(labels[1:] != labels[:-1]) + 1,
            len(labels),
        ]

        changed = False

        for i in range(1, len(boundaries) - 2):
            start = boundaries[i]
            end = boundaries[i + 1]

            if end - start >= min_frames:
                continue

            previous_label = labels[start - 1]
            current_label = labels[start]
            next_label = labels[end]

            if (
                previous_label == next_label
                and current_label != previous_label
            ):
                labels[start:end] = previous_label
                total_merged += 1
                changed = True
                break  # Recalculate runs after each correction.

        if not changed:
            break

    return labels, total_merged


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    if len(df) != EXPECTED_FRAMES:
        raise ValueError(
            f"Expected {EXPECTED_FRAMES:,} rows, "
            f"found {len(df):,}."
        )

    if not np.array_equal(
        df["frame"].to_numpy(),
        np.arange(EXPECTED_FRAMES),
    ):
        raise ValueError("Frame indices are incorrect.")

    labels = df["label"].astype(str).to_numpy(copy=True)

    if not np.isin(labels, ["H", "A", "D", "X"]).all():
        raise ValueError("Unexpected prediction label.")

    total_merged = 0

    for start, stop in HALVES:
        corrected, merged = merge_short_runs(
            labels[start:stop],
            MIN_RUN_FRAMES,
        )

        labels[start:stop] = corrected
        total_merged += merged

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "label"])
        writer.writerows(enumerate(labels))

    print("POST-PROCESSING COMPLETE")
    print(f"Input:  {INPUT_FILE.name}")
    print(f"Output: {OUTPUT_FILE.name}")
    print(f"Minimum segment duration: {MIN_RUN_FRAMES / 25:.1f}s")
    print(f"Short runs merged: {total_merged:,}")

    print("\nFinal label counts:")
    for label in ["H", "A", "D", "X"]:
        count = int(np.sum(labels == label))
        print(f"  {label}: {count:,}")

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()