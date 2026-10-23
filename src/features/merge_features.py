from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

TRACKING_FILE = PROCESSED_DIR / "tactical_features.csv"
TAGGING_FILE = PROCESSED_DIR / "tactical_tagging_features.csv"
OUTPUT_FILE = PROCESSED_DIR / "merged_features.csv"

EXPECTED_FRAMES = 199_319


def main():
    print("Loading tracking features...")
    tracking = pd.read_csv(TRACKING_FILE)

    print("Loading Smart Tagging features...")
    tagging = pd.read_csv(TAGGING_FILE)

    # Check file sizes before merging.
    if len(tracking) != EXPECTED_FRAMES:
        raise ValueError(
            f"Tracking features have {len(tracking):,} rows; "
            f"expected {EXPECTED_FRAMES:,}."
        )

    if len(tagging) != EXPECTED_FRAMES:
        raise ValueError(
            f"Tagging features have {len(tagging):,} rows; "
            f"expected {EXPECTED_FRAMES:,}."
        )

    # Require exact frame-by-frame alignment.
    tracking_frames = tracking["frame"].to_numpy()
    tagging_frames = tagging["frame"].to_numpy()

    expected_indices = np.arange(EXPECTED_FRAMES)

    if not np.array_equal(tracking_frames, expected_indices):
        raise ValueError(
            "Tracking features are not indexed from 0 to 199318."
        )

    if not np.array_equal(tagging_frames, expected_indices):
        raise ValueError(
            "Tagging features are not indexed from 0 to 199318."
        )

    # Keep tracking data as the left table and attach tagging
    # features to matching tactical video frames.
    merged = tracking.merge(
        tagging,
        on="frame",
        how="inner",
        validate="one_to_one",
        suffixes=("_tracking", "_tagging"),
    )

    if len(merged) != EXPECTED_FRAMES:
        raise ValueError(
            f"Merged table has {len(merged):,} rows; "
            f"expected {EXPECTED_FRAMES:,}."
        )

    if merged.isna().all(axis=1).any():
        raise ValueError(
            "At least one merged row contains only missing values."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print("\nFEATURE MERGE COMPLETE")
    print(f"Tracking rows: {len(tracking):,}")
    print(f"Tagging rows:  {len(tagging):,}")
    print(f"Merged rows:   {len(merged):,}")
    print(f"Total columns: {len(merged.columns)}")

    print("\nTagging coverage in merged table:")
    for column in [
        "tagged_phase_count",
        "tagged_home_phase_count",
        "tagged_away_phase_count",
        "set_piece_count",
        "tag_throw_in",
        "tag_corner_kick",
        "tag_goal_kick",
        "tag_free_kick",
    ]:
        frames = int((merged[column] > 0).sum())
        print(f"  {column}: {frames:,} frames")

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()