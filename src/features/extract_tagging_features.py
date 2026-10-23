import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from starter.load_smart_tagging import load

EXPECTED_FRAMES = 199_319

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "tactical_tagging_features.csv"

COLUMNS = [
    "frame",
    "tagged_phase_count",
    "tagged_home_phase_count",
    "tagged_away_phase_count",
    "tagged_unknown_team_phase_count",
    "tagged_team_conflict",
    "set_piece_count",
    "set_piece_home",
    "set_piece_away",
    "set_piece_unknown",
    "tag_throw_in",
    "tag_corner_kick",
    "tag_goal_kick",
    "tag_free_kick",
    "tag_kickoff_after_goal",
    "tag_kickoff_start",
]


def map_team(team_value):
    """Map Smart Tagging team names to H, A or X."""
    value = str(team_value or "").strip().lower()

    if "barcelona" in value:
        return "H"

    # The repository documents variants of the away-team spelling.
    if "athletic" in value or "atletic" in value:
        return "A"

    return "X"


def normalize_type(value):
    return str(value or "").strip().lower().replace("–", "-")


def increment(array, start, end):
    """Count an annotation over a half-open frame interval."""
    if end > start:
        array[start:end] += 1


def main():
    print("Loading Smart Tagging annotations...")

    events = load()

    # Count annotations independently for every tactical video frame.
    arrays = {
        column: np.zeros(EXPECTED_FRAMES, dtype=np.int32)
        for column in COLUMNS
        if column != "frame"
        and column != "tagged_team_conflict"
    }

    home_active = np.zeros(EXPECTED_FRAMES, dtype=bool)
    away_active = np.zeros(EXPECTED_FRAMES, dtype=bool)

    valid_events = 0
    skipped_events = 0
    set_piece_events = 0

    type_columns = {
        "throw-in": "tag_throw_in",
        "throw in": "tag_throw_in",
        "corner kick": "tag_corner_kick",
        "goal kick": "tag_goal_kick",
        "free kick": "tag_free_kick",
        "kick off (after goal)": "tag_kickoff_after_goal",
        "kick off (start)": "tag_kickoff_start",
    }

    for event in events:
        start = max(0, int(event["start_frame"]))
        end = min(EXPECTED_FRAMES, int(event["end_frame"]))

        if end <= start:
            skipped_events += 1
            continue

        valid_events += 1

        code = str(event.get("code") or "").strip().lower()
        tags = event.get("tags", {})

        team = map_team(tags.get("Team"))

        increment(arrays["tagged_phase_count"], start, end)

        if team == "H":
            increment(
                arrays["tagged_home_phase_count"], start, end
            )
            home_active[start:end] = True

        elif team == "A":
            increment(
                arrays["tagged_away_phase_count"], start, end
            )
            away_active[start:end] = True

        else:
            increment(
                arrays["tagged_unknown_team_phase_count"],
                start,
                end,
            )

        if code == "set pieces":
            set_piece_events += 1

            increment(arrays["set_piece_count"], start, end)

            if team == "H":
                increment(arrays["set_piece_home"], start, end)
            elif team == "A":
                increment(arrays["set_piece_away"], start, end)
            else:
                increment(
                    arrays["set_piece_unknown"], start, end
                )

            set_piece_type = normalize_type(tags.get("Type"))
            type_column = type_columns.get(set_piece_type)

            if type_column:
                increment(arrays[type_column], start, end)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(
            destination, fieldnames=COLUMNS
        )
        writer.writeheader()

        for frame in range(EXPECTED_FRAMES):
            row = {"frame": frame}

            for column, values in arrays.items():
                row[column] = int(values[frame] > 0)

            # Preserve counts for overlapping phases rather than
            # converting these two fields to binary flags.
            row["tagged_phase_count"] = int(
                arrays["tagged_phase_count"][frame]
            )
            row["tagged_home_phase_count"] = int(
                arrays["tagged_home_phase_count"][frame]
            )
            row["tagged_away_phase_count"] = int(
                arrays["tagged_away_phase_count"][frame]
            )
            row["tagged_unknown_team_phase_count"] = int(
                arrays["tagged_unknown_team_phase_count"][frame]
            )
            row["set_piece_count"] = int(
                arrays["set_piece_count"][frame]
            )

            row["tagged_team_conflict"] = int(
                home_active[frame] and away_active[frame]
            )

            writer.writerow(row)

    print("\nSMART TAGGING EXTRACTION COMPLETE")
    print(f"Annotations loaded: {len(events)}")
    print(f"Valid intervals: {valid_events}")
    print(f"Skipped intervals: {skipped_events}")
    print(f"Set-piece intervals: {set_piece_events}")

    print("\nFrames covered by each signal:")
    for column in [
        "tagged_phase_count",
        "tagged_home_phase_count",
        "tagged_away_phase_count",
        "set_piece_count",
        "tag_throw_in",
        "tag_corner_kick",
        "tag_goal_kick",
        "tag_free_kick",
        "tag_kickoff_after_goal",
        "tag_kickoff_start",
    ]:
        print(
            f"  {column}: "
            f"{int(np.count_nonzero(arrays[column])):,}"
        )

    print(f"\nOutput frames: {EXPECTED_FRAMES:,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()