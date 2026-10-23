import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FEATURES_FILE = ROOT / "data" / "processed" / "merged_features.csv"
OUTPUT_FILE = ROOT / "evaluation" / "results" / "candidate_temporal_v3.csv"

EXPECTED_FRAMES = 199_319

# Published tactical-video frame boundaries.
FIRST_HALF = (11_185, 90_652)       # Stop index is exclusive.
SECOND_HALF = (110_808, 190_198)
SECOND_HALF_KICKOFF = 113_742

LABELS = np.array(["H", "A", "D"])


def load_features():
    if not FEATURES_FILE.is_file():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_FILE}\n"
            "Run the feature extraction and merge scripts first."
        )

    df = pd.read_csv(FEATURES_FILE)

    if len(df) != EXPECTED_FRAMES:
        raise ValueError(
            f"Expected {EXPECTED_FRAMES:,} rows, got {len(df):,}."
        )

    if not np.array_equal(
        df["frame"].to_numpy(), np.arange(EXPECTED_FRAMES)
    ):
        raise ValueError("Feature frames are not correctly aligned.")

    required = [
        "ball_valid",
        "home_nearest_distance_m",
        "away_nearest_distance_m",
        "nearest_distance_m",
        "ball_speed_mps",
        "tagged_home_phase_count",
        "tagged_away_phase_count",
        "set_piece_count",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df


def numeric(df, column):
    return pd.to_numeric(df[column], errors="coerce").to_numpy(
        dtype=np.float64
    )


def build_emissions(df):
    """
    Estimate H/A/D observation probabilities from tracking and tags.

    This is a heuristic candidate, not a trained or calibrated model.
    """
    n = len(df)

    home_distance = numeric(df, "home_nearest_distance_m")
    away_distance = numeric(df, "away_nearest_distance_m")
    nearest_distance = numeric(df, "nearest_distance_m")
    ball_speed = numeric(df, "ball_speed_mps")

    ball_valid = numeric(df, "ball_valid") == 1

    home_tag = numeric(df, "tagged_home_phase_count") > 0
    away_tag = numeric(df, "tagged_away_phase_count") > 0
    set_piece = numeric(df, "set_piece_count") > 0

    # Positive margin means a home player is closer to the ball.
    margin = np.zeros(n, dtype=np.float64)

    distance_available = (
        np.isfinite(home_distance)
        & np.isfinite(away_distance)
    )

    margin[distance_available] = (
        away_distance[distance_available]
        - home_distance[distance_available]
    )

    # Estimate which team has control when the ball is in play.
    reliability = np.ones(n, dtype=np.float64)
    nearest_known = np.isfinite(nearest_distance)

    reliability[nearest_known] = 1.0 / (
        1.0
        + np.maximum(
            nearest_distance[nearest_known] - 3.0, 0.0
        ) / 8.0
    )

    team_logit = margin / 2.5 * reliability

    # Tagged teams provide supporting evidence, not ground truth.
    home_only = home_tag & ~away_tag
    away_only = away_tag & ~home_tag

    team_logit[home_only] += 0.35
    team_logit[away_only] -= 0.35

    team_logit = np.clip(team_logit, -6.0, 6.0)
    p_home_if_live = 1.0 / (1.0 + np.exp(-team_logit))

    # Estimate dead-ball likelihood separately from team control.
    # Negative starting logit means D is not the default state.
    dead_logit = np.full(n, -2.3, dtype=np.float64)

    # Missing ball observations provide only weak evidence of D.
    dead_logit[~ball_valid] += 0.45

    # The farther team's nearest-player distance can be informative
    # when one team is much farther from the ball than the other.
    far_team_distance = np.maximum(
        home_distance, away_distance
    )
    far_known = np.isfinite(far_team_distance)

    dead_logit[far_known & (far_team_distance > 8.0)] += 0.35
    dead_logit[far_known & (far_team_distance > 12.0)] += 0.90
    dead_logit[far_known & (far_team_distance > 16.0)] += 0.50

    # A large team-distance imbalance can indicate a restart or
    # stoppage, but should not independently force a D prediction.
    large_margin = distance_available & (np.abs(margin) > 8.0)
    very_large_margin = distance_available & (
        np.abs(margin) > 12.0
    )

    dead_logit[large_margin] += 1.25
    dead_logit[very_large_margin] += 0.45

    # Slow ball movement is additional, uncertain evidence.
    speed_known = np.isfinite(ball_speed)
    slow_ball = speed_known & (ball_speed < 0.35)
    very_slow_ball = speed_known & (ball_speed < 0.15)

    dead_logit[slow_ball] += 0.50
    dead_logit[very_slow_ball] += 0.35

    slow_and_far = (
        slow_ball
        & far_known
        & (far_team_distance > 10.0)
    )
    dead_logit[slow_and_far] += 0.50

    # Smart Tagging marks phases, not exact dead-ball intervals.
    # Therefore, use a moderate signal across tagged set pieces.
    dead_logit[set_piece] += 0.55

    # Give additional weight to the onset of a set-piece annotation.
    # The decaying pulse limits the influence of the boundary.
    set_piece_start = set_piece & ~np.r_[False, set_piece[:-1]]

    for start in np.flatnonzero(set_piece_start):
        end = min(start + 25, n)

        dead_logit[start:end] += np.linspace(
            2.2, 0.5, end - start
        )

    p_dead = 1.0 / (
        1.0 + np.exp(-np.clip(dead_logit, -8.0, 8.0))
    )
    p_dead = np.clip(p_dead, 0.02, 0.90)

    # Convert the three estimates into normalized emissions.
    emission = np.column_stack([
        (1.0 - p_dead) * p_home_if_live,
        (1.0 - p_dead) * (1.0 - p_home_if_live),
        p_dead,
    ])

    return emission, p_dead


def viterbi_decode(emission):
    """Find the highest-probability state sequence under an HMM."""
    if len(emission) == 0:
        return np.array([], dtype=np.int8)

    # Rows are previous state, columns are next state.
    # Same-team continuation is preferred, but turnovers and stoppages
    # remain possible.
    transition = np.array([
        [0.9975, 0.0005, 0.0020],  # Previous H
        [0.0005, 0.9975, 0.0020],  # Previous A
        [0.0150, 0.0150, 0.9700],  # Previous D
    ], dtype=np.float64)

    log_transition = np.log(transition)
    log_emission = np.log(np.clip(emission, 1e-12, 1.0))

    n = len(emission)
    scores = np.empty((n, 3), dtype=np.float64)
    backpointers = np.zeros((n, 3), dtype=np.int8)

    # Uniform initial state prior.
    scores[0] = -np.log(3.0) + log_emission[0]

    for t in range(1, n):
        candidates = scores[t - 1][:, None] + log_transition

        backpointers[t] = np.argmax(candidates, axis=0)
        scores[t] = (
            np.max(candidates, axis=0) + log_emission[t]
        )

    states = np.empty(n, dtype=np.int8)
    states[-1] = np.argmax(scores[-1])

    for t in range(n - 1, 0, -1):
        states[t - 1] = backpointers[t, states[t]]

    return states


def main():
    print("Loading merged tracking and tagging features...")
    df = load_features()

    print("Building H/A/D observation probabilities...")
    emission, p_dead = build_emissions(df)

    # Outside the graded halves, use X. Decode the two halves
    # separately so that halftime cannot merge possessions.
    labels = np.full(EXPECTED_FRAMES, "X", dtype="<U1")

    start, stop = FIRST_HALF
    states = viterbi_decode(emission[start:stop])
    labels[start:stop] = LABELS[states]

    # The published second-half block starts before kickoff.
    # These frames are graded and documented as dead ball.
    kickoff = SECOND_HALF_KICKOFF
    s2, e2 = SECOND_HALF

    labels[s2:kickoff] = "D"

    states = viterbi_decode(emission[kickoff:e2])
    labels[kickoff:e2] = LABELS[states]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "label"])
        writer.writerows(enumerate(labels))

    print("\nTEMPORAL MODEL COMPLETE")
    print(f"Frames: {len(labels):,}")
    print(f"Output: {OUTPUT_FILE}")

    for label in ("H", "A", "D", "X"):
        count = int(np.sum(labels == label))
        print(f"  {label}: {count:,} ({100 * count / len(labels):.1f}%)")

        mean_dead = np.mean(
            np.concatenate([
                p_dead[FIRST_HALF[0]:FIRST_HALF[1]],
                p_dead[kickoff:e2],
            ])
        )

        print(
            "\nMean heuristic dead-ball evidence within both halves: "
            f"{mean_dead:.3f}"
        )


if __name__ == "__main__":
    main()