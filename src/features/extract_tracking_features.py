import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "tactical"
OUTPUT_DIR = ROOT / "data" / "processed"

EXPECTED_FRAMES = 199_319
PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

TRACKING_FILE = DATA_DIR / "BAR-ATH_tactical_atd.txt"
METADATA_FILE = DATA_DIR / "BAR-ATH_tactical_atd.xml"

OUTPUT_FILE = OUTPUT_DIR / "tactical_features.csv"

COLUMNS = [
    "frame",
    "ball_x_norm",
    "ball_y_norm",
    "ball_speed_mps",
    "ball_valid",
    "tracked_home_players",
    "tracked_away_players",
    "home_nearest_distance_m",
    "away_nearest_distance_m",
    "nearest_distance_m",
    "distance_margin_away_minus_home_m",
    "nearest_team",
]


def load_team_mapping():
    """Map each tracking slot to H or A using the metadata XML."""
    root = ET.parse(METADATA_FILE).getroot()

    teams = [team.get("id") for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError(f"Expected exactly 2 teams, found {len(teams)}.")

    home_id, away_id = teams

    players = list(root.iter("Player"))
    if not players:
        raise ValueError("No Player entries found in tracking metadata.")

    mapping = []

    for player in players:
        team_id = player.get("teamId")

        if team_id == home_id:
            mapping.append("H")
        elif team_id == away_id:
            mapping.append("A")
        else:
            raise ValueError(
                f"Unknown team ID in tracking metadata: {team_id}"
            )

    print(f"Tracking slots mapped: {len(mapping)}")
    print("Home = H, Away = A")

    return mapping


def parse_track(group):
    """Return normalized x, y, speed, or None for a missing track."""
    group = group.strip()

    if not group or group.startswith("N"):
        return None

    values = group.split(",")
    if len(values) < 3:
        return None

    try:
        x = float(values[0])
        y = float(values[1])
        speed = float(values[2])
    except ValueError:
        return None

    if not math.isfinite(x) or not math.isfinite(y):
        return None

    if not math.isfinite(speed):
        speed = None

    return x, y, speed


def distance_metres(player, ball):
    """Distance between two normalized pitch coordinates in metres."""
    dx = (player[0] - ball[0]) * PITCH_LENGTH_M
    dy = (player[1] - ball[1]) * PITCH_WIDTH_M

    return math.hypot(dx, dy)


def format_number(value):
    """Write missing measurements as empty CSV cells."""
    if value is None:
        return ""

    return f"{value:.6f}"


def extract_features():
    if not TRACKING_FILE.is_file():
        raise FileNotFoundError(TRACKING_FILE)

    if not METADATA_FILE.is_file():
        raise FileNotFoundError(METADATA_FILE)

    team_mapping = load_team_mapping()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    ball_visible_count = 0
    player_total = 0
    nearest_team_counts = {"H": 0, "A": 0, "X": 0}

    with (
        TRACKING_FILE.open("r", encoding="utf-8") as source,
        OUTPUT_FILE.open("w", newline="", encoding="utf-8") as destination,
    ):
        writer = csv.DictWriter(destination, fieldnames=COLUMNS)
        writer.writeheader()

        for line_number, line in enumerate(source, start=1):
            parts = line.rstrip("\r\n").split(":")

            if len(parts) != 3:
                raise ValueError(
                    f"Unexpected tracking format at line {line_number}."
                )

            frame_id = int(parts[0])

            if frame_id != line_number:
                raise ValueError(
                    f"Frame alignment error at line {line_number}: "
                    f"found frame ID {frame_id}."
                )

            player_groups = parts[1].split(";")

            if len(player_groups) != len(team_mapping):
                raise ValueError(
                    f"Expected {len(team_mapping)} player slots at "
                    f"frame {frame_id}, found {len(player_groups)}."
                )

            ball = parse_track(parts[2])
            ball_valid = ball is not None

            if ball_valid:
                ball_visible_count += 1

            home_count = 0
            away_count = 0
            home_min = None
            away_min = None

            for slot, group in enumerate(player_groups):
                player = parse_track(group)

                if player is None:
                    continue

                team = team_mapping[slot]

                if team == "H":
                    home_count += 1
                else:
                    away_count += 1

                if ball is None:
                    continue

                distance = distance_metres(player, ball)

                if team == "H":
                    if home_min is None or distance < home_min:
                        home_min = distance
                else:
                    if away_min is None or distance < away_min:
                        away_min = distance

            player_total += home_count + away_count

            nearest_distance = None
            distance_margin = None

            if home_min is not None and away_min is not None:
                nearest_distance = min(home_min, away_min)

                # Positive: home is closer.
                # Negative: away is closer.
                distance_margin = away_min - home_min

            if ball is None or (
                home_min is None and away_min is None
            ):
                nearest_team = "X"
            elif home_min is None:
                nearest_team = "A"
            elif away_min is None:
                nearest_team = "H"
            else:
                nearest_team = "H" if home_min <= away_min else "A"

            nearest_team_counts[nearest_team] += 1

            writer.writerow({
                "frame": frame_id - 1,
                "ball_x_norm": format_number(ball[0] if ball else None),
                "ball_y_norm": format_number(ball[1] if ball else None),
                "ball_speed_mps": format_number(ball[2] if ball else None),
                "ball_valid": int(ball_valid),
                "tracked_home_players": home_count,
                "tracked_away_players": away_count,
                "home_nearest_distance_m": format_number(home_min),
                "away_nearest_distance_m": format_number(away_min),
                "nearest_distance_m": format_number(nearest_distance),
                "distance_margin_away_minus_home_m": format_number(
                    distance_margin
                ),
                "nearest_team": nearest_team,
            })

            frame_count += 1

            if frame_count % 50_000 == 0:
                print(f"Processed {frame_count:,} frames...")

    if frame_count != EXPECTED_FRAMES:
        raise ValueError(
            f"Expected {EXPECTED_FRAMES:,} frames, got {frame_count:,}."
        )

    print("\nFEATURE EXTRACTION COMPLETE")
    print(f"Frames processed: {frame_count:,}")
    print(
        f"Ball visible: {ball_visible_count:,} "
        f"({100 * ball_visible_count / frame_count:.1f}%)"
    )
    print(f"Average tracked players per frame: "
          f"{player_total / frame_count:.2f}")
    print(f"Nearest-team evidence: {nearest_team_counts}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_features()