import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "tactical"
EXPECTED_FRAMES = 199_319

REQUIRED_FILES = [
    "BAR-ATH_tactical.mp4",
    "BAR-ATH_tactical_atd.xml",
    "BAR-ATH_tactical_atd.txt",
    "BAR-ATH_tactical_homography.csv",
    "BAR-ATH_tactical_smart_tagging.xml",
]


def check_files():
    errors = []

    print("Checking required tactical assets...\n")

    for filename in REQUIRED_FILES:
        path = DATA_DIR / filename

        if not path.is_file():
            errors.append(f"Missing file: {path}")
            print(f"FAIL  {filename}")
        elif path.stat().st_size == 0:
            errors.append(f"Empty file: {path}")
            print(f"FAIL  {filename}")
        else:
            size_mb = path.stat().st_size / (1024 ** 2)
            print(f"OK    {filename} ({size_mb:.2f} MB)")

    return errors


def check_tracking():
    path = DATA_DIR / "BAR-ATH_tactical_atd.txt"
    if not path.is_file():
        return ["Cannot check tracking: file is missing."]

    errors = []
    count = 0
    first_frame = None
    last_frame = None
    bad_frames = []

    print("\nChecking tracking frame alignment...")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            count = line_number

            try:
                frame_id = int(line.split(":", 1)[0])
            except (ValueError, IndexError):
                errors.append(
                    f"Invalid tracking frame ID at line {line_number}."
                )
                break

            if first_frame is None:
                first_frame = frame_id

            last_frame = frame_id

            if frame_id != line_number and len(bad_frames) < 5:
                bad_frames.append((line_number, frame_id))

    print(f"Tracking rows: {count:,}")
    print(f"First frame ID: {first_frame}")
    print(f"Last frame ID:  {last_frame}")

    if count != EXPECTED_FRAMES:
        errors.append(
            f"Expected {EXPECTED_FRAMES:,} tracking rows, got {count:,}."
        )

    if bad_frames:
        errors.append(
            f"Tracking frame IDs do not match row numbers: {bad_frames}"
        )

    if not errors:
        print("OK    Tracking IDs are contiguous and correctly aligned.")

    return errors


def check_homography():
    path = DATA_DIR / "BAR-ATH_tactical_homography.csv"

    if not path.is_file():
        return ["Cannot check homography: file is missing."]

    errors = []
    count = 0
    bad_frames = []

    print("\nChecking homography frame alignment...")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "frame" not in reader.fieldnames:
            return ["Homography CSV is missing the 'frame' column."]

        for expected_frame, row in enumerate(reader):
            count += 1

            try:
                actual_frame = int(row["frame"])
            except (ValueError, TypeError):
                errors.append(
                    f"Invalid homography frame at data row {count}."
                )
                break

            if actual_frame != expected_frame and len(bad_frames) < 5:
                bad_frames.append((expected_frame, actual_frame))

    print(f"Homography rows: {count:,}")

    if count != EXPECTED_FRAMES:
        errors.append(
            f"Expected {EXPECTED_FRAMES:,} homography rows, got {count:,}."
        )

    if bad_frames:
        errors.append(
            f"Homography frame indices are incorrect: {bad_frames}"
        )

    if not errors:
        print("OK    Homography indices run from 0 to 199318.")

    return errors


def check_xml(filename):
    path = DATA_DIR / filename

    if not path.is_file():
        return [f"Cannot parse XML: {filename} is missing."]

    try:
        ET.parse(path)
        print(f"OK    XML parsed: {filename}")
        return []
    except ET.ParseError as exc:
        return [f"Invalid XML in {filename}: {exc}"]


def main():
    errors = check_files()

    for filename in (
        "BAR-ATH_tactical_atd.xml",
        "BAR-ATH_tactical_smart_tagging.xml",
    ):
        errors.extend(check_xml(filename))

    errors.extend(check_tracking())
    errors.extend(check_homography())

    print("\n" + "=" * 55)

    if errors:
        print("ASSET CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    print("ASSET CHECK PASSED")
    print("All required files exist and the frame indices align.")


if __name__ == "__main__":
    main()