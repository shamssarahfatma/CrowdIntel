"""Prepare CrowdIntel's demo from the official UMN crowd-activity AVI.

The source video is not downloaded or redistributed by this repository because
the official project page does not state explicit redistribution terms.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "demo" / "demo.mp4"
EXPECTED_SOURCE_NAME = "Crowd-Activity-All.avi"
START_SECONDS = 176.0
DURATION_SECONDS = 28.0
OUTPUT_SIZE = (640, 480)
OUTPUT_FPS = 30.0


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract an authentic 28-second UMN panic sequence for CrowdIntel."
    )
    parser.add_argument(
        "source",
        type=existing_file,
        help=f"Official UMN source AVI ({EXPECTED_SOURCE_NAME})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def prepare(source: Path, output: Path) -> int:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        capture.release()
        raise RuntimeError("The source video does not report a valid frame rate")

    start_frame = round(START_SECONDS * source_fps)
    source_frames_needed = round(DURATION_SECONDS * source_fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        OUTPUT_SIZE,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output}")

    written = 0
    try:
        for _ in range(source_frames_needed):
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()

    expected = round(DURATION_SECONDS * OUTPUT_FPS)
    if written < expected:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Source ended early: wrote {written} of {expected} expected frames"
        )

    print(f"Created authentic UMN demo: {output}")
    print(f"Frames: {written} | Resolution: 640x480 | FPS: 30 | Duration: 28 seconds")
    print("Normal movement comes first; panic begins within the first 10 seconds.")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return prepare(args.source, args.output)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
