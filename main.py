"""CrowdIntel: weapon and abnormal crowd behaviour detection."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Sequence

# modelnew.h5 was created with Keras 2. TensorFlow 2.16 otherwise selects
# Keras 3, which cannot deserialize this model's legacy DepthwiseConv2D config.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np


LOGGER = logging.getLogger("crowdintel")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEMO = PROJECT_ROOT / "demo" / "demo.mp4"
DEFAULT_MODEL = PROJECT_ROOT / "modelnew.h5"
DEFAULT_YOLO_CONFIG = PROJECT_ROOT / "yolov3_testing.cfg"
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "yolov3_training_2000.weights"
DEFAULT_CLASS_NAMES = PROJECT_ROOT / "classes.names"
DEFAULT_DETECTIONS_DIR = PROJECT_ROOT / "detections"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "session_log.txt"
ALERT_WINDOW_NAME = "CrowdIntel - Threat Alert"
VIOLENCE_WARMUP_FRAMES = 60
VIOLENCE_CONFIRMATION_SECONDS = 1.0
DEFAULT_DEMO_FRAME_SKIP = 9

MENU_TEXT = """
====================================

          CrowdIntel

====================================

Choose Input Source

1. Live Webcam

2. Demo Video

3. Exit

====================================
"""

MISSING_DEMO_MESSAGE = f"""Demo video not found: {DEFAULT_DEMO}

CrowdIntel does not redistribute dataset footage without explicit permission.
Download the official UMN source file:
  https://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi

Then prepare the verified 28-second demo:
  python scripts/prepare_umn_demo.py path/to/Crowd-Activity-All.avi
"""


class AlertSender:
    """Send rate-limited Twilio alerts when environment variables are present."""

    ENVIRONMENT_VARIABLES = (
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
        "TWILIO_TO_NUMBER",
    )

    def __init__(self, cooldown_seconds: float = 60.0) -> None:
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.last_sent_at = 0.0
        self.client = None
        self.from_number = ""
        self.to_number = ""

        values = {name: os.getenv(name, "").strip() for name in self.ENVIRONMENT_VARIABLES}
        configured = [name for name, value in values.items() if value]
        if not configured:
            LOGGER.info("SMS alerts disabled (Twilio environment variables are not set).")
            return
        if len(configured) != len(self.ENVIRONMENT_VARIABLES):
            missing = sorted(set(self.ENVIRONMENT_VARIABLES) - set(configured))
            LOGGER.warning("SMS alerts disabled; missing environment variables: %s", ", ".join(missing))
            return

        try:
            from twilio.rest import Client
        except ImportError:
            LOGGER.warning(
                "SMS alerts disabled; install optional dependencies with "
                "'pip install -r requirements-alerts.txt'."
            )
            return

        self.client = Client(values["TWILIO_ACCOUNT_SID"], values["TWILIO_AUTH_TOKEN"])
        self.from_number = values["TWILIO_FROM_NUMBER"]
        self.to_number = values["TWILIO_TO_NUMBER"]
        LOGGER.info("SMS alerts enabled.")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def send(self, message: str) -> bool:
        """Send an alert unless alerts are disabled or still in cooldown."""
        if not self.enabled:
            return False

        now = time.monotonic()
        if now - self.last_sent_at < self.cooldown_seconds:
            return False

        try:
            self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number,
            )
        except Exception as exc:  # A network alert failure must not stop video processing.
            LOGGER.error("Could not send SMS alert: %s", exc)
            return False

        self.last_sent_at = now
        LOGGER.info("SMS alert sent.")
        return True


class SessionLogger:
    """Write one tab-separated record for every processed frame."""

    HEADER = (
        "Timestamp\tInput Source\tWeapon Confidence\tViolence Confidence"
        "\tAlert Triggered\tSMS Sent\n"
    )

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("w", encoding="utf-8", newline="")
        self.file.write(self.HEADER)
        self.file.flush()

    def write(
        self,
        input_source: str,
        weapon_confidence: float,
        violence_confidence: float,
        alert_triggered: bool,
        sms_sent: bool,
    ) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.file.write(
            f"{timestamp}\t{input_source}\t{weapon_confidence:.4f}\t"
            f"{violence_confidence:.4f}\t"
            f"{'YES' if alert_triggered else 'NO'}\t"
            f"{'YES' if sms_sent else 'NO'}\n"
        )
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class EvidenceRecorder:
    """Save annotated evidence without producing one image per video frame."""

    def __init__(self, directory: Path, cooldown_seconds: float = 5.0) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.last_saved_at = {"weapon": float("-inf"), "violence": float("-inf")}

    def record(
        self,
        image: np.ndarray,
        weapon_found: bool,
        violence_found: bool,
    ) -> list[Path]:
        now = time.monotonic()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        saved: list[Path] = []

        for threat, detected in (("weapon", weapon_found), ("violence", violence_found)):
            if not detected or now - self.last_saved_at[threat] < self.cooldown_seconds:
                continue
            path = self.directory / f"{timestamp}_{threat}.jpg"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Could not save evidence image: {path}")
            self.last_saved_at[threat] = now
            saved.append(path)
            LOGGER.warning("Evidence saved: %s", path)

        return saved


class PopupAlert:
    """Show a separate, non-blocking OpenCV alert window."""

    def __init__(
        self,
        enabled: bool,
        duration_seconds: float = 3.0,
        snapshot_path: Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.duration_seconds = max(0.5, duration_seconds)
        self.snapshot_path = snapshot_path
        self.snapshot_saved = False
        self.visible_until = 0.0
        self.image: np.ndarray | None = None
        self.window_created = False

    def show(self, reasons: Sequence[str]) -> None:
        if not self.enabled:
            return

        reason_text = " & ".join(reason.upper() for reason in reasons)
        image = np.full((220, 640, 3), (25, 25, 180), dtype=np.uint8)
        cv2.putText(
            image,
            "CROWDINTEL ALERT",
            (105, 75),
            cv2.FONT_HERSHEY_DUPLEX,
            1.35,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            image,
            f"{reason_text} DETECTED",
            (45, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        self.image = image
        self.visible_until = time.monotonic() + self.duration_seconds
        if self.snapshot_path is not None and not self.snapshot_saved:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.snapshot_path), image):
                raise RuntimeError(f"Could not save popup snapshot: {self.snapshot_path}")
            self.snapshot_saved = True

    def update(self) -> None:
        if not self.enabled or self.image is None:
            return
        if time.monotonic() >= self.visible_until:
            self.close()
            return
        cv2.namedWindow(ALERT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ALERT_WINDOW_NAME, 640, 220)
        cv2.imshow(ALERT_WINDOW_NAME, self.image)
        self.window_created = True

    def close(self) -> None:
        if self.window_created:
            try:
                cv2.destroyWindow(ALERT_WINDOW_NAME)
            except cv2.error:
                pass
        self.window_created = False
        self.image = None


def existing_file(value: str) -> Path:
    """Argparse converter that returns an absolute path to an existing file."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def output_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def choose_input_source() -> str | None:
    """Display the required startup menu and return webcam, demo, or None."""
    while True:
        print(MENU_TEXT)
        try:
            choice = input("Enter choice (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting CrowdIntel.")
            return None
        if choice == "1":
            return "webcam"
        if choice == "2":
            return "demo"
        if choice == "3":
            print("Exiting CrowdIntel.")
            return None
        print("Invalid choice. Please enter 1, 2, or 3.\n")


def resolve_input_source(args: argparse.Namespace) -> tuple[int | str, str] | None:
    """Resolve menu/CLI input without changing the shared detection pipeline."""
    if args.input is not None:
        return str(args.input), f"Video: {args.input.name}"

    mode = args.mode
    if args.check and mode is None:
        mode = "demo"
    if mode is None:
        mode = choose_input_source()
    if mode is None:
        return None

    if mode == "webcam":
        return 0, "Live Webcam"
    if not DEFAULT_DEMO.is_file():
        raise RuntimeError(MISSING_DEMO_MESSAGE)
    return str(DEFAULT_DEMO), "Demo Video"


def load_class_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {path}")
    return names


def load_yolo(weights_path: Path, config_path: Path):
    LOGGER.info("Loading YOLO network...")
    net = cv2.dnn.readNet(str(weights_path), str(config_path))
    try:
        output_layers = list(net.getUnconnectedOutLayersNames())
    except AttributeError:
        layer_names = net.getLayerNames()
        indexes = np.asarray(net.getUnconnectedOutLayers()).reshape(-1)
        output_layers = [layer_names[int(index) - 1] for index in indexes]
    return net, output_layers


def load_violence_detection_model(model_path: Path):
    LOGGER.info("Loading violence detection model...")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    from tensorflow.keras.models import load_model

    # compile=False avoids restoring obsolete training-only optimizer state.
    return load_model(str(model_path), compile=False)


def detect_objects(
    image: np.ndarray,
    net,
    output_layers: Sequence[str],
    confidence_threshold: float = 0.7,
    nms_threshold: float = 0.4,
) -> tuple[list[int], list[list[int]], list[int], list[float]]:
    """Run the existing YOLOv3 detector on one frame."""
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1 / 255.0,
        size=(416, 416),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids: list[int] = []
    confidences: list[float] = []
    boxes: list[list[int]] = []

    for output in outputs:
        for detection in output:
            objectness = float(detection[4])
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = objectness * float(scores[class_id])
            if confidence < confidence_threshold:
                continue

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            box_width = int(detection[2] * width)
            box_height = int(detection[3] * height)
            boxes.append(
                [
                    int(center_x - box_width / 2),
                    int(center_y - box_height / 2),
                    box_width,
                    box_height,
                ]
            )
            confidences.append(confidence)
            class_ids.append(class_id)

    raw_indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        confidence_threshold,
        nms_threshold,
    )
    indexes = [int(index) for index in np.asarray(raw_indexes).reshape(-1)]
    return indexes, boxes, class_ids, confidences


def annotate_objects(
    image: np.ndarray,
    indexes: Sequence[int],
    boxes: Sequence[Sequence[int]],
    class_ids: Sequence[int],
    confidences: Sequence[float],
    classes: Sequence[str],
) -> bool:
    """Draw retained YOLO detections and report whether any were found."""
    for index in indexes:
        x, y, width, height = boxes[index]
        class_id = class_ids[index]
        label = classes[class_id] if class_id < len(classes) else f"class-{class_id}"
        color = (0, 191, 255)
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            image,
            f"{label}: {confidences[index]:.2f}",
            (max(0, x), max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    return bool(indexes)


def predict_violence(image: np.ndarray, model) -> float:
    """Run the unchanged 128x128 RGB violence-classifier preprocessing."""
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    prediction = model.predict(np.expand_dims(frame, axis=0), verbose=0)
    return float(np.asarray(prediction).reshape(-1)[0])


def resize_for_processing(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize once before both detectors while preserving aspect ratio."""
    height, width = image.shape[:2]
    if width == target_width:
        return image
    target_height = max(1, round(height * target_width / width))
    interpolation = cv2.INTER_AREA if target_width < width else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def detection_status(
    weapon_found: bool,
    violence_found: bool,
    violence_warming_up: bool = False,
    violence_pending: bool = False,
) -> str:
    if weapon_found and violence_found:
        return "THREAT: WEAPON + VIOLENCE"
    if weapon_found:
        return "THREAT: WEAPON"
    if violence_found:
        return "THREAT: VIOLENCE"
    if violence_warming_up:
        return "WARMING UP"
    if violence_pending:
        return "VERIFYING"
    return "NORMAL"


def draw_status_panel(
    image: np.ndarray,
    mode: str,
    fps: float,
    weapon_confidence: float,
    violence_confidence: float,
    status: str,
) -> None:
    """Draw the requested live telemetry without affecting model input."""
    cv2.rectangle(image, (10, 10), (465, 170), (18, 18, 18), -1)
    if status.startswith("THREAT"):
        status_color = (0, 0, 255)
    elif status.startswith(("WARMING", "VERIFYING")):
        status_color = (0, 191, 255)
    else:
        status_color = (0, 210, 0)
    lines = (
        (f"Current Mode: {mode}", (230, 230, 230)),
        (f"FPS: {fps:.1f}", (230, 230, 230)),
        (f"Weapon Confidence: {weapon_confidence:.2f}", (0, 191, 255)),
        (f"Violence Confidence: {violence_confidence:.2f}", (0, 210, 0)),
        (f"Detection Status: {status}", status_color),
    )
    for row, (text, color) in enumerate(lines):
        cv2.putText(
            image,
            text,
            (24, 38 + row * 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )


def create_video_writer(path: Path, fps: float, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CrowdIntel weapon and abnormal crowd behaviour detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("webcam", "demo"),
        help="Skip the startup menu and select an input mode",
    )
    parser.add_argument(
        "--input",
        type=existing_file,
        help="Advanced: process a custom video instead of showing the menu",
    )
    parser.add_argument("--model", type=existing_file, default=DEFAULT_MODEL, help="Keras H5 model")
    parser.add_argument(
        "--yolo-config",
        type=existing_file,
        default=DEFAULT_YOLO_CONFIG,
        help="Darknet YOLO configuration",
    )
    parser.add_argument(
        "--yolo-weights",
        type=existing_file,
        default=DEFAULT_YOLO_WEIGHTS,
        help="Darknet YOLO weights",
    )
    parser.add_argument(
        "--classes",
        type=existing_file,
        default=DEFAULT_CLASS_NAMES,
        help="One class name per line",
    )
    parser.add_argument("--confidence", type=float, default=0.7, help="YOLO confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.4, help="YOLO NMS threshold")
    parser.add_argument("--violence-threshold", type=float, default=0.5)
    parser.add_argument("--smoothing-window", type=int, default=128)
    parser.add_argument(
        "--processing-width",
        type=int,
        default=480,
        help="Demo-only pre-inference width; webcam frames are left unchanged",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        help=(
            "Frames to skip after each processed frame "
            f"(demo default: {DEFAULT_DEMO_FRAME_SKIP}; webcam default: 0)"
        ),
    )
    parser.add_argument("--alert-cooldown", type=float, default=60.0, help="Seconds between SMS alerts")
    parser.add_argument(
        "--evidence-cooldown",
        type=float,
        default=5.0,
        help="Seconds between evidence images for the same threat type",
    )
    parser.add_argument(
        "--detections-dir",
        type=output_path,
        default=DEFAULT_DETECTIONS_DIR,
        help="Evidence image directory",
    )
    parser.add_argument(
        "--log-path",
        type=output_path,
        default=DEFAULT_LOG_PATH,
        help="Session log path",
    )
    parser.add_argument("--output", type=output_path, help="Optional annotated MP4 output path")
    parser.add_argument("--snapshot", type=output_path, help="Save the first annotated frame")
    parser.add_argument(
        "--popup-snapshot",
        type=output_path,
        help="Save the first popup alert exactly as displayed",
    )
    parser.add_argument("--no-display", action="store_true", help="Do not open the main OpenCV window")
    parser.add_argument("--no-popup", action="store_true", help="Disable the separate popup alert window")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load every asset, process one demo frame headlessly, then exit",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for option in ("model", "yolo_config", "yolo_weights", "classes"):
        path = getattr(args, option)
        if not path.is_file():
            raise ValueError(f"Required file not found: {path}")
    if args.input is not None and args.mode is not None:
        raise ValueError("Use either --input or --mode, not both")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if not 0.0 <= args.nms_threshold <= 1.0:
        raise ValueError("--nms-threshold must be between 0 and 1")
    if not 0.0 <= args.violence_threshold <= 1.0:
        raise ValueError("--violence-threshold must be between 0 and 1")
    if args.smoothing_window < 1:
        raise ValueError("--smoothing-window must be at least 1")
    if args.processing_width < 128:
        raise ValueError("--processing-width must be at least 128")
    if args.frame_skip is not None and args.frame_skip < 0:
        raise ValueError("--frame-skip cannot be negative")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    if args.check:
        args.no_display = True
        args.no_popup = True
        args.max_frames = 1

    resolved_source = resolve_input_source(args)
    if resolved_source is None:
        return 0
    capture_source, mode_label = resolved_source
    is_demo_mode = mode_label == "Demo Video"
    effective_frame_skip = (
        args.frame_skip
        if args.frame_skip is not None
        else DEFAULT_DEMO_FRAME_SKIP if is_demo_mode else 0
    )

    classes = load_class_names(args.classes)
    net, output_layers = load_yolo(args.yolo_weights, args.yolo_config)
    model = load_violence_detection_model(args.model)
    alerts = AlertSender(args.alert_cooldown)
    evidence = EvidenceRecorder(args.detections_dir, args.evidence_cooldown)
    popup = PopupAlert(
        enabled=not args.no_display and not args.no_popup,
        snapshot_path=args.popup_snapshot,
    )
    session_log = SessionLogger(args.log_path)

    # Keep the required webcam constructor exact; both modes join the same loop below.
    capture = cv2.VideoCapture(0) if capture_source == 0 else cv2.VideoCapture(capture_source)
    if capture_source == 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not capture.isOpened():
        session_log.close()
        source_description = "webcam index 0" if capture_source == 0 else str(capture_source)
        raise RuntimeError(f"Could not open input source: {source_description}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0

    probabilities: deque[float] = deque(maxlen=args.smoothing_window)
    fps_samples: deque[float] = deque(maxlen=30)
    writer = None
    processed_frames = 0
    source_frames_seen = 0
    snapshot_saved = False
    previous_frame_at: float | None = None
    consecutive_high_predictions = 0
    violence_high_since: float | None = None

    LOGGER.info("Input source: %s", mode_label)
    if is_demo_mode:
        LOGGER.info(
            "Demo optimization: width=%d, frame skip=%d.",
            args.processing_width,
            effective_frame_skip,
        )
    LOGGER.info("Session log: %s", session_log.path)
    LOGGER.info("Press Esc or Q to stop.")

    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            source_frames_seen += 1
            if is_demo_mode:
                image = resize_for_processing(image, args.processing_width)

            indexes, boxes, class_ids, confidences = detect_objects(
                image,
                net,
                output_layers,
                args.confidence,
                args.nms_threshold,
            )
            weapon_found = annotate_objects(
                image,
                indexes,
                boxes,
                class_ids,
                confidences,
                classes,
            )
            weapon_confidence = max((confidences[index] for index in indexes), default=0.0)

            # Keep the rolling prediction history, but require a warm-up and
            # sustained consecutive raw predictions before an alert. This
            # prevents an isolated startup spike from becoming an incident.
            violence_confidence = predict_violence(image, model)
            probabilities.append(violence_confidence)
            queue_is_full = len(probabilities) >= probabilities.maxlen
            violence_warmup_complete = (
                source_frames_seen > VIOLENCE_WARMUP_FRAMES or queue_is_full
            )

            if is_demo_mode:
                timeline_position = (source_frames_seen - 1) / source_fps
            else:
                timeline_position = time.monotonic()

            violence_prediction_high = (
                violence_warmup_complete
                and violence_confidence >= args.violence_threshold
            )
            if violence_prediction_high:
                if consecutive_high_predictions == 0:
                    violence_high_since = timeline_position
                consecutive_high_predictions += 1
            else:
                consecutive_high_predictions = 0
                violence_high_since = None

            sustained_seconds = (
                timeline_position - violence_high_since
                if violence_high_since is not None
                else 0.0
            )
            violence_found = (
                consecutive_high_predictions >= 2
                and sustained_seconds >= VIOLENCE_CONFIRMATION_SECONDS
            )

            now = time.perf_counter()
            if previous_frame_at is not None:
                elapsed = now - previous_frame_at
                if elapsed > 0:
                    fps_samples.append(1.0 / elapsed)
            previous_frame_at = now
            current_fps = float(np.mean(fps_samples)) if fps_samples else 0.0

            status = detection_status(
                weapon_found,
                violence_found,
                violence_warming_up=not violence_warmup_complete,
                violence_pending=violence_prediction_high and not violence_found,
            )
            draw_status_panel(
                image,
                mode_label,
                current_fps,
                weapon_confidence,
                violence_confidence,
                status,
            )

            threat_found = weapon_found or violence_found
            reasons: list[str] = []
            if weapon_found:
                reasons.append("weapon")
            if violence_found:
                reasons.append("violence")

            saved_evidence: list[Path] = []
            sms_sent = False
            if threat_found:
                saved_evidence = evidence.record(image, weapon_found, violence_found)
                if saved_evidence:
                    popup.show(reasons)
                sms_sent = alerts.send(
                    f"CrowdIntel alert: {' and '.join(reasons)} detected in {mode_label}."
                )

            session_log.write(
                mode_label,
                weapon_confidence,
                violence_confidence,
                threat_found,
                sms_sent,
            )

            if writer is None and args.output:
                height, width = image.shape[:2]
                output_fps = max(0.1, source_fps / (effective_frame_skip + 1))
                writer = create_video_writer(args.output, output_fps, width, height)
            if writer is not None:
                writer.write(image)

            if args.snapshot and not snapshot_saved:
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.snapshot), image):
                    raise RuntimeError(f"Could not write snapshot: {args.snapshot}")
                snapshot_saved = True

            processed_frames += 1
            if not args.no_display:
                cv2.imshow("CrowdIntel - Surveillance Monitor", image)
                popup.update()
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.max_frames is not None and processed_frames >= args.max_frames:
                break

            # Optional sampling is disabled by default. It is useful when CPU
            # inference is slower than the source FPS and never changes the
            # models, preprocessing, thresholds, or processed-frame pipeline.
            reached_end = False
            for _ in range(effective_frame_skip):
                if not capture.grab():
                    reached_end = True
                    break
                source_frames_seen += 1
            if reached_end:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        session_log.close()
        popup.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    if processed_frames == 0:
        raise RuntimeError(f"No frames could be read from input source: {mode_label}")

    LOGGER.info("Finished after %d frame(s).", processed_frames)
    if args.check:
        LOGGER.info("Self-check passed: both models, demo video, evidence path, and logging are usable.")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
