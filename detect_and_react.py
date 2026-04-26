#!/usr/bin/env python3
"""
Object-triggered arm reaction using YOLOv8 + SO101.

Watches a webcam for a water bottle. When one is detected, the arm pans
to point at it and waves. Runs inference on the Jetson Orin GPU.

Usage:
    python detect_and_react.py --port /dev/ttyUSB0
    python detect_and_react.py --port /dev/ttyUSB0 --confidence 0.5 --cooldown 5
"""

import argparse
import time

import cv2
from ultralytics import YOLO

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

# COCO class name for bottles
TARGET_CLASS = "bottle"

# Arm joint limits for panning to track the bottle (degrees)
PAN_MIN = -60.0
PAN_MAX = 60.0

# Poses
REST_POSE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}

POINT_POSE_BASE = {
    "shoulder_lift.pos": -30.0,
    "elbow_flex.pos": 45.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}


def make_action(**kwargs) -> dict:
    base = dict(REST_POSE)
    base.update({f"{k}.pos" if ".pos" not in k else k: v for k, v in kwargs.items()})
    return base


def move_to(robot: SOFollower, action: dict, duration: float, hz: float = 20.0) -> None:
    steps = max(2, int(duration * hz))
    dt = 1.0 / hz
    obs = robot.get_observation()
    start = {key: obs[key] for key in action}
    for i in range(1, steps + 1):
        t = i / steps
        interp = {key: start[key] + t * (action[key] - start[key]) for key in action}
        robot.send_action(interp)
        time.sleep(dt)


def x_to_pan(x_center: float, frame_width: int) -> float:
    """Map horizontal pixel position to shoulder_pan angle."""
    normalized = (x_center / frame_width) - 0.5  # -0.5 to +0.5
    # Mirror: object on left → pan left (negative)
    return -normalized * (PAN_MAX - PAN_MIN)


def react(robot: SOFollower, pan_deg: float) -> None:
    """Point at the detected object and wave."""
    print(f"  Bottle detected! Panning to {pan_deg:.1f}°")

    point_pose = {
        "shoulder_pan.pos": pan_deg,
        **POINT_POSE_BASE,
    }
    move_to(robot, point_pose, duration=0.8)

    # Wave toward the bottle
    for _ in range(2):
        move_to(robot, {**point_pose, "wrist_flex.pos": 30.0}, duration=0.35)
        move_to(robot, {**point_pose, "wrist_flex.pos": -30.0}, duration=0.35)
    move_to(robot, point_pose, duration=0.3)

    # Return to rest
    move_to(robot, REST_POSE, duration=1.2)


def draw_overlay(frame, boxes, confidences, labels, triggered: bool) -> None:
    for box, conf, label in zip(boxes, confidences, labels):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if triggered else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status = "REACTING" if triggered else "SCANNING"
    status_color = (0, 255, 0) if triggered else (200, 200, 200)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)


def run(robot: SOFollower, args) -> None:
    model = YOLO("yolov8n.pt")
    # Warm up the model on the GPU
    model.predict(source="assets/dummy.jpg", verbose=False) if False else None

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera at index 0")

    print("Camera open. Press 'q' to quit.")
    print(f"Watching for: {TARGET_CLASS}  |  confidence threshold: {args.confidence}")

    last_reaction_time = 0.0
    reacting = False

    move_to(robot, REST_POSE, duration=1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — exiting.")
            break

        results = model.predict(source=frame, conf=args.confidence, verbose=False)
        detections = results[0]

        target_boxes = []
        target_confs = []
        target_labels = []

        for box in detections.boxes:
            cls_name = model.names[int(box.cls)]
            if cls_name == TARGET_CLASS:
                target_boxes.append(box.xyxy[0].tolist())
                target_confs.append(float(box.conf))
                target_labels.append(cls_name)

        now = time.time()
        cooldown_elapsed = (now - last_reaction_time) >= args.cooldown

        if target_boxes and cooldown_elapsed and not reacting:
            # Pick the highest-confidence detection
            best = max(range(len(target_confs)), key=lambda i: target_confs[i])
            x1, y1, x2, y2 = target_boxes[best]
            x_center = (x1 + x2) / 2.0
            pan_deg = x_to_pan(x_center, frame.shape[1])

            reacting = True
            draw_overlay(frame, target_boxes, target_confs, target_labels, triggered=True)
            cv2.imshow("detect_and_react", frame)
            cv2.waitKey(1)

            react(robot, pan_deg)
            last_reaction_time = time.time()
            reacting = False
        else:
            draw_overlay(frame, target_boxes, target_confs, target_labels, triggered=reacting)

        cv2.imshow("detect_and_react", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()
    move_to(robot, REST_POSE, duration=1.0)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 bottle detection → SO101 arm reaction")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0",
                        help="Serial port for the SO101 (default: /dev/ttyUSB0)")
    parser.add_argument("--id", type=str, default="follower",
                        help="Robot ID for calibration lookup")
    parser.add_argument("--confidence", type=float, default=0.45,
                        help="YOLO detection confidence threshold (default: 0.45)")
    parser.add_argument("--cooldown", type=float, default=6.0,
                        help="Seconds to wait between reactions (default: 6)")
    args = parser.parse_args()

    config = SOFollowerRobotConfig(port=args.port, id=args.id)
    robot = SOFollower(config)

    print(f"Connecting to SO101 on {args.port}...")
    try:
        robot.connect(calibrate=False)
        time.sleep(0.5)
        run(robot, args)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
