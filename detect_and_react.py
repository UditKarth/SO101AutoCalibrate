#!/usr/bin/env python3
"""
Object-triggered arm reaction using YOLOv8 + SO101.

When idle, the arm slowly pans left and right searching for a water bottle.
When one is detected, it pans to point at it and waves, then resumes scanning.

Usage:
    python detect_and_react.py --port /dev/ttyUSB0
    python detect_and_react.py --port /dev/ttyUSB0 --confidence 0.5 --cooldown 5
"""

import argparse
import time
from enum import Enum, auto

import cv2
from ultralytics import YOLO

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

TARGET_CLASS = "bottle"

PAN_MIN = -60.0
PAN_MAX = 60.0
SCAN_STEP_DEG = 1.5   # degrees per frame while scanning
SCAN_HZ = 20.0        # target loop rate during scanning

REST_POSE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}

SCAN_POSE_BASE = {
    "shoulder_lift.pos": -20.0,
    "elbow_flex.pos": 30.0,
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


class State(Enum):
    SCANNING = auto()
    REACTING = auto()
    COOLDOWN = auto()


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
    normalized = (x_center / frame_width) - 0.5
    return -normalized * (PAN_MAX - PAN_MIN)


def react(robot: SOFollower, pan_deg: float) -> None:
    print(f"  Bottle detected! Reacting at pan={pan_deg:.1f}°")
    point_pose = {"shoulder_pan.pos": pan_deg, **POINT_POSE_BASE}
    move_to(robot, point_pose, duration=0.8)
    for _ in range(2):
        move_to(robot, {**point_pose, "wrist_flex.pos": 30.0}, duration=0.35)
        move_to(robot, {**point_pose, "wrist_flex.pos": -30.0}, duration=0.35)
    move_to(robot, point_pose, duration=0.3)
    move_to(robot, REST_POSE, duration=1.0)


def draw_overlay(frame, boxes, confidences, labels, state: State) -> None:
    color_map = {
        State.SCANNING: (200, 200, 200),
        State.REACTING: (0, 255, 0),
        State.COOLDOWN: (0, 165, 255),
    }
    status_color = color_map[state]

    for box, conf, label in zip(boxes, confidences, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.putText(frame, state.name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)


def run(robot: SOFollower, args) -> None:
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera at index 0")

    print("Camera open. Press 'q' to quit.")
    print(f"Watching for: {TARGET_CLASS}  |  confidence: {args.confidence}  |  cooldown: {args.cooldown}s")

    state = State.SCANNING
    scan_pan = 0.0
    scan_direction = 1.0       # +1 sweeping right, -1 sweeping left
    cooldown_until = 0.0

    move_to(robot, {**SCAN_POSE_BASE, "shoulder_pan.pos": scan_pan}, duration=1.0)

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — exiting.")
            break

        results = model.predict(source=frame, conf=args.confidence, verbose=False)
        detections = results[0]

        target_boxes, target_confs, target_labels = [], [], []
        for box in detections.boxes:
            if model.names[int(box.cls)] == TARGET_CLASS:
                target_boxes.append(box.xyxy[0].tolist())
                target_confs.append(float(box.conf))
                target_labels.append(TARGET_CLASS)

        now = time.time()

        if state == State.COOLDOWN:
            if now >= cooldown_until:
                print("Cooldown done — resuming scan.")
                state = State.SCANNING
                move_to(robot, {**SCAN_POSE_BASE, "shoulder_pan.pos": scan_pan}, duration=0.5)

        elif state == State.SCANNING:
            if target_boxes:
                best = max(range(len(target_confs)), key=lambda i: target_confs[i])
                x1, y1, x2, y2 = target_boxes[best]
                pan_deg = x_to_pan((x1 + x2) / 2.0, frame.shape[1])

                state = State.REACTING
                draw_overlay(frame, target_boxes, target_confs, target_labels, state)
                cv2.imshow("detect_and_react", frame)
                cv2.waitKey(1)

                react(robot, pan_deg)

                cooldown_until = time.time() + args.cooldown
                state = State.COOLDOWN
                print(f"Cooling down for {args.cooldown}s...")
            else:
                # Advance the scan sweep
                scan_pan += scan_direction * SCAN_STEP_DEG
                if scan_pan >= PAN_MAX:
                    scan_pan = PAN_MAX
                    scan_direction = -1.0
                elif scan_pan <= PAN_MIN:
                    scan_pan = PAN_MIN
                    scan_direction = 1.0

                robot.send_action({**SCAN_POSE_BASE, "shoulder_pan.pos": scan_pan})

        draw_overlay(frame, target_boxes, target_confs, target_labels, state)
        cv2.imshow("detect_and_react", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit requested.")
            break

        # Pace the scan loop so the arm moves smoothly
        if state == State.SCANNING:
            elapsed = time.time() - frame_start
            sleep_for = max(0.0, (1.0 / SCAN_HZ) - elapsed)
            time.sleep(sleep_for)

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
                        help="Seconds between reactions (default: 6)")
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
