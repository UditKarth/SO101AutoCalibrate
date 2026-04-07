#!/usr/bin/env python3
"""
Auto-Calibration Script for SO-101 Robot Arm (LeRobot / Feetech STS3215)

This script semi-automatically calibrates a SO-101 follower arm by:
  1. Asking the user to manually place the arm in its rest/home pose.
  2. Reading the raw encoder positions at home to compute homing offsets.
  3. For each joint, slowly sweep toward 0 then 4095
     while monitoring for stalls, to discover the mechanical range limits.
  4. Presenting discovered values for user confirmation.
  5. Saving the calibration JSON.

Usage:
    python auto_calibrate_so101.py --port /dev/ttyACM0

Requirements:
    pip install lerobot[feetech]   (or: pip install lerobot scservo-sdk)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# LeRobot imports
# ---------------------------------------------------------------------------
try:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode
except ImportError:
    print("ERROR: Could not import lerobot motor modules.")
    print("Make sure you have lerobot installed with feetech support:")
    print("  pip install lerobot[feetech]")
    sys.exit(1)


# ---------------------------------------------------------------------------
# SO-101 joint definitions (follower arm)
# ---------------------------------------------------------------------------
# Calibration order: safest joints first (outermost → innermost).
JOINT_NAMES_CALIBRATION_ORDER = [
    "gripper",
    "wrist_roll",
    "wrist_flex",
    "elbow_flex",
    "shoulder_lift",
    "shoulder_pan",
]

JOINT_MOTOR_IDS = {
    "shoulder_pan":  1,
    "shoulder_lift": 2,
    "elbow_flex":    3,
    "wrist_flex":    4,
    "wrist_roll":    5,
    "gripper":       6,
}

# wrist_roll is continuous rotation – it can spin 360° so mechanical
# limit detection doesn't apply; we give it the full encoder range.
CONTINUOUS_JOINTS = {"wrist_roll"}

# ---------------------------------------------------------------------------
# Tuning constants (mutable dict so CLI can override without `global`)
# ---------------------------------------------------------------------------
CONFIG = {
    "sweep_step_size": 15,   # encoder steps per increment during sweep
    "grace_steps":     30,   # iterations before stall detection activates (~2.4s)
    "stall_threshold": 3,    # minimum movement to count as progress
    "stall_checks":    15,   # consecutive no-progress reads to declare stall
    "poll_interval":   0.08, # seconds between position reads / steps
    "settle_time":     0.3,  # seconds to wait after stopping before reading pos
    "safety_margin":   10,   # encoder steps to pull back from the hard stop
    "move_step_size":  30,   # encoder steps per increment for non-sweep moves
    "move_step_delay": 0.03, # seconds between steps for non-sweep moves
}


def build_motor_bus(port: str) -> FeetechMotorsBus:
    """Create the FeetechMotorsBus with the SO-101 motor definitions."""
    motors = {
        name: Motor(mid, "sts3215", MotorNormMode.RANGE_M100_100)
        for name, mid in JOINT_MOTOR_IDS.items()
    }
    bus = FeetechMotorsBus(port=port, motors=motors, protocol_version=0)
    return bus


def read_position(bus: FeetechMotorsBus, joint: str) -> int:
    """Read the raw (unnormalized) present position of a single joint."""
    return int(bus.read("Present_Position", joint, normalize=False))


def write_goal(bus: FeetechMotorsBus, joint: str, position: int):
    """Write a raw goal position to a single joint."""
    bus.write("Goal_Position", joint, position, normalize=False)


def move_to_position(bus: FeetechMotorsBus, joint: str, target: int,
                     step_size: int = 30, step_delay: float = 0.05,
                     timeout: float = 10.0):
    """
    Move *joint* to *target* in small increments.  This avoids needing
    a speed register — we control velocity by choosing step_size and
    step_delay.  Blocks until the target is reached (or close enough),
    or until *timeout* seconds have elapsed.
    """
    start = time.time()
    last_pos = read_position(bus, joint)
    stuck_count = 0

    while True:
        pos = read_position(bus, joint)
        diff = target - pos
        if abs(diff) <= step_size:
            write_goal(bus, joint, target)
            time.sleep(step_delay)
            return

        if time.time() - start > timeout:
            print(f"\n    ⚠ move_to_position timeout for {joint} "
                  f"(at {pos}, target {target})")
            write_goal(bus, joint, pos)  # hold where we are
            return

        # Detect if we're stuck (servo can't reach target)
        if abs(pos - last_pos) <= 2:
            stuck_count += 1
            if stuck_count > 20:  # ~1s of no movement
                print(f"\n    ⚠ {joint} stuck at {pos} (target {target}), stopping")
                write_goal(bus, joint, pos)
                return
        else:
            stuck_count = 0
        last_pos = pos

        direction = 1 if diff > 0 else -1
        write_goal(bus, joint, pos + direction * step_size)
        time.sleep(step_delay)


def enable_torque_single(bus: FeetechMotorsBus, joint: str):
    """Enable torque on a single joint."""
    bus.enable_torque(joint)


def disable_torque_single(bus: FeetechMotorsBus, joint: str):
    """Disable torque on a single joint."""
    bus.disable_torque(joint)


# ---------------------------------------------------------------------------
# Stall-detection sweep
# ---------------------------------------------------------------------------
def sweep_to_limit(bus: FeetechMotorsBus, joint: str, target: int) -> int:
    """
    Move *joint* toward *target* (0 or 4095) and detect when it stalls
    at a mechanical limit.

    Uses a single unified loop:
      - Commands progressively further goal positions toward target.
      - Tracks the furthest position the servo has actually reached.
      - Once the servo stops making progress for enough consecutive
        iterations, declares a stall.
      - Ignores the first N iterations (grace period) so the servo has
        time to start moving.

    Returns the raw encoder position where the joint stalled.
    """
    step_size = CONFIG["sweep_step_size"]
    step_delay = CONFIG["poll_interval"]
    grace_steps = CONFIG["grace_steps"]
    stall_checks = CONFIG["stall_checks"]

    start_pos = read_position(bus, joint)
    direction = 1 if target > start_pos else -1

    # We'll advance the commanded goal steadily toward target.
    commanded = start_pos
    best_pos = start_pos        # furthest position reached in target direction
    no_progress_count = 0       # consecutive iterations with no new progress
    iteration = 0

    while True:
        # Advance commanded goal by one step
        remaining = abs(target - commanded)
        if remaining <= 2:
            time.sleep(CONFIG["settle_time"])
            return read_position(bus, joint)

        step = min(step_size, remaining)
        commanded += direction * step
        commanded = max(0, min(4095, commanded))
        write_goal(bus, joint, commanded)
        time.sleep(step_delay)

        # Read actual position
        actual = read_position(bus, joint)
        iteration += 1

        # Track the furthest the servo has reached toward target
        if direction == 1:
            if actual > best_pos:
                best_pos = actual
                no_progress_count = 0
            else:
                no_progress_count += 1
        else:
            if actual < best_pos:
                best_pos = actual
                no_progress_count = 0
            else:
                no_progress_count += 1

        # Don't check for stalls during grace period — servo needs time
        # to start moving, especially heavy joints under load
        if iteration <= grace_steps:
            no_progress_count = 0
            continue

        # If we haven't moved at all after the grace period, we're at limit
        if iteration == grace_steps + 1 and abs(best_pos - start_pos) <= CONFIG["stall_threshold"]:
            print(f" (at limit)", end="", flush=True)
            write_goal(bus, joint, start_pos)
            time.sleep(CONFIG["settle_time"])
            return read_position(bus, joint)

        # Stall detection: no progress for enough consecutive iterations
        if no_progress_count >= stall_checks:
            write_goal(bus, joint, best_pos)
            time.sleep(CONFIG["settle_time"])
            return read_position(bus, joint)


# ---------------------------------------------------------------------------
# Main calibration flow
# ---------------------------------------------------------------------------
def run_calibration(port: str, output_path: str, dry_run: bool = False):
    bus = build_motor_bus(port)

    mode_label = "DRY RUN" if dry_run else "Auto-Calibration"
    print("=" * 60)
    print(f"  SO-101 {mode_label} Script")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 0: Connect
    # ------------------------------------------------------------------
    print("[1/5] Connecting to motors on", port, "...")
    bus.connect()
    print("       Connected successfully.\n")

    # Disable torque so the user can move the arm freely
    bus.disable_torque()

    # ------------------------------------------------------------------
    # Step 1: Home pose
    # ------------------------------------------------------------------
    print("[2/5] MANUAL STEP — Set home pose")
    print("       Please move the arm to its HOME / REST position.")
    print("       This is the pose where all joints are at their")
    print("       logical zero (typically arm straight up, gripper open).")
    print()
    input("       Press ENTER when the arm is in the home position... ")

    home_positions: dict[str, int] = {}
    for name in JOINT_MOTOR_IDS:
        home_positions[name] = read_position(bus, name)

    print()
    print("       Home positions recorded:")
    for name, pos in home_positions.items():
        print(f"         {name:16s} = {pos}")
    print()

    # Lock the arm in the home position so the user can let go
    print("       Locking arm in home position (torque enabled)...")
    for name in JOINT_MOTOR_IDS:
        enable_torque_single(bus, name)
        write_goal(bus, name, home_positions[name])
    time.sleep(0.5)
    print("       Arm is now holding itself — you can let go.\n")

    # ------------------------------------------------------------------
    # Step 2: Automated range sweep
    # ------------------------------------------------------------------
    print("[3/5] Automated range-of-motion sweep")
    print("       The arm will move each joint slowly in both directions")
    print("       to find the mechanical limits.")
    print()
    print("  ⚠  CAUTION: Keep hands clear of the arm during this step!")
    print()
    input("       Press ENTER to begin the sweep... ")
    print()

    range_results: dict[str, dict] = {}

    for joint in JOINT_NAMES_CALIBRATION_ORDER:
        if joint in CONTINUOUS_JOINTS:
            print(f"  [{joint}] Continuous joint — assigning full range 0–4095")
            range_results[joint] = {"range_min": 0, "range_max": 4095}
            continue

        home_pos = home_positions[joint]
        print(f"  [{joint}] (home={home_pos})")

        # Enable torque on all joints to hold them steady, except the
        # one we're testing (which also needs torque to move).
        for other in JOINT_MOTOR_IDS:
            enable_torque_single(bus, other)
            if other != joint:
                write_goal(bus, other, home_positions[other])

        time.sleep(0.5)

        # --- Sweep toward 0 (find range_min) ---
        print(f"    ↓ Sweeping toward 0 ...", end="", flush=True)
        raw_min = sweep_to_limit(bus, joint, 0)
        range_min = raw_min + CONFIG["safety_margin"]
        print(f" stall at {raw_min}, range_min = {range_min}")

        # Return to home before sweeping the other way
        move_to_position(bus, joint, home_pos,
                         CONFIG["move_step_size"], CONFIG["move_step_delay"])
        time.sleep(0.3)

        # --- Sweep toward 4095 (find range_max) ---
        print(f"    ↑ Sweeping toward 4095 ...", end="", flush=True)
        raw_max = sweep_to_limit(bus, joint, 4095)
        range_max = raw_max - CONFIG["safety_margin"]
        print(f" stall at {raw_max}, range_max = {range_max}")

        range_results[joint] = {"range_min": range_min, "range_max": range_max}

        # Return to home
        move_to_position(bus, joint, home_pos,
                         CONFIG["move_step_size"], CONFIG["move_step_delay"])
        time.sleep(1.0)

        print(f"    ✓ {joint}: [{range_min}, {range_max}]  (span = {range_max - range_min})")
        print()

    # ------------------------------------------------------------------
    # Step 3: Compute homing offsets
    # ------------------------------------------------------------------
    # The STS3215 "Homing_Offset" register shifts the reported position so
    # that "Present_Position = raw_position + homing_offset".  LeRobot's
    # calibration stores the offset that, when applied, maps the home pose
    # reading to the midpoint of the encoder range (2048).
    #
    # offset = -(home_raw - 2048)  =>  so that home becomes ~2048 after offset.
    # But the existing calibration files use a simpler convention:
    #   homing_offset = -(home_position)
    # which maps home to 0.  We'll match the convention from the user's
    # existing file: homing_offset = -(home_position).
    # Actually, looking at the provided file, homing_offset values like -1312
    # with home positions around 1312 confirm:  homing_offset ≈ -home_pos.

    print("[4/5] Computing homing offsets from home positions")
    print()

    calibration: dict[str, dict] = {}
    for name in JOINT_MOTOR_IDS:
        homing_offset = -home_positions[name]
        rmin = range_results[name]["range_min"]
        rmax = range_results[name]["range_max"]

        calibration[name] = {
            "id": JOINT_MOTOR_IDS[name],
            "drive_mode": 0,
            "homing_offset": homing_offset,
            "range_min": rmin,
            "range_max": rmax,
        }

    # ------------------------------------------------------------------
    # Step 4: Present results for confirmation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Calibration Results")
    print("=" * 60)
    print()
    print(f"  {'Joint':16s} {'ID':>3s}  {'Offset':>8s}  {'Min':>6s}  {'Max':>6s}  {'Span':>6s}")
    print(f"  {'-'*16} {'-'*3}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")
    for name in JOINT_MOTOR_IDS:
        c = calibration[name]
        span = c["range_max"] - c["range_min"]
        print(
            f"  {name:16s} {c['id']:3d}  {c['homing_offset']:8d}"
            f"  {c['range_min']:6d}  {c['range_max']:6d}  {span:6d}"
        )
    print()

    # Show the full JSON for inspection
    print("Full calibration JSON:")
    print(json.dumps(calibration, indent=4))
    print()

    # Return all joints to home and disable torque
    for joint in JOINT_MOTOR_IDS:
        move_to_position(bus, joint, home_positions[joint],
                         CONFIG["move_step_size"], CONFIG["move_step_delay"])
    time.sleep(0.5)
    bus.disable_torque()

    # ------------------------------------------------------------------
    # Step 5: Save or report (dry run)
    # ------------------------------------------------------------------
    if dry_run:
        # In dry-run mode, also try to load and compare against any
        # existing calibration file so the user can see the diff.
        existing = None
        if os.path.exists(output_path):
            try:
                with open(output_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        if existing:
            print("=" * 60)
            print("  Comparison with existing calibration")
            print(f"  ({output_path})")
            print("=" * 60)
            print()
            print(
                f"  {'Joint':16s} {'Field':14s} {'Existing':>8s}  {'Measured':>8s}  {'Delta':>7s}"
            )
            print(f"  {'-'*16} {'-'*14} {'-'*8}  {'-'*8}  {'-'*7}")
            for name in JOINT_MOTOR_IDS:
                old = existing.get(name, {})
                new = calibration[name]
                for field in ("homing_offset", "range_min", "range_max"):
                    old_val = old.get(field, "N/A")
                    new_val = new[field]
                    if isinstance(old_val, (int, float)):
                        delta = new_val - old_val
                        delta_str = f"{delta:+d}"
                    else:
                        delta_str = "—"
                    print(
                        f"  {name:16s} {field:14s} {str(old_val):>8s}"
                        f"  {new_val:8d}  {delta_str:>7s}"
                    )
                print()

        print("  DRY RUN complete — no files were written.")
    else:
        confirm = input("Save this calibration? [Y/n] ").strip().lower()
        if confirm in ("", "y", "yes"):
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(calibration, f, indent=4)
            print(f"\n  ✓ Calibration saved to {out}")
        else:
            print("\n  ✗ Calibration NOT saved.")

    bus.disconnect()
    print("\nDone. Disconnected from motors.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Auto-calibrate an SO-101 follower arm."
    )
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="Serial port the arm is connected to (e.g. /dev/ttyACM0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.expanduser(
            "~/.cache/huggingface/lerobot/calibration/robots/so101_auto_calibration.json"
        ),
        help="Path to write the calibration JSON file.",
    )
    parser.add_argument(
        "--sweep-step-size",
        type=int,
        default=CONFIG["sweep_step_size"],
        help=f"Encoder steps per increment during sweep (default: {CONFIG['sweep_step_size']}). Smaller = slower/safer.",
    )
    parser.add_argument(
        "--safety-margin",
        type=int,
        default=CONFIG["safety_margin"],
        help=f"Encoder steps to pull back from hard stops (default: {CONFIG['safety_margin']}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run the full sweep but don't save — just print results and compare with existing calibration.",
    )
    args = parser.parse_args()

    # Apply CLI overrides to config
    CONFIG["sweep_step_size"] = args.sweep_step_size
    CONFIG["safety_margin"] = args.safety_margin

    run_calibration(args.port, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
