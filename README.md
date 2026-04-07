# SO-101 Auto-Calibration

A semi-automatic calibration script for the SO-101 robot arm (LeRobot / Feetech STS3215). Instead of manually moving each joint through its full range of motion, this script drives each servo toward its mechanical limits and records where it stalls — discovering `range_min`, `range_max`, and `homing_offset` automatically.

## Requirements

- Python 3.10+
- [LeRobot](https://github.com/huggingface/lerobot) with Feetech support:
  ```
  pip install lerobot[feetech]
  ```
- An assembled and motor-configured SO-101 follower arm connected via USB

## Quick Start

```bash
python auto_calibrate_so101.py --port /dev/ttyACM0
```

The script will walk you through two steps:

1. **Set home pose** — With torque disabled, manually position the arm at its rest/zero pose, then press ENTER. The arm locks in place so you can let go.
2. **Automated sweep** — Press ENTER again and the script takes over. Each joint sweeps slowly toward both ends of its range while the others hold steady. Takes ~3–4 minutes with default settings.

After the sweep, you'll see a summary table and the full calibration JSON. Confirm to save.

## Dry Run Mode

If you've already calibrated manually and want to verify, use `--dry-run`. The script performs the full sweep but doesn't write anything. It loads your existing calibration file and prints a side-by-side comparison:

```bash
python auto_calibrate_so101.py --port /dev/ttyACM0 --dry-run
```

Output looks like:

```
  Joint            Field          Existing  Measured    Delta
  ---------------- -------------- --------  --------  -------
  shoulder_pan     homing_offset    -1312     -1308       +4
  shoulder_pan     range_min          962       958       -4
  shoulder_pan     range_max         3490      3495       +5
  ...
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | *(required)* | Serial port (e.g. `/dev/ttyACM0`, `COM3`) |
| `--output` | `~/.cache/huggingface/lerobot/calibration/robots/so101_auto_calibration.json` | Where to save the calibration file |
| `--sweep-step-size` | `15` | Encoder steps per increment. Larger = faster but less gentle |
| `--safety-margin` | `10` | Steps to pull back from discovered hard stops |
| `--dry-run` | off | Run the sweep but don't save; compare with existing calibration |

## Speed Tuning

The default settings are conservative. For faster runs (e.g. demos), increase `--sweep-step-size`:

| Preset | `--sweep-step-size` | Approx. time per joint | Notes |
|--------|---------------------|------------------------|-------|
| Safe (default) | `15` | ~30–40s | Gentle on hardware |
| Demo | `40` | ~10–15s | Good balance of speed and safety |
| Aggressive | `60` | ~5–8s | Audible stops, fine for short-term use |

Example:

```bash
python auto_calibrate_so101.py --port /dev/ttyACM0 --sweep-step-size 40
```

## How It Works

### Calibration values

The script determines three values per joint, matching the format used by LeRobot's calibration files:

- **`homing_offset`**: Negative of the raw encoder position at the home pose. Maps the home pose to logical zero.
- **`range_min`**: Lowest encoder position the joint can reach (plus a safety margin).
- **`range_max`**: Highest encoder position the joint can reach (minus a safety margin).

### Stall detection

The script commands each servo to move in small increments toward position 0 (for `range_min`) and 4095 (for `range_max`). It tracks the furthest position the servo actually reaches. When the servo stops making forward progress for enough consecutive iterations, a mechanical limit has been found.

A grace period at the start of each sweep prevents false stalls from servo startup lag — heavier joints like the shoulder need time to begin moving under load.

### Joint ordering

Joints are calibrated from outermost to innermost (gripper → wrist → elbow → shoulder) to minimize risk. While one joint is being swept, all others hold their home positions with torque enabled.

### Special cases

- **wrist_roll**: Treated as a continuous rotation joint (no mechanical stops). Automatically assigned the full 0–4095 range.
- **Return-to-home failures**: If a joint can't return to its home position after a sweep (e.g. due to gravity or load), the script times out gracefully and continues rather than hanging.

## Output Format

The calibration file is a JSON object keyed by joint name:

```json
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -1312,
        "range_min": 962,
        "range_max": 3490
    },
    ...
}
```

This is the same format used by LeRobot's built-in calibration. To use it, copy or symlink the output file to your LeRobot calibration directory (typically `~/.cache/huggingface/lerobot/calibration/robots/`).

## Troubleshooting

**"Could not import lerobot motor modules"** — Make sure LeRobot is installed with Feetech support: `pip install lerobot[feetech]`.

**Servo doesn't move during sweep** — Check that the power supply is connected and providing adequate voltage (5V for 7.4V motors, 12V for 12V motors). The script will print "(at limit)" if a joint truly can't move.

**Unexpected range values** — Run with `--dry-run` to compare against a known-good calibration. If ranges are too narrow, try decreasing `--sweep-step-size` for finer detection.

**Joint gets stuck returning to home** — This is handled automatically (the script times out and continues). It typically happens when a heavy joint sweeps far from home and gravity prevents a clean return. The calibration values are still valid.

**Permission denied on serial port** — On Linux, add your user to the `dialout` group: `sudo usermod -a -G dialout $USER`, then log out and back in.
