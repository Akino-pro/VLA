import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np

# Force RTSP over TCP (good practice for stability)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"


def record_rgb_rtsp(
    base_ip: str,
    out_dir: str,
    duration_s: float = 10.0,
    sample_hz: float = 1.0,
):
    """
    Records Kinova Vision Module RGB RTSP stream:
      - rtsp://<base_ip>/color

    Saves:
      out_dir/
        rgb/000000.png
        index.csv   (t_sec, rgb_file, depth_file, depth_dtype, depth_shape)
      NOTE: depth fields are left blank for compatibility.
    """
    rgb_url = f"rtsp://{base_ip}:554/color"  # explicit RTSP port
    # dep_url removed

    rgb_dir = os.path.join(out_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "index.csv")

    # FFmpeg backend is typically best for RTSP on Windows
    cap_rgb = cv2.VideoCapture(rgb_url, cv2.CAP_FFMPEG)

    if not cap_rgb.isOpened():
        raise RuntimeError(f"Failed to open RGB RTSP stream: {rgb_url}")

    period = 1.0 / float(sample_hz)
    t0 = time.perf_counter()
    next_t = t0
    idx = 0

    # Grab one frame to verify decoding
    ok_r, rgb0 = cap_rgb.read()
    if not ok_r or rgb0 is None:
        raise RuntimeError("Could not read initial RGB frame from RTSP.")

    print("RGB frame:", rgb0.dtype, rgb0.shape)   # expect uint8 (H,W,3)
    print("Output dir:", out_dir)

    def save_rgb(t_sec: float, rgb: np.ndarray, k: int, writer):
        rgb_name = f"{k:06d}.png"
        rgb_rel = os.path.join("rgb", rgb_name)

        cv2.imwrite(os.path.join(out_dir, rgb_rel), rgb)

        # Keep depth columns for compatibility, but leave blank
        writer.writerow([t_sec, rgb_rel, "", "", ""])

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "rgb_file", "depth_file", "depth_dtype", "depth_shape"])

        # Save first frame as sample 0
        save_rgb(0.0, rgb0, 0, w)
        idx = 1
        next_t = t0 + period

        while True:
            now = time.perf_counter()
            t = now - t0
            if t >= duration_s:
                break

            ok_r, rgb = cap_rgb.read()
            if not ok_r or rgb is None:
                print("Warning: missed RGB frame")
                continue

            save_rgb(t, rgb, idx, w)
            idx += 1

            next_t += period
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    cap_rgb.release()
    print(f"Done. Saved {idx} RGB samples.")
    print(f"CSV index: {csv_path}")


def main():
    BASE_IP = "192.168.1.10"
    DURATION_S = 10.0
    SAMPLE_HZ = 1.0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"rgb_{stamp}"
    record_rgb_rtsp(BASE_IP, out_dir, duration_s=DURATION_S, sample_hz=SAMPLE_HZ)


if __name__ == "__main__":
    main()
