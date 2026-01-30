import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np


def record_rgbd_rtsp(
    base_ip: str,
    out_dir: str,
    duration_s: float = 10.0,
    sample_hz: float = 10.0,
):
    """
    Records Kinova Vision Module RTSP streams:
      - rtsp://<base_ip>/color
      - rtsp://<base_ip>/depth

    Saves:
      out_dir/
        rgb/000000.png
        depth/000000.png
        index.csv   (t_sec, rgb_file, depth_file, depth_dtype, depth_shape)
    """
    rgb_url = f"rtsp://{base_ip}/color"
    dep_url = f"rtsp://{base_ip}/depth"

    rgb_dir = os.path.join(out_dir, "rgb")
    dep_dir = os.path.join(out_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(dep_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "index.csv")

    # FFmpeg backend is typically best for RTSP on Windows
    cap_rgb = cv2.VideoCapture(rgb_url, cv2.CAP_FFMPEG)
    cap_dep = cv2.VideoCapture(dep_url, cv2.CAP_FFMPEG)

    if not cap_rgb.isOpened():
        raise RuntimeError(f"Failed to open RGB RTSP stream: {rgb_url}")
    if not cap_dep.isOpened():
        raise RuntimeError(f"Failed to open Depth RTSP stream: {dep_url}")

    period = 1.0 / sample_hz
    t0 = time.perf_counter()
    next_t = t0
    idx = 0

    # Grab one frame to report what OpenCV actually decodes
    ok_r, rgb0 = cap_rgb.read()
    ok_d, dep0 = cap_dep.read()
    if not ok_r or rgb0 is None:
        raise RuntimeError("Could not read initial RGB frame from RTSP.")
    if not ok_d or dep0 is None:
        raise RuntimeError("Could not read initial Depth frame from RTSP.")

    print("RGB frame:", rgb0.dtype, rgb0.shape)   # expect uint8 (H,W,3)
    print("DEP frame:", dep0.dtype, dep0.shape)   # ideally uint16 (H,W) or (H,W,1)
    print("Output dir:", out_dir)

    def save_pair(t_sec: float, rgb: np.ndarray, dep: np.ndarray, k: int, writer):
        rgb_name = f"{k:06d}.png"
        dep_name = f"{k:06d}.png"
        rgb_rel = os.path.join("rgb", rgb_name)
        dep_rel = os.path.join("depth", dep_name)

        # Save RGB (PNG preserves full 8-bit values)
        cv2.imwrite(os.path.join(out_dir, rgb_rel), rgb)

        # Save Depth:
        # Best case: decoded as uint16 single channel => save as 16-bit PNG
        # Sometimes OpenCV decodes RTSP as uint8/3-channel; we still save what we got,
        # but you should verify the dtype printed above.
        depth_dtype = str(dep.dtype)
        depth_shape = str(dep.shape)

        if dep.dtype == np.uint16:
            # Ensure single channel for PNG write
            dep2 = dep if dep.ndim == 2 else dep[:, :, 0]
            cv2.imwrite(os.path.join(out_dir, dep_rel), dep2)
            depth_dtype = "uint16"
            depth_shape = str(dep2.shape)
        else:
            # Save as-is (may be visualized depth if not uint16)
            cv2.imwrite(os.path.join(out_dir, dep_rel), dep)

        writer.writerow([t_sec, rgb_rel, dep_rel, depth_dtype, depth_shape])

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "rgb_file", "depth_file", "depth_dtype", "depth_shape"])

        # Save the first frames as sample 0
        save_pair(0.0, rgb0, dep0, 0, w)
        idx = 1
        next_t = t0 + period

        while True:
            now = time.perf_counter()
            t = now - t0
            if t >= duration_s:
                break

            ok_r, rgb = cap_rgb.read()
            ok_d, dep = cap_dep.read()

            if not ok_r or rgb is None:
                print("Warning: missed RGB frame")
                continue
            if not ok_d or dep is None:
                print("Warning: missed Depth frame")
                continue

            save_pair(t, rgb, dep, idx, w)
            idx += 1

            next_t += period
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    cap_rgb.release()
    cap_dep.release()
    print(f"Done. Saved {idx} samples.")
    print(f"CSV index: {csv_path}")


def main():
    # ---- EDIT THESE ----
    BASE_IP = "192.168.1.10"   # or your actual base controller IP
    DURATION_S = 10.0
    SAMPLE_HZ = 10.0           # start at 10; increase if stable
    # --------------------

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"rgbd_{stamp}"
    record_rgbd_rtsp(BASE_IP, out_dir, duration_s=DURATION_S, sample_hz=SAMPLE_HZ)


if __name__ == "__main__":
    main()
