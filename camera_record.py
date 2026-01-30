import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    # --------- EDIT THESE ----------
    BASE_IP = "192.168.1.10"      # robot base controller IP
    SAMPLE_HZ = 1             # start conservative for RTSP + disk write
    DURATION_S = 10.0             # seconds
    # ------------------------------

    rgb_url = f"rtsp://{BASE_IP}/color"
    depth_url = f"rtsp://{BASE_IP}/depth"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"rgbd_{stamp}"
    rgb_dir = os.path.join(out_dir, "rgb")
    dep_dir = os.path.join(out_dir, "depth")
    ensure_dir(rgb_dir)
    ensure_dir(dep_dir)

    csv_path = os.path.join(out_dir, "timestamps.csv")

    # Use FFmpeg backend for RTSP (best chance on Windows)
    cap_rgb = cv2.VideoCapture(rgb_url, cv2.CAP_FFMPEG)
    cap_dep = cv2.VideoCapture(depth_url, cv2.CAP_FFMPEG)

    if not cap_rgb.isOpened():
        raise RuntimeError(f"Failed to open RGB stream: {rgb_url}")
    if not cap_dep.isOpened():
        raise RuntimeError(f"Failed to open Depth stream: {depth_url}")

    period = 1.0 / SAMPLE_HZ
    t0 = time.perf_counter()
    next_t = t0
    idx = 0

    # Grab one frame to report formats
    ok_r, frame_r = cap_rgb.read()
    ok_d, frame_d = cap_dep.read()
    if not ok_r or frame_r is None:
        raise RuntimeError("Could not read initial RGB frame.")
    if not ok_d or frame_d is None:
        raise RuntimeError("Could not read initial depth frame.")

    print("Initial RGB frame:", frame_r.dtype, frame_r.shape)
    print("Initial DEP frame:", frame_d.dtype, frame_d.shape)
    print("Saving to:", out_dir)

    # Write CSV header
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "rgb_file", "depth_file", "depth_dtype", "depth_shape"])

        # We already consumed one frame each; save them as idx=0, then continue
        def save_pair(t_sec, rgb, dep, k):
            rgb_name = f"{k:06d}.png"
            dep_name = f"{k:06d}.png"
            rgb_path = os.path.join(rgb_dir, rgb_name)
            dep_path = os.path.join(dep_dir, dep_name)

            # RGB is typically BGR uint8; PNG preserves it
            cv2.imwrite(rgb_path, rgb)

            # Depth handling:
            # - If dep is uint16 single-channel, write as 16-bit PNG (best)
            # - If dep is 8-bit or 3-channel (decoder gives weird format), still save
            #   but you should verify whether it’s true metric depth.
            if dep.dtype == np.uint16 and (dep.ndim == 2 or (dep.ndim == 3 and dep.shape[2] == 1)):
                # Ensure 2D for writing
                dep2 = dep if dep.ndim == 2 else dep[:, :, 0]
                cv2.imwrite(dep_path, dep2)
                depth_dtype = "uint16"
                depth_shape = str(dep2.shape)
            else:
                cv2.imwrite(dep_path, dep)
                depth_dtype = str(dep.dtype)
                depth_shape = str(dep.shape)

            w.writerow([t_sec, os.path.join("rgb", rgb_name), os.path.join("depth", dep_name),
                        depth_dtype, depth_shape])

        # Save the first grabbed frames as sample 0
        save_pair(0.0, frame_r, frame_d, 0)
        idx = 1
        next_t = t0 + period

        while True:
            now = time.perf_counter()
            t = now - t0
            if t >= DURATION_S:
                break

            ok_r, rgb = cap_rgb.read()
            ok_d, dep = cap_dep.read()

            if not ok_r or rgb is None:
                print("Warning: missed RGB frame")
                continue
            if not ok_d or dep is None:
                print("Warning: missed DEP frame")
                continue

            save_pair(t, rgb, dep, idx)
            idx += 1

            # Rate control
            next_t += period
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    cap_rgb.release()
    cap_dep.release()
    print(f"Done. Wrote {idx} samples.")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
