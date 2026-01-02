import argparse
import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="running average")
    p.add_argument("--input", default=0, help="file path or camera idx (default 0)")
    p.add_argument("--bg-method", default="running_avg", choices=["running_avg"], help="background control")
    p.add_argument("--threshold", type=int, default=25)
    p.add_argument("--min-area", type=int, default=500, help="min contour area (pixels)")
    p.add_argument("--learning-rate", type=float, default=0.01, help="background learning rate")
    p.add_argument("--output-video", default=None, help="ex: out.mp4")
    p.add_argument("--output-json", default=None, help="save detection results as JSON")
    p.add_argument("--generate-test-video", default=None, help="generate short test video")
    p.add_argument("--visualize", action="store_true", help="interface (cv2.imshow)")
    return p.parse_args()

def open_capture(source):
    # is it camera or file
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except Exception:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open the source: {source}")
    return cap