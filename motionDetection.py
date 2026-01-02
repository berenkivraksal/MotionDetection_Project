import argparse
import cv2
import numpy as np
from datetime import datetime
import json
import sys

def parse_args():
    p = argparse.ArgumentParser(description="running average")
    p.add_argument("--input", default=0, help="file path or camera idx (default 0)")
    p.add_argument("--threshold", type=int, default=25)
    p.add_argument("--min-area", type=int, default=500, help="min contour area (pixels)")
    p.add_argument("--learning-rate", type=float, default=0.01, help="background learning rate")
    p.add_argument("--output-video", default=None, help="ex: out.mp4")
    p.add_argument("--output-json", default=None, help="save detection results as JSON")
    p.add_argument("--save-background", default=None, help="save final background image (ex: bg.png)")
    p.add_argument("--generate-test-video", default=None, help="generate short test video")
    p.add_argument("--visualize", action="store_true", help="interface (cv2.imshow)")

    return p.parse_args()

def generate_test_video(path, width=640, height=480, duration=4, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    total = int(duration * fps)

    for i in range(total):
        frame = np.full((height, width, 3), 200, dtype=np.uint8)
        # moving rectangle
        x = int((i / total) * (width - 100))
        y = height // 3
        cv2.rectangle(frame, (x, y), (x + 100, y + 60), (0, 0, 255), -1)
        out.write(frame)

    out.release()
    print(f"Synthetic test video created: {path}")

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

def process_video(input_src, output_path=None, json_path=None, bg_path=None, thresh=25, 
                        min_area=500, learning_rate=0.01, visualize=False):
    cap = open_capture(input_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0 or np.isnan(fps):
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # background as float image (initialized with the first frame)
    bg_float = None
    bg_uint8 = None
    frame_idx = 0
    detections = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

        if bg_float is None:
            bg_float = gray_blur.astype("float")
            # background initialized from first frame; diff will be zero
            bg_uint8 = cv2.convertScaleAbs(bg_float)

        else:
            # update background using running average
            cv2.accumulateWeighted(gray_blur, bg_float, learning_rate)
            bg_uint8 = cv2.convertScaleAbs(bg_float)

        diff = cv2.absdiff(gray_blur, bg_uint8)
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        # morphological operations: opening + dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_dets = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            frame_dets.append({"bbox": [int(x), int(y), int(w), int(h)], "area": float(area)})
            # draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # timestamp
        ts = datetime.now().isoformat()
        cv2.putText(frame, f"Frame: {frame_idx}  Dets: {len(frame_dets)}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if writer:
            writer.write(frame)

        if visualize:
            cv2.imshow("Mask", mask)
            cv2.imshow("Frame", frame)
            cv2.imshow("Background", bg_uint8)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        detections.append({"frame": frame_idx, "timestamp": ts, "detections": frame_dets})

    cap.release()
    if writer:
        writer.release()

    if bg_path and bg_uint8 is not None:
        cv2.imwrite(bg_path, bg_uint8)
        print(f"Background image saved: {bg_path}")

    if visualize:
        cv2.destroyAllWindows()

    if json_path:
        with open(json_path, "w") as f:
            json.dump(detections, f, indent=2)
        print(f"Detections JSON saved: {json_path}")

    print("Processing complete.")

if __name__ == "__main__":
    args = parse_args()

    if args.generate_test_video:
        generate_test_video(args.generate_test_video)
        # if only the test video was requested, exit
        if args.input == 0:
            sys.exit(0)

    try:
        process_video(args.input, output_path=args.output_video, json_path=args.output_json, 
                    bg_path=args.save_background, thresh=args.threshold, min_area=args.min_area, 
                    learning_rate=args.learning_rate, visualize=args.visualize)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)