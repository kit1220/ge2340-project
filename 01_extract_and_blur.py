"""Extract frames at 5s intervals, blur person head regions for privacy, and generate YOLO pre-labels."""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

RAW_VIDEO_DIR = Path("raw_videos")
OUTPUT_DIR = Path("dataset")
MAX_WIDTH = 1280
INTERVAL_SEC = 5
EMPTY_INTERVAL_SEC = 6
TEST_NUM_STEPS = 60
STATIC_DIFF_THRESH = 2.0
BLUR_CONF = 0.18
LABEL_CONF = 0.25

SESSION_SPLIT = {
    "empty": "train", "bag": "train", "s1": "train", "s2": "train", "s3": "train",
    "s4": "val", "s5": "test", "s6": "test",
}
COCO_BELONGING_IDS = {24, 25, 26, 28, 39, 41, 63, 64, 66, 67, 73}


def blur_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    k = min(99, max(45, ((x2 - x1) // 3) | 1))
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def main() -> None:
    model = YOLO("yolo11s.pt")
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        if split in {"train", "val"}:
            (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for vpath in sorted(RAW_VIDEO_DIR.glob("*.mp4")):
        session_id, cam_id = vpath.stem.lower().split("_")[:2]
        if session_id not in SESSION_SPLIT:
            continue
        split = SESSION_SPLIT[session_id]
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps)
        step = EMPTY_INTERVAL_SEC if session_id == "empty" else INTERVAL_SEC
        timestamps = [i * INTERVAL_SEC for i in range(TEST_NUM_STEPS)] if split == "test" else list(range(0, max(1, duration - 2), step))

        prev_gray = None
        for idx, sec in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if w > MAX_WIDTH:
                frame = cv2.resize(frame, (MAX_WIDTH, int(round(h * MAX_WIDTH / w))), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]

            if split in {"train", "val"} and session_id != "empty":
                gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and float(np.mean(cv2.absdiff(gray, prev_gray))) < STATIC_DIFF_THRESH:
                    continue
                prev_gray = gray

            res = model.predict(frame, conf=BLUR_CONF, verbose=False)[0]
            blurred = frame.copy()
            lines = []
            for box in res.boxes:
                cls_id, conf = int(box.cls[0]), float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if cls_id == 0:
                    bw, bh = x2 - x1, y2 - y1
                    blur_region(blurred, int(x1 - 0.08 * bw), int(y1 + 0.05 * bh), int(x2 + 0.08 * bw), int(y1 + 0.58 * bh))
                if split in {"train", "val"} and session_id != "empty" and conf >= LABEL_CONF:
                    mapped = 1 if cls_id == 0 else (0 if cls_id in COCO_BELONGING_IDS else None)
                    if mapped is not None:
                        lines.append(f"{mapped} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")

            stem = f"{session_id}_{cam_id}_t{idx * step:04d}"
            cv2.imwrite(str(OUTPUT_DIR / "images" / split / f"{stem}.jpg"), blurred)
            if split in {"train", "val"}:
                (OUTPUT_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        cap.release()


if __name__ == "__main__":
    main()
