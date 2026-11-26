"""Run YOLO detection + ROI mapping + temporal majority smoothing (W=3) + timer state machine (T=120s)."""
import csv
import json
import time
from collections import Counter, deque
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

TEST_IMG_DIR = Path("dataset/images/test")
ROIS_PATH = Path("configs/rois.json")
BEST_PT = Path("runs/yolo11s_seats_v1/weights/best.pt")
OUT_DIR = Path("outputs")

DT_SEC = 5
THRESHOLD_T = 120
WINDOW_W = 3
CONF_THRESH = 0.25
PERSON_Y_RATIO = 0.65
COCO_BELONGING_IDS = {24, 25, 26, 28, 39, 41, 63, 64, 66, 67, 73}


def majority_vote(win: deque) -> str:
    counts = Counter(win)
    m = max(counts.values())
    top = [k for k, v in counts.items() if v == m]
    return win[-1] if len(top) > 1 else top[0]


def run_exp(name: str, model: YOLO, rois: dict, test_index: dict, is_pretrained: bool, use_timer: bool) -> dict:
    rows, latencies = [], []
    for sess in sorted(test_index):
        windows = {s: deque(maxlen=WINDOW_W) for c in rois for s in rois[c]}
        timers = {s: 0 for c in rois for s in rois[c]}
        for sec in sorted(test_index[sess]):
            for cam in ["cam1", "cam2"]:
                img = cv2.imread(str(test_index[sess][sec][cam]))
                t0 = time.perf_counter()
                res = model.predict(img, conf=CONF_THRESH, verbose=False)[0]
                flags = {s: {"person": False, "belonging": False} for s in rois[cam]}
                for box in res.boxes:
                    cid = int(box.cls[0])
                    if is_pretrained:
                        role = "person" if cid == 0 else ("belonging" if cid in COCO_BELONGING_IDS else None)
                    else:
                        cname = str(model.names[cid]).lower()
                        role = cname if cname in {"person", "belonging"} else None
                    if not role:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + PERSON_Y_RATIO * (y2 - y1)) if role == "person" else ((y1 + y2) / 2.0)
                    for s_id, poly in rois[cam].items():
                        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                            flags[s_id][role] = True

                for s_id in sorted(rois[cam]):
                    raw = "raw_occupied" if flags[s_id]["person"] else ("raw_items_only" if flags[s_id]["belonging"] else "raw_empty")
                    windows[s_id].append(raw)
                    sm = majority_vote(windows[s_id])
                    if sm == "raw_occupied":
                        timers[s_id], pred = 0, "occupied"
                    elif sm == "raw_empty":
                        timers[s_id], pred = 0, "empty"
                    else:
                        if use_timer:
                            timers[s_id] += DT_SEC
                            pred = "suspected_hogging" if timers[s_id] >= THRESHOLD_T else "temporarily_away"
                        else:
                            timers[s_id], pred = 0, "temporarily_away"
                    rows.append({
                        "session": sess, "cam": cam, "timestamp_sec": sec,
                        "seat_id": s_id, "pred_state": pred, "unattended_sec": timers[s_id],
                    })
                latencies.append((time.perf_counter() - t0) * 1000.0)

    rows.sort(key=lambda r: (r["session"], r["cam"], r["seat_id"], r["timestamp_sec"]))
    out_csv = OUT_DIR / f"preds_{name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["session", "cam", "timestamp_sec", "seat_id", "pred_state", "unattended_sec"])
        w.writeheader()
        w.writerows(rows)
    mean_ms = float(np.mean(latencies))
    return {"mean_ms": round(mean_ms, 2), "fps": round(1000.0 / mean_ms, 1)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rois_raw = json.loads(ROIS_PATH.read_text(encoding="utf-8"))
    rois = {c: {s: np.array(p, np.int32) for s, p in cfg["seats"].items()} for c, cfg in rois_raw.items()}
    test_index = {}
    for p in sorted(TEST_IMG_DIR.glob("*.jpg")):
        sess, cam, t_tok = p.stem.split("_")
        test_index.setdefault(sess, {}).setdefault(int(t_tok[1:]), {})[cam] = p

    ft_model = YOLO(str(BEST_PT))
    pt_model = YOLO("yolo11s.pt")
    stats = {
        "rule_baseline": run_exp("rule_baseline", ft_model, rois, test_index, False, False),
        "pretrained_timer": run_exp("pretrained_timer", pt_model, rois, test_index, True, True),
        "finetuned_timer": run_exp("finetuned_timer", ft_model, rois, test_index, False, True),
    }
    (OUT_DIR / "speed_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("Done:", stats)


if __name__ == "__main__":
    main()
