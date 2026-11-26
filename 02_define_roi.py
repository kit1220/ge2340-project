"""Interactive 4-point polygon ROI picker for fixed camera seats -> configs/rois.json."""
import json
import cv2
import numpy as np
from pathlib import Path

CONFIG_OUT = Path("configs/rois.json")
CAM_TARGETS = {
    "cam1": ("dataset/images/train/s1_cam1_*.jpg", ["seat_1", "seat_2", "seat_3"]),
    "cam2": ("dataset/images/train/s1_cam2_*.jpg", ["seat_4", "seat_5", "seat_6"]),
}


def pick_cam_rois(cam: str, img_path: Path, seats: list[str]) -> dict:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    cam_rois, pts, idx = {}, [], 0
    win = f"ROI {cam} (Left: Add | Right: Undo | Space/Enter: Next | R: Reset)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while idx < len(seats):
        canvas = img.copy()
        for s_id, poly in cam_rois.items():
            cv2.polylines(canvas, [np.array(poly, np.int32)], True, (0, 220, 0), 2)
        for i, p in enumerate(pts):
            cv2.circle(canvas, tuple(p), 5, (0, 255, 255), -1)
            if i > 0:
                cv2.line(canvas, tuple(pts[i - 1]), tuple(p), (0, 255, 255), 2)
        if len(pts) == 4:
            cv2.line(canvas, tuple(pts[3]), tuple(pts[0]), (0, 255, 255), 2)
        cv2.putText(canvas, f"{cam} - {seats[idx]} ({len(pts)}/4)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and len(pts) == 4:
            cam_rois[seats[idx]] = list(pts)
            pts.clear()
            idx += 1
        elif key in (ord("r"), ord("R")):
            pts.clear()
    cv2.destroyAllWindows()
    return {"image_size": [w, h], "seats": cam_rois}


def main() -> None:
    data = {}
    for cam, (pattern, seats) in CAM_TARGETS.items():
        matches = sorted(Path(".").glob(pattern))
        data[cam] = pick_cam_rois(cam, matches[len(matches) // 2], seats)
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved ROI config to {CONFIG_OUT}")


if __name__ == "__main__":
    main()
