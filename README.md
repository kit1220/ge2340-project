# Identifying Truly Available Library Seats (GE2340 Group 17)

Fixed-camera library seat occupancy and suspected seat-hogging monitoring pipeline using fine-tuned **YOLO11s**, 4-point polygon seat ROIs, temporal majority smoothing (`W = 3`, 15 s), and an unattended-item timer state machine (`dt = 5 s`, evaluation threshold `T = 120 s`).

## Project Structure

```text
library-seat-monitor/
├── 01_extract_and_blur.py   # 5s frame sampling + automatic face blurring + YOLO pre-labeling
├── 02_define_roi.py         # Interactive 4-point polygon ROI picker -> configs/rois.json
├── 03_run_pipeline.py       # YOLO inference + ROI assignment + smoothing + timer state machine
├── 04_evaluate.py           # Evaluation vs. Ground Truth (P/R/F1, Macro-F1, Confusion Matrices)
├── configs/
│   └── rois.json            # Fixed camera seat polygons (cam1: seat_1-3, cam2: seat_4-6)
├── data.yaml                # Session-level dataset split configuration (2 classes)
└── requirements.txt
```

## Dataset Split (Session-Level, Zero Data Leakage)

- **Train (457 frames):** `empty`, `bag`, `s1` (standard), `s2` (jackets on chair backs & small items), `s3` (half lights off)
- **Val (134 frames):** `s4` (mixed lighting and items; Validation `mAP50 = 0.944`, `mAP50-95 = 0.744`)
- **Test (240 frames / 720 seat-step samples):** `s5` (standard test + unseen tiny items), `s6` (stress test: lights off, moved chairs, cross-desk sleeping, hanging bags)

## Quantitative Results (Tesla T4 GPU)

| Method | Scope | Accuracy | Macro-F1 | Hogging P | Hogging R | Hogging F1 | Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Rule Baseline (No Timer) | Overall (720) | 0.6819 | 0.5489 | 0.0000 | 0.0000 | 0.0000 | 72.6 FPS |
| Pretrained YOLO11s + Timer | Overall (720) | 0.5111 | 0.4260 | 0.0476 | 0.0105 | 0.0172 | 77.8 FPS |
| **Fine-tuned YOLO11s + Timer** | **Overall (720)** | **0.7208** | **0.6734** | **1.0000** | **0.2947** | **0.4553** | **62.9 FPS** |
| **Fine-tuned YOLO11s + Timer** | **S5 Easy (360)** | **0.8833** | **0.8606** | **1.0000** | **0.6667** | **0.8000** | **62.9 FPS** |
| **Fine-tuned YOLO11s + Timer** | **S6 Hard (360)** | **0.5583** | **0.4911** | **1.0000** | **0.0968** | **0.1765** | **62.9 FPS** |

## Quick Start

```bash
pip install -r requirements.txt
python 01_extract_and_blur.py
python 02_define_roi.py
python 03_run_pipeline.py
python 04_evaluate.py
```

## Privacy & Ethics Notice

All participants provided informed consent and wore masks during recording. Faces were automatically blurred during frame extraction. Raw videos and identifiable images are strictly excluded from this repository and deleted after evaluation.
