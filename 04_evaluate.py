"""Evaluate predictions against test_ground_truth.csv and export P/R/F1, Macro-F1, and confusion matrices."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

OUT_DIR = Path("outputs")
EVAL_DIR = OUT_DIR / "eval_results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["empty", "occupied", "temporarily_away", "suspected_hogging"]
EXPERIMENTS = [
    ("rule_baseline", "Rule Baseline (No Timer)", OUT_DIR / "preds_rule_baseline.csv"),
    ("pretrained_timer", "Pretrained YOLO + Timer", OUT_DIR / "preds_pretrained_timer.csv"),
    ("finetuned_timer", "Fine-tuned YOLO + Timer", OUT_DIR / "preds_finetuned_timer.csv"),
]


def main() -> None:
    gt = pd.read_csv(OUT_DIR / "test_ground_truth.csv")
    keys = ["session", "cam", "timestamp_sec", "seat_id"]
    summary = []
    for key, label, path in EXPERIMENTS:
        merged = pd.merge(gt, pd.read_csv(path), on=keys, how="inner")
        for scope, sub in [("Overall", merged), ("S5_Easy", merged[merged["session"] == "s5"]), ("S6_Hard", merged[merged["session"] == "s6"])]:
            y_true, y_pred = sub["ground_truth_state"], sub["pred_state"]
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
            summary.append({
                "Model": label, "Scope": scope,
                "Accuracy": round(accuracy_score(y_true, y_pred), 4),
                "Macro-F1": round(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0), 4),
                "Hogging_P": round(p[3], 4), "Hogging_R": round(r[3], 4), "Hogging_F1": round(f1[3], 4),
            })
        cm = confusion_matrix(merged["ground_truth_state"], merged["pred_state"], labels=CLASSES)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(label)
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=20, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(EVAL_DIR / f"cm_{key}.png", dpi=300)
        plt.close(fig)

    df = pd.DataFrame(summary)
    df.to_csv(EVAL_DIR / "summary_comparison.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
