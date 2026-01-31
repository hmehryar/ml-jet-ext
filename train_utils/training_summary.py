# train_utils/training_summary.py
import os
import json
import datetime

def init_training_summary(cfg):
    summary = {
        "model_tag": cfg.model_tag,
        "backbone": cfg.backbone,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "dataset_root_dir": cfg.dataset_root_dir,
        "global_max": cfg.global_max,
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_status": "interrupted_or_incomplete",
        "metrics_file": os.path.join(cfg.output_dir, "epoch_metrics.json"),
        "best_model_path": os.path.join(cfg.output_dir, "best_model.pth")
    }
    save_training_summary(summary, cfg.output_dir)
    return summary

def finalize_training_summary(summary, best_epoch, best_acc, best_metrics, output_dir):
    summary.update({
        "best_epoch": best_epoch,
        "best_accuracy": best_acc,
        "summary_status": "completed",
        "end_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_metrics": best_metrics
    })
    save_training_summary(summary, output_dir)


def save_training_summary(summary, output_dir):
    path = os.path.join(output_dir, "training_summary.json")
    # build the path if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] 📄 Training summary saved to: {path}")

def print_best_model_summary(best_epoch, best_acc, best_metrics):
    print(f"\n🏁 Best Model @ Epoch {best_epoch}")
    print(f"Total Accuracy: {best_acc:.4f}")
    for task in ['energy', 'alpha', 'q0']:
        print(f"\n🔹 {task.upper()} Task")
        print(f"  Accuracy : {best_metrics[task]['accuracy']:.4f}")
        print(f"  Precision: {best_metrics[task]['precision']:.4f}")
        print(f"  Recall   : {best_metrics[task]['recall']:.4f}")
        print(f"  F1-Score : {best_metrics[task]['f1']:.4f}")
        print(f"  Confusion Matrix:\n{best_metrics[task]['confusion_matrix']}")
