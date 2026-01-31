# train_utils/resume.py
import datetime
import os
import torch
import json
from train_utils.training_summary import init_training_summary
from numpy.core.multiarray import scalar
torch.serialization.add_safe_globals([scalar])

def init_resume_state(model, optimizer, device,config):
    print(f"[INFO] Init Resume/Training Parameters")
    early_stop_counter,start_epoch,best_acc,best_epoch,best_metrics,all_epoch_metrics,summary_status= 0, 0, 0.0, 0, {}, [],""

    summary_path = os.path.join(config.output_dir, "training_summary.json")
    best_path = os.path.join(config.output_dir, "best_model.pth")
    resume_path = os.path.join(config.output_dir, "last_model.pth")

    best_metrics = {}
    training_summary= {}

    # Try loading resume state
    if os.path.exists(resume_path) and os.path.exists(summary_path):
        print(f"[INFO] 🔁 Resuming training from checkpoint and summary")
        
        # Load model checkpoint
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        # Load the last (most recent) state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        # Load best model metrics for tracking
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            best_epoch = best_ckpt['epoch']
            best_acc = best_ckpt['metrics']['accuracy']
            best_metrics = best_ckpt.get('metrics', {})

        # Load summary info (optional counters/history)
        with open(summary_path, "r") as f:
            training_summary = json.load(f)
            early_stop_counter = training_summary.get("early_stop_counter", 0)
            metric_file_path= training_summary.get("metrics_file", "")
            summary_status=training_summary.get("summary_status","")# "interrupted_or_incomplete",
            if os.path.exists(metric_file_path):
                with open(metric_file_path, "r") as m:
                    all_epoch_metrics = json.load(m)

        print(f"[INFO] Resumed at epoch {start_epoch} with total acc {best_acc:.4f} from epoch {best_epoch} and early stop counter {early_stop_counter}")
    else:
        print(f"[INFO] Starting fresh training run by initializing training summary")
        training_summary=init_training_summary(config)
    return model, optimizer, start_epoch, best_acc, early_stop_counter, best_epoch, best_metrics, training_summary, all_epoch_metrics, summary_status
        
def fill_trackers_from_history(all_epoch_metrics,
                               train_loss_energy_list, train_loss_alpha_list,
                               train_loss_q0_list, train_loss_list,
                               train_acc_energy_list, train_acc_alpha_list,
                               train_acc_q0_list, train_acc_list,
                               val_loss_energy_list, val_loss_alpha_list,
                               val_loss_q0_list, val_loss_list,
                               val_acc_energy_list, val_acc_alpha_list,
                               val_acc_q0_list, val_acc_list,
                               summary_status, best_epoch):
    """
    If summary_status indicates an interrupted/incomplete run,
    extract metrics from all_epoch_metrics and append into the provided lists.
    """
    if summary_status != "interrupted_or_incomplete":
        return

    # # Trim metrics in-place
    # trimmed = [r for r in all_epoch_metrics if r["epoch"] <= best_epoch]
    # all_epoch_metrics[:] = trimmed

    # for record in trimmed:
    for record in all_epoch_metrics:
        # training
        train_loss_energy_list.append(record["train_loss_energy"])
        train_loss_alpha_list.append(record["train_loss_alpha"])
        train_loss_q0_list.append(record["train_loss_q0"])
        train_loss_list.append(record["train_loss"])
        train_acc_energy_list.append(record["train_acc_energy"])
        train_acc_alpha_list.append(record["train_acc_alpha"])
        train_acc_q0_list.append(record["train_acc_q0"])
        train_acc_list.append(record["train_acc"])

        # validation
        val_loss_energy_list.append(record["val_loss_energy"])
        val_loss_alpha_list.append(record["val_loss_alpha"])
        val_loss_q0_list.append(record["val_loss_q0"])
        val_loss_list.append(record["val_loss"])
        val_acc_energy_list.append(record["val_energy"]["accuracy"])
        val_acc_alpha_list.append(record["val_alpha"]["accuracy"])
        val_acc_q0_list.append(record["val_q0"]["accuracy"])
        val_acc_list.append(record["val_acc"])

def load_pretrained_model(model, device, config, strict=False):
    """
    Loads a pretrained model from config.preload_model_path if it exists.
    Only loads model weights (no optimizer state).
    """
    preload_path = getattr(config, "preload_model_path", None)

    if not preload_path:
        print("[INFO] No preload_model_path specified — skipping preload.")
        return model, False

    if not os.path.exists(preload_path):
        print(f"[WARN] preload_model_path specified but file not found: {preload_path}")
        return model, False

    print(f"[INFO] 🔄 Preloading model weights from {preload_path}")
    checkpoint = torch.load(preload_path, map_location=device, weights_only=False)

    # Accept raw state dict or wrapped dict
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=strict)

    print("[INFO] ✅ Pretrained weights loaded successfully.")
    return model, True
