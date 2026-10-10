# config.py
import os
import platform
import socket
import getpass
import re
import argparse
import yaml
from types import SimpleNamespace

def get_config(config_path=None):
    # --- Parse command-line args only if not running in notebook ---
    if config_path is None:
        try:
        # --- Parse command-line args ---
            parser = argparse.ArgumentParser(description="ML-JET Model Config")
            parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
            args = parser.parse_args()
            config_path = args.config
        except SystemExit:
            raise ValueError("Must provide config_path explicitly when running in notebook mode.")

    print(f"[INFO] Config Path: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # T02: preserve legacy mode by default. Moment mode reads precomputed splits.
    dataset_type = cfg_dict.get("dataset_type", "legacy")
    if dataset_type not in ("legacy", "moment"):
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")
    moments = cfg_dict.get("moments", [1, 2]) if dataset_type == "moment" else None
    if dataset_type == "moment":
        if not isinstance(moments, list) or not moments or any(type(m) is not int or m not in range(1, 6) for m in moments) or len(set(moments)) != len(moments):
            raise ValueError("moments must be a list of distinct integers between 1 and 5")
    moment_fold = cfg_dict.get("moment_fold", None)
    if dataset_type == "moment" and (moment_fold is not None and (type(moment_fold) is not int or moment_fold not in range(5))):
        raise ValueError("moment_fold must be null or 0..4")
    if dataset_type == "moment" and cfg_dict.get("use_val_folds", False):
        raise ValueError("Use moment_fold (not use_val_folds) for moment dataset mode")

    # --- System/User Info ---
    system = platform.system()
    release = platform.release().lower()
    hostname = socket.gethostname().lower()
    user = getpass.getuser().lower()

    # --- Dataset Path Based on System ---
    if cfg_dict.get("dataset_root") or (dataset_type == "moment" and os.environ.get("DATASET_ROOT")):
        base_path = ""
    elif system == "Linux" and "hm-srv1" in hostname and "arsalan" in user and "wsl" in release :
        base_path = "/mnt/d/Projects/110_JetscapeML/hm_jetscapeml_source/data"
        print("[INFO] Detected WSL environment")
    elif system == "Linux" and "ds044955"  in hostname and "arsalan" in user:
        base_path = "/home/arsalan/wsu_grid_rs_grp_mljet/"
        # base_path = "/home/arsalan/Projects/110_JetscapeML/hm_jetscapeml_source/data"
        # base_path = "/home/arsalan/wsu-grid/hm_jetscapeml_source/data/"
        print("[INFO] Detected native Ubuntu host: DS044955")
    elif system == "Linux" and "ds028160" in hostname and "arsalan" in user:
        base_path = "/mnt/Research/arsalan/wsu_grid_rs_grp_mljet/ml-jet-ext/"
        # base_path = "/home/arsalan/Projects/110_JetscapeML/hm_jetscapeml_source/data"
        # base_path = "/home/arsalan/wsu-grid/hm_jetscapeml_source/data/"
        print("[INFO] Detected native Mint host: DS028160")
    elif system == "Linux" and "gy4065" in user:
        base_path = "/rs/rs_grp_mljet/ml-jet-ext/" 
        #in previous research it was /wsu/home/gy/gy40/gy4065/hm_jetscapeml_source/data
        print("[INFO] Detected WSU Grid environment (user: gy4065)")
    else:
        raise RuntimeError("\u274c Unknown system. Please define the dataset path for this host.")

    # --- Dataset Subdir ---
    dataset_subdir = cfg_dict.get("dataset_subdir")
    dataset_root_override = cfg_dict.get("dataset_root") or (os.environ.get("DATASET_ROOT") if dataset_type == "moment" else None)
    dataset_root_dir = os.path.abspath(os.path.expanduser(str(dataset_root_override))) if dataset_root_override else os.path.join(base_path, dataset_subdir)
    print(f"[INFO] Using dataset root: {dataset_root_dir}")

    # --- Dataset Size ---
    if "dataset_size" in cfg_dict:
        dataset_size = int(cfg_dict["dataset_size"])
        print(f"[INFO] Using dataset_size from config: {dataset_size}")
    else:
        match = re.search(r"size_(\d+)", dataset_root_dir)
        dataset_size = match.group(1) if match else "unknown"
        print(f"[INFO] Extracted dataset_size from path: {dataset_size}")

    # match = re.search(r"size_(\d+)", dataset_root_dir)
    # dataset_size = match.group(1) if match else "unknown"
    # print(f"[INFO] Detected dataset size: {dataset_size}")

    # --- Model Configs ---
    model_tag = cfg_dict.get("model_tag")
    backbone = cfg_dict.get("backbone")
    input_shape = tuple(cfg_dict.get("input_shape", [1, 32, 32]))
    if dataset_type == "moment":
        expected_shape = (len(moments), 32, 32)
        if "input_shape" in cfg_dict and input_shape != expected_shape:
            raise ValueError(f"Moment input_shape should be {expected_shape}; got {input_shape}")
        input_shape = expected_shape
    batch_size = cfg_dict.get("batch_size", 512)
    epochs = cfg_dict.get("epochs", 50)
    learning_rate = cfg_dict.get("learning_rate", 1e-4)
    patience = cfg_dict.get("patience", 5)
    global_max = cfg_dict.get("global_max", float('nan'))
    output_base = cfg_dict.get("output_dir", "training_output/")
    group_size    = cfg_dict.get("group_size", 1)

    # Set default scheduler settings if not present
    scheduler_defaults = {
        'type': 'ReduceLROnPlateau',
        'mode': 'max',
        'factor': 0.5,
        'patience': 4,
        'verbose': True
    }
    scheduler=cfg_dict.get('scheduler', scheduler_defaults)

    preloaded = ""
    preload_model_path = cfg_dict.get("preload_model_path")
    if preload_model_path:
        preloaded="_preloaded"

    freeze_backbone = cfg_dict.get("freeze_backbone", False)
    frozen_backbone_tag=""
    if freeze_backbone == True:
        frozen_backbone_tag="_frozen"
    else:
        frozen_backbone_tag="_unfrozen"

    initialization_defaults = {
        'type': 'gaussian',
        'mean': 0.0,
        'std': 0.02
    }
    initialization = cfg_dict.get("initialization", initialization_defaults)
    init_tag=""
    if freeze_backbone == False and preloaded=="":
        init_tag = f"_{initialization['type']}"
    
    # after scheduler / preload_model_path parsing
    loss_cfg = cfg_dict.get("loss", {}) or {}
    loss_weights = (loss_cfg.get("weights") or {
        "energy_loss_output": 1.0,
        "alpha_output": 1.0,
        "q0_output": 1.0,
    })
    if not (loss_cfg.get("weights") is None):
        # concatenate loss weights into a string and separate them with __annotations__
        weighted_loss="_weighted_loss{}".format("_".join([f"{k}_{v}" for k, v in loss_weights.items()]))
    else:
        weighted_loss = ""
    # --- Split CSV Paths based on group_size ---
    if group_size > 1 :
        basename     = f"file_labels_aggregated_ds{dataset_size}_g{group_size}"
    else:
        basename     = "file_labels"
    train_csv    = os.path.join(dataset_root_dir, f"{basename}_train.csv")
    val_csv      = os.path.join(dataset_root_dir, f"{basename}_val.csv")
    test_csv     = os.path.join(dataset_root_dir, f"{basename}_test.csv")

    # Optional: fold-aware routing over the OLD validation set
    use_val_folds = bool(cfg_dict.get("use_val_folds", False))
    test_csv_same_as_val_fold = bool(cfg_dict.get("test_csv_same_as_val_fold", False))
    if use_val_folds and dataset_type == "legacy":
        fold_index = cfg_dict.get("fold_index", None)
        if fold_index is None:
            raise ValueError("use_val_folds=True requires 'fold_index' (0..n_splits-1) in the config.")

        # Your convention:
        #  - aggregated: file_labels_aggregated_ds{dataset_size}_g{group_size}_val_folds_out
        #  - non-aggregated: file_labels_val_folds_out
        if group_size > 1:
            folds_dir = f"{basename}_val_folds_out"
        else:
            folds_dir = "c0005_alpha_q0_agg500_14000jets_5folds"

        folds_root = os.path.join(dataset_root_dir,"metadata", folds_dir)
        train_csv = os.path.join(folds_root, f"c0005_alpha_q0_agg500_14000jets_fold{fold_index}_train.csv")
        val_csv   = os.path.join(folds_root, f"c0005_alpha_q0_agg500_14000jets_fold{fold_index}_test.csv")

        # For pure CV you can either:
        #  (a) keep the original test set, or
        #  (b) evaluate the held-out fold as "test" per run:
        if test_csv_same_as_val_fold:
            test_csv = val_csv
        else:
            test_csv=""

    if dataset_type == "moment":
        split_root = os.path.join(dataset_root_dir, "splits")
        train_csv = os.path.join(split_root, "train.csv" if moment_fold is None else f"five_fold/fold_{moment_fold}_train.csv")
        val_csv = os.path.join(split_root, "val.csv" if moment_fold is None else f"five_fold/fold_{moment_fold}_val.csv")
        test_csv = os.path.join(split_root, "test.csv")
        loss_weights = {"alpha_output": 1.0, "q0_output": 1.0} if not loss_cfg.get("weights") else loss_weights
        if "energy_loss_output" in loss_weights:
            raise ValueError("Moment dataset has no energy_loss_output target")

    scheduler_type = scheduler.get('type', 'NoScheduler')
    run_tag = f"{model_tag}_bs{batch_size}_ep{epochs}_lr{learning_rate:.0e}_ds{dataset_size}_g{group_size}_sched_{scheduler_type}{preloaded}{frozen_backbone_tag}{init_tag}{weighted_loss}"
    if dataset_type == "moment":
        run_tag += "_moments_" + "-".join(map(str, moments))
    output_dir = os.path.join(output_base, run_tag)
    # Add fold suffix if using CV folds
    if use_val_folds and dataset_type == "legacy" and fold_index is not None:
        output_dir = os.path.join(output_dir, f"fold_{fold_index}")
        
    if dataset_type == "moment" and moment_fold is not None:
        output_dir = os.path.join(output_dir, f"fold_{moment_fold}")

    return SimpleNamespace(**{
        "dataset_type": dataset_type,
        "moments": moments,
        "moment_fold": moment_fold,
        "num_workers": int(cfg_dict.get("num_workers", 0)),
        "pin_memory": cfg_dict.get("pin_memory", None),
        "seed": int(cfg_dict.get("seed", 42)),
        "model_tag": model_tag,
        "backbone": backbone,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "patience": patience,
        "input_shape": input_shape,
        "global_max": global_max,
        "dataset_root_dir": dataset_root_dir,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
        "output_dir": output_dir,
        "group_size": group_size,
        "scheduler": scheduler,
        "dataset_size": dataset_size,
        "preload_model_path": preload_model_path,
        "loss_weights": loss_weights,
        "freeze_backbone": freeze_backbone,
        "initialization": initialization,
    })

if __name__ == "__main__":
    cfg=get_config()
    print(cfg)

# # config.py
# import os
# import platform
# import socket
# import getpass
# import re
# from dataclasses import dataclass
# import os

# from types import SimpleNamespace


# def get_config():
#     # --- System/User Info ---
#     system = platform.system()
#     release = platform.release().lower()
#     hostname = socket.gethostname().lower()
#     user = getpass.getuser().lower()

#     # --- Dataset Path Based on System ---
#     if system == "Linux" and "wsl" in release and "arsi" in user:
#         base_path = "/mnt/d/Projects/110_JetscapeML/hm_jetscapeml_source/data"
#         print("[INFO] Detected WSL environment")
#     elif system == "Linux" and "ds044955" in hostname and "arsalan" in user:
#         base_path = "/home/arsalan/Projects/110_JetscapeML/hm_jetscapeml_source/data"
#         print("[INFO] Detected native Ubuntu host: DS044955")
#     elif system == "Linux" and "gy4065" in user:
#         base_path = "/wsu/home/gy/gy40/gy4065/hm_jetscapeml_source/data"
#         print("[INFO] Detected WSU Grid environment (user: gy4065)")
#     else:
#         raise RuntimeError("\u274c Unknown system. Please define the dataset path for this host.")

#     # --- Dataset Subdir ---
#     dataset_subdir = "jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled"
#     dataset_root_dir = os.path.join(base_path, dataset_subdir)
#     print(f"[INFO] Using dataset root: {dataset_root_dir}")

#     # --- Extract dataset size from path ---
#     match = re.search(r"size_(\d+)", dataset_root_dir)
#     dataset_size = match.group(1) if match else "unknown"
#     print(f"[INFO] Detected dataset size: {dataset_size}")

#     # === Define all backbone-model_tag pairs to evaluate ===
#     experiments = [
#         {"model_tag": "EfficientNet", "backbone": "efficientnet"},
#         {"model_tag": "ConvNeXt", "backbone": "convnext"},
#         {"model_tag": "SwinTransformerV2", "backbone": "swin"},
#         {"model_tag": "Mamba", "backbone": "mamba"},
#         {"model_tag": "VisionMamba", "backbone": "vision_mamba"},
#     ]
#     model_idx = 0
#     model_tag = experiments[model_idx]["model_tag"]
#     backbone = experiments[model_idx]["backbone"]
#     # ==========================================================
#     print(f"[INFO] model tag: {model_tag}, backbone: {backbone}")

#     input_shape=(1, 32, 32)
#     batch_size = 512
#     epochs = 50
#     learning_rate = 1e-4
#     patience= 5
#     global_max = 121.79151153564453
#     output_dir = 'training_output/'
#     # Build dynamic output directory
#     run_tag = f"{model_tag}_bs{batch_size}_ep{epochs}_lr{learning_rate:.0e}_ds{dataset_size}"
    

#     # --- Config Dictionary ---
#     return SimpleNamespace(**{
#         "model_tag": model_tag,
#         "backbone": backbone,
#         "batch_size": batch_size,
#         "epochs": epochs,
#         "learning_rate": learning_rate,
#         "patience": patience,
#         "input_shape": input_shape,
#         "global_max": global_max,
#         "dataset_root_dir": dataset_root_dir,
#         "train_csv": os.path.join(dataset_root_dir, "train_files.csv"),
#         "val_csv": os.path.join(dataset_root_dir, "val_files.csv"),
#         "test_csv": os.path.join(dataset_root_dir, "test_files.csv"),
#         "output_dir": os.path.join(output_dir, run_tag)
#     })

