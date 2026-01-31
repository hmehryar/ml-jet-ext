from train_utils.train_epoch import train_one_epoch
from train_utils.train_metrics_logger import update_train_logs
from train_utils.evaluate import evaluate
from train_utils.train_metrics_logger import update_val_logs
from train_utils.train_metrics_logger import record_and_save_epoch
from train_utils.early_stopping import check_early_stopping

def run_training_loop(cfg,train_loader,val_loader,
                      device, model,criterion,
                      optimizer,scheduler,
                      start_epoch,early_stop_counter,
                      best_acc,best_metrics,best_epoch,
                      train_loss_list,
                        train_loss_energy_list,
                        train_loss_alpha_list,
                        train_loss_q0_list,
                        train_acc_list,
                        train_acc_energy_list,
                        train_acc_alpha_list,
                        train_acc_q0_list,
                        val_loss_list,
                        val_loss_energy_list,
                        val_loss_alpha_list,
                        val_loss_q0_list,
                        val_acc_list,
                        val_acc_energy_list,
                        val_acc_alpha_list,
                        val_acc_q0_list,
                        all_epoch_metrics):
    for epoch in range(start_epoch, cfg.epochs):
        print(f"[INFO] Epoch {epoch+1}/{cfg.epochs}")
        train_metrics={}
        train_metrics = train_one_epoch(train_loader, model, criterion, optimizer, device,
                                        loss_weights=cfg.loss_weights)
        (train_loss_list,
        train_loss_energy_list,
        train_loss_alpha_list,
        train_loss_q0_list,
        train_acc_list,
        train_acc_energy_list,
        train_acc_alpha_list,
        train_acc_q0_list
        ) = update_train_logs(
            train_metrics,
            train_loss_list,
            train_loss_energy_list,
            train_loss_alpha_list,
            train_loss_q0_list,
            train_acc_list,
            train_acc_energy_list,
            train_acc_alpha_list,
            train_acc_q0_list
        )
        # ensure evaluate() computes weighted total loss with same weights
        val_metrics = evaluate(val_loader, model, criterion, device, loss_weights=cfg.loss_weights)
        (val_loss_list,
        val_loss_energy_list,
        val_loss_alpha_list,
        val_loss_q0_list,
        val_acc_list,
        val_acc_energy_list,
        val_acc_alpha_list,
        val_acc_q0_list,
        ) = update_val_logs(
            val_metrics,
            val_loss_list,
            val_loss_energy_list,
            val_loss_alpha_list,
            val_loss_q0_list,
            val_acc_list,
            val_acc_energy_list,
            val_acc_alpha_list,
            val_acc_q0_list,
        )
        print(f"[INFO] Epoch {epoch+1}: Energy Acc ={val_metrics['energy']['accuracy']:.4f}, αs Acc = {val_metrics['alpha']['accuracy']:.4f}, Q0 Acc = {val_metrics['q0']['accuracy']:.4f}, Total Acc = {val_metrics['accuracy']:.4f}")
        print(f"[INFO] Epoch {epoch+1}: Energy Loss ={val_metrics['loss_energy']:.4f}, αs Loss = {val_metrics['loss_alpha']:.4f}, Q0 Loss = {val_metrics['loss_q0']:.4f}, Total Loss = {val_metrics['loss']:.4f}")
        
        scheduler.step(val_metrics['accuracy'])  # or macro average accuracy if defined
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print(f"📉 Current LR: {current_lr}")
        
        all_epoch_metrics=record_and_save_epoch(epoch, train_metrics, val_metrics, current_lr, all_epoch_metrics, cfg.output_dir)
        
        # save_epoch_checkpoint(
        #     epoch=epoch,
        #     model=model,
        #     optimizer=optimizer,
        #     metrics=val_metrics,
        #     output_dir=cfg.output_dir
        # )

        best_acc, best_metrics, best_epoch, early_stop_counter, should_stop = check_early_stopping(
            best_acc=best_acc,
            best_metrics=best_metrics,
            early_stop_counter=early_stop_counter,
            best_epoch=best_epoch,
            model=model,
            optimizer=optimizer,
            val_metrics=val_metrics,
            output_dir=cfg.output_dir,
            patience=cfg.patience,
            epoch=epoch
        )
        
        if should_stop:
            break
        
        print("="*150)
    return best_epoch,best_acc,best_metrics
        

#################################
# --- High-level adapter for HPO/CLI ---

def run_training_with_config(cfg, trial_callback=None):
    """
    High-level convenience wrapper that:
    - builds loaders/model/criterion/optimizer/scheduler from cfg
    - initializes tracking lists
    - calls the low-level run_training_loop(...)
    - returns final validation metrics
    """
    import torch
    from train_utils.evaluate import evaluate
    from train_utils.train_epoch import train_one_epoch
    # from train_utils.scheduler_utils import build_scheduler  # if you have one
    from data.loader import get_dataloaders
    from train_utils.scheduler_utils import create_scheduler
    # ^ adjust imports to your actual file structure

    # 1) device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) data loaders (reuse your existing builder)
    # Replace the following with your actual dataset/loader creation
    train_loader, val_loader, test_loader = get_dataloaders(cfg, device=device)

    # 3) model + criterion
    model, criterion, optimizer = build_model_optimizer_and_criterion_from_cfg(cfg, device)

    # 4) optimizer + scheduler
    # optimizer = build_optimizer_from_cfg(cfg, model.parameters())
    scheduler = create_scheduler(optimizer, cfg) if 'scheduler' in globals() or 'create_scheduler' in dir() else None

    # 5) bookkeeping / state
    start_epoch = 0
    early_stop_counter = 0
    best_acc = 0.0
    best_metrics = None
    best_epoch = -1

    train_loss_list = []
    train_loss_energy_list, train_loss_alpha_list, train_loss_q0_list = [], [], []
    train_acc_list = []
    train_acc_energy_list, train_acc_alpha_list, train_acc_q0_list = [], [], []

    val_loss_list = []
    val_loss_energy_list, val_loss_alpha_list, val_loss_q0_list = [], [], []
    val_acc_list = []
    val_acc_energy_list, val_acc_alpha_list, val_acc_q0_list = [], [], []

    all_epoch_metrics = []

    # 6) call your existing low-level loop
    # final_val_metrics = run_training_loop(
    #     train_loader, val_loader, device, model, criterion,
    #     optimizer, scheduler, start_epoch, early_stop_counter,
    #     best_acc, best_metrics, best_epoch,
    #     train_loss_list, train_loss_energy_list, train_loss_alpha_list, train_loss_q0_list,
    #     train_acc_list, train_acc_energy_list, train_acc_alpha_list, train_acc_q0_list,
    #     val_loss_list, val_loss_energy_list, val_loss_alpha_list, val_loss_q0_list,
    #     val_acc_list, val_acc_energy_list, val_acc_alpha_list, val_acc_q0_list,
    #     all_epoch_metrics,
    #     # trial_callback=trial_callback,             # <- only if your function supports it
    #     # loss_weights=getattr(cfg, "loss_weights", None)  # pass λs if supported
    # )
    best_epoch,best_acc,final_val_metrics = run_training_loop(cfg,train_loader,val_loader,
                      device, model,criterion,
                      optimizer,scheduler,
                      start_epoch,early_stop_counter,
                      best_acc,best_metrics,best_epoch,
                      train_loss_list,
                        train_loss_energy_list,
                        train_loss_alpha_list,
                        train_loss_q0_list,
                        train_acc_list,
                        train_acc_energy_list,
                        train_acc_alpha_list,
                        train_acc_q0_list,
                        val_loss_list,
                        val_loss_energy_list,
                        val_loss_alpha_list,
                        val_loss_q0_list,
                        val_acc_list,
                        val_acc_energy_list,
                        val_acc_alpha_list,
                        val_acc_q0_list,
                        all_epoch_metrics)
    return final_val_metrics

import torch
# --- Helper builders (STUBS) ---
def build_loaders_from_cfg(cfg):
    """Replace this stub with your actual DataLoader construction."""
    # Example:
    # train_ds = JetDataset(cfg, split="train")
    # val_ds   = JetDataset(cfg, split="val")
    # train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    # val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    # return train_loader, val_loader
    raise NotImplementedError("Implement build_loaders_from_cfg(cfg) for your dataset.")

from models.model_vit import create_model
from torch import nn
def build_model_optimizer_and_criterion_from_cfg(cfg, device):
    model, optimizer = create_model(cfg.backbone, cfg.input_shape, cfg.learning_rate)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Parallelizing model across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    elif torch.cuda.device_count() == 1:
        print("No parallelization, using single GPU")
    elif torch.cuda.device_count() == 0:
        print("No GPU available, using CPU")

    criterion = {
        # 'energy_loss_output': nn.BCELoss(),
        'energy_loss_output': nn.BCEWithLogitsLoss(),
        'alpha_output': nn.CrossEntropyLoss(),
        'q0_output': nn.CrossEntropyLoss()
    }
    print(f"[INFO] Loss functions:{criterion}")
    return model, criterion, optimizer
    """Replace this stub with your actual model + criterion creation."""
    # model = build_model(cfg).to(device)
    # criterion = {
    #   "energy_loss_output": BCEWithLogitsLoss(),
    #   "alpha_output":       CrossEntropyLoss(),
    #   "q0_output":          CrossEntropyLoss(),
    # }
    # return model, criterion

    raise NotImplementedError("Implement build_model_and_criterion_from_cfg(cfg, device).")

def build_optimizer_from_cfg(cfg, params):
    """Replace this stub with your actual optimizer."""
    # import torch.optim as optim
    # return optim.Adam(params, lr=cfg.learning_rate, weight_decay=getattr(cfg,"weight_decay",0.0))
    raise NotImplementedError("Implement build_optimizer_from_cfg(cfg, params).")

