import os
import json
import pandas as pd
def update_train_logs(train_metrics={},
                      train_loss_list=[],
                      train_loss_energy_list=[],
                      train_loss_alpha_list=[],
                      train_loss_q0_list=[],
                      train_acc_list=[],
                      train_acc_energy_list=[],
                      train_acc_alpha_list=[],
                      train_acc_q0_list=[],
                      ):

    train_loss_list.append(train_metrics['loss'])
    train_loss_energy_list.append(train_metrics['loss_energy'])
    train_loss_alpha_list.append(train_metrics['loss_alpha'])
    train_loss_q0_list.append(train_metrics['loss_q0'])

    train_acc_list.append(train_metrics['accuracy'])
    train_acc_energy_list.append(train_metrics['accuracy_energy'])
    train_acc_alpha_list.append(train_metrics['accuracy_alpha'])
    train_acc_q0_list.append(train_metrics['accuracy_q0'])
    

    return (train_loss_list,
            train_loss_energy_list,
            train_loss_alpha_list,
            train_loss_q0_list,
            train_acc_list,
            train_acc_energy_list,
            train_acc_alpha_list,
            train_acc_q0_list
            )
            
    

def update_val_logs(val_metrics,
                    val_loss_list,
                    val_loss_energy_list,
                    val_loss_alpha_list,
                    val_loss_q0_list,
                    val_acc_list,
                    val_acc_energy_list,
                    val_acc_alpha_list,
                    val_acc_q0_list):
    val_loss_list.append(val_metrics["loss"])
    val_loss_energy_list.append(val_metrics["loss_energy"])
    val_loss_alpha_list.append(val_metrics["loss_alpha"])
    val_loss_q0_list.append(val_metrics["loss_q0"])

    val_acc_list.append(val_metrics["accuracy"])
    val_acc_energy_list.append(val_metrics['energy']['accuracy'])
    val_acc_alpha_list.append(val_metrics['alpha']['accuracy'])
    val_acc_q0_list.append(val_metrics['q0']['accuracy'])

    

    return (val_loss_list,
            val_loss_energy_list,
            val_loss_alpha_list,
            val_loss_q0_list,
            val_acc_list,
            val_acc_energy_list,
            val_acc_alpha_list,
            val_acc_q0_list)

def record_and_save_epoch(epoch, train_metrics, val_metrics, lr, all_epoch_metrics, output_dir):
    epoch_record = {
        "epoch": epoch + 1,
        "learning_rate": lr,
        # Train losses
        "train_loss": train_metrics["loss"],
        "train_loss_energy": train_metrics["loss_energy"],
        "train_loss_alpha": train_metrics["loss_alpha"],
        "train_loss_q0": train_metrics["loss_q0"],

        # Train accuracies
        "train_acc": train_metrics["accuracy"],
        "train_acc_energy": train_metrics["accuracy_energy"],
        "train_acc_alpha": train_metrics["accuracy_alpha"],
        "train_acc_q0": train_metrics["accuracy_q0"],

        # Validation loss
        "val_loss": val_metrics["loss"],
        "val_loss_energy": val_metrics["loss_energy"],
        "val_loss_alpha": val_metrics["loss_alpha"],
        "val_loss_q0": val_metrics["loss_q0"],

        # Validation metrics
        "val_acc": val_metrics["accuracy"],
        "val_energy": val_metrics["energy"],
        "val_alpha": val_metrics["alpha"],
        "val_q0": val_metrics["q0"],
    }
    all_epoch_metrics.append(epoch_record)
    
    print(f"[INFO] Epoch {epoch + 1}: Saving metrics to disk")
    save_metrics_to_disk(all_epoch_metrics, output_dir)
    return all_epoch_metrics

def save_metrics_to_disk(all_epoch_metrics, output_dir):
    # Save to JSON
    with open(os.path.join(output_dir, "epoch_metrics.json"), "w") as f:
        json.dump(all_epoch_metrics, f, indent=2)

    # Save to CSV
    df = pd.DataFrame([
        {
            'epoch': m['epoch'],
            'train_loss': m['train_loss'],
            'train_loss_energy': m['train_loss_energy'],
            'train_loss_alpha': m['train_loss_alpha'],
            'train_loss_q0': m['train_loss_q0'],

            'train_acc': m['train_acc'],
            'train_acc_energy': m['train_acc_energy'],
            'train_acc_alpha': m['train_acc_alpha'],
            'train_acc_q0': m['train_acc_q0'],
            
            'val_loss': m['val_loss'],
            'val_loss_energy': m['val_loss_energy'],
            'val_loss_alpha': m['val_loss_alpha'],
            'val_loss_q0': m['val_loss_q0'],

            'val_acc': m['val_acc'],
            'val_energy_acc': m['val_energy']['accuracy'],
            'val_energy_prec': m['val_energy']['precision'],
            'val_energy_rec': m['val_energy']['recall'],
            'val_energy_f1': m['val_energy']['f1'],
            'val_alpha_acc': m['val_alpha']['accuracy'],
            'val_alpha_prec': m['val_alpha']['precision'],
            'val_alpha_rec': m['val_alpha']['recall'],
            'val_alpha_f1': m['val_alpha']['f1'],
            'val_q0_acc': m['val_q0']['accuracy'],
            'val_q0_prec': m['val_q0']['precision'],
            'val_q0_rec': m['val_q0']['recall'],
            'val_q0_f1': m['val_q0']['f1'],

        }
        for m in all_epoch_metrics
    ])
    df.to_csv(os.path.join(output_dir, "epoch_metrics.csv"), index=False)