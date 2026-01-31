# train_utils/early_stopping.py
import torch
import os

def check_early_stopping(best_acc, best_metrics,early_stop_counter, best_epoch,
                          model, optimizer, val_metrics, output_dir, patience, epoch):
    """
    At each epoch, save the last model checkpoint and, if improved, save the best model checkpoint.
    Returns updated state and a flag indicating whether training should stop.
    """
    # Always save the last model
    last_path = os.path.join(output_dir, "last_model.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': val_metrics,
    }, last_path)
    print(f"💾 Last model saved at epoch {epoch+1}")

    improved = val_metrics["accuracy"] > best_acc
    if improved:
        best_acc = val_metrics["accuracy"]
        best_metrics = val_metrics
        best_epoch = epoch + 1
        early_stop_counter = 0

        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
        }, os.path.join(output_dir, "best_model.pth"))

        print(f"✅ Best model saved at epoch {best_epoch} with total accuracy: {best_acc:.4f}")
    else:
        early_stop_counter += 1
        print(f"⏳ No improvement. Early stop counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}. Best was at epoch {best_epoch}.")
            return best_acc, best_metrics, best_epoch, early_stop_counter,  True

    return best_acc, best_metrics, best_epoch, early_stop_counter,  False