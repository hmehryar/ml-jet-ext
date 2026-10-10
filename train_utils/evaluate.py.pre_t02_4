# train_utils/evaluate.py
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import os
import numpy as np

# def evaluate(loader, model, criterion, device):
#     model.eval()
#     y_true = {'energy': [], 'alpha': [], 'q0': []}
#     y_pred = {'energy': [], 'alpha': [], 'q0': []}
#     correct_all = 0
#     total = 0

#     val_loss_total = 0.0
#     val_loss_energy = 0.0
#     val_loss_alpha = 0.0
#     val_loss_q0 = 0.0

#     with torch.no_grad():
#         for x, labels in loader:
#             x = x.to(device)
#             for key in labels:
#                 labels[key] = labels[key].to(device)

#             outputs = model(x)

#             # Energy loss: binary thresholding
#             # pred_energy = (outputs['energy_loss_output'] > 0.5).long().squeeze()
#             energy_logits = outputs['energy_loss_output'].squeeze()
#             pred_energy = (torch.sigmoid(energy_logits) > 0.5).long()
            
#             pred_alpha = torch.argmax(outputs['alpha_output'], dim=1)
#             pred_q0 = torch.argmax(outputs['q0_output'], dim=1)

#             gt_energy = labels['energy_loss_output'].squeeze()
#             gt_alpha = labels['alpha_output'].squeeze()
#             gt_q0 = labels['q0_output'].squeeze()

#             y_true['energy'].extend(labels['energy_loss_output'].squeeze().cpu().numpy())
#             y_true['alpha'].extend(labels['alpha_output'].squeeze().cpu().numpy())
#             y_true['q0'].extend(labels['q0_output'].squeeze().cpu().numpy())

#             y_pred['energy'].extend(pred_energy.cpu().numpy())
#             y_pred['alpha'].extend(pred_alpha.cpu().numpy())
#             y_pred['q0'].extend(pred_q0.cpu().numpy())

#             # Total accuracy = all 3 correct
#             correct_batch = ((pred_energy == gt_energy) &
#                              (pred_alpha == gt_alpha) &
#                              (pred_q0 == gt_q0)).sum().item()
#             correct_all += correct_batch
#             total += x.size(0)

#             # Compute loss per task
#             energy_out = outputs['energy_loss_output'].squeeze()
#             alpha_out = outputs['alpha_output']
#             q0_out = outputs['q0_output']

#             loss_energy = criterion['energy_loss_output'](energy_out, gt_energy.float())
#             loss_alpha = criterion['alpha_output'](alpha_out, gt_alpha)
#             loss_q0 = criterion['q0_output'](q0_out, gt_q0)

#             val_loss_energy += loss_energy.item()
#             val_loss_alpha += loss_alpha.item()
#             val_loss_q0 += loss_q0.item()
#             val_loss_total += (loss_energy + loss_alpha + loss_q0).item()

            
#     # Compute individual accuracies
#     acc_total = correct_all / total
#     avg_loss_energy = val_loss_energy / len(loader)
#     avg_loss_alpha = val_loss_alpha / len(loader)
#     avg_loss_q0 = val_loss_q0 / len(loader)
#     avg_val_loss = val_loss_total / len(loader)

#     # cm_energy = confusion_matrix(y_true["energy"], y_pred["energy"])
#     # cm_alpha = confusion_matrix(y_true["alpha"], y_pred["alpha"])
#     # cm_q0 = confusion_matrix(y_true["q0"], y_pred["q0"])

#     # All metrics + losses in one dict
#     metrics = {
#         "loss": avg_val_loss,
#         "loss_energy": avg_loss_energy,
#         "loss_alpha": avg_loss_alpha,
#         "loss_q0": avg_loss_q0,
#         "accuracy": acc_total,
#         "energy": {
            
#             "accuracy": accuracy_score(y_true['energy'], y_pred['energy']),
#             "precision": precision_score(y_true['energy'], y_pred['energy'], average='macro',zero_division=0),
#             "recall": recall_score(y_true['energy'], y_pred['energy'], average='macro'),
#             "f1": f1_score(y_true['energy'], y_pred['energy'], average='macro'),
#             "confusion_matrix": confusion_matrix(y_true["energy"], y_pred["energy"]).tolist(),
#             # "confusion_matrix": cm_energy.tolist()
#         },
#         "alpha": {
#             "accuracy": accuracy_score(y_true['alpha'], y_pred['alpha']),
#             "precision": precision_score(y_true['alpha'], y_pred['alpha'], average='macro',zero_division=0),
#             "recall": recall_score(y_true['alpha'], y_pred['alpha'], average='macro'),
#             "f1": f1_score(y_true['alpha'], y_pred['alpha'], average='macro'),
#             "confusion_matrix": confusion_matrix(y_true["alpha"], y_pred["alpha"]).tolist(),
#             # "confusion_matrix": cm_alpha.tolist()
#         },
#         "q0": {
#             "accuracy": accuracy_score(y_true['q0'], y_pred['q0']),
#             "precision": precision_score(y_true['q0'], y_pred['q0'], average='macro',zero_division=0),
#             "recall": recall_score(y_true['q0'], y_pred['q0'], average='macro'),
#             "f1": f1_score(y_true['q0'], y_pred['q0'], average='macro'),
#             "confusion_matrix": confusion_matrix(y_true["q0"], y_pred["q0"]).tolist(),
#             # "confusion_matrix": cm_q0.tolist()
#         }
#     }

#     return metrics


from train_utils.prob_utils import collect_head_probs, plot_prob_heatmap, save_probs_csv
from train_utils.prob_utils import plot_alpha_histograms, plot_alpha_avgprob_histograms, plot_q0_avgprob_histograms

def evaluate(loader, model, criterion, device, loss_weights=None,*,
              make_alpha_fig=False, alpha_fig_path=None, alpha_class_names=("0.2","0.3","0.4"),
              make_alpha_avgprob_fig=False, alpha_avgprob_fig_path=None,
              make_q0_avgprob_fig=False, q0_avgprob_fig_path=None, q0_class_names=("1.0","1.5","2.0","2.5")):
    model.eval()
    y_true = {'energy': [], 'alpha': [], 'q0': []}
    y_pred = {'energy': [], 'alpha': [], 'q0': []}
    
    # NEW: collect α_s probabilities so we can pass to the prob-hist method
    alpha_proba_rows = []   # will become [N, C] numpy

    # NEW: collect Q0 probabilities (softmax) to build avg-prob plots
    q0_proba_rows = []     # will become [N, 4] numpy

    correct_all = 0
    total = 0

    val_loss_total = 0.0
    val_loss_energy = 0.0
    val_loss_alpha = 0.0
    val_loss_q0 = 0.0

    loss_weights = loss_weights or {}
    w_energy = float(loss_weights.get("energy_loss_output", 1.0))
    w_alpha  = float(loss_weights.get("alpha_output", 1.0))
    w_q0     = float(loss_weights.get("q0_output", 1.0))

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            for key in labels:
                labels[key] = labels[key].to(device)

            outputs = model(x)

            # Energy loss: binary thresholding
            # pred_energy = (outputs['energy_loss_output'] > 0.5).long().squeeze()
            energy_logits = outputs['energy_loss_output'].squeeze()
            pred_energy = (torch.sigmoid(energy_logits) > 0.5).long()

            alpha_logits = outputs['alpha_output']
            pred_alpha = torch.argmax(alpha_logits, dim=1)
            # alpha_proba = torch.softmax(alpha_logits, dim=1)
            # alpha_proba_rows.append(alpha_proba.cpu().numpy())

            # pred_q0 = torch.argmax(outputs['q0_output'], dim=1)
            q0_logits = outputs['q0_output']
            pred_q0=torch.argmax(q0_logits,dim=1)
            # q0_proba=torch.softmax(q0_logits,dim=1)
            # q0_proba_rows.append(q0_proba.cpu().numpy())

            gt_energy = labels['energy_loss_output'].squeeze()
            gt_alpha = labels['alpha_output'].squeeze()
            gt_q0 = labels['q0_output'].squeeze()

            y_true['energy'].extend(labels['energy_loss_output'].squeeze().cpu().numpy())
            y_true['alpha'].extend(labels['alpha_output'].squeeze().cpu().numpy())
            y_true['q0'].extend(labels['q0_output'].squeeze().cpu().numpy())

            y_pred['energy'].extend(pred_energy.cpu().numpy())
            y_pred['alpha'].extend(pred_alpha.cpu().numpy())
            y_pred['q0'].extend(pred_q0.cpu().numpy())

            # Total accuracy = all 3 correct
            correct_batch = ((pred_energy == gt_energy) &
                             (pred_alpha == gt_alpha) &
                             (pred_q0 == gt_q0)).sum().item()
            correct_all += correct_batch
            total += x.size(0)

            # Compute loss per task
            energy_out = outputs['energy_loss_output'].squeeze()
            alpha_out = outputs['alpha_output']
            q0_out = outputs['q0_output']

            loss_energy = criterion['energy_loss_output'](energy_out, gt_energy.float())
            loss_alpha = criterion['alpha_output'](alpha_out, gt_alpha)
            loss_q0 = criterion['q0_output'](q0_out, gt_q0)

            total_batch_loss = (w_energy * loss_energy
                                + w_alpha  * loss_alpha
                                + w_q0     * loss_q0)
            
            val_loss_energy += loss_energy.item()
            val_loss_alpha += loss_alpha.item()
            val_loss_q0 += loss_q0.item()
            # val_loss_total += (loss_energy + loss_alpha + loss_q0).item()
            val_loss_total += total_batch_loss.item()

    # # Aggregate α_s probabilities (N, C)
    # if len(alpha_proba_rows):
    #     y_alpha_proba = np.concatenate(alpha_proba_rows, axis=0)
    # else:
    #     y_alpha_proba = np.zeros((0, len(alpha_class_names)), dtype=float)

    # # Aggregate Q0 probabilities (N, C)
    # if len(q0_proba_rows):
    #     y_q0_proba = np.concatenate(q0_proba_rows, axis=0)
    # else:
    #     y_q0_proba = np.zeros((0, len(q0_class_names)), dtype=float)



    # Compute individual accuracies
    acc_total = correct_all / total
    avg_loss_energy = val_loss_energy / len(loader)
    avg_loss_alpha = val_loss_alpha / len(loader)
    avg_loss_q0 = val_loss_q0 / len(loader)
    avg_val_loss = val_loss_total / len(loader)

    # cm_energy = confusion_matrix(y_true["energy"], y_pred["energy"])
    # cm_alpha = confusion_matrix(y_true["alpha"], y_pred["alpha"])
    # cm_q0 = confusion_matrix(y_true["q0"], y_pred["q0"])

    # All metrics + losses in one dict
    metrics = {
        "loss": avg_val_loss,
        "loss_energy": avg_loss_energy,
        "loss_alpha": avg_loss_alpha,
        "loss_q0": avg_loss_q0,
        "accuracy": acc_total,
        "energy": {
            
            "accuracy": accuracy_score(y_true['energy'], y_pred['energy']),
            "precision": precision_score(y_true['energy'], y_pred['energy'], average='macro',zero_division=0),
            "recall": recall_score(y_true['energy'], y_pred['energy'], average='macro',zero_division=0),
            "f1": f1_score(y_true['energy'], y_pred['energy'], average='macro',zero_division=0),
            "confusion_matrix": confusion_matrix(y_true["energy"], y_pred["energy"]).tolist(),
            # "confusion_matrix": cm_energy.tolist()
        },
        "alpha": {
            "accuracy": accuracy_score(y_true['alpha'], y_pred['alpha']),
            "precision": precision_score(y_true['alpha'], y_pred['alpha'], average='macro',zero_division=0),
            "recall": recall_score(y_true['alpha'], y_pred['alpha'], average='macro',zero_division=0),
            "f1": f1_score(y_true['alpha'], y_pred['alpha'], average='macro',zero_division=0),
            "confusion_matrix": confusion_matrix(y_true["alpha"], y_pred["alpha"]).tolist(),
            # "confusion_matrix": cm_alpha.tolist()
        },
        "q0": {
            "accuracy": accuracy_score(y_true['q0'], y_pred['q0']),
            "precision": precision_score(y_true['q0'], y_pred['q0'], average='macro',zero_division=0),
            "recall": recall_score(y_true['q0'], y_pred['q0'], average='macro',zero_division=0),
            "f1": f1_score(y_true['q0'], y_pred['q0'], average='macro',zero_division=0),
            "confusion_matrix": confusion_matrix(y_true["q0"], y_pred["q0"]).tolist(),
            # "confusion_matrix": cm_q0.tolist()
        },
        # # expose α_s soft info so notebooks can reuse without re-running evaluation
        # "alpha_soft": {
        #     "y_true_alpha": np.asarray(y_true['alpha']).tolist(),
        #     "y_alpha_proba": y_alpha_proba.tolist(),
        #     "class_names": list(alpha_class_names),
        # },
        # # expose q_0 soft info so notebooks can reuse without re-running evaluation
        # "q0_soft": {
        #     "y_true_q0": np.asarray(y_true['q0']).tolist(),
        #     "y_q0_proba": y_q0_proba.tolist(),
        #     "class_names": list(q0_class_names),
        # }
    }
    
    # Collect softmax probability tables only if needed (avoids extra pass otherwise)
    need_soft = make_alpha_avgprob_fig or make_q0_avgprob_fig
    if need_soft:
        (alpha_probs_soft, alpha_true_soft), (q0_probs_soft, q0_true_soft) = collect_head_probs(
            loader, model, device, alpha_key="alpha_output", q0_key="q0_output"
        )
        alpha_csv_path =save_probs_csv(
            path_prefix=alpha_fig_path,
            probs=alpha_probs_soft,
            y_true=alpha_true_soft,
            class_names=alpha_class_names
        )
        metrics["alpha_probs_csv"] = str(alpha_csv_path)
        q0_csv_path = save_probs_csv(
            path_prefix=q0_avgprob_fig_path,
            probs=q0_probs_soft,
            y_true=q0_true_soft,
            class_names=q0_class_names
        )
        metrics["q0_probs_csv"] = str(q0_csv_path)
    else:
        alpha_probs_soft = np.zeros((0, len(alpha_class_names)), dtype=float)
        q0_probs_soft    = np.zeros((0, len(q0_class_names)), dtype=float)
    # --- NEW: optionally create α_s-focused figure here ---

    if make_alpha_fig:
        plot_alpha_histograms(
            y_true_alpha=y_true['alpha'],
            y_pred_alpha=y_pred['alpha'],
            class_names=[rf"$\alpha_s={c}$" for c in alpha_class_names],
            save_path=alpha_fig_path,
            show=False  # toggle True in notebooks if you want it displayed immediately
        )
        
        if alpha_fig_path:
            base = os.path.splitext(alpha_fig_path)[0]
            metrics["alpha_hist_path"] = base + ".png"

    if make_alpha_avgprob_fig:
        metrics["alpha_hist_path"] = alpha_fig_path  # report where it was saved (if any)
        plot_alpha_avgprob_histograms(
            y_true_alpha=y_true['alpha'],
            # y_alpha_proba=y_alpha_proba,
            y_alpha_proba=alpha_probs_soft,

            class_names=[rf"$\alpha_s={c}$" for c in alpha_class_names],
            save_path=alpha_avgprob_fig_path,
            show=False
        )
        metrics["alpha_avgprob_hist_path"] = (alpha_avgprob_fig_path + ".png") if alpha_avgprob_fig_path else None
        
        plot_prob_heatmap(
            probs=alpha_probs_soft,
            y_true=alpha_true_soft,
            class_names=[rf" $\alpha_s={c}$ " for c in alpha_class_names],
            title=rf"$\alpha_s$: mean predicted distribution per true class",
            save_path=alpha_avgprob_fig_path + "_heat_map",
            show=False
        )
        metrics["alpha_heatmap_path"] = (alpha_avgprob_fig_path + "_heat_map.png") if alpha_avgprob_fig_path else None

    if make_q0_avgprob_fig:
        plot_q0_avgprob_histograms(
            y_true_q0=y_true['q0'],
            # y_q0_proba=y_q0_proba,
            y_q0_proba=q0_probs_soft,
            class_names=[rf"$Q_0={c}$" for c in q0_class_names],
            save_path=q0_avgprob_fig_path,
            show=False
        )
        metrics["q0_avgprob_hist_path"] = (q0_avgprob_fig_path + ".png") if q0_avgprob_fig_path else None
        plot_prob_heatmap(
            probs=q0_probs_soft,
            # y_true=y_true['q0'],
            y_true=q0_true_soft,
            class_names=[rf" $Q_0={c}$ " for c in q0_class_names],
            title=rf"$Q_0$: mean predicted distribution per true class",
            save_path=q0_avgprob_fig_path + "_heat_map",
            show=False
        )
        metrics["q0_heatmap_path"] = (q0_avgprob_fig_path + "_heat_map.png") if q0_avgprob_fig_path else None

    return metrics

