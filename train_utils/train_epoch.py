import torch
from tqdm import tqdm

def train_one_epoch(loader, model, criterion, optimizer, device, loss_weights=None):
    model.train()
    total_loss = 0.0
    loss_energy_total = 0.0
    loss_alpha_total = 0.0
    loss_q0_total = 0.0

    loss_weights = loss_weights or {}
    w_energy = float(loss_weights.get("energy_loss_output", 1.0))
    w_alpha  = float(loss_weights.get("alpha_output", 1.0))
    w_q0     = float(loss_weights.get("q0_output", 1.0))

    correct_energy = 0
    correct_alpha = 0
    correct_q0 = 0
    correct_all = 0
    total = 0

    # scaler = torch.amp.GradScaler('cuda', enabled=True)

    for x, labels in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        for key in labels:
            labels[key] = labels[key].to(device)
        
        
        optimizer.zero_grad()

        # with torch.amp.autocast('cuda', enabled=True):
            # Forward
        outputs = model(x)
        energy_out = outputs['energy_loss_output'].squeeze()
        alpha_out = outputs['alpha_output']
        q0_out = outputs['q0_output']

        # Labels
        gt_energy = labels['energy_loss_output'].squeeze()
        gt_alpha = labels['alpha_output'].squeeze()
        gt_q0 = labels['q0_output'].squeeze()

        # Loss
        loss_energy = criterion['energy_loss_output'](energy_out, gt_energy.float())
        loss_alpha = criterion['alpha_output'](alpha_out, gt_alpha)
        loss_q0 = criterion['q0_output'](q0_out, gt_q0)
        # total_batch_loss = loss_energy + loss_alpha + loss_q0
        total_batch_loss = (w_energy * loss_energy
                            + w_alpha  * loss_alpha
                            + w_q0     * loss_q0)

        total_batch_loss.backward()
        optimizer.step()
        # scaler.scale(total_batch_loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # running_loss += loss.item()
        total_loss += total_batch_loss.item()
        loss_energy_total += loss_energy.item()
        loss_alpha_total += loss_alpha.item()
        loss_q0_total += loss_q0.item()

        # Accuracy
        pred_energy = (energy_out > 0.5).long()
        pred_alpha = torch.argmax(alpha_out, dim=1)
        pred_q0 = torch.argmax(q0_out, dim=1)

        correct_energy += (pred_energy == gt_energy).sum().item()
        correct_alpha += (pred_alpha == gt_alpha).sum().item()
        correct_q0 += (pred_q0 == gt_q0).sum().item()

        correct_all += ((pred_energy == gt_energy) &
                        (pred_alpha == gt_alpha) &
                        (pred_q0 == gt_q0)).sum().item()

        total += x.size(0)

    return {
        'loss': total_loss / len(loader),
        'loss_energy': loss_energy_total / len(loader),
        'loss_alpha': loss_alpha_total / len(loader),
        'loss_q0': loss_q0_total / len(loader),
        'accuracy': correct_all / total,
        'accuracy_energy': correct_energy / total,
        'accuracy_alpha': correct_alpha / total,
        'accuracy_q0': correct_q0 / total
    }

