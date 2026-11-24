"""Apply Stage 3 to the existing Stage 2 checkout; refuses unknown source anchors."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def change(path, before, after):
    p = ROOT / path
    src = p.read_text()
    if after in src:
        print('Already applied:', path)
        return
    count = src.count(before)
    if count != 1:
        raise RuntimeError('%s: expected one exact anchor, found %s: %r' % (path, count, before[:100]))
    p.write_text(src.replace(before, after, 1))
    print('Updated:', path)

# FP16 overflow detection: count completed optimizer updates separately from attempted windows.
p = 'train_utils/t04_precision.py'
change(p, '    both = total = steps = 0', '    both = total = steps = 0\n    completed_steps = skipped_steps = 0')
# Stage 2 streaming accumulation flush() must have this exact sequence.
change(p, '        nonlocal steps, window_batches, window_samples',
          '        nonlocal steps, window_batches, window_samples, completed_steps, skipped_steps')
change(p, """        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        steps += 1  # attempted steps; FP16 overflow may skip an update""", """        if scaler is None:
            optimizer.step()
            completed_steps += 1
        else:
            # Normal optimizer.step may return None, so detect skipped updates
            # using the dynamic loss-scale decrease after scaler.update().
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped_steps += 1
            else:
                completed_steps += 1
        optimizer.zero_grad(set_to_none=True)
        steps += 1  # accumulation windows attempted""")
change(p, "    result['optimizer_steps']=steps", "    result['optimizer_steps']=steps\n    result['optimizer_steps_completed']=completed_steps\n    result['optimizer_steps_skipped']=skipped_steps")

p = 'train_utils/t04_session_training.py'
change(p, 'from train_utils.moment_training import train_moment_epoch, evaluate_moment',
          'from train_utils.moment_training import train_moment_epoch, evaluate_moment\nfrom train_utils.t04_telemetry import cuda_reset, epoch_telemetry, save_session_epoch_telemetry')
change(p, "                  precision='fp32', accumulation=1):", "                  precision='fp32', accumulation=1, microbatch_size=None):")
change(p, '        seed_loader_epoch(train_loader, seed, epoch)',
          '        seed_loader_epoch(train_loader, seed, epoch)\n        telemetry_start = cuda_reset(device)')
# Append sidecar only AFTER checkpoint commit and CSV append. Do not alter immutable checkpoint/CSV schema.
change(p, '        _append_csv(history, row)\n        _retain_checkpoints',
          '''        _append_csv(history, row)
        telemetry = epoch_telemetry(telemetry_start, device, optimizer, tr, va,
                                    microbatch_size, accumulation)
        save_session_epoch_telemetry(session, epoch, telemetry)
        print('[T04-04] epoch telemetry: ' + json.dumps(telemetry, sort_keys=True), flush=True)
        _retain_checkpoints''')
p = 'train_utils/t04_session_worker.py'
change(p, "        precision=cfg['precision'], accumulation=cfg['gradient_accumulation'])",
          "        precision=cfg['precision'], accumulation=cfg['gradient_accumulation'],\n        microbatch_size=cfg['batch_size'])")
print('Stage 3 installed. Validate before preparing fresh immutable runs.')
