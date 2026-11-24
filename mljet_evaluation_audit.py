#!/usr/bin/env python3
"""Read-only ML-JET-Ext T04 evaluation/provenance audit. Python standard library only.

Scans executed T04-06/T04-07/T04-10 run directories; does not load model weights,
run inference, modify source runs, or claim test evaluation from filenames alone.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

STUDIES = {
    't04_06_v2_production': ('multi', 80),
    't04_07_loss_pilot': ('multi', 80),
    't04_07_top4_production': ('multi', 80),
    't04_10_single_head': ('single', 200),
}
MULTI_METRICS = ('val_loss', 'val_joint_accuracy', 'val_alpha_accuracy',
                 'val_q0_accuracy', 'val_alpha_macro_f1', 'val_q0_macro_f1',
                 'val_alpha_mae', 'val_q0_mae', 'val_alpha_rmse', 'val_q0_rmse',
                 'val_alpha_normalized_mae', 'val_q0_normalized_mae',
                 'val_alpha_physical_mae_loss', 'val_q0_physical_mae_loss')
SINGLE_METRICS = ('val_loss', 'val_accuracy', 'val_macro_f1',
                  'val_top1_accuracy', 'val_top2_accuracy', 'val_top3_accuracy',
                  'val_within_1_class_accuracy', 'val_within_2_class_accuracy',
                  'val_physical_mae', 'val_physical_rmse')
TEST_FILE_TERMS = ('test', 'prediction', 'evaluation', 'eval_report', 'inference')
EXCLUDE_EVIDENCE_NAMES = {'run_manifest.json', 'source_manifest.json',
                          'config.resolved.json', 'split_manifest.json',
                          'session_manifest.json', 'latest_checkpoint.json',
                          'loss_ablation_manifest.json', 'history.json',
                          'moment_epoch_metrics.csv', 'single_head_epoch_metrics.csv'}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dig(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def get_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def find_best_checkpoint(run, best_epoch):
    if best_epoch is None:
        return []
    name = f'best_epoch_{best_epoch:06d}.pth'
    return sorted((run / 'sessions').glob(f'session-*/checkpoints/{name}'))


def session_details(run):
    result = []
    for p in sorted((run / 'sessions').glob('session-*/session_manifest.json')):
        try:
            j = load_json(p)
            result.append((p, j))
        except (OSError, ValueError):
            result.append((p, {'status': 'unreadable'}))
    return result


def test_record_candidates(run):
    """Only user-authored experiment artifacts; never inspect source copies or model files.

    A candidate is not proof of test evaluation. Real provenance must establish
    selected checkpoint, data split/hash, and evaluation metrics.
    """
    candidates = []
    for p in run.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in ('.json', '.csv'):
            continue
        rel = p.relative_to(run)
        if p.name in EXCLUDE_EVIDENCE_NAMES:
            continue
        if any(part in {'source', 'source_snapshot', 'snapshots', 'code', 'tests'}
               for part in rel.parts[:-1]):
            continue
        if not any(term in p.name.lower() for term in TEST_FILE_TERMS):
            continue
        # Do not imply test evaluation from filename. Yield review candidates.
        candidates.append(p)
    return sorted(candidates)


def inspect_candidate(path):
    """Conservatively describe any *explicit* test-partition report."""
    rec = {'artifact': str(path), 'explicit_test_partition': False,
           'has_metric_fields': False, 'has_split_hash': False,
           'has_checkpoint_reference': False, 'status': 'candidate_only'}
    if path.suffix == '.json':
        try:
            obj = load_json(path)
        except (ValueError, OSError):
            rec['status'] = 'unreadable'
            return rec
        if not isinstance(obj, dict):
            return rec
        explicit = any(str(obj.get(k, '')).lower() == 'test'
                       for k in ('split', 'partition', 'subset', 'dataset_split', 'phase'))
        rec['explicit_test_partition'] = explicit
        rec['has_metric_fields'] = any(k in obj for k in ('metrics', 'accuracy', 'macro_f1', 'joint_accuracy', 'predictions'))
        rec['has_split_hash'] = bool(obj.get('split_sha256') or obj.get('test_split_sha256') or obj.get('test_csv_sha256'))
        rec['has_checkpoint_reference'] = bool(obj.get('checkpoint') or obj.get('checkpoint_sha256') or obj.get('checkpoint_path'))
        if explicit and rec['has_metric_fields']:
            rec['status'] = 'explicit_test_result_needs_linkage_review'
    elif path.suffix == '.csv':
        # CSV naming is never treated as evidence of actual test inference.
        rec['status'] = 'candidate_only_csv'
    return rec


def audit(root, out, hash_best=False):
    out.mkdir(parents=True, exist_ok=True)
    runs, errors, candidates, sessions_report = [], [], [], []

    for study, (kind, target) in STUDIES.items():
        base = root / study
        if not base.is_dir():
            errors.append(dict(study=study, run_id='', issue='study_directory_missing', detail=str(base)))
            continue
        for manifest_path in sorted(base.rglob('run_manifest.json')):
            run = manifest_path.parent
            archived = any(part.startswith('_archive') for part in run.relative_to(base).parts)
            if archived:
                # Archived records remain excluded from primary, but counted separately.
                continue
            rid = run.name
            try:
                manifest = load_json(manifest_path)
                cfg = load_json(run / 'config.resolved.json')
            except (ValueError, OSError) as exc:
                errors.append(dict(study=study, run_id=rid, issue='metadata_unreadable', detail=str(exc)))
                continue

            metric_file = run / ('moment_epoch_metrics.csv' if kind == 'multi' else 'single_head_epoch_metrics.csv')
            try:
                history = get_csv(metric_file)
            except (OSError, ValueError, csv.Error) as exc:
                errors.append(dict(study=study, run_id=rid, issue='history_unreadable', detail=str(exc)))
                history = []
            selector = 'val_joint_accuracy' if kind == 'multi' else 'val_macro_f1'
            metric_cols = MULTI_METRICS if kind == 'multi' else SINGLE_METRICS
            epochs = []
            for r in history:
                v = number(r.get('epoch'))
                if v is not None and int(v) == v:
                    epochs.append(int(v))
            continuous = epochs == list(range(1, target + 1))
            if not continuous:
                errors.append(dict(study=study, run_id=rid, issue='incomplete_or_discontinuous_history', detail=f'{len(epochs)} recorded, last {epochs[-1] if epochs else None}, target {target}'))
            # Python max preserves first occurrence: earliest epoch wins exact ties.
            eligible = [(i, number(r.get(selector))) for i, r in enumerate(history)]
            eligible = [(i, v) for i, v in eligible if v is not None]
            best_i, best_val = max(eligible, key=lambda item: item[1]) if eligible else (None, None)
            best_row = history[best_i] if best_i is not None else {}
            best_epoch = int(number(best_row.get('epoch'))) if best_row and number(best_row.get('epoch')) is not None else None
            last_row = history[-1] if history else {}
            best_paths = find_best_checkpoint(run, best_epoch)
            if len(best_paths) != 1:
                errors.append(dict(study=study, run_id=rid, issue='selected_checkpoint_missing_or_ambiguous', detail=f'best epoch={best_epoch}, matches={len(best_paths)}'))
            if not eligible:
                errors.append(dict(study=study, run_id=rid, issue='selector_metric_missing', detail=selector))

            session_list = session_details(run)
            statuses = [data.get('status', 'missing') for _, data in session_list]
            terminal_status = statuses[-1] if statuses else 'missing'
            if terminal_status != 'completed':
                errors.append(dict(study=study, run_id=rid, issue='final_session_not_completed', detail=terminal_status))
            for path, sdata in session_list:
                sessions_report.append(dict(study=study, run_id=rid, session=path.parent.name,
                                            status=sdata.get('status'), mode=sdata.get('mode'),
                                            target_epochs=sdata.get('target_epochs'),
                                            parent_epoch=dig(sdata, 'parent_checkpoint', 'epoch'),
                                            parent_sha256=dig(sdata, 'parent_checkpoint', 'sha256')))

            latest = {}
            pointer_path = run / 'latest_checkpoint.json'
            if pointer_path.is_file():
                try:
                    latest = load_json(pointer_path)
                except (ValueError, OSError):
                    errors.append(dict(study=study, run_id=rid, issue='latest_pointer_unreadable', detail=str(pointer_path)))
            latest_epoch = latest.get('epoch')
            if latest_epoch != target:
                errors.append(dict(study=study, run_id=rid, issue='latest_epoch_not_target', detail=str(latest_epoch)))
            latest_path = (run / latest['path']) if latest.get('path') else None
            if latest_path is not None and not latest_path.is_file():
                errors.append(dict(study=study, run_id=rid, issue='latest_checkpoint_file_missing', detail=str(latest_path)))

            # Multi-head session result provides an additional independent selection record.
            final_result = {}
            results = sorted((run / 'sessions').glob('session-*/result.json')) if kind == 'multi' else []
            if results:
                try:
                    final_result = load_json(results[-1])
                except (ValueError, OSError) as exc:
                    errors.append(dict(study=study, run_id=rid, issue='session_result_unreadable', detail=str(exc)))
            if kind == 'single' and best_val is not None and history:
                stored_best = number(history[-1].get('best_val_macro_f1'))
                if stored_best is None or not math.isclose(stored_best, best_val, abs_tol=1e-8, rel_tol=1e-8):
                    errors.append(dict(study=study, run_id=rid, issue='single_head_running_best_mismatch', detail=f'{stored_best} vs {best_val}'))
            if final_result:
                if final_result.get('best_epoch') != best_epoch:
                    errors.append(dict(study=study, run_id=rid, issue='best_epoch_result_mismatch', detail=f"result={final_result.get('best_epoch')} history={best_epoch}"))
                recorded = number(final_result.get('best_metric'))
                if recorded is None or best_val is None or not math.isclose(recorded, best_val, abs_tol=1e-8, rel_tol=1e-8):
                    errors.append(dict(study=study, run_id=rid, issue='best_metric_result_mismatch', detail=f'{recorded} vs {best_val}'))

            selected_hash = sha256_file(best_paths[0]) if hash_best and len(best_paths) == 1 else ''
            split_inputs = dig(manifest, 'inputs', 'input_records') if kind == 'multi' else dig(manifest, 'inputs', 'inputs')
            split_inputs = split_inputs if isinstance(split_inputs, dict) else {}
            def split_hash(label):
                item = split_inputs.get(label if kind == 'multi' else f'{label}.csv')
                return item.get('sha256', '') if isinstance(item, dict) else ''
            loss_spec = cfg.get('moment_loss') or {}
            if kind == 'single':
                extra_loss_path = run / 'loss_ablation_manifest.json'
                if extra_loss_path.is_file():
                    try:
                        extra_loss = load_json(extra_loss_path)
                        loss_spec = extra_loss.get('loss') or loss_spec
                    except (OSError, ValueError):
                        errors.append(dict(study=study, run_id=rid, issue='loss_manifest_unreadable', detail=str(extra_loss_path)))
            run_row = dict(study=study, run_id=rid, model=cfg.get('model'), task=cfg.get('task', 'joint'),
                           strategy=cfg.get('strategy'), moments=json.dumps(cfg.get('moments', [])), seed=cfg.get('seed'),
                           loss_spec=json.dumps(loss_spec, sort_keys=True),
                           config_budget_epochs=cfg.get('epochs'), executed_target_epochs=target,
                           recorded_epochs=len(epochs), last_epoch=epochs[-1] if epochs else '',
                           history_continuous=continuous, session_count=len(session_list),
                           session_modes=';'.join(str(s.get('mode', '')) for _, s in session_list),
                           final_session_status=terminal_status, selection_metric=selector,
                           best_epoch=best_epoch, best_validation_selection_value=best_val,
                           best_checkpoint_present=(len(best_paths) == 1),
                           best_checkpoint_path=str(best_paths[0]) if len(best_paths) == 1 else '',
                           best_checkpoint_sha256=selected_hash,
                           latest_checkpoint_epoch=latest_epoch, latest_checkpoint_sha256=latest.get('sha256', ''),
                           git_commit=manifest.get('git_commit') or dig(manifest, 'git', 'commit') or '',
                           train_split_sha256=split_hash('train'), val_split_sha256=split_hash('val'),
                           test_split_sha256=split_hash('test'),
                           source_overlap=dig(manifest, 'inputs', 'source_overlap') if kind == 'single' else dig(manifest, 'inputs', 'global_cross_split_overlap'),
                           recorded_test_flag=manifest.get('test_set_evaluated', 'not_recorded'),
                           test_evaluation_status='not_verified', run_directory=str(run))
            for metric in metric_cols:
                run_row[f'best_{metric}'] = best_row.get(metric, '')
                run_row[f'final_{metric}'] = last_row.get(metric, '')
            discovered = test_record_candidates(run)
            for p in discovered:
                c = inspect_candidate(p)
                c.update(study=study, run_id=rid)
                candidates.append(c)
                if c['status'] == 'explicit_test_result_needs_linkage_review':
                    run_row['test_evaluation_status'] = 'explicit_record_needs_linkage_review'
            runs.append(run_row)

    run_cols = list(dict.fromkeys(['study', 'run_id', 'model', 'task', 'strategy', 'moments', 'seed',
      'loss_spec', 'config_budget_epochs', 'executed_target_epochs', 'recorded_epochs', 'last_epoch',
      'history_continuous', 'session_count', 'session_modes', 'final_session_status',
      'selection_metric', 'best_epoch', 'best_validation_selection_value',
      'best_checkpoint_present', 'best_checkpoint_path', 'best_checkpoint_sha256',
      'latest_checkpoint_epoch', 'latest_checkpoint_sha256', 'git_commit', 'train_split_sha256',
      'val_split_sha256', 'test_split_sha256', 'source_overlap', 'recorded_test_flag',
      'test_evaluation_status', 'run_directory'] +
      [f'best_{m}' for m in MULTI_METRICS + SINGLE_METRICS] +
      [f'final_{m}' for m in MULTI_METRICS + SINGLE_METRICS]))
    write_csv(out / 'evaluation_run_inventory.csv', runs, run_cols)
    write_csv(out / 'evaluation_audit_exceptions.csv', errors, ['study', 'run_id', 'issue', 'detail'])
    write_csv(out / 'evaluation_session_provenance.csv', sessions_report,
              ['study', 'run_id', 'session', 'status', 'mode', 'target_epochs', 'parent_epoch', 'parent_sha256'])
    write_csv(out / 'explicit_test_evidence_candidates.csv', candidates,
              ['study', 'run_id', 'artifact', 'explicit_test_partition', 'has_metric_fields',
               'has_split_hash', 'has_checkpoint_reference', 'status'])
    counts = Counter(r['study'] for r in runs)
    expected_counts = {'t04_06_v2_production': 120, 't04_07_loss_pilot': 21,
                       't04_07_top4_production': 84, 't04_10_single_head': 96}
    for study, expected in expected_counts.items():
        if counts[study] != expected:
            errors.append(dict(study=study, run_id='', issue='unexpected_run_count',
                               detail=f'expected {expected}, scanned {counts[study]}'))
    for partition in ('train', 'val', 'test'):
        hashes = {r[f'{partition}_split_sha256'] for r in runs if r[f'{partition}_split_sha256']}
        if len(hashes) > 1:
            errors.append(dict(study='all', run_id='', issue='split_hash_disagreement',
                               detail=f'{partition}: {len(hashes)} distinct SHA256 hashes'))
    # Rewrite the exceptions file after cross-study reconciliation.
    write_csv(out / 'evaluation_audit_exceptions.csv', errors, ['study', 'run_id', 'issue', 'detail'])
    summary = dict(version='1.0',
      studies={study: dict(records=counts[study], expected=expected_counts[study])
               for study in STUDIES},
      total_active_runs=len(runs),
      total_continuous_histories=sum(bool(r['history_continuous']) for r in runs),
      total_selected_checkpoints_found=sum(bool(r['best_checkpoint_present']) for r in runs),
      total_final_sessions_completed=sum(r['final_session_status'] == 'completed' for r in runs),
      explicit_test_result_candidates=sum(c['status'] == 'explicit_test_result_needs_linkage_review' for c in candidates),
      test_evaluations_fully_verified=0,
      test_policy='No test evaluation is called verified without dataset split hash, checkpoint linkage, and explicit evaluation record.',
      exception_count=len(errors), exceptions_by_kind=dict(Counter(e['issue'] for e in errors)),
      optional_best_checkpoint_sha256=hash_best)
    (out / 'evaluation_audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Reports:', out)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/rs/rs_grp_mljet/ml-jet-ext/training_output/T04'))
    parser.add_argument('--out', type=Path, default=Path('/tmp/mljet_evaluation_audit'))
    parser.add_argument('--hash-best-checkpoints', action='store_true', help='Optional: stream-read all best checkpoint weights to compute SHA-256 (more I/O).')
    args = parser.parse_args()
    audit(args.root, args.out, args.hash_best_checkpoints)
