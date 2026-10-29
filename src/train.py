"""Retain every forward trial, then freeze rule-based final refits before test inference."""
import argparse
import csv
import gzip
import json
import time
from datetime import date, timedelta
from pathlib import Path
from model import ROOT, FAMILIES, PENALTIES, FOLDS, csv_gzip, fit, load_pairs, matrix, metrics, predict, read_panel, sha, write_json


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def validate(root=ROOT, reuse_existing=False):
    pairs = [pair for pair in load_pairs(root) if pair['language'] == 'es' and pair['role'] == 'development']
    assert len(pairs) >= 6
    panel, trials, predictions = read_panel(root), [], []
    models = {}
    if reuse_existing:
        archive = root / 'results/search-v1'
        receipt = json.loads((archive / 'SEARCH_RECEIPT.json').read_text(encoding='utf-8'))
        assert receipt['status'] == 'PASS_ORIGINAL_VALIDATION_ONLY_SEARCH'
        for filename, digest in receipt['sha256'].items():
            assert sha((archive / filename).read_bytes()) == digest, filename
        models = json.loads((archive / 'validation_models.json').read_text(encoding='utf-8'))
        with (archive / 'validation_trials.csv').open(encoding='utf-8', newline='') as stream:
            trials = list(csv.DictReader(stream))
        for row in trials:
            for key in ['fold', 'training_n', 'n']:
                row[key] = int(row[key])
            for key in ['penalty', 'runtime_seconds', 'msle', 'rmsle', 'mae', 'mse', 'rmse']:
                row[key] = float(row[key])
            assert row['status'] == 'ok'
        with gzip.open(archive / 'validation_predictions.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
            predictions = list(csv.DictReader(stream))
        for row in predictions:
            row['fold'], row['actual'] = int(row['fold']), int(row['actual'])
            for key in ['pred_log_raw', 'pred_log_clipped', 'prediction']:
                row[key] = float(row[key])
    for family in FAMILIES:
        for penalty in PENALTIES:
            for pair in pairs:
                for fold, (start, end) in enumerate(FOLDS):
                    name = f"{family}_l{penalty:g}_{pair['qid']}_f{fold}"
                    if name in models:
                        continue
                    row = {'model': name, 'family': family, 'penalty': penalty, 'qid': pair['qid'],
                           'topic': pair['topic'], 'language': 'es', 'role': 'development', 'fold': fold,
                           'status': 'failed', 'runtime_seconds': None, 'training_n': None, 'n': None,
                           'msle': None, 'rmsle': None, 'mae': None, 'mse': None, 'rmse': None, 'error': ''}
                    timer = time.perf_counter()
                    try:
                        cutoff = (date.fromisoformat(start) - timedelta(days=2)).isoformat()
                        days, x, target, names = matrix(pair, panel, family, end=cutoff)
                        assert len(days) >= 500 and days[-1] <= cutoff
                        model = fit(x, target, penalty, names, days)
                        model.update(qid=pair['qid'], language='es', role='development', family=family, fold=fold,
                                     fitting_origin=start, training_label_available_lag_days=2)
                        val_days, val_x, val_y, val_names = matrix(pair, panel, family, start=start, end=end)
                        assert names == val_names and len(val_days) >= 100 and set(days).isdisjoint(val_days)
                        raw, clipped, counts = predict(model, val_x)
                        score = metrics(val_y, raw, counts)
                        row.update({key: score[key] for key in ['n', 'msle', 'rmsle', 'mae', 'mse', 'rmse']})
                        row.update(status='ok', training_n=len(days))
                        models[name] = model
                        for day, actual, raw_value, clipped_value, prediction in zip(val_days, val_y, raw, clipped, counts):
                            predictions.append({'model': name, 'qid': pair['qid'], 'language': 'es', 'date': day,
                                                'fold': fold, 'actual': int(actual), 'pred_log_raw': float(raw_value),
                                                'pred_log_clipped': float(clipped_value), 'prediction': float(prediction)})
                    except Exception as error:
                        row['error'] = type(error).__name__ + ': ' + str(error)
                    row['runtime_seconds'] = time.perf_counter() - timer
                    trials.append(row)
    results = root / 'results'
    write_csv(results / 'validation_trials.csv', trials)
    csv_gzip(results / 'validation_predictions.csv.gz', predictions, list(predictions[0]))
    write_json(results / 'validation_models.json', models)
    candidates = []
    for family in FAMILIES:
        for penalty in PENALTIES:
            rows = [row for row in trials if row['family'] == family and row['penalty'] == penalty]
            ok = all(row['status'] == 'ok' for row in rows) and len(rows) == len(pairs) * 3
            candidates.append({'family': family, 'penalty': penalty, 'fits': len(rows),
                               'failed_fits': sum(row['status'] != 'ok' for row in rows),
                               'mean_page_fold_msle': sum(row['msle'] for row in rows) / len(rows) if ok else None,
                               'eligible': ok})
    selected = {}
    for family in FAMILIES:
        viable = [row for row in candidates if row['family'] == family and row['eligible']]
        assert viable, 'No complete successful configuration in ' + family
        selected[family] = min(viable, key=lambda row: (row['mean_page_fold_msle'], row['penalty']))
    selection = {'status': 'VALIDATION_ONLY_SELECTION', 'selected': selected, 'candidates': candidates,
                 'fitted_models': len(models), 'trial_rows': len(trials), 'prediction_rows': len(predictions),
                 'selection_language': 'es', 'selection_role': 'development',
                 'selection_topics': [pair['qid'] for pair in pairs],
                 'source_manifest_sha256': sha((root / 'data/source-v2/manifest.json').read_bytes()),
                 'source_accounting_sha256': sha((results / 'source_accounting.json').read_bytes()),
                 'test_inference_performed': False, 'test_outcome_summary_inspected': False}
    write_json(results / 'selection.json', selection)
    print(json.dumps({'validation_fits': len(trials), 'failed': sum(row['status'] != 'ok' for row in trials),
                      'validation_predictions': len(predictions), 'selected': selected}))


def freeze(root=ROOT):
    selection = json.loads((root / 'results/selection.json').read_text(encoding='utf-8'))
    assert selection['test_inference_performed'] is False
    panel, models = read_panel(root), {}
    for pair in load_pairs(root):
        for family in FAMILIES:
            name = f"{family}_{pair['language']}_{pair['qid']}"
            days, x, target, names = matrix(pair, panel, family, end='2024-12-30')
            assert days[-1] <= '2024-12-30'
            model = fit(x, target, selection['selected'][family]['penalty'], names, days)
            model.update(qid=pair['qid'], language=pair['language'], role=pair['role'], topic=pair['topic'],
                         family=family, fitting_origin='2025-01-01', training_label_available_lag_days=2)
            models[name] = model
    write_json(root / 'results/frozen_models.json', models)
    write_json(root / 'results/freeze.json', {'status': 'FROZEN_BEFORE_TEST_INFERENCE', 'models': len(models),
        'selection_sha256': sha((root / 'results/selection.json').read_bytes()),
        'models_sha256': sha((root / 'results/frozen_models.json').read_bytes()),
        'panel_sha256': sha((root / 'data/panel.csv.gz').read_bytes()),
        'protocol_sha256': sha((root / 'PROTOCOL.md').read_bytes()),
        'normalized_source_sha256': {name: sha((root / name).read_text(encoding='utf-8').encode('utf-8')) for name in ['src/model.py', 'src/train.py', 'src/evaluate.py', 'PROTOCOL.md', 'SOURCE_AMENDMENT_V2.md', 'SEARCH_EXTENSION_V2.md']},
        'test_start': '2025-01-01', 'test_end': '2025-12-31', 'test_inference_performed': False,
        'scope': 'Hyperparameter transfer; final models fit their own pre-2025 page history. Retrospective source snapshot.'})
    print('Frozen', len(models), 'models; no 2025 prediction or scoring performed. Commit before evaluation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['validate', 'freeze'])
    parser.add_argument('--reuse-existing', action='store_true')
    args = parser.parse_args()
    validate(reuse_existing=args.reuse_existing) if args.mode == 'validate' else freeze()
