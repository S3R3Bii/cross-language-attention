"""Score committed frozen models once, retaining all methods and common-month draws."""
from datetime import date, timedelta
import csv
import gzip
import json
import math
import os
import subprocess
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import numpy as np
from model import ROOT, FAMILIES, csv_gzip, load_pairs, matrix, metrics, predict, read_panel, sha, write_json
from train import write_csv


def normalized_hash(path):
    return sha(path.read_text(encoding='utf-8').encode('utf-8'))


def evaluate(root=ROOT, require_committed=True):
    freeze = json.loads((root / 'results/freeze.json').read_text(encoding='utf-8'))
    assert freeze['status'] == 'FROZEN_BEFORE_TEST_INFERENCE'
    for key, path in [('models_sha256', 'results/frozen_models.json'), ('selection_sha256', 'results/selection.json'), ('panel_sha256', 'data/panel.csv.gz')]:
        assert freeze[key] == sha((root / path).read_bytes()), path
    selection = json.loads((root / 'results/selection.json').read_text(encoding='utf-8'))
    assert selection['source_accounting_sha256'] == sha((root / 'results/source_accounting.json').read_bytes())
    for path, digest in freeze['normalized_source_sha256'].items():
        assert normalized_hash(root / path) == digest, path
    commit = 'temporary_full_replay'
    if require_committed:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        for path in ['results/freeze.json', 'results/frozen_models.json', 'results/selection.json', *freeze['normalized_source_sha256']]:
            blob = subprocess.check_output(['git', 'show', commit + ':' + path], cwd=root)
            work = (root / path).read_bytes()
            assert blob.replace(b'\r\n', b'\n') == work.replace(b'\r\n', b'\n'), 'Uncommitted frozen dependency: ' + path
    models = json.loads((root / 'results/frozen_models.json').read_text(encoding='utf-8'))
    panel, rows, scores = read_panel(root), [], []
    for pair in load_pairs(root):
        own = panel[(pair['qid'], pair['language'])]
        fitting = [day for day in pair['eligible_dates'] if day <= '2024-12-30']
        scale = sum(abs(own[day] - own[(date.fromisoformat(day) - timedelta(days=7)).isoformat()]) for day in fitting) / len(fitting)
        for method in [*FAMILIES, 'lag7', 'weekly4']:
            family = method if method in FAMILIES else 'own'
            days, x, actual, _ = matrix(pair, panel, family, start='2025-01-01', end='2025-12-31')
            if method in FAMILIES:
                model = models[f"{method}_{pair['language']}_{pair['qid']}"]
                raw, clipped, prediction = predict(model, x)
            else:
                lags = [7] if method == 'lag7' else [7, 14, 21, 28]
                prediction = np.array([sum(own[(date.fromisoformat(day) - timedelta(days=lag)).isoformat()] for lag in lags) / len(lags) for day in days], dtype=float)
                raw = clipped = np.log1p(prediction)
            score = metrics(actual, raw, prediction, mase_scale=scale)
            scores.append({'qid': pair['qid'], 'topic': pair['topic'], 'language': pair['language'], 'role': pair['role'],
                           'method': method, 'mase_scale': scale, **score})
            for day, target, raw_value, clipped_value, pred in zip(days, actual, raw, clipped, prediction):
                rows.append({'qid': pair['qid'], 'language': pair['language'], 'role': pair['role'], 'method': method,
                             'date': day, 'actual': int(target), 'pred_log_raw': float(raw_value),
                             'pred_log_clipped': float(clipped_value), 'prediction': float(pred)})
    csv_gzip(root / 'results/test_predictions.csv.gz', rows, list(rows[0]))
    write_csv(root / 'results/test_metrics.csv', scores)
    summaries = []
    for lang in ['es', 'ja']:
        for role in ['development', 'held', 'all']:
            for method in [*FAMILIES, 'lag7', 'weekly4']:
                selected = [row for row in scores if row['language'] == lang and (role == 'all' or row['role'] == role) and row['method'] == method]
                msle = sum(row['msle'] for row in selected) / len(selected)
                summaries.append({'language': lang, 'role': role, 'method': method, 'pages': len(selected),
                    'n': sum(row['n'] for row in selected), 'page_balanced_msle': msle,
                    'page_balanced_rmsle': math.sqrt(msle),
                    'page_balanced_mae': sum(row['mae'] for row in selected) / len(selected),
                    'page_balanced_mse': sum(row['mse'] for row in selected) / len(selected),
                    'mase_defined_pages': sum(row['mase'] is not None for row in selected),
                    'mean_defined_mase': sum(row['mase'] for row in selected if row['mase'] is not None) / sum(row['mase'] is not None for row in selected) if any(row['mase'] is not None for row in selected) else None})
    write_csv(root / 'results/test_summary.csv', summaries)

    rng = np.random.default_rng(20261028)
    draw_ids, draws, intervals = {}, [], []
    # All pages and both comparisons within a group share each sampled month.
    for lang in ['es', 'ja']:
        for role in ['development', 'held']:
            qids = sorted({row['qid'] for row in rows if row['language'] == lang and row['role'] == role})
            common = rng.integers(0, 12, size=(2000, 12))
            draw_ids[lang + '_' + role] = common.tolist()
            buckets = {}
            for qid in qids:
                for method in FAMILIES:
                    for month in range(1, 13):
                        part = [row for row in rows if row['qid'] == qid and row['language'] == lang and row['method'] == method and int(row['date'][5:7]) == month]
                        buckets[(qid, method, month - 1)] = (sum((row['pred_log_clipped'] - math.log1p(row['actual'])) ** 2 for row in part), len(part))
            for method in ['english2', 'english7']:
                changes = []
                for index, months in enumerate(common):
                    losses = {}
                    for current in ['own', method]:
                        losses[current] = sum(sum(buckets[(qid, current, int(month))][0] for month in months) / sum(buckets[(qid, current, int(month))][1] for month in months) for qid in qids) / len(qids)
                    delta = losses[method] - losses['own']
                    changes.append(delta)
                    draws.append({'language': lang, 'role': role, 'method': method, 'draw': index,
                                  'own_msle': losses['own'], 'cross_msle': losses[method], 'difference': delta})
                observed = {row['method']: row['page_balanced_msle'] for row in summaries if row['language'] == lang and row['role'] == role}
                low, high = np.quantile(changes, [0.025, 0.975])
                intervals.append({'language': lang, 'role': role, 'method': method, 'pages': len(qids),
                                  'point_difference': observed[method] - observed['own'],
                                  'conditional_95_low': float(low), 'conditional_95_high': float(high), 'draws': 2000})
    (root / 'results/bootstrap_month_indices.json.gz').write_bytes(gzip.compress(json.dumps(draw_ids).encode('utf-8'), mtime=0))
    csv_gzip(root / 'results/bootstrap_draws.csv.gz', draws, list(draws[0]))
    write_csv(root / 'results/paired_intervals.csv', intervals)
    report = {'status': 'EVALUATED_FROZEN_2025_COMPARISON', 'frozen_commit': commit,
              'test_prediction_rows': len(rows), 'page_method_metrics': len(scores), 'summary_rows': len(summaries),
              'bootstrap_draws': len(draws), 'preexisting_test_reused_for_correction': False,
              'scope': 'Retrospective source snapshot, fixed title convenience panel, assumed two-day input availability; no causal or prospective performance claim.'}
    write_json(root / 'results/evaluation_summary.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    evaluate()
