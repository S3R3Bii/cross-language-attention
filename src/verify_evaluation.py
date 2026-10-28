"""Reconstruct every final prediction, metric and common-month draw independently."""
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
import csv
import gzip
import json
import math
import numpy as np
from verify_models import close, independent_row, load_context, scalar_prediction

ROOT = Path(__file__).resolve().parents[1]


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def verify(root=ROOT):
    data, pairs = load_context(root)
    models = json.loads((root / 'results/frozen_models.json').read_text(encoding='utf-8'))
    grouped, seen = defaultdict(list), set()
    with gzip.open(root / 'results/test_predictions.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            qid, lang, method, day = row['qid'], row['language'], row['method'], row['date']
            key = (qid, lang, method)
            pair = pairs[(qid, lang)]
            assert row['role'] == pair['role'] and '2025-01-01' <= day <= '2025-12-31' and day in pair['eligible_dates']
            assert (key, day) not in seen
            seen.add((key, day))
            actual = data[(qid, lang, day)]
            assert int(row['actual']) == actual
            if method in ['own', 'english2', 'english7']:
                raw, clipped, pred = scalar_prediction(models[f'{method}_{lang}_{qid}'], independent_row(data, qid, lang, method, day))
            else:
                assert method in ['lag7', 'weekly4']
                lags = [7] if method == 'lag7' else [7, 14, 21, 28]
                pred = math.fsum(data[(qid, lang, (date.fromisoformat(day) - timedelta(days=lag)).isoformat())] for lag in lags) / len(lags)
                raw = clipped = math.log1p(pred)
            close(float(row['pred_log_raw']), raw)
            close(float(row['pred_log_clipped']), clipped)
            close(float(row['prediction']), pred)
            grouped[key].append((day, clipped - math.log1p(actual), pred - actual))
    scores = {}
    with (root / 'results/test_metrics.csv').open(encoding='utf-8', newline='') as stream:
        exported_scores = list(csv.DictReader(stream))
    assert len(exported_scores) == len(pairs) * 5
    for row in exported_scores:
        qid, lang, method = row['qid'], row['language'], row['method']
        pair, values = pairs[(qid, lang)], grouped[(qid, lang, method)]
        assert row['role'] == pair['role'] and row['topic'] == pair['topic']
        expected_days = {day for day in pair['eligible_dates'] if day.startswith('2025')}
        assert {value[0] for value in values} == expected_days and len(values) == int(row['n'])
        fitting = [day for day in pair['eligible_dates'] if day <= '2024-12-30']
        scale = math.fsum(abs(data[(qid, lang, day)] - data[(qid, lang, (date.fromisoformat(day) - timedelta(days=7)).isoformat())]) for day in fitting) / len(fitting)
        score = {'msle': math.fsum(value[1] ** 2 for value in values) / len(values),
                 'mse': math.fsum(value[2] ** 2 for value in values) / len(values),
                 'mae': math.fsum(abs(value[2]) for value in values) / len(values)}
        score.update(rmsle=math.sqrt(score['msle']), rmse=math.sqrt(score['mse']), mase_scale=scale,
                     mase=score['mae'] / scale if scale else None, n=len(values))
        for metric, expected in score.items():
            if expected is None:
                assert row[metric] == ''
            else:
                close(float(row[metric]), expected)
        scores[(qid, lang, method)] = score
    summaries = {}
    with (root / 'results/test_summary.csv').open(encoding='utf-8', newline='') as stream:
        summary_rows = list(csv.DictReader(stream))
    assert len(summary_rows) == 30
    for row in summary_rows:
        lang, role, method = row['language'], row['role'], row['method']
        chosen = [score for (qid, language, current), score in scores.items() if language == lang and current == method and (role == 'all' or pairs[(qid, language)]['role'] == role)]
        assert int(row['pages']) == len(chosen) and int(row['n']) == sum(value['n'] for value in chosen)
        msle = math.fsum(value['msle'] for value in chosen) / len(chosen)
        close(float(row['page_balanced_msle']), msle)
        close(float(row['page_balanced_rmsle']), math.sqrt(msle))
        close(float(row['page_balanced_mae']), math.fsum(value['mae'] for value in chosen) / len(chosen))
        close(float(row['page_balanced_mse']), math.fsum(value['mse'] for value in chosen) / len(chosen))
        valid = [value['mase'] for value in chosen if value['mase'] is not None]
        assert int(row['mase_defined_pages']) == len(valid)
        if valid:
            close(float(row['mean_defined_mase']), math.fsum(valid) / len(valid))
        summaries[(lang, role, method)] = msle

    indices = json.loads(gzip.decompress((root / 'results/bootstrap_month_indices.json.gz').read_bytes()))
    generator = np.random.Generator(np.random.PCG64(20261028))
    for lang in ['es', 'ja']:
        for role in ['development', 'held']:
            assert indices[lang + '_' + role] == generator.integers(0, 12, size=(2000, 12)).tolist()
    buckets = {}
    for key, values in grouped.items():
        if key[2] in ['own', 'english2', 'english7']:
            for month in range(1, 13):
                selected = [value for value in values if int(value[0][5:7]) == month]
                buckets[(*key, month - 1)] = (math.fsum(value[1] ** 2 for value in selected), len(selected))
    changes, draw_seen = defaultdict(list), set()
    with gzip.open(root / 'results/bootstrap_draws.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            lang, role, method, index = row['language'], row['role'], row['method'], int(row['draw'])
            assert method in ['english2', 'english7'] and (lang, role, method, index) not in draw_seen
            draw_seen.add((lang, role, method, index))
            months = indices[lang + '_' + role][index]
            qids = sorted(qid for (qid, language), pair in pairs.items() if language == lang and pair['role'] == role)
            expected = {}
            for current in ['own', method]:
                page_losses = []
                for qid in qids:
                    numerator = math.fsum(buckets[(qid, lang, current, month)][0] for month in months)
                    denominator = sum(buckets[(qid, lang, current, month)][1] for month in months)
                    page_losses.append(numerator / denominator)
                expected[current] = math.fsum(page_losses) / len(page_losses)
            delta = expected[method] - expected['own']
            close(float(row['own_msle']), expected['own'])
            close(float(row['cross_msle']), expected[method])
            close(float(row['difference']), delta)
            changes[(lang, role, method)].append(delta)
    assert len(draw_seen) == 16000
    with (root / 'results/paired_intervals.csv').open(encoding='utf-8', newline='') as stream:
        interval_rows = list(csv.DictReader(stream))
    assert len(interval_rows) == 8
    for row in interval_rows:
        key = (row['language'], row['role'], row['method'])
        assert len(changes[key]) == int(row['draws']) == 2000
        close(float(row['point_difference']), summaries[key] - summaries[(key[0], key[1], 'own')])
        close(float(row['conditional_95_low']), quantile(changes[key], 0.025))
        close(float(row['conditional_95_high']), quantile(changes[key], 0.975))
    result = {'status': 'PASS', 'scalar_predictions_reconstructed': len(seen),
              'page_method_metric_rows': len(exported_scores), 'summary_rows': len(summary_rows),
              'independent_common_month_draws': len(draw_seen), 'interval_rows': len(interval_rows),
              'scope': 'Every retained final prediction, target join, date/denominator, MASE scale, scalar metric, group weight, month draw and interval; no fitting-code metric helper imported.',
              'absolute_tolerance': 1e-7, 'relative_tolerance': 1e-9}
    (root / 'results/INDEPENDENT_EVALUATION_CHECK.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    verify()
