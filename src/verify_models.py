"""Independent lag reconstruction, scalar errors and augmented-QR ridge certificates."""
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import numpy as np
from scipy.linalg import lstsq

ROOT = Path(__file__).resolve().parents[1]
ABS, REL = 1e-7, 1e-9


def close(actual, expected, absolute=ABS, relative=REL):
    assert math.isfinite(float(actual)) and math.isfinite(float(expected))
    assert abs(actual - expected) <= absolute + relative * abs(expected), (actual, expected)


def load_context(root):
    data = {}
    with gzip.open(root / 'data/panel.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            data[(row['qid'], row['language'], row['date'])] = int(row['views'])
    source = json.loads((root / 'results/source_accounting.json').read_text(encoding='utf-8'))
    assert source['status'] == 'PASS_SOURCE_COVERAGE_GATE'
    pairs = {(row['qid'], row['language']): row for row in source['pairs'] if row['eligible']}
    return data, pairs


def independent_row(data, qid, lang, family, stamp):
    day = date.fromisoformat(stamp)
    def log_count(language, lag):
        return math.log1p(data[(qid, language, (day - timedelta(days=lag)).isoformat())])
    row = [log_count(lang, lag) for lag in (2, 7, 14, 28)]
    row.append(sum(log_count(lang, lag) for lag in (2, 3, 4, 5, 6, 7, 8)) / 7)
    row.extend(int(day.weekday() == j) for j in (0, 1, 2, 3, 4, 5))
    ordinal = (day - date(day.year, 1, 1)).days
    row.extend([math.sin(2 * math.pi * ordinal / 365.25), math.cos(2 * math.pi * ordinal / 365.25)])
    if family == 'english2':
        row.extend(log_count('en', lag) for lag in (2, 7, 14, 28))
        row.append(sum(log_count('en', lag) for lag in (2, 3, 4, 5, 6, 7, 8)) / 7)
    elif family == 'english7':
        row.extend(log_count('en', lag) for lag in (7, 14, 21, 35))
        row.append(sum(log_count('en', lag) for lag in (7, 8, 9, 10, 11, 12, 13)) / 7)
    else:
        assert family == 'own'
    return row


def certificate(name, model, data, pair, cache):
    qid, lang, family = model['qid'], model['language'], model['family']
    origin = date.fromisoformat(model['fitting_origin'])
    cutoff = (origin - timedelta(days=2)).isoformat()
    days = [day for day in pair['eligible_dates'] if day <= cutoff]
    assert model['training_n'] == len(days) and model['training_start'] == days[0] and model['training_end'] == days[-1]
    assert model['training_dates_sha256'] == hashlib.sha256(('\n'.join(days) + '\n').encode('utf-8')).hexdigest()
    key = (qid, lang, family, cutoff)
    if key not in cache:
        x = np.array([independent_row(data, qid, lang, family, day) for day in days])
        y = np.array([math.log1p(data[(qid, lang, day)]) for day in days])
        means = np.array([math.fsum(float(v) for v in x[:, j]) / len(days) for j in range(x.shape[1])])
        scales = np.array([math.sqrt(math.fsum((float(v) - means[j]) ** 2 for v in x[:, j]) / len(days)) for j in range(x.shape[1])])
        scales[scales == 0] = 1
        cache[key] = (x, y, means, scales)
    x, y, means, scales = cache[key]
    for actual, expected in zip(model['mean'], means):
        close(actual, expected, 1e-12, 1e-12)
    for actual, expected in zip(model['scale'], scales):
        close(actual, expected, 1e-12, 1e-12)
    z = (x - np.array(model['mean'])) / np.array(model['scale'])
    b, beta, lam = model['intercept'], np.array(model['coefficient']), model['penalty']
    assert lam in [0.01, 0.1, 1, 10, 100]
    residual = np.array([b + math.fsum(float(v * w) for v, w in zip(row, beta)) - target for row, target in zip(z, y)])
    gradient = [2 * math.fsum(float(v) for v in residual) / len(y)]
    gradient.extend(2 * math.fsum(float(a * r) for a, r in zip(z[:, j], residual)) / len(y) + 2 * lam * beta[j] for j in range(z.shape[1]))
    largest = max(abs(value) for value in gradient)
    assert largest <= 1e-9, (name, largest)
    objective = math.fsum(float(value * value) for value in residual) / len(y) + lam * math.fsum(float(value * value) for value in beta)
    close(model['objective'], objective)
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.zeros((len(beta), len(beta) + 1))
    penalty[:, 1:] = math.sqrt(len(z) * lam) * np.eye(len(beta))
    qr, _, rank, _ = lstsq(np.vstack([design, penalty]), np.concatenate([y, np.zeros(len(beta))]), lapack_driver='gelsy')
    assert rank == len(beta) + 1
    for actual, expected in zip([b, *beta], qr):
        close(actual, expected)
    hessian = 2 * design.T @ design / len(y)
    hessian[1:, 1:] += 2 * lam * np.eye(len(beta))
    minimum = float(np.linalg.eigvalsh(hessian)[0])
    assert minimum > 0
    return {'model': name, 'training_n': len(days), 'objective': objective, 'max_gradient': largest,
            'minimum_hessian_eigenvalue': minimum, 'max_augmented_qr_coefficient_delta': float(np.max(np.abs(qr - np.array([b, *beta]))))}


def scalar_prediction(model, row):
    raw = model['intercept'] + math.fsum((value - mean) / scale * coefficient for value, mean, scale, coefficient in zip(row, model['mean'], model['scale'], model['coefficient']))
    return raw, max(0, raw), math.expm1(max(0, raw))


def verify(stage, root=ROOT):
    data, pairs = load_context(root)
    filename = 'validation_models.json' if stage == 'validation' else 'frozen_models.json'
    models = json.loads((root / 'results' / filename).read_text(encoding='utf-8'))
    cache, certificates = {}, []
    selection = json.loads((root / 'results/selection.json').read_text(encoding='utf-8'))
    for name, model in models.items():
        pair = pairs[(model['qid'], model['language'])]
        if stage == 'validation':
            assert model['language'] == 'es' and pair['role'] == model['role'] == 'development'
            assert model['fitting_origin'] in ['2024-01-01', '2024-05-01', '2024-09-01']
        else:
            assert model['penalty'] == selection['selected'][model['family']]['penalty']
            assert model['fitting_origin'] == '2025-01-01'
        certificates.append(certificate(name, model, data, pair, cache))
    prediction_count = 0
    if stage == 'validation':
        grouped = defaultdict(list)
        with gzip.open(root / 'results/validation_predictions.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                model = models[row['model']]
                qid, lang, day = row['qid'], row['language'], row['date']
                assert model['qid'] == qid and lang == model['language'] == 'es'
                bounds = [('2024-01-01', '2024-04-30'), ('2024-05-01', '2024-08-31'), ('2024-09-01', '2024-12-31')][model['fold']]
                assert bounds[0] <= day <= bounds[1] and day in pairs[(qid, lang)]['eligible_dates']
                actual = data[(qid, lang, day)]
                assert int(row['actual']) == actual
                raw, clipped, prediction = scalar_prediction(model, independent_row(data, qid, lang, model['family'], day))
                close(float(row['pred_log_raw']), raw)
                close(float(row['pred_log_clipped']), clipped)
                close(float(row['prediction']), prediction)
                grouped[row['model']].append((day, clipped - math.log1p(actual), prediction - actual))
                prediction_count += 1
        with (root / 'results/validation_trials.csv').open(encoding='utf-8', newline='') as stream:
            trials = list(csv.DictReader(stream))
        expected = len([pair for pair in pairs.values() if pair['language'] == 'es' and pair['role'] == 'development']) * 45
        assert len(trials) == len(models) == expected
        objectives = defaultdict(list)
        for trial in trials:
            assert trial['status'] == 'ok'
            values = grouped[trial['model']]
            assert len(values) == int(trial['n']) == len({value[0] for value in values})
            msle = math.fsum(value[1] ** 2 for value in values) / len(values)
            mse = math.fsum(value[2] ** 2 for value in values) / len(values)
            mae = math.fsum(abs(value[2]) for value in values) / len(values)
            for metric, expected_value in [('msle', msle), ('rmsle', math.sqrt(msle)), ('mse', mse), ('rmse', math.sqrt(mse)), ('mae', mae)]:
                close(float(trial[metric]), expected_value)
            objectives[(trial['family'], float(trial['penalty']))].append(msle)
        scores = {key: math.fsum(values) / len(values) for key, values in objectives.items()}
        for candidate in selection['candidates']:
            close(candidate['mean_page_fold_msle'], scores[(candidate['family'], candidate['penalty'])])
        for family in ['own', 'english2', 'english7']:
            winner = min([(score, penalty) for (f, penalty), score in scores.items() if f == family])
            assert selection['selected'][family]['penalty'] == winner[1]
            close(selection['selected'][family]['mean_page_fold_msle'], winner[0])
    else:
        assert len(models) == len(pairs) * 3
        freeze = json.loads((root / 'results/freeze.json').read_text(encoding='utf-8'))
        assert freeze['models_sha256'] == hashlib.sha256((root / 'results/frozen_models.json').read_bytes()).hexdigest()
        assert freeze['selection_sha256'] == hashlib.sha256((root / 'results/selection.json').read_bytes()).hexdigest()
    report = {'status': 'PASS', 'stage': stage, 'independent_optimizer_certificates': len(certificates),
              'max_gradient': max(row['max_gradient'] for row in certificates),
              'min_hessian_eigenvalue': min(row['minimum_hessian_eigenvalue'] for row in certificates),
              'max_augmented_qr_coefficient_delta': max(row['max_augmented_qr_coefficient_delta'] for row in certificates),
              'scalar_predictions_and_metrics_checked': prediction_count,
              'certificates': certificates, 'absolute_tolerance': ABS, 'relative_tolerance': REL}
    (root / 'results' / ('INDEPENDENT_' + stage.upper() + '_CHECK.json')).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'certificates'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['validation', 'frozen'])
    verify(parser.parse_args().stage)
