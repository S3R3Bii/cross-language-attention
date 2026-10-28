"""Declared lag features and deterministic fixed-objective ridge fits."""
from datetime import date, timedelta
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PENALTIES = [0.01, 0.1, 1.0, 10.0, 100.0]
FAMILIES = ['own', 'english2', 'english7']
FOLDS = [('2024-01-01', '2024-04-30'), ('2024-05-01', '2024-08-31'), ('2024-09-01', '2024-12-31')]


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n', encoding='utf-8')


def read_panel(root=ROOT):
    panel = {}
    with gzip.open(root / 'data/panel.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['qid'], row['language'])
            values = panel.setdefault(key, {})
            assert row['date'] not in values
            values[row['date']] = int(row['views'])
    return panel


def load_pairs(root=ROOT):
    source = json.loads((root / 'results/source_accounting.json').read_text(encoding='utf-8'))
    assert source['status'] == 'PASS_SOURCE_COVERAGE_GATE', 'Source gate must pass before fitting.'
    return [row for row in source['pairs'] if row['eligible']]


def feature_row(day, own, english, family):
    day = date.fromisoformat(day) if isinstance(day, str) else day
    def logged(values, lag):
        return math.log1p(values[(day - timedelta(days=lag)).isoformat()])
    features = [logged(own, lag) for lag in [2, 7, 14, 28]]
    features += [math.fsum(logged(own, lag) for lag in range(2, 9)) / 7]
    names = ['own_lag2', 'own_lag7', 'own_lag14', 'own_lag28', 'own_mean2_8']
    features += [float(day.weekday() == weekday) for weekday in range(6)]
    names += ['weekday_' + str(weekday) for weekday in range(6)]
    phase = 2 * math.pi * (day.timetuple().tm_yday - 1) / 365.25
    features += [math.sin(phase), math.cos(phase)]
    names += ['annual_sin', 'annual_cos']
    if family != 'own':
        assert family in ['english2', 'english7']
        lags, window = ([2, 7, 14, 28], range(2, 9)) if family == 'english2' else ([7, 14, 21, 35], range(7, 14))
        features += [logged(english, lag) for lag in lags]
        features += [math.fsum(logged(english, lag) for lag in window) / 7]
        names += ['english_lag' + str(lag) for lag in lags] + ['english_mean' + str(window.start) + '_' + str(window.stop - 1)]
    return features, names


def matrix(pair, panel, family, start=None, end=None):
    days = [day for day in pair['eligible_dates'] if (start is None or day >= start) and (end is None or day <= end)]
    own, english = panel[(pair['qid'], pair['language'])], panel[(pair['qid'], 'en')]
    rows = [feature_row(day, own, english, family)[0] for day in days]
    names = feature_row(days[0], own, english, family)[1]
    y = np.array([own[day] for day in days], dtype=np.float64)
    return days, np.array(rows, dtype=np.float64), y, names


def fit(x, counts, penalty, names, days):
    target = np.log1p(counts)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    intercept = float(target.mean())
    normal = z.T @ z / len(z) + penalty * np.eye(z.shape[1])
    coefficient = np.linalg.solve(normal, z.T @ (target - intercept) / len(z))
    residual = intercept + z @ coefficient - target
    gradient = 2 * z.T @ residual / len(z) + 2 * penalty * coefficient
    assert np.isfinite(coefficient).all() and np.max(np.abs(gradient)) <= 1e-9
    return {'feature_names': names, 'mean': mean.tolist(), 'scale': scale.tolist(),
            'coefficient': coefficient.tolist(), 'intercept': intercept, 'penalty': penalty,
            'training_n': len(days), 'training_start': days[0], 'training_end': days[-1],
            'training_dates_sha256': sha(('\n'.join(days) + '\n').encode('utf-8')),
            'objective': float(np.mean(residual ** 2) + penalty * np.dot(coefficient, coefficient)),
            'fit_max_gradient': float(max(np.max(np.abs(gradient)), abs(2 * residual.mean()))),
            'normal_condition': float(np.linalg.cond(normal))}


def predict(model, x):
    raw_log = model['intercept'] + ((x - np.array(model['mean'])) / np.array(model['scale'])) @ np.array(model['coefficient'])
    clipped_log = np.maximum(raw_log, 0)
    counts = np.expm1(clipped_log)
    assert np.isfinite(counts).all()
    return raw_log, clipped_log, counts


def metrics(actual, raw_log, prediction, mase_scale=None):
    error = prediction - actual
    log_error = np.maximum(raw_log, 0) - np.log1p(actual)
    mse, msle = float(np.mean(error ** 2)), float(np.mean(log_error ** 2))
    return {'n': len(actual), 'msle': msle, 'rmsle': math.sqrt(msle),
            'mae': float(np.mean(np.abs(error))), 'mse': mse, 'rmse': math.sqrt(mse),
            'mase': None if not mase_scale else float(np.mean(np.abs(error)) / mase_scale)}


def csv_gzip(path, rows, names):
    import io
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, names, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(buffer.getvalue().encode('utf-8'), mtime=0))
