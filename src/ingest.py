"""Reconcile fixed-title response identities and construct a nonimputed panel."""
from datetime import date, timedelta
from pathlib import Path
import csv
import gzip
import io
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/source'


def load(name):
    return json.loads(gzip.decompress((SOURCE / (name + '.json.gz')).read_bytes()))


def dates(start, end):
    while start <= end:
        yield start
        start += timedelta(days=1)


def ingest():
    topics = json.loads((SOURCE / 'topics.json').read_text(encoding='utf-8'))
    rows, identities, lookups = [], [], {}
    expected = {day.isoformat() for day in dates(date(2022, 1, 1), date(2025, 12, 31))}
    for lang in ['en', 'es', 'ja']:
        for topic in topics:
            qid, title = topic['qid'], topic['titles'][lang]
            info = load(f'{lang}-{qid}-page-identity')
            moves = load(f'{lang}-{qid}-moves')
            pages = info['query']['pages']
            assert len(pages) == 1
            page = pages[0]
            reasons = []
            if page.get('missing') or page.get('ns') != 0:
                reasons.append('Missing or non-article current page')
            if page.get('title', '').replace('_', ' ') != title.replace('_', ' '):
                reasons.append('Current title resolves to a different title')
            if not page.get('revisions') or page['revisions'][0]['timestamp'] >= '2022-01-01T00:00:00Z':
                reasons.append('No retained pre-2022 revision identity')
            events = moves['query']['logevents']
            if events:
                reasons.append('Current-title move log contains an event since 2022')
            if 'continue' in moves:
                reasons.append('Move log truncated at the declared bound')
            values = {}
            for year in [2022, 2023, 2024, 2025]:
                response = load(f'{lang}-{qid}-{year}')
                for record in response['items']:
                    stamp = record['timestamp']
                    day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8])).isoformat()
                    assert stamp.endswith('00') and int(stamp[:4]) == year
                    assert record['project'] == lang + '.wikipedia'
                    assert record['article'].replace('_', ' ') == title.replace('_', ' ')
                    assert record['access'] == 'all-access' and record['agent'] == 'user'
                    value = record['views']
                    assert isinstance(value, int) and value >= 0 and day not in values
                    values[day] = value
                    rows.append([qid, topic['topic'], topic['role'], lang, title, day, value])
            assert set(values) <= expected
            lookups[(qid, lang)] = values
            identities.append({'qid': qid, 'topic': topic['topic'], 'role': topic['role'], 'language': lang,
                               'title': title, 'pageid': page.get('pageid'),
                               'pre_2022_revision': page.get('revisions', []),
                               'bounded_move_events': events, 'exclusion_reasons': reasons,
                               'observed_days': len(values), 'missing_days': sorted(expected - set(values)),
                               'lineage_limit': 'Current-title log only; aliases are not aggregated and exhaustive historical equivalence is unverified.'})

    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['qid', 'topic', 'role', 'language', 'title', 'date', 'views'])
    writer.writerows(sorted(rows, key=lambda row: (row[0], row[3], row[5])))
    (ROOT / 'data/panel.csv.gz').write_bytes(gzip.compress(output.getvalue().encode('utf-8'), mtime=0))
    pairs, violations = [], []
    by_key = {(x['qid'], x['language']): x for x in identities}
    history_lags = sorted(set([2, 7, 14, 28, 21, 35] + list(range(2, 9)) + list(range(7, 14))))
    for topic in topics:
        for lang in ['es', 'ja']:
            reasons = by_key[(topic['qid'], lang)]['exclusion_reasons'] + by_key[(topic['qid'], 'en')]['exclusion_reasons']
            target, english = lookups[(topic['qid'], lang)], lookups[(topic['qid'], 'en')]
            eligible = []
            for day in dates(date(2022, 2, 5), date(2025, 12, 31)):
                histories = [(day - timedelta(days=lag)).isoformat() for lag in history_lags]
                if day.isoformat() in target and all(key in target and key in english for key in histories):
                    eligible.append(day.isoformat())
            counts = {'fit_first_fold': sum(day <= '2023-12-30' for day in eligible),
                      'validation_0': sum('2024-01-01' <= day <= '2024-04-30' for day in eligible),
                      'validation_1': sum('2024-05-01' <= day <= '2024-08-31' for day in eligible),
                      'validation_2': sum('2024-09-01' <= day <= '2024-12-31' for day in eligible),
                      'test': sum(day >= '2025-01-01' for day in eligible)}
            if counts['fit_first_fold'] < 500 or min(counts['validation_' + str(i)] for i in range(3)) < 100 or counts['test'] < 300:
                reasons.append('Insufficient prespecified date coverage')
            pairs.append({'qid': topic['qid'], 'topic': topic['topic'], 'role': topic['role'], 'language': lang,
                          'eligible': not reasons, 'exclusion_reasons': reasons, 'counts': counts,
                          'eligible_dates': eligible})
    for lang in ['es', 'ja']:
        for role, minimum in [('development', 6), ('held', 3)]:
            count = sum(x['eligible'] and x['language'] == lang and x['role'] == role for x in pairs)
            if count < minimum:
                violations.append({'language': lang, 'role': role, 'eligible_topics': count, 'minimum': minimum})
    result = {'status': 'PASS_SOURCE_COVERAGE_GATE' if not violations else 'SOURCE_COVERAGE_HOLD',
              'source_rows': len(rows), 'expected_days_per_title': len(expected), 'title_count': len(identities),
              'identities': identities, 'pairs': pairs, 'coverage_gate_violations': violations,
              'outcome_summary_inspected': False, 'historical_api_vintages': 'Unavailable; retrospective snapshot only'}
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/source_accounting.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'source_rows': len(rows),
                      'eligible_pairs': sum(x['eligible'] for x in pairs),
                      'excluded_pairs': [{k: x[k] for k in ['qid', 'language', 'exclusion_reasons']} for x in pairs if not x['eligible']],
                      'coverage_gate_violations': violations}, ensure_ascii=False))
    assert not violations, 'Retain source coverage hold; do not fit.'


if __name__ == '__main__':
    ingest()
