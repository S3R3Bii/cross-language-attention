"""Independent standard-library/SQL reconciliation; imports no ingestion/model code."""
from datetime import date, timedelta
from pathlib import Path
import csv
import gzip
import hashlib
import json
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/source-v2'


def verify(root=ROOT):
    source = root / 'data/source-v2'
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['requests']) == 289
    for row in manifest['requests']:
        assert row['status'] == 'ok'
        blob = (root / row['file']).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == row['stored_sha256']
        assert hashlib.sha256(gzip.decompress(blob)).hexdigest() == row['response_sha256']
        if 'reused_from' in row:
            assert blob == (root / row['reused_from']).read_bytes()
        def no_editor_fields(value):
            if isinstance(value, dict):
                assert not {'user', 'userid', 'comment', 'parsedcomment'}.intersection(value)
                for child in value.values():
                    no_editor_fields(child)
            elif isinstance(value, list):
                for child in value:
                    no_editor_fields(child)
        no_editor_fields(json.loads(gzip.decompress(blob)))

    def raw(name):
        return json.loads(gzip.decompress((source / (name + '.json.gz')).read_bytes()))

    topics = json.loads((source / 'topics.json').read_text(encoding='utf-8'))
    entities = raw('topic-sitelinks')['entities']
    assert len(topics) == len(entities) == 16
    held = {'Moon', 'Tsunami', 'Machine learning', 'Olympic Games'}
    for topic in topics:
        entity = entities[topic['qid']]
        assert topic['topic'] == entity['sitelinks']['enwiki']['title']
        assert topic['role'] == ('held' if topic['topic'] in held else 'development')
        assert topic['titles'] == {lang: entity['sitelinks'][lang + 'wiki']['title'] for lang in ['en', 'es', 'ja']}

    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE original(qid TEXT,language TEXT,day TEXT,views INTEGER,title TEXT,PRIMARY KEY(qid,language,day))')
    db.execute('CREATE TABLE exported(qid TEXT,language TEXT,day TEXT,views INTEGER,title TEXT,PRIMARY KEY(qid,language,day))')
    for topic in topics:
        for lang in ['en', 'es', 'ja']:
            for year in range(2022, 2026):
                for item in raw(f"{lang}-{topic['qid']}-{year}")['items']:
                    stamp = item['timestamp']
                    day = '-'.join([stamp[:4], stamp[4:6], stamp[6:8]])
                    assert str(year) == stamp[:4] and stamp[8:] == '00'
                    assert item['project'] == lang + '.wikipedia' and item['access'] == 'all-access' and item['agent'] == 'user'
                    assert item['article'].replace('_', ' ') == topic['titles'][lang]
                    assert type(item['views']) is int and item['views'] >= 0
                    db.execute('INSERT INTO original VALUES(?,?,?,?,?)', (topic['qid'], lang, day, item['views'], topic['titles'][lang]))
    with gzip.open(root / 'data/panel.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            topic = next(x for x in topics if x['qid'] == row['qid'])
            assert row['topic'] == topic['topic'] and row['role'] == topic['role']
            db.execute('INSERT INTO exported VALUES(?,?,?,?,?)', (row['qid'], row['language'], row['date'], int(row['views']), row['title']))
    assert not db.execute('SELECT * FROM original EXCEPT SELECT * FROM exported').fetchall()
    assert not db.execute('SELECT * FROM exported EXCEPT SELECT * FROM original').fetchall()

    accounting = json.loads((root / 'results/source_accounting.json').read_text(encoding='utf-8'))
    assert len(accounting['identities']) == 48
    identity_ok = {}
    for identity in accounting['identities']:
        lang, qid = identity['language'], identity['qid']
        topic = next(x for x in topics if x['qid'] == qid)
        page = raw(f'{lang}-{qid}-page-identity')['query']['pages'][0]
        moves = raw(f'{lang}-{qid}-moves')
        events = [event for event in moves['query']['logevents'] if '2022-01-01T00:00:00Z' <= event['timestamp'] < '2026-01-01T00:00:00Z']
        ok = not page.get('missing') and page.get('ns') == 0 and page.get('title') == topic['titles'][lang]
        ok = bool(ok and page.get('revisions') and page['revisions'][0]['timestamp'] < '2022-01-01T00:00:00Z' and not events and 'continue' not in moves)
        assert identity['pageid'] == page.get('pageid')
        assert identity['analysis_period_move_events'] == events
        assert bool(not identity['exclusion_reasons']) == ok
        identity_ok[(qid, lang)] = ok
    calendar = [(date(2022, 1, 1) + timedelta(days=i)).isoformat() for i in range(1461)]
    lagset = set(range(2, 9)) | set(range(7, 14)) | {14, 21, 28, 35}
    eligible_counts = {}
    for pair in accounting['pairs']:
        qid, lang = pair['qid'], pair['language']
        own = {day for (day,) in db.execute('SELECT day FROM original WHERE qid=? AND language=?', (qid, lang))}
        english = {day for (day,) in db.execute('SELECT day FROM original WHERE qid=? AND language="en"', (qid,))}
        dates = []
        for day in calendar[35:]:
            history = {(date.fromisoformat(day) - timedelta(days=lag)).isoformat() for lag in lagset}
            if day in own and history <= own and history <= english:
                dates.append(day)
        assert pair['eligible_dates'] == dates
        counts = {'fit_first_fold': sum(day <= '2023-12-30' for day in dates),
                  'validation_0': sum('2024-01-01' <= day <= '2024-04-30' for day in dates),
                  'validation_1': sum('2024-05-01' <= day <= '2024-08-31' for day in dates),
                  'validation_2': sum('2024-09-01' <= day <= '2024-12-31' for day in dates),
                  'test': sum(day.startswith('2025') for day in dates)}
        assert pair['counts'] == counts
        sufficient = counts['fit_first_fold'] >= 500 and min(counts['validation_' + str(i)] for i in range(3)) >= 100 and counts['test'] >= 300
        ok = identity_ok[(qid, lang)] and identity_ok[(qid, 'en')] and sufficient
        assert pair['eligible'] == ok
        if ok:
            key = (lang, pair['role'])
            eligible_counts[key] = eligible_counts.get(key, 0) + 1
    assert all(eligible_counts.get((lang, role), 0) >= minimum for lang in ['es', 'ja'] for role, minimum in [('development', 6), ('held', 3)])
    assert accounting['status'] == 'PASS_SOURCE_COVERAGE_GATE'
    result = {'status': 'PASS', 'responses_hashed': 289, 'source_rows_independently_joined': db.execute('SELECT COUNT(*) FROM original').fetchone()[0],
              'identities_checked': 48, 'pair_dates_and_eligibility_checked': 32,
              'eligible_counts': [{'language': key[0], 'role': key[1], 'topics': value} for key, value in sorted(eligible_counts.items())],
              'scope': 'All retained source counts, date keys, selected metadata fields, reuse bytes and cohort rules; bounded current-title lineage only; no live-vintage claim.'}
    (root / 'results/INDEPENDENT_SOURCE_CHECK.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    verify()
