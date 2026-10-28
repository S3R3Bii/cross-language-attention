"""Fetch a bounded, sequential public snapshot without inspecting test outcomes."""
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/source-v2'
USER_AGENT = 'CrossLanguageAttentionResearch/0.1 (https://github.com/S3R3Bii/cross-language-attention) Python-urllib/3.10'
TITLES = ['Earth', 'Moon', 'Mars', 'Sun', 'Volcano', 'Earthquake', 'Tsunami',
          'Artificial_intelligence', 'Machine_learning', 'Climate_change',
          'Olympic_Games', 'Solar_eclipse', 'Water', 'Photosynthesis', 'Computer', 'Music']
HELD = {'Moon', 'Tsunami', 'Machine learning', 'Olympic Games'}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def request(url):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if attempt == 0 and error.code in [429, 500, 502, 503, 504]:
                time.sleep(min(10, max(1, int(error.headers.get('Retry-After', '2')))))
                continue
            raise


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for row in manifest['requests']:
            if row['status'] == 'ok':
                stored = (ROOT / row['file']).read_bytes()
                assert digest(stored) == row['stored_sha256'], row['file']
                assert digest(gzip.decompress(stored)) == row['response_sha256'], row['file']
        print('Verified the exact retained public source snapshot; no network refresh.')
        return

    records = []
    recovered = {}
    original_manifest = ROOT / 'data/source/manifest.json'
    reusable = {row['url']: row for row in json.loads(original_manifest.read_text(encoding='utf-8'))['requests'] if row['status'] == 'ok'} if original_manifest.exists() else {}
    failed_path = DATA / 'failed-attempt.json'
    if failed_path.exists():
        previous = json.loads(failed_path.read_text(encoding='utf-8'))
        history = DATA / 'attempts'
        history.mkdir(exist_ok=True)
        saved = history / f'failed-{len(list(history.glob("failed-*.json"))) + 1}.json'
        shutil.copy2(failed_path, saved)
        recovered = {row['file']: row for row in previous['requests'] if row['status'] == 'ok'}
    started = datetime.now(timezone.utc).isoformat()

    def save(url, name):
        path = DATA / (name + '.json.gz')
        if path.exists():
            previous = recovered.get(path.relative_to(ROOT).as_posix())
            assert previous is not None and previous['url'] == url, path.name
            stored = path.read_bytes()
            assert digest(stored) == previous['stored_sha256'], path.name
            blob = gzip.decompress(stored)
            assert digest(blob) == previous['response_sha256'], path.name
            records.append(previous)
            return json.loads(blob)
        if url in reusable:
            previous = reusable[url]
            stored = (ROOT / previous['file']).read_bytes()
            assert digest(stored) == previous['stored_sha256']
            blob = gzip.decompress(stored)
            assert digest(blob) == previous['response_sha256']
            path.write_bytes(stored)
            entry = dict(previous, file=path.relative_to(ROOT).as_posix(), reused_from=previous['file'])
            records.append(entry)
            return json.loads(blob)
        entry = {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                 'file': path.relative_to(ROOT).as_posix()}
        try:
            blob = request(url)
            parsed = json.loads(blob)
            if 'error' in parsed:
                raise ValueError('API returned an error: ' + str(parsed['error'].get('code')))
            stored = gzip.compress(blob, mtime=0)
            path.write_bytes(stored)
            entry.update(status='ok', response_sha256=digest(blob), stored_sha256=digest(stored))
        except Exception as error:
            entry.update(status='failed', error_type=type(error).__name__, error=str(error))
            records.append(entry)
            (DATA / 'failed-attempt.json').write_text(json.dumps({'started_utc': started, 'requests': records}, indent=2) + '\n', encoding='utf-8')
            raise
        records.append(entry)
        return parsed

    params = {'action': 'wbgetentities', 'sites': 'enwiki', 'titles': '|'.join(TITLES),
              'props': 'sitelinks', 'sitefilter': 'enwiki|eswiki|jawiki', 'format': 'json'}
    entities = save('https://www.wikidata.org/w/api.php?' + urllib.parse.urlencode(params), 'topic-sitelinks')
    assert len(entities['entities']) == 16
    topics = []
    for qid, entity in entities['entities'].items():
        links = entity['sitelinks']
        assert set(links) == {'enwiki', 'eswiki', 'jawiki'}
        title = links['enwiki']['title']
        topics.append({'qid': qid, 'topic': title, 'role': 'held' if title in HELD else 'development',
                       'titles': {lang: links[lang + 'wiki']['title'] for lang in ['en', 'es', 'ja']}})
    topics.sort(key=lambda x: x['topic'])
    (DATA / 'topics.json').write_text(json.dumps(topics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    for lang in ['en', 'es', 'ja']:
        for topic in topics:
            title = topic['titles'][lang]
            # Date-bounded revision queries accept one page per request.
            params = {'action': 'query', 'prop': 'info|revisions', 'titles': title,
                      'rvprop': 'ids|timestamp', 'rvlimit': 1, 'rvstart': '2021-12-31T23:59:59Z',
                      'format': 'json', 'formatversion': 2, 'redirects': 1}
            save('https://' + lang + '.wikipedia.org/w/api.php?' + urllib.parse.urlencode(params),
                 f"{lang}-{topic['qid']}-page-identity")
            params = {'action': 'query', 'list': 'logevents', 'letype': 'move', 'letitle': title,
                      'leprop': 'ids|title|type|timestamp|details', 'leend': '2022-01-01T00:00:00Z',
                      'lelimit': 500, 'format': 'json', 'formatversion': 2}
            save('https://' + lang + '.wikipedia.org/w/api.php?' + urllib.parse.urlencode(params),
                 f"{lang}-{topic['qid']}-moves")

    for lang in ['en', 'es', 'ja']:
        for topic in topics:
            article = urllib.parse.quote(topic['titles'][lang].replace(' ', '_'), safe='')
            for year in [2022, 2023, 2024, 2025]:
                url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia.org/all-access/user/{article}/daily/{year}0101/{year}1231'
                save(url, f"{lang}-{topic['qid']}-{year}")
            print(lang, topic['qid'], 'four annual responses retained', flush=True)

    manifest = {'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                'source': 'Wikimedia Analytics API and selected factual MediaWiki/Wikidata metadata',
                'user_agent': USER_AGENT, 'pageview_license': 'CC0-1.0', 'wikidata_structured_license': 'CC0-1.0',
                'scope': 'Fixed titles; aggregate classified-user daily requests; no editor names or comments requested',
                'requests': records, 'test_outcomes_inspected': False}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Retained', len(records), 'responses. No outcome summary or fitting was performed.')


if __name__ == '__main__':
    fetch()
