"""Recompute plotted values without importing the plotting or scoring code."""
from collections import defaultdict
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
ROOT=Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    result=root/'results'
    plotted=json.loads((result/'figure_values.json').read_text(encoding='utf-8'))
    def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-7,rel_tol=1e-9),(a,b)
    def read(name):
        with (result/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    for name,digest in plotted['source_sha256'].items():assert hashlib.sha256((result/name).read_bytes()).hexdigest()==digest,name
    rows=read('test_metrics.csv');lookup={(r['language'],r['role'],r['method'],r['topic']):r for r in rows}
    assert len(plotted['groups'])==20
    for r in plotted['groups']:
        chosen=[x for x in rows if all(x[k]==r[k] for k in ['language','role','method'])]
        assert len(chosen)==r['pages'] and sum(int(x['n']) for x in chosen)==r['n']
        close(r['page_balanced_rmsle'],math.sqrt(math.fsum(float(x['msle']) for x in chosen)/len(chosen)))
        close(r['page_balanced_msle'],math.fsum(float(x['msle']) for x in chosen)/len(chosen))
    expected=read('paired_intervals.csv');assert len(expected)==len(plotted['intervals'])==8
    for a,b in zip(expected,plotted['intervals']):
        for k in a:
            if k in ['language','role','method']:assert a[k]==b[k]
            else:close(a[k],b[k])
    assert len(plotted['japanese_held'])==8
    for r in plotted['japanese_held']:
        close(r['difference'],float(lookup[('ja','held',r['method'],r['topic'])]['msle'])-float(lookup[('ja','held','own',r['topic'])]['msle']))
    trials=read('validation_trials.csv');assert len(trials)==567
    assert len(plotted['validation'])==21
    scores={}
    for r in plotted['validation']:
        chosen=[x for x in trials if x['family']==r['family'] and float(x['penalty'])==r['penalty']]
        assert len(chosen)==r['fits']==27 and all(x['status']=='ok' for x in chosen)
        score=math.fsum(float(x['msle']) for x in chosen)/len(chosen)
        close(score,r['mean_page_fold_msle']);scores[(r['family'],r['penalty'])]=score
    for family,penalty in plotted['selected_penalties'].items():
        assert penalty==min((score,p) for (f,p),score in scores.items() if f==family)[1]
    buckets=defaultdict(list)
    with gzip.open(result/'test_predictions.csv.gz','rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r['role']=='held' and r['method'] in ['own','english2','english7']:
                buckets[(r['language'],r['method'],int(r['date'][5:7]),r['qid'])].append((max(0,float(r['pred_log_raw']))-math.log1p(int(r['actual'])))**2)
    assert len(plotted['held_months'])==48
    for r in plotted['held_months']:
        losses={}
        for method in ['own',r['method']]:
            parts=[math.fsum(v)/len(v) for (lang,m,month,qid),v in buckets.items() if (lang,m,month)==(r['language'],method,r['month'])]
            assert len(parts)==(3 if r['language']=='es' else 4)
            losses[method]=math.fsum(parts)/len(parts)
        close(r['difference'],losses[r['method']]-losses['own'])
    for name in ['01_predictive_comparison','02_paired_transfer','03_search_and_months']:
        assert (root/'figures'/f'{name}.pdf').read_bytes().startswith(b'%PDF-')
        assert (root/'figures'/f'{name}.png').stat().st_size>10000
    report={'status':'PASS','groups':20,'intervals':8,'japanese_topic_contrasts':8,'validation_settings':21,'held_month_contrasts':48,
            'scope':'All plotted values reconstructed; source hashes and three rendered file pairs checked. Visual layout requires separate inspection.',
            'absolute_tolerance':1e-7,'relative_tolerance':1e-9}
    (result/'INDEPENDENT_PRESENTATION_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':verify()
