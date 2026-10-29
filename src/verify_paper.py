"""Check numerical manuscript claims against retained independent evidence."""
from pathlib import Path
import csv
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]


def main():
    paper=ROOT/'paper';text=(paper/'main.tex').read_text(encoding='utf-8')
    clean=text.replace(r'\allowbreak ','')
    def read(name):
        with (ROOT/'results'/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    summary=read('test_summary.csv');lookup={(r['language'],r['role'],r['method']):r for r in summary}
    roles=[('es','development','Spanish development'),('es','held','Spanish held topics'),('ja','development','Japanese development'),('ja','held','Japanese held topics')]
    for lang,role,label in roles:
        selected=[lookup[(lang,role,m)] for m in ['own','english2','english7']]
        expected=label+' & '+str(selected[0]['pages'])+' & '+' & '.join(f"{float(r['page_balanced_rmsle']):.4f}" for r in selected)
        assert expected in text,expected
    intervals=read('paired_intervals.csv')
    for lang in ['es','ja']:
        for method in ['english2','english7']:
            r=next(r for r in intervals if (r['language'],r['role'],r['method'])==(lang,'held',method))
            point=f"{float(r['point_difference']):+.6f}"
            if point.startswith('-'):point=point
            assert point in clean,point
            bounds='['+f"{float(r['conditional_95_low']):+.6f}"+','+f"{float(r['conditional_95_high']):+.6f}"+']'
            assert bounds in clean,bounds
    metrics=read('test_metrics.csv')
    for topic in ['Machine learning','Moon','Olympic Games','Tsunami']:
        part={r['method']:float(r['msle']) for r in metrics if r['language']=='ja' and r['topic']==topic}
        assert f"{part['english2']-part['own']:+.6f}" in text,topic
    for lang in ['es','ja']:
        qids={r['qid'] for r in metrics if r['language']==lang};wins=0
        for qid in qids:
            vals={r['method']:float(r['msle']) for r in metrics if r['language']==lang and r['qid']==qid}
            wins+=vals['english2']<vals['own']
        assert f'{wins} of {len(qids)}' in text
        formatted='/'.join(f"{float(lookup[(lang,'all',m)]['page_balanced_rmsle']):.4f}" for m in ['own','english2','english7'])
        assert formatted in text
    for method in ['own','english2']:
        r=lookup[('ja','held',method)]
        assert f"{float(r['page_balanced_mae']):.2f}" in text
        assert f"{float(r['page_balanced_mse']):,.1f}" in text
    source=json.loads((ROOT/'results/source_accounting.json').read_text(encoding='utf-8'))
    exclusions=[r for r in source['pairs'] if not r['eligible']]
    assert len(exclusions)==6
    names=[{'es':'Spanish','ja':'Japanese'}[r['language']]+' '+r['topic'] for r in exclusions]
    assert 'Excluded pairs are '+', '.join(names[:-1])+' and '+names[-1]+'.' in text
    selection=json.loads((ROOT/'results/selection.json').read_text(encoding='utf-8'))
    assert [selection['selected'][f]['penalty'] for f in ['own','english2','english7']]==[.001,.01,.001]
    assert '567' in text and '69,174' in text and '47,450' in text and '16,000' in text
    certificates=[json.loads((ROOT/'results'/f'INDEPENDENT_{stage}_CHECK.json').read_text(encoding='utf-8')) for stage in ['VALIDATION','FROZEN']]
    assert sum(r['independent_optimizer_certificates'] for r in certificates)==645
    assert f"{max(r['max_gradient'] for r in certificates)/1e-14:.2f}" in text
    assert f"{min(r['min_hessian_eigenvalue'] for r in certificates):.5f}" in text
    assert f"{max(r['max_augmented_qr_coefficient_delta'] for r in certificates)/1e-14:.2f}" in text
    scope=json.loads((paper/'PUBLICATION_SCOPE.json').read_text(encoding='utf-8'))
    assert scope['scope']=='independent-program-study' and scope['formal_submission_target'] is False
    for name,digest in scope['public_files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for p in (paper/'images').glob('*.pdf'):assert p.read_bytes()==(ROOT/'figures'/p.name).read_bytes(),p.name
    report={'status':'PASS','scope':'All four headline table rows; eight held interval endpoints/four points; four Japanese page contrasts; page wins; all-language aggregates; Japanese raw losses; six exclusions; selected penalties and fitting certificate extrema; approved manuscript/source/figure hashes.',
            'limitations':'Data-to-source claim check and exact reviewed PDF hashes. PDF visual/content review is retained separately; this does not establish peer review or universal correctness.'}
    (ROOT/'results/INDEPENDENT_PAPER_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
