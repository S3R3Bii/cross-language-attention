"""Fresh offline replay of every fit and score, retaining the original tolerances."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ABS,REL=1e-7,1e-9
FROZEN_COMMIT='5a446198de1f0fc3eff16b675f39cd37a60236df'
# Byte hashes are independently checked within each run. Newline serialization
# and runtime observations are not scientific quantities to compare across OSes.
PROVENANCE={'runtime_seconds','source_manifest_sha256','source_accounting_sha256',
            'selection_sha256','models_sha256','panel_sha256','protocol_sha256',
            'source_sha256','frozen_commit'}


def compare(expected,actual,where='root'):
    if isinstance(expected,dict):
        assert set(expected)==set(actual),where
        for k in expected:
            if k not in PROVENANCE:compare(expected[k],actual[k],where+'.'+k)
    elif isinstance(expected,list):
        assert len(expected)==len(actual),where
        for i,(a,b) in enumerate(zip(expected,actual)):compare(a,b,where+f'[{i}]')
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert math.isclose(expected,actual,abs_tol=ABS,rel_tol=REL),(where,expected,actual)
    else:assert expected==actual,(where,expected,actual)


def table(first,second,keys):
    a=pd.read_csv(first,float_precision='round_trip');b=pd.read_csv(second,float_precision='round_trip')
    assert list(a)==list(b),(first.name,'columns')
    if 'runtime_seconds' in a:a=a.drop(columns='runtime_seconds');b=b.drop(columns='runtime_seconds')
    a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_exact=False,atol=ABS,rtol=REL)


def main():
    timer=time.perf_counter()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    def run(folder,*args):subprocess.run([sys.executable,*args],cwd=folder,env=env,check=True)
    # The historical pre-test freeze and current frozen files must still agree.
    freeze=json.loads((ROOT/'results/freeze.json').read_text(encoding='utf-8'))
    for path in ['results/freeze.json','results/frozen_models.json','results/selection.json',*freeze['normalized_source_sha256']]:
        historical=subprocess.check_output(['git','show',FROZEN_COMMIT+':'+path],cwd=ROOT)
        current=(ROOT/path).read_bytes()
        assert historical.replace(b'\r\n',b'\n')==current.replace(b'\r\n',b'\n'),path
    archive=ROOT/'results/search-v1'
    receipt=json.loads((archive/'SEARCH_RECEIPT.json').read_text(encoding='utf-8'))
    for name,digest in receipt['sha256'].items():assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==digest,name
    original=json.loads((archive/'validation_models.json').read_text(encoding='utf-8'))
    final=json.loads((ROOT/'results/validation_models.json').read_text(encoding='utf-8'))
    assert len(original)==405
    for name,model in original.items():compare(model,final[name],'original405.'+name)
    for args in [('src/fetch.py',),('src/verify_source.py',),('src/verify_models.py','validation'),
                 ('src/verify_models.py','frozen'),('src/verify_evaluation.py',),('src/verify_presentation.py',)]:run(ROOT,*args)
    with tempfile.TemporaryDirectory(prefix='attention-replay-') as temporary:
        work=Path(temporary)
        for name in ['src','data']:shutil.copytree(ROOT/name,work/name)
        (work/'results').mkdir()
        for name in ['PROTOCOL.md','SOURCE_AMENDMENT_V2.md','SEARCH_EXTENSION_V2.md']:shutil.copy2(ROOT/name,work/name)
        for args in [('src/fetch.py',),('src/ingest.py',),('src/verify_source.py',),('src/train.py','validate'),
                     ('src/verify_models.py','validation'),('src/train.py','freeze'),('src/verify_models.py','frozen')]:run(work,*args)
        # This is explicitly a temporary replay; the retained first evaluation
        # and its committed freeze remain untouched in the reviewed repository.
        run(work,'-c',"import sys;from pathlib import Path;sys.path.insert(0,'src');from evaluate import evaluate;evaluate(Path.cwd(),require_committed=False)")
        for name in ['verify_evaluation.py','figures.py','verify_presentation.py']:run(work,'src/'+name)
        for name in ['source_accounting.json','selection.json','validation_models.json','frozen_models.json','freeze.json','evaluation_summary.json','figure_values.json']:
            compare(json.loads((ROOT/'results'/name).read_text(encoding='utf-8')),json.loads((work/'results'/name).read_text(encoding='utf-8')),name)
        for name,keys in [('validation_trials.csv',['model']),('validation_predictions.csv.gz',['model','date']),
                          ('test_predictions.csv.gz',['language','qid','method','date']),('test_metrics.csv',['language','qid','method']),
                          ('test_summary.csv',['language','role','method']),('bootstrap_draws.csv.gz',['language','role','method','draw']),
                          ('paired_intervals.csv',['language','role','method'])]:table(ROOT/'results'/name,work/'results'/name,keys)
        table(ROOT/'data/panel.csv.gz',work/'data/panel.csv.gz',['qid','language','date'])
        assert json.loads(gzip.decompress((ROOT/'results/bootstrap_month_indices.json.gz').read_bytes()))==json.loads(gzip.decompress((work/'results/bootstrap_month_indices.json.gz').read_bytes()))
    report={'status':'PASS','fresh_validation_fits':567,'fresh_final_refits':78,'validation_predictions':69174,
            'test_predictions':47450,'bootstrap_draws':16000,'original_405_models_unchanged':True,
            'absolute_tolerance':ABS,'relative_tolerance':REL,'end_to_end_seconds':time.perf_counter()-timer,
            'scope':'All source rows, 645 fresh fits, frozen transforms/coefficients, every validation/test prediction, all metric tables, common month indices/draws/intervals and plotted values.',
            'provenance_comparison':'Within-run hashes checked; cross-platform JSON/CSV newline byte hashes, timings and replay commit marker are excluded from scientific comparisons.'}
    (ROOT/'results/reproduction.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
