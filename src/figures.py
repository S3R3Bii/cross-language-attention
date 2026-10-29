"""Publication figures bound to the complete retained comparisons."""
from pathlib import Path
import hashlib
import json
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'private/mpl-config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILIES = ['own', 'english2', 'english7']
METHODS = FAMILIES + ['lag7', 'weekly4']
LABELS = {'own': 'Own history', 'english2': '+ Fast English', 'english7': '+ Slower English',
          'lag7': 'Last week', 'weekly4': 'Four-week mean'}
COLORS = {'own': '#263f59', 'english2': '#067b83', 'english7': '#c16a2d', 'lag7': '#8a919a', 'weekly4': '#bbc0c6'}
GROUPS = [('es', 'development'), ('es', 'held'), ('ja', 'development'), ('ja', 'held')]
LANGS = {'es': 'Spanish', 'ja': 'Japanese'}


def make(root=ROOT):
    results, out = root / 'results', root / 'figures'
    out.mkdir(exist_ok=True)
    summary = pd.read_csv(results / 'test_summary.csv', float_precision='round_trip')
    intervals = pd.read_csv(results / 'paired_intervals.csv', float_precision='round_trip')
    metric = pd.read_csv(results / 'test_metrics.csv', float_precision='round_trip')
    predictions = pd.read_csv(results / 'test_predictions.csv.gz', float_precision='round_trip')
    selection = json.loads((results / 'selection.json').read_text(encoding='utf-8'))
    plotted = {'groups': [], 'intervals': intervals.to_dict('records'), 'japanese_held': [],
               'validation': selection['candidates'], 'selected_penalties': {k: v['penalty'] for k,v in selection['selected'].items()},
               'held_months': []}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    def save(fig, name):
        fig.savefig(out / (name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(out / (name + '.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.9), sharex=True)
    for ax, (lang, role) in zip(axes.flat, GROUPS):
        rows = summary[(summary.language == lang) & (summary.role == role)].set_index('method').loc[METHODS]
        values = rows.page_balanced_rmsle.to_numpy()
        plotted['groups'] += rows.reset_index().to_dict('records')
        ax.barh(np.arange(5), values, color=[COLORS[m] for m in METHODS], height=.67)
        ax.set_yticks(np.arange(5), [LABELS[m] for m in METHODS]); ax.invert_yaxis()
        ax.set_title(f'{LANGS[lang]} / {role} topics ({int(rows.pages.iloc[0])} pages)', loc='left')
        ax.set_xlim(0, .46); ax.set_xlabel('Page-balanced RMSLE (lower is better)')
        ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
        for i,v in enumerate(values): ax.text(v+.005, i, f'{v:.4f}', va='center', fontsize=9)
    fig.suptitle('English history offers uneven gains beyond own-page history', x=.02, ha='left', fontweight='bold', fontsize=14)
    fig.text(.02, .01, '2025 test year; 365 dates/page. RMSLE = sqrt(equal-page mean MSLE). Roles concern parameter selection; every page uses its own earlier history.', fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94)); save(fig, '01_predictive_comparison')

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), gridspec_kw={'width_ratios':[1.35,1]})
    ax = axes[0]; labels=[]; positions=[]
    for i,(lang,role) in enumerate(GROUPS):
        for j,method in enumerate(['english2','english7']):
            row = intervals[(intervals.language==lang)&(intervals.role==role)&(intervals.method==method)].iloc[0]
            y=i*2.5+j; positions.append(y); labels.append(f'{LANGS[lang]} {role}\n{LABELS[method]}')
            ax.plot([row.conditional_95_low,row.conditional_95_high],[y,y],color=COLORS[method],lw=2)
            ax.plot(row.point_difference,y,'o',color=COLORS[method],ms=5)
    ax.set_yticks(positions,labels,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='#30343a',lw=.8)
    ax.set_xlabel('Change in MSLE: English model minus own history')
    ax.set_title('All paired comparisons',loc='left');ax.grid(axis='x',alpha=.18)
    ax=axes[1]
    held=metric[(metric.language=='ja')&(metric.role=='held')].pivot(index='topic',columns='method',values='msle').sort_index()
    for j,method in enumerate(['english2','english7']):
        values=held[method]-held['own']
        ax.barh(np.arange(len(values))+(j-.5)*.30,values,height=.28,label=LABELS[method],color=COLORS[method])
        plotted['japanese_held'] += [{'topic':topic,'method':method,'difference':float(value)} for topic,value in values.items()]
    ax.set_yticks(np.arange(4),['Machine learning','Moon','Olympic Games','Tsunami']);ax.invert_yaxis()
    ax.axvline(0,color='#30343a',lw=.8);ax.set_xlabel('Per-page change in MSLE')
    ax.set_title('Japanese held topics',loc='left');ax.legend(frameon=False,fontsize=8,loc='upper right');ax.grid(axis='x',alpha=.18)
    fig.suptitle('A larger Tsunami-page loss reverses three fast-English wins',x=.02,ha='left',fontweight='bold',fontsize=13)
    fig.text(.02,.02,'Left: conditional 95% percentile intervals, 2,000 common-month draws/group. Negative favors English inputs.\nIntervals condition on this fixed panel and 12 months; they are not population or causal guarantees. Right: point differences, no page-level intervals.',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.93));save(fig,'02_paired_transfer')

    fig=plt.figure(figsize=(10.4,7.1));grid=fig.add_gridspec(2,6,hspace=.60,wspace=.90)
    for i,family in enumerate(FAMILIES):
        ax=fig.add_subplot(grid[0,i*2:i*2+2])
        candidates=sorted([r for r in selection['candidates'] if r['family']==family],key=lambda r:r['penalty'])
        ax.semilogx([r['penalty'] for r in candidates],[r['mean_page_fold_msle'] for r in candidates],'-o',color=COLORS[family],ms=4)
        chosen=selection['selected'][family]
        ax.plot(chosen['penalty'],chosen['mean_page_fold_msle'],'o',mec='black',mfc='white',ms=8)
        ax.set_title(f"{LABELS[family]}\nSelected penalty {chosen['penalty']:g}",loc='left')
        ax.set_xlabel('Ridge penalty (log axis)');ax.set_ylim(0,.46);ax.grid(alpha=.18)
        if i==0:ax.set_ylabel('Mean page-fold validation MSLE')
    predictions['month']=predictions.date.str.slice(5,7).astype(int)
    predictions['loss']=(predictions.pred_log_clipped-np.log1p(predictions.actual))**2
    for i,lang in enumerate(['es','ja']):
        ax=fig.add_subplot(grid[1,i*3:i*3+3])
        part=predictions[(predictions.language==lang)&(predictions.role=='held')]
        means=part.groupby(['qid','month','method']).loss.mean().groupby(['month','method']).mean().unstack('method')
        for method in ['english2','english7']:
            values=means[method]-means['own']
            ax.plot(values.index,values.values,'-o',color=COLORS[method],ms=3,label=LABELS[method])
            plotted['held_months'] += [{'language':lang,'method':method,'month':int(month),'difference':float(value)} for month,value in values.items()]
        ax.axhline(0,color='#30343a',lw=.8);ax.set_xticks(range(1,13));ax.set_xlabel('Calendar month of 2025');ax.set_ylim(-.012,.030)
        ax.set_title(f'{LANGS[lang]} held topics: monthly point differences',loc='left',fontsize=10)
        ax.set_ylabel('Change in equal-page MSLE');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('The complete validation grid and test-month sensitivity',x=.02,ha='left',fontweight='bold',fontsize=14)
    fig.text(.02,.01,'Top: all 567 validation fits, 21 settings, 9 Spanish development pages × 3 forward folds; white circles mark finite-grid selections.\nBottom: all held-topic months and both English families on common axes; descriptive sensitivity only, with no post-test retuning.',fontsize=8)
    fig.subplots_adjust(top=.86,bottom=.12,left=.09,right=.98);save(fig,'03_search_and_months')
    sources=['test_summary.csv','paired_intervals.csv','test_metrics.csv','test_predictions.csv.gz','selection.json']
    plotted['source_sha256']={name:hashlib.sha256((results/name).read_bytes()).hexdigest() for name in sources}
    (results/'figure_values.json').write_text(json.dumps(plotted,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Rendered three PDF/PNG figures with complete numeric bindings.')


if __name__=='__main__':make()
