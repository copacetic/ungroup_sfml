import json, sys, numpy as np
for name in sys.argv[1:]:
    r=json.load(open(f'/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/spatial/res_{name}.json'))
    print('=====', name)
    for k,v in r['ladder'].items(): print(' ', k, v['prog'], 'early',round(v['early'],2),'len',round(v['len']),'grp',round(v['grp'],2),'sp',round(v['spills'],1),'lv',round(v['leaves'],1),'bk',round(v['banks'],1))
    for c in ('crowd12','crowd20'):
        rr=r[c]; f=lambda k: round(float(np.mean([x[k] for x in rr])),3)
        print(' ', c, {k:f(k) for k in ('chir30','chir_all','wave','wave_abs','peak_lag','peak_corr','depletions','dep_prop','mean_stock','mining_frac','spills','length')})
        print('   progress by type', {s: round(float(np.mean([np.mean([p for i,p in enumerate(x['progress']) if (['solo','bail','loyal']*8)[i]==s]) for x in rr])),3) for s in ('solo','bail','loyal')})
