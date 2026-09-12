"""Islands statistic: after t>150 s, of the players inside a lobe (within isl_r of an inner mine), the fraction whose
largest remaining need equals the lobe's mine type (chance ~0.25). Base uses the nearest inner mine as the 'lobe'."""
import os, sys, json, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from ungroup.native import NativeBatch, Config
name=sys.argv[1]; n=20; E=8; seats=(['solo','bail','loyal']*8)[:n]
b=NativeBatch(E, Config(n_players=n), seed=5000, decide_every=5)
for e in range(E): b.set_seats(e,seats); b.reset(e,5000+e)
metas=[b.meta(e) for e in range(E)]; acts=np.zeros((E,n,4),dtype=np.int32); done=[False]*E
hit=np.zeros(E); tot=np.zeros(E); cross=np.zeros(E); last=[dict() for _ in range(E)]; k=0
while not all(done):
    _,_,_,ep=b.step(acts,auto_reset=False)
    for d in ep: done[d['env']]=True
    k+=1
    if k%2: continue
    for e in range(E):
        if done[e]: continue
        fr=b.frame(e)
        if fr['t']<150: continue
        m=metas[e]; mp=np.array(m['mine_pos']); mt=m['mine_type']; inner=[i for i in range(8) if i%2==1]
        frac=fr['t']/240; ri=0.6+(0.25-0.6)*min(1,(frac-0.5)/0.5)
        for bd in fr['bodies']:
            x=np.array([bd['x'],bd['y']]); d=[np.linalg.norm(x-mp[i]) for i in inner]; j=int(np.argmin(d))
            if d[j]>ri: continue
            lobe=mt[inner[j]]
            for pid in bd['m']:
                need=np.array(m['needs'][pid]); bk=np.array(fr['players'][pid]['banked']); rem=np.maximum(0,need-bk)/need
                top=int(np.argmax(rem)); tot[e]+=1; hit[e]+= (top==lobe)
                if pid in last[e] and last[e][pid]!=j: cross[e]+=1
                last[e][pid]=j
print(name, 'lobe-need match', round(float((hit/np.maximum(1,tot)).mean()),3), 'lobe switches/player', round(float(cross.mean()/n),2))
