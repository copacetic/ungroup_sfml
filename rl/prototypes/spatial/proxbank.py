"""Pad-well statistic: for each group bank, which member pad was nearest when the group's pool first reached 10 units
(the loyal threshold), and did the bank go there? Also the furthest-behind member (loyal's intended receiver)."""
import os, sys, json, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from ungroup.native import NativeBatch, Config
name=sys.argv[1]; seats=sys.argv[2].split(','); n=len(seats); E=24
b=NativeBatch(E, Config(n_players=n), seed=1000, decide_every=5)
for e in range(E): b.set_seats(e,seats); b.reset(e,1000+e)
metas=[b.meta(e) for e in range(E)]; acts=np.zeros((E,n,4),dtype=np.int32); done=[False]*E
armed=[dict() for _ in range(E)]  # key: frozenset(members) -> (nearest pad member, furthest-behind member)
near_hit=0; behind_hit=0; total=0; k=0
def padpos(e, i, R):
    r=max(R-0.06-0.02,0.1); a=metas[e]['pads'][i]; return np.array([r*np.cos(a), r*np.sin(a)])
while not all(done):
    _,_,_,ep=b.step(acts,auto_reset=False)
    for d in ep: done[d['env']]=True
    k+=1
    for e in range(E):
        if done[e]: continue
        fr=b.frame(e); R=fr['R']
        for ev in fr['events']:
            if ev.get('kind')=='bank' and len(ev['group'])>1:
                key=frozenset(ev['group'])
                if key in armed[e]:
                    nearest,behind=armed[e].pop(key); total+=1; near_hit+= ev['player']==nearest; behind_hit+= ev['player']==behind
        for bd in fr['bodies']:
            if len(bd['m'])<2: continue
            key=frozenset(bd['m']); pool=sum(bd['pool'])
            if pool>=10 and key not in armed[e]:
                x=np.array([bd['x'],bd['y']])
                dists={i: np.linalg.norm(padpos(e,i,R)-x) for i in bd['m']}
                prog={i: np.mean([min(fr['players'][i]['banked'][t]/metas[e]['needs'][i][t],1) for t in range(4)]) for i in bd['m']}
                armed[e][key]=(min(dists,key=dists.get), min(prog,key=prog.get))
            elif pool<1 and key in armed[e]: armed[e].pop(key)
print(name, seats[0], 'group banks', total, 'to nearest-at-threshold %.2f' % (near_hit/max(1,total)), 'to furthest-behind %.2f' % (behind_hit/max(1,total)))
