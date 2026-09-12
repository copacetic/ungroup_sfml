import gzip, pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/ecology"
fig, axes = plt.subplots(4, 1, figsize=(11, 11))
for ax, (name, ln, e, K) in zip(axes, [("legacy","6loyal",0,30),("D","6loyal",0,6),("D","3bail3loyal",0,6),("D","12equal",0,6)]):
    R = pickle.load(gzip.open(f"{S}/rec_{name}.pkl.gz"))[ln]
    t = np.array(R["ts"][e]); st = np.array(R["stocks"][e]); al = np.array(R["alive"][e], dtype=bool)
    x = st / K; x[~al] = np.nan
    mpos = np.array(R["meta"][e]["mine_pos"]); ang = np.degrees(np.arctan2(mpos[:,1], mpos[:,0])) % 360
    order = np.argsort(ang)  # ring order
    im = ax.imshow(x[:, order].T, aspect="auto", origin="lower", extent=[t[0], t[-1], -0.5, 7.5], vmin=0, vmax=1.2, cmap="YlGn")
    ax.set_yticks(range(8)); ax.set_yticklabels([f"m{m} {'in' if np.hypot(*mpos[m])<0.5 else 'out'} T{R['meta'][e]['mine_type'][m]}" for m in order], fontsize=7)
    ax.set_title(f"{name} {ln} seed {R['meta'][e]['seed']}: stock / K (K={K}) per mine in ring order; dark = dormant, > 1 = autumn superbloom", fontsize=9)
    ax.axvline(120, color="k", lw=0.5); ax.axvline(192, color="k", lw=0.5)
plt.colorbar(im, ax=axes, fraction=0.02); axes[-1].set_xlabel("t (s)")
plt.savefig(f"{S}/fig_stock_heatmaps.png", dpi=110); print("fig ok")
