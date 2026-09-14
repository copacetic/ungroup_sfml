import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SC = os.path.dirname(os.path.abspath(__file__))
fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
for k, lk in enumerate(["12bh", "20eq", "6b3l3"]):
    for v, c in [("base", "0.6"), ("contag", "C3")]:
        try:
            g = np.load(os.path.join(SC, f"groupseries_{v}_{lk}.npy")); t = np.arange(len(g)) * 0.2
            ax[k].plot(t, g, color=c, lw=0.8, label=f"groups (2+), {v}")
        except Exception as e: print(e)
    try:
        br = np.load(os.path.join(SC, f"brandseries_contag_{lk}.npy")); t = np.arange(len(br)) * 0.2
        n = {"12bh": 12, "20eq": 20, "6b3l3": 6}[lk]
        ax[k].plot(t, br * n, color="C1", lw=1.2, label="branded players (package)")
    except Exception as e: print(e)
    ax[k].set_title(f"{lk}: seed round 0"); ax[k].legend(fontsize=7, loc="upper right"); ax[k].set_xlabel("t (s)")
plt.tight_layout(); plt.savefig(os.path.join(SC, "fig_brand_groups.png"), dpi=110); print("saved")
