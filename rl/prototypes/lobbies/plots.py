import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/lobbies"
S = json.load(open(f"{OUT}/series.json"))
C = {"equal": "#2a78d6", "loyal": "#eb6834", "bail": "#1baf7a", "policy": "#eda100"}
LAB = {"equal": "equal solo/bail/loyal", "loyal": "loyal-heavy", "bail": "bail-heavy", "policy": "10 policy + bots (control)"}
t = np.arange(241)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7"})
def marks(ax):
    for x, lab in ((120, "shrink starts"), (192, "outer mines die")):
        ax.axvline(x, color="#9a9a94", lw=0.8, ls="--"); ax.text(x + 1, ax.get_ylim()[1] * 0.97, lab, fontsize=7, color="#52514e", va="top")

# Figure 1: crowd structure, 20-player lineups
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
panels = [("n_bodies", "bodies on the floor"), ("largest", "largest group (members)"), ("n_groups", "groups (bodies with 2+ members)"), ("nn_ratio", "clustering: mean NN distance / uniform-disc baseline")]
for ax, (key, title) in zip(axes.flat, panels):
    for kind in ("equal", "loyal", "bail", "policy"):
        v = np.array(S[f"20_{kind}"][key], dtype=float)
        ax.plot(t, v, color=C[kind], lw=1.6, label=LAB[kind])
    ax.set_title(title, loc="left", fontsize=10); marks(ax)
    if key == "nn_ratio": ax.axhline(1.0, color="#52514e", lw=0.8); ax.set_ylim(0.5, 1.1)
axes[1, 0].set_xlabel("time (s)"); axes[1, 1].set_xlabel("time (s)")
axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("20-player lobbies: crowd structure over time (mean of 8 rounds; frames every 10 ticks)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_structure_20.png", dpi=140); plt.close(fig)

# Figure 2: resources and spills across lobby sizes (equal lineup) + loyal-32
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
sizes = [("12_equal", "#2a78d6", "12 players, equal"), ("20_equal", "#eb6834", "20 players, equal"), ("32_equal", "#1baf7a", "32 players, equal"), ("32_loyal", "#eda100", "32 players, loyal-heavy")]
ax = axes[0, 0]
for name, col, lab in sizes:
    ax.plot(t, S[name]["stock_outer"], color=col, lw=1.6, label=lab); ax.plot(t, S[name]["stock_inner"], color=col, lw=1.2, ls=":")
ax.set_title("mine stock: outer ring (solid) and inner ring (dotted), mean of 4 mines", loc="left", fontsize=10); ax.set_ylim(0, 31); marks(ax); ax.legend(frameon=False, fontsize=8, loc="lower left")
ax = axes[0, 1]
for name, col, lab in sizes: ax.plot(t, S[name]["mine_frac"], color=col, lw=1.6, label=lab)
ax.set_title("fraction of players touching a live mine", loc="left", fontsize=10); ax.set_ylim(0, 1); marks(ax)
ax = axes[1, 0]
for name, col, lab in sizes: ax.plot(t, S[name]["picks"], color=col, lw=1.6, label=lab)
ax.set_title("floor pickups (units lying on the floor)", loc="left", fontsize=10); marks(ax); ax.set_xlabel("time (s)")
ax = axes[1, 1]
edges = np.arange(0, 241, 10); w = 2.2
for k, (name, col, lab) in enumerate(sizes):
    h = np.array(S[name]["spills_hist"]) / 8 / 10 * 60  # spills per minute per round
    ax.bar(edges[:-1] + 1 + k * w, h, width=w, color=col, label=lab, linewidth=0)
ax.set_title("spill rate (spills per minute, per round)", loc="left", fontsize=10); marks(ax); ax.set_xlabel("time (s)")
fig.suptitle("Resource depletion and spills by lobby size (mean of 8 rounds)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_resources_spills.png", dpi=140); plt.close(fig)

# Figure 3: one round in detail (20 policy env 0) - group oscillation + per-mine stock heatmap; alliance formation histogram
fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 1]})
ax = axes[0]
ax.plot(t, S["20_policy"]["n_groups_env0"], color="#2a78d6", lw=1.6, label="groups (2+ members)")
ax.plot(t, S["20_policy"]["largest_env0"], color="#eb6834", lw=1.6, label="largest group size")
ax.set_title("single round, 20 players (10 policy + bots), seed 5020: cooperation oscillates on a ~25 s bank cycle", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="upper left"); marks(ax)
ax = axes[1]
st = np.array(S["20_policy"]["stock_env0"]).T  # 8 x 241
order = [0, 2, 4, 6, 1, 3, 5, 7]
im = ax.imshow(st[order], aspect="auto", cmap="Blues", vmin=0, vmax=30, extent=[0, 240, 8, 0], interpolation="nearest")
ax.set_yticks(np.arange(8) + 0.5); ax.set_yticklabels([f"outer {m}" for m in (0, 2, 4, 6)] + [f"inner {m}" for m in (1, 3, 5, 7)], fontsize=7)
ax.set_title("per-mine stock (units; dark = full, white = depleted); outer ring dies at 192 s", loc="left", fontsize=10); ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03); cb.set_label("stock", fontsize=8)
ax = axes[2]
for kind in ("equal", "loyal", "bail", "policy"):
    h = np.array(S[f"20_{kind}"]["form_hist"]) / 10 * 60
    ax.plot(edges[:-1] + 5, h, color=C[kind], lw=1.6, label=LAB[kind])
ax.set_title("alliance formation rate, 20-player lineups (new co-member pairs per minute, mean of 8 rounds)", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2); marks(ax); ax.set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_round_detail.png", dpi=140); plt.close(fig)
print("ok")
