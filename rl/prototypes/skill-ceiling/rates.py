# Event rates per player-minute for several lineups (24 rounds each): how many "decisions" a round actually contains.
import sys, json
sys.path.insert(0, "rl")
from ladder_native import run
lineups = {
 "6 loyal": ["loyal"]*6,
 "6 bail": ["bail"]*6,
 "3 bail 3 loyal": ["bail"]*3+["loyal"]*3,
 "2 bail 2 loyal 2 rammer": ["bail"]*2+["loyal"]*2+["rammer"]*2,
 "2 bail 2 loyal 2 kidnap": ["bail"]*2+["loyal"]*2+["kidnap"]*2,
}
out = {}
for name, seats in lineups.items():
    r = run(seats, 24, 1000, None, quiet=True)
    n = len(seats); pm = r["length"]/60.0*n  # player-minutes per round
    row = dict(length=r["length"], early=r["finished_early"], merges=r["merges"], leaves=r["leaves"], banks=r["banks"],
               group_banks=r["group_banks"], spills=r["spills"], alliances=r["alliances"], alliance_dur=r["alliance_dur"],
               alliances_long=r["alliances_long"], avg_group=r["avg_group"],
               per_player_min=dict(merges=r["merges"]/pm, leaves=r["leaves"]/pm, banks=r["banks"]/pm, spills=r["spills"]/pm),
               progress={k: v["progress"] for k, v in r["by_type"].items()})
    out[name] = row
    print(f"{name:26s} len={r['length']:.0f}s early={r['finished_early']:.2f} group={r['avg_group']:.2f} merges={r['merges']:.1f} leaves={r['leaves']:.1f} banks={r['banks']:.1f} (group {r['group_banks']:.1f}) spills={r['spills']:.1f} alliances={r['alliances']:.1f} dur={r['alliance_dur']:.1f}s long={r['alliances_long']:.1f}")
    print(f"   per player-minute: merges {row['per_player_min']['merges']:.2f} leaves {row['per_player_min']['leaves']:.2f} banks {row['per_player_min']['banks']:.2f} spills {row['per_player_min']['spills']:.2f}  progress {row['progress']}")
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/skill-ceiling/rates.json","w"), indent=1)
