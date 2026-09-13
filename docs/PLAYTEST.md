# Playtest notes (13 September 2026)

Played from a terminal with `rl/play_console.py`, one macro decision every 4 to 12 seconds of game time,
on the life preset. Round one: my seat with loyal, bail, grudge, solo and bail bots. Round two: my seat among
four bails and one loyal. Both rounds were played blind to other players' needs, as a human would be.

## What was fun

- **Steering the group by pushing.** Pulling a trio off its consensus mine to the D mine I needed worked and
  felt like influence. It only works while the pool is below the bots' bank threshold; above it two bots
  outvote one push and the group heads for the pad. That threshold is invisible to a human today.
- **The endgame leave.** At 0.85 progress, four units from finishing, with the crown feeding the two members
  behind me, leaving with a small share and mining the last units solo was a clear, satisfying decision and
  it won the round at 181 s. The rules produce this decision on their own.
- **The crown rotation reads as a story.** Being the sucker for one bank, then head, then handing it on.
  In the bail lobby the bail banked first at its own pad, I became head, and it left with 42 percent of the
  pool as we reached my pad, branded for 16 s. Legible betrayal.
- **Blooming mines keep bodies moving.** A trio empties a mine in about five seconds, so mine-hopping is
  constant and the map never settles.

## What was not

- **The last 0.03 of a unit.** Progress read 0.99 with every need displayed as met (C 6/6 was 5.97) and no win.
  Fixed by whole-unit banking (`bank_round=1` in the life preset: banked amounts round to whole units, the
  fraction stays in the pool), so 6/6 means six.
- **Passenger time.** While not head I had nothing to decide for 30 to 40 percent of the round except where to
  push. Tried giving the crown double steering weight (`head_steer=2`): six loyal fell from 0.765 to 0.713,
  six bail rose to 0.695, and the kidnapper climbed from 0.12 to 0.54 because a crowned kidnapper could drag
  the group. Rejected; the field stays at 1.0.
- **Intent is locked in a group,** so you cannot prepare the type of share you will leave with. By design, but
  it needs to be shown, not discovered.
- **Bank timing is the bots' decision.** As head I could not hold the group at a mine to fill a bigger pool.
  A human-only lobby will not have this problem; a mixed lobby needs the head's intent to matter, which is the
  same lever as passenger time. The clean fix is a group-level "bank now" vote or a head-only bank call,
  which is a rules change to test after the web version exists.

## Rules state after the playtest

Life preset: crown (head_vest 10), brand, bloom (K = players + 2), whole-unit banking. Gates at 24 rounds:
six loyal 0.747, six bail 0.699, six solo 0.632; five bail + one loyal 0.732 vs 0.674; kidnap 0.113 and
rammer winless. The macro-action agent v10 update 100 is evaluated under the same rules in
docs/SKILL_CEILING.md.
