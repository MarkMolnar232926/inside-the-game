"""Print the hero player's attacking-third interceptions and goals for a seed (story/clip-window picking)."""
import sys

from pitchlens.state.match_state import MatchState
from pitchlens.synth.generate import generate

seed, nar = int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "late_comeback"
m = generate(seed, nar)
st = MatchState(m.meta)
for e in m.events:
    st.ingest(e)
for e in st.evidence[("KBY-07", "interceptions_attacking_third")]:
    print("K7 att3 interception", e.clock, round(e.t, 1), round(e.x, 1), round(e.y, 1), "period", e.period)
for e in m.events:
    if e.type.value == "goal":
        print("goal", e.clock, round(e.t, 1), e.team_id, e.player_id)
print("final", st.score)
