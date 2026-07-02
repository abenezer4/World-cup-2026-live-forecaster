"""bracket_view.py — builds the real 2026 World Cup knockout bracket (with
actual results locked in) and renders it as an animated, two-sided HTML
bracket, styled after the official bracket layout (match IDs M73-M104).

Undetermined matches reveal the model's predicted winner with a staggered
fade-in animation and a confidence percentage.
"""
from __future__ import annotations
from simulate import BRACKET_R32, predict_bracket_path

# Real Round-of-32 match data: (match_id, home, away, home_score, away_score)
# score=None means not yet played. This is display data (both real opponents
# + real scores where known); BRACKET_R32 in simulate.py only needs the
# *winner* of each, so this lives separately.
R32_REAL = {
    "left": [
        ("M74", "Germany", "Paraguay", 1, 1, "PAR", "4-3 pens"),
        ("M77", "France", "Sweden", 3, 0, "FRA", None),
        ("M73", "South Africa", "Canada", 0, 1, "CAN", None),
        ("M75", "Netherlands", "Morocco", 1, 1, "MAR", "3-2 pens"),
        ("M83", "Portugal", "Croatia", None, None, None, None),
        ("M84", "Spain", "Austria", None, None, None, None),
        ("M81", "United States", "Bosnia and Herzegovina", 2, 0, "United States", None),
        ("M82", "Belgium", "Senegal", 3, 2, "Belgium", None),
    ],
    "right": [
        ("M76", "Brazil", "Japan", 2, 1, "Brazil", None),
        ("M78", "Ivory Coast", "Norway", 1, 2, "Norway", None),
        ("M79", "Mexico", "Ecuador", 2, 0, "Mexico", None),
        ("M80", "England", "DR Congo", 2, 1, "England", None),
        ("M86", "Argentina", "Cape Verde", None, None, None, None),
        ("M88", "Australia", "Egypt", None, None, None, None),
        ("M85", "Switzerland", "Algeria", None, None, None, None),
        ("M87", "Colombia", "Ghana", None, None, None, None),
    ],
}

FLAGS = {
    "Germany": "GER", "Paraguay": "PAR", "France": "FRA", "Sweden": "SWE",
    "South Africa": "RSA", "Canada": "CAN", "Netherlands": "NED", "Morocco": "MAR",
    "Portugal": "POR", "Croatia": "CRO", "Spain": "ESP", "Austria": "AUT",
    "United States": "USA", "Bosnia and Herzegovina": "BIH", "Belgium": "BEL",
    "Senegal": "SEN", "Brazil": "BRA", "Japan": "JPN", "Ivory Coast": "CIV",
    "Norway": "NOR", "Mexico": "MEX", "Ecuador": "ECU", "England": "ENG",
    "DR Congo": "COD", "Argentina": "ARG", "Cape Verde": "CPV",
    "Australia": "AUS", "Egypt": "EGY", "Switzerland": "SUI", "Algeria": "ALG",
    "Colombia": "COL", "Ghana": "GHA",
}

# deterministic pseudo-random but pleasant hue per team, for the code badge
def _hue(team):
    return (sum(ord(c) for c in team) * 47) % 360


def _box(team, score=None, is_winner=False, pending=False, flag_only=False):
    code = FLAGS.get(team, team[:3].upper())
    if pending:
        return ('<div class="team pending"><span class="flag" style="background:#3a4a44">?'
                '</span><span class="name">TBD</span></div>')
    cls = "team winner" if is_winner else "team"
    hue = _hue(team)
    score_html = f'<span class="score">{score}</span>' if score is not None else ""
    return (f'<div class="{cls}"><span class="flag" style="background:hsl({hue},55%,32%)">'
            f'{code}</span><span class="name">{team}</span>{score_html}</div>')


def _match_box(home, away, home_score=None, away_score=None, winner=None,
               note=None, confidence=None, match_id="", predicted=False):
    played = home_score is not None
    hw = played and home_score > away_score
    aw = played and away_score > home_score
    if not played and winner:
        hw = winner == home
        aw = winner == away
    conf_html = (f'<div class="conf">{confidence*100:.0f}% confidence</div>'
                if (predicted and confidence) else "")
    note_html = f'<div class="note">{note}</div>' if note else ""
    predicted_cls = "predicted-pending" if predicted else ""
    return f'''
    <div class="match {predicted_cls}" data-matchid="{match_id}">
      <div class="mid">{match_id}</div>
      {_box(home, home_score, hw)}
      {_box(away, away_score, aw)}
      {note_html}{conf_html}
    </div>'''


def build_bracket_html(model) -> str:
    path = predict_bracket_path(model)

    # ---- Round of 32 (real data, both sides) ----
    def render_r32_side(side):
        html = ""
        for mid, h, a, hs, as_, winner, note in R32_REAL[side]:
            if hs is not None:
                html += _match_box(h, a, hs, as_, winner, note, match_id=mid)
            else:
                # find the model's predicted pick for this pending pair
                pred = next((m for m in path["Round of 32 (remaining)"]
                            if {m["home"], m["away"]} == {h, a}), None)
                conf = pred["confidence"] if pred else None
                pick = pred["winner"] if pred else None
                html += _match_box(h, a, None, None, pick, None, conf,
                                   match_id=mid, predicted=True)
        return html

    left_r32 = render_r32_side("left")
    right_r32 = render_r32_side("right")

    # ---- later rounds, split evenly left/right the way the real bracket does ----
    def side_matches(round_name, half):
        matches = path[round_name]
        n = len(matches)
        return matches[:n // 2] if half == "left" else matches[n // 2:]

    def render_round(round_name, half, id_prefix):
        html = ""
        for i, m in enumerate(side_matches(round_name, half)):
            html += _match_box(m["home"], m["away"], None, None, m["winner"],
                               None, m["confidence"], match_id=f"{id_prefix}{i+1}",
                               predicted=True)
        return html

    left_r16 = render_round("Round of 16", "left", "R16-L")
    right_r16 = render_round("Round of 16", "right", "R16-R")
    left_qf = render_round("Quarterfinals", "left", "QF-L")
    right_qf = render_round("Quarterfinals", "right", "QF-R")
    left_sf = render_round("Semifinals", "left", "SF-L")
    right_sf = render_round("Semifinals", "right", "SF-R")

    final = path["Final"][0]
    champion = path["champion"]
    champ_code = FLAGS.get(champion, champion[:3].upper())
    champ_hue = _hue(champion)

    html = f"""
<div class="bracket-wrap">
  <style>
    .bracket-wrap {{
      --pitch: #0B3D2E; --pitch2: #0E5038; --chalk: #F2F4F0;
      --lime: #C6FF4F; --muted: #9FB3AA; --slate: #16241F;
      background: radial-gradient(1400px 700px at 50% -10%, var(--pitch2) 0%, var(--pitch) 55%, var(--slate) 100%);
      color: var(--chalk); font-family: -apple-system, Segoe UI, Roboto, sans-serif;
      padding: 28px 12px; border-radius: 16px; overflow-x: auto;
    }}
    .bracket-grid {{
      display: grid;
      grid-template-columns: repeat(4, 200px) 220px repeat(4, 200px);
      gap: 8px; align-items: center; min-width: 1900px;
    }}
    .col {{ display: flex; flex-direction: column; justify-content: space-around; height: 1500px; }}
    .col-final {{ display: flex; flex-direction: column; align-items: center; justify-content: center; height: 1500px; gap: 24px; }}
    .match {{
      background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12);
      border-radius: 10px; padding: 8px 10px; margin: 4px 0; position: relative;
      opacity: 0; transform: translateX(-8px); animation: fadein .6s ease forwards;
    }}
    .col:nth-child(6) .match, .col:nth-child(7) .match, .col:nth-child(8) .match, .col:nth-child(9) .match {{
      transform: translateX(8px);
    }}
    @keyframes fadein {{ to {{ opacity: 1; transform: translateX(0); }} }}
    .mid {{ font-size: 10px; color: var(--muted); letter-spacing: .05em; margin-bottom: 3px; }}
    .team {{ display: flex; align-items: center; gap: 6px; padding: 3px 2px; border-radius: 6px; font-size: 13px; }}
    .team .flag {{ font-size: 9px; font-weight: 800; letter-spacing: .02em; color: #fff;
      border-radius: 4px; padding: 2px 5px; min-width: 26px; text-align: center; flex-shrink: 0; }}
    .team .name {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .team .score {{ font-weight: 700; color: var(--chalk); }}
    .team.winner {{ background: rgba(198,255,79,.14); font-weight: 700; }}
    .team.winner .name {{ color: var(--lime); }}
    .team.pending .name {{ color: var(--muted); }}
    .note {{ font-size: 10px; color: var(--muted); margin-top: 2px; }}
    .conf {{ font-size: 10px; color: var(--lime); margin-top: 2px; font-weight: 600; }}
    .predicted-pending {{ border-style: dashed; border-color: rgba(198,255,79,.35); }}
    .predicted-pending .winner {{ animation: pulse 2.2s ease-in-out infinite; animation-delay: 1s; }}
    @keyframes pulse {{ 0%,100% {{ background: rgba(198,255,79,.14); }} 50% {{ background: rgba(198,255,79,.28); }} }}
    .round-label {{ text-align: center; font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
      color: var(--muted); margin-bottom: 10px; }}
    .champion-box {{
      text-align: center; background: rgba(198,255,79,.10); border: 2px solid var(--lime);
      border-radius: 16px; padding: 22px 26px; opacity: 0; animation: champFade 1s ease forwards;
      animation-delay: 3.2s;
    }}
    @keyframes champFade {{ to {{ opacity: 1; }} }}
    .champion-box .cflag {{ font-size: 15px; font-weight: 800; color: #fff; border-radius: 8px;
      padding: 8px 16px; display: inline-block; margin-bottom: 10px; }}
    .champion-box .cname {{ display: block; font-size: 20px; font-weight: 800; color: var(--lime); }}
    .champion-box .clabel {{ font-size: 10px; letter-spacing: .15em; text-transform: uppercase; color: var(--muted); margin-top: 4px; }}
    .final-match {{ text-align: center; }}
  </style>

  <div class="bracket-grid">
    <div class="col"><div class="round-label">Round of 32</div>{left_r32}</div>
    <div class="col"><div class="round-label">Round of 16</div>{left_r16}</div>
    <div class="col"><div class="round-label">Quarterfinal</div>{left_qf}</div>
    <div class="col"><div class="round-label">Semifinal</div>{left_sf}</div>
    <div class="col-final">
      <div class="round-label">Final</div>
      <div class="final-match">{_match_box(final['home'], final['away'], None, None, final['winner'], None, final['confidence'], match_id='M104', predicted=True)}</div>
      <div class="champion-box">
        <span class="cflag" style="background:hsl({champ_hue},55%,32%)">{champ_code}</span>
        <span class="cname">{champion}</span>
        <div class="clabel">Model's Predicted Champion</div>
      </div>
    </div>
    <div class="col"><div class="round-label">Semifinal</div>{right_sf}</div>
    <div class="col"><div class="round-label">Quarterfinal</div>{right_qf}</div>
    <div class="col"><div class="round-label">Round of 16</div>{right_r16}</div>
    <div class="col"><div class="round-label">Round of 32</div>{right_r32}</div>
  </div>
</div>
"""
    return html
