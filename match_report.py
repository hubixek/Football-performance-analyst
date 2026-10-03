#!/usr/bin/env python3
"""Match report as one self-contained HTML page, in the style of the statistics page of a football portal:
a header with the teams and the score, tabs (match / 1st half / 2nd half) and rows with the value of each team at
the sides, the name of the statistic in the middle and bars growing from the centre.

The bar of a team is its share of the sum of both values (as on the portal), the higher value is highlighted.

Made from the results of the analysis of a match (nothing is calculated again except two small drawings):
  summary.json        possession
  stats.json          zones, recoveries, possession sequences, team shape        (match_stats.py)
  team_distance.csv   running distance of the field players                      (dual_tracks.py, optional)
  possession.csv      the possession during the match: the timeline              (dual_possession.py, optional)
  tracks.csv          positions of the players: the heat maps                    (dual_tracks.py, optional)
  teams_colors.json   colours of the kits                                        (dual_teams.py, optional)

The data of the report are also stored in the page as JSON (<script id="match-data">), so that other programs
(for instance a tool that assesses the performance of a team) can read them.

Usage:
  python match_report.py ~/football/analysis/mecz1_dual --pitch pitch/pitch_6v6.json \
      --names "Team A,Team B" --score 11-2 --date "2026-09-20" --gif overlay_38m39_15s.gif
"""
import argparse
import base64
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import pass_calibration

TEXT = {
    "pl": {
        "sport": "PIŁKA NOŻNA 6v6", "analysis": "Analiza meczu", "status": "KONIEC",
        "tabs": ["MECZ", "1. POŁOWA", "2. POŁOWA"],
        "xg": "Oczekiwane gole (xG) *", "shots": "Strzały łącznie *", "shots_on": "Strzały na bramkę *", "big_chances": "Wielkie szanse *",
        "corners": "Rzuty rożne *", "passes": "Celność podań (szacunek) *", "box_entries": "Wejścia w pole karne *",
        "est_note": "* Szacunek z widocznej piłki (piłka widoczna w {vis:.0f}% klatek): liczby podań są zaniżone (program znajduje mniej więcej połowę), a procent udanych podań jest orientacyjny.",
        "pass_note": "Celność podań to szacunek skorygowany o błąd programu: za często nazywa on podanie „straconym” (na podstawie ocen człowieka dla {n} podań z {m} meczów); w nawiasie surowe liczby programu (udane/wszystkie znalezione). Przedział 95%: drużyna 0 {ca}, drużyna 1 {cb}.",
        "shots_hidden": "Strzały i xG są ukryte: tor piłki znalazł {pct}% goli jako strzały (poniżej 50%), więc liczby byłyby mocno zaniżone (--shots always je pokazuje).",
        "est_goals": " Z {given} goli {found} znaleziono jako strzały z toru piłki, resztę dodano z pozycji ostatniej kontroli.",
        "top": "TOP STATYSTYKI", "possession": "Posiadanie piłki", "possession_opp": "Posiadanie na połowie rywala",
        "recoveries": "Odbiory piłki", "recoveries_opp": "Odbiory na połowie rywala", "losses": "Straty piłki",
        "possessions": "Akcje z piłką", "avg_possession": "Średni czas akcji (s)", "long_possessions": "Akcje dłuższe niż 10 s",
        "distance": "Dystans zawodników z pola (km)",
        "thirds": "GDZIE DRUŻYNA MIAŁA PIŁKĘ", "own_third": "Własna tercja", "middle_third": "Środkowa tercja", "final_third": "Tercja ataku",
        "shape": "USTAWIENIE DRUŻYNY", "avg_position": "Średnia pozycja (m od własnej bramki)",
        "length": "Długość ustawienia (m)", "width": "Szerokość ustawienia (m)",
        "goals_missing": "Program wykrył {n} z {total} goli; brakujących nie ma na liście.", "goals": "GOLE", "goals_auto": "wykryte automatycznie z wznowień i toru piłki; czas gola jest oszacowaniem (±kilka sekund)",
        "score_auto": "wynik z automatycznie wykrytych goli", "goal_minute": "{m}. minuta",
        "timeline": "PRZEBIEG POSIADANIA", "timeline_note": "Udział drużyny w posiadaniu w ostatniej minucie, oś: minuta gry",
        "half": "{n}. połowa", "heat": "GDZIE GRALI ZAWODNICY", "heat_note": "Mapy ciepła, każda drużyna atakuje w prawo",
        "animation": "ANIMACJA", "about": "O ANALIZIE",
        "about_text": ("Statystyki policzone automatycznie z nagrania z dwóch obiektywów. Posiadanie jest przypisane w {assigned:.0f}% czasu gry; "
                       "przez {dead:.0f}% czasu piłka była poza boiskiem, leżała w miejscu albo trwało wznowienie po bramce. "
                       "Dystans to oszacowanie (około ±10%); zawodnicy nie są identyfikowani indywidualnie."),
        "team": "Drużyna {n}",
        "quality_title": {"ok": "Analiza wiarygodna w granicach opisanych niżej", "warn": "Analiza z zastrzeżeniami", "bad": "Analiza niewiarygodna"},
        "score_approx": "wynik orientacyjny: strzelca części goli zgadnięto",
        "shows_title": "CO TEN RAPORT POKAZUJE, A CZEGO NIE",
        "shows": ["Gole i wynik: gole znalezione po piłce są prawdziwe (na {matches} meczach {goals_found} z {goals_total}, żaden fałszywy); program może pominąć gol na końcu połowy albo wznowienie bez widocznej piłki.",
                  "Strefy, ustawienie drużyn, dystans: liczone z pozycji zawodników w metrach (dystans ±10%).",
                  "Posiadanie: orientacyjne. Dokładność nie była mierzona na losowych momentach, a dwa podobne modele różniły się o około 4 punkty.",
                  "Podania: orientacyjne. Na {matches} meczach około {passes_precision}% znalezionych to prawdziwe podania (próba {passes_judged}), program znajduje mniej więcej połowę, "
                  "surowy werdykt \"udane / stracone\" zgadza się z człowiekiem w około {passes_completed_agree}% przypadków i zaniża celność, dlatego pokazana celność jest szacunkiem skorygowanym "
                  "(przedział niepewności około 9 punktów procentowych; dotyczy podań, które program znajduje).",
                  "Strzały i xG: niedostępne. Na {shots_judged} ocenionych strzałach tylko około {shots_precision}% było prawdziwymi strzałami.",
                  "Statystyki pojedynczych zawodników: niedostępne, zawodnicy są śledzeni, ale nie identyfikowani."],
    },
    "en": {
        "sport": "FOOTBALL 6v6", "analysis": "Match analysis", "status": "FULL TIME",
        "tabs": ["MATCH", "1ST HALF", "2ND HALF"],
        "xg": "Expected goals (xG) *", "shots": "Total shots *", "shots_on": "Shots on target *", "big_chances": "Big chances *",
        "corners": "Corners *", "passes": "Pass accuracy (estimate) *", "box_entries": "Entries into the penalty area *",
        "est_note": "* Estimate from the visible ball (the ball is visible in {vis:.0f}% of the frames): the numbers of passes are too low (the program finds roughly half) and the pass accuracy is approximate.",
        "pass_note": "The pass accuracy is an estimate corrected for an error of the program: it names too many passes 'lost' (from a human's judgement of {n} passes of {m} matches); the raw counts of the program (completed / all found) are in brackets. 95% interval: team 0 {ca}, team 1 {cb}.",
        "shots_hidden": "Shots and xG are hidden: the ball track found {pct}% of the goals as shots (below 50%), so the numbers would be far too low (--shots always shows them).",
        "est_goals": " Of {given} goals {found} were found as shots in the ball track, the rest were added from the position of the last control.",
        "top": "TOP STATS", "possession": "Ball possession", "possession_opp": "Possession in the opponent's half",
        "recoveries": "Ball recoveries", "recoveries_opp": "Recoveries in the opponent's half", "losses": "Ball losses",
        "possessions": "Possessions", "avg_possession": "Average possession (s)", "long_possessions": "Possessions over 10 s",
        "distance": "Running distance of field players (km)",
        "thirds": "WHERE THE TEAM HAD THE BALL", "own_third": "Own third", "middle_third": "Middle third", "final_third": "Final third",
        "shape": "TEAM SHAPE", "avg_position": "Average position (m from own goal)",
        "length": "Length of the team (m)", "width": "Width of the team (m)",
        "goals_missing": "The program found {n} of the {total} goals; the missing ones are not listed.", "goals": "GOALS", "goals_auto": "found automatically from the kick-offs and the ball track; the time of a goal is an estimate (a few seconds)",
        "score_auto": "score from the goals found automatically", "goal_minute": "minute {m}",
        "quality_title": {"ok": "The analysis is trustworthy within the limits below", "warn": "The analysis has reservations", "bad": "The analysis is not trustworthy"},
        "score_approx": "approximate score: the scorer of some goals was guessed",
        "shows_title": "WHAT THIS REPORT SHOWS AND WHAT IT DOES NOT",
        "shows": ["Goals and score: the goals found by the ball are real (on {matches} matches {goals_found} of {goals_total}, none false); the program can miss a goal at the end of a half or a restart without a visible ball.",
                  "Zones, team shape, distance: computed from the positions of the players in metres (distance +-10%).",
                  "Possession: approximate. Its accuracy was not measured on random moments, and two similar models differed by about 4 points.",
                  "Passes: approximate. On {matches} matches about {passes_precision}% of the passes found were real (sample {passes_judged}), the program finds roughly half, "
                  "the raw verdict completed / lost agrees with a human in about {passes_completed_agree}% of the cases and understates the accuracy, so the accuracy shown is a corrected "
                  "estimate (an interval of about 9 points; it concerns the passes the program finds).",
                  "Shots and xG: not available. Of {shots_judged} judged shots only about {shots_precision}% were real shots.",
                  "Statistics of single players: not available, players are tracked but not identified."],
        "timeline": "POSSESSION TIMELINE", "timeline_note": "Share of possession in the last minute, axis: minute of play",
        "half": "Half {n}", "heat": "WHERE THE PLAYERS PLAYED", "heat_note": "Heat maps, every team attacks to the right",
        "animation": "ANIMATION", "about": "ABOUT THE ANALYSIS",
        "about_text": ("Statistics calculated automatically from the dual-lens recording. Possession is assigned in {assigned:.0f}% of the playing time; "
                       "for {dead:.0f}% of the time the ball was out of the pitch, lying still or being restarted after a goal. "
                       "The distance is an estimate (about ±10%); players are not identified individually."),
        "team": "Team {n}",
    },
}

CSS = """
:root{--bg:#0c1a23;--panel:#0f202a;--line:#1d3441;--text:#e8eef2;--muted:#8fa3ae;--red:#c8102e;--white:#f2f4f5;--track:#142935}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:720px;margin:0 auto;padding:18px 16px 40px}
.crumbs{color:#b9c7cf;font-size:12px;font-weight:600;letter-spacing:.4px}
.crumbs span{color:var(--muted);margin:0 6px}
.head{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px;margin:26px 0 8px;text-align:center}
.team{display:flex;flex-direction:column;align-items:center;gap:10px;font-weight:700;font-size:16px}
.badge{width:76px;height:76px;border-radius:14px;background:#fff;display:flex;align-items:center;justify-content:center}
.badge i{display:block;width:44px;height:44px;border-radius:50%;border:3px solid rgba(255,255,255,.85);box-shadow:0 0 0 2px rgba(0,0,0,.15)}
.mid .date{color:#b9c7cf;font-size:13px}
.mid .score{font-size:44px;font-weight:800;letter-spacing:3px;margin:2px 0}
.mid .status{color:#b9c7cf;font-size:12px;font-weight:700;letter-spacing:.6px}
.tabs{display:flex;gap:8px;margin:26px 0 0}
.tab{background:#1a3140;color:#cfdbe2;border:0;border-radius:8px;padding:9px 18px;font:700 12px/1 inherit;letter-spacing:.6px;cursor:pointer}
.tab.active{background:#33505f;color:#fff}
.panel{background:var(--panel);border-radius:10px;padding:14px 18px 8px;margin-top:14px}
.panel h3{margin:0 0 14px;text-align:center;font-size:11px;font-weight:700;letter-spacing:.9px;color:#b9c7cf}
.row{margin:0 0 16px}
.vals{display:grid;grid-template-columns:1fr auto 1fr;align-items:baseline;margin-bottom:5px}
.vals .v{font-weight:700;font-size:14px}
.vals .v.right{text-align:right}
.vals .lab{font-size:14px;color:#dfe8ed;text-align:center;padding:0 10px}
.bars{display:flex;gap:6px}
.half{flex:1;height:7px;border-radius:2px;background:var(--track);display:flex}
.half.left{justify-content:flex-end}
.bar{height:100%;border-radius:2px;background:var(--white)}
.bar.lead{background:var(--red)}
.style-team .bar{opacity:.45}
.style-team .bar.lead{opacity:1}
.pane.hidden{display:none}
.goal{display:flex;align-items:center;gap:10px;padding:7px 0;border-top:1px solid var(--line);font-size:14px}.goal:first-of-type{border-top:0}
.goal .m{width:92px;color:#b9c7cf;font-weight:600}.goal .rough{color:var(--muted);font-size:12px}
.stack{display:flex;height:9px;border-radius:2px;overflow:hidden;margin-top:6px;gap:2px}
.stack b{display:block;height:100%}
.note{color:var(--muted);font-size:12px;text-align:center;margin:0 0 12px}
svg{display:block;width:100%;height:auto}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.cap{color:#dfe8ed;font-size:13px;font-weight:600;margin:0 0 6px;display:flex;align-items:center;gap:8px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
img.anim{width:100%;border-radius:8px;display:block}
.about{color:var(--muted);font-size:12.5px;line-height:1.55;padding-bottom:10px}
.quality{border-radius:10px;padding:12px 16px;margin:10px 0;border:1px solid}
.quality.ok{background:#e8f5ec;border-color:#9bd0aa;color:#14532d}.quality.warn{background:#fff6e0;border-color:#e8c36a;color:#6b4a00}.quality.bad{background:#fde8e8;border-color:#e08a8a;color:#7f1d1d}
.quality h3{margin:0 0 6px;font-size:14px}.quality li{margin:3px 0;font-size:13px}
@media (max-width:560px){.two{grid-template-columns:1fr}.mid .score{font-size:32px;letter-spacing:1px}.badge{width:56px;height:56px}.badge i{width:32px;height:32px}.team{font-size:14px}.wrap{padding:14px 10px 30px}.panel{padding:12px 12px 6px}.tab{padding:9px 12px}.vals .lab{font-size:13px;padding:0 6px}}
"""

JS = """
document.querySelectorAll('.tab').forEach(function(b){
  b.addEventListener('click', function(){
    document.querySelectorAll('.tab').forEach(function(x){x.classList.toggle('active', x===b)});
    document.querySelectorAll('.pane').forEach(function(p){p.classList.toggle('hidden', p.dataset.scope !== b.dataset.scope)});
  });
});
"""


def fmt(v, kind):
    if v is None:
        return None
    if kind == "pct":
        return f"{v:.0f}%"
    if kind == "int":
        return f"{v:.0f}"
    if kind == "dec2":
        return f"{v:.2f}"
    return f"{v:.1f}"


def row(label, a, b, kind, text_a=None, text_b=None):
    """One row of the statistics table: values at the sides, the name in the middle, bars from the centre.
    text_a / text_b replace the printed values (e.g. '88% (467/531)'); the bars always follow a and b."""
    if a is None or b is None:
        return ""
    total = a + b
    pa, pb = (100 * a / total, 100 * b / total) if total > 0 else (0, 0)
    ca, cb = ("lead" if a > b else ""), ("lead" if b > a else "")
    return (f'<div class="row"><div class="vals"><span class="v left">{text_a or fmt(a, kind)}</span><span class="lab">{html.escape(label)}</span>'
            f'<span class="v right">{text_b or fmt(b, kind)}</span></div><div class="bars">'
            f'<div class="half left"><div class="bar {ca}" style="width:{pa:.1f}%"></div></div>'
            f'<div class="half right"><div class="bar {cb}" style="width:{pb:.1f}%"></div></div></div></div>')


def stack_row(label, a_parts, b_parts, colors):
    """Where a team had the ball: one stacked bar per team (own third / middle / final third)."""
    def bar(parts, c):
        alphas = (0.35, 0.65, 1.0)
        return '<div class="stack">' + "".join(f'<b style="width:{p:.1f}%;background:{c};opacity:{al}"></b>' for p, al in zip(parts, alphas)) + "</div>"
    def txt(parts):
        return " / ".join(f"{p:.0f}%" for p in parts)
    return (f'<div class="row"><div class="vals"><span class="v left">{txt(a_parts)}</span><span class="lab">{html.escape(label)}</span>'
            f'<span class="v right">{txt(b_parts)}</span></div><div class="bars"><div class="half left" style="background:none">{bar(a_parts, colors[0])}</div>'
            f'<div class="half right" style="background:none">{bar(b_parts, colors[1])}</div></div></div>')


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def load_distance(path):
    out = defaultdict(dict)                                    # (scope) -> {team: km}
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            half, team, km = int(r["half"]), int(r["team"]), float(r["team_field_km"])
            out[f"half_{half}"][team] = km
            out["match"][team] = out["match"].get(team, 0.0) + km
    return out


def collect(summary, stats, dist, actions, scope):
    """The numbers of both teams for a scope: 'match', 'half_1' or 'half_2'."""
    data = {0: {}, 1: {}}
    poss = summary.get("possession_percent" if scope == "match" else scope, {})
    for t in (0, 1):
        s = stats.get(scope, {}).get(f"team_{t}", {})
        d = data[t]
        d["possession"] = poss.get(f"team_{t}")
        d["possession_opp"] = s.get("possession_in_opponent_half_percent")
        d["recoveries"], d["recoveries_opp"], d["losses"] = s.get("recoveries"), s.get("recoveries_in_opponent_half"), s.get("ball_losses")
        d["possessions"], d["avg_possession"], d["long_possessions"] = s.get("possessions"), s.get("possession_avg_s"), s.get("possessions_over_10s")
        th = s.get("possession_by_third_percent", {})
        d["thirds"] = [th.get("own"), th.get("middle"), th.get("final")] if th else None
        d["avg_position"], d["length"], d["width"] = s.get("average_position_m"), s.get("team_length_m"), s.get("team_width_m")
        d["distance"] = dist.get(scope, {}).get(t)
        ac = actions.get("scopes", {}).get(scope, {}).get(f"team_{t}", {})
        d.update({"xg": ac.get("xg"), "shots": ac.get("shots"), "shots_on": ac.get("shots_on_target"), "big_chances": ac.get("big_chances"),
                  "corners": ac.get("corners"), "box_entries": ac.get("box_entries"), "pass_att": ac.get("passes_attempted"),
                  "pass_ok": ac.get("passes_completed"), "pass_acc": ac.get("pass_accuracy_percent")})
        att, ok = ac.get("passes_attempted") or 0, ac.get("passes_completed") or 0
        est = pass_calibration.estimate(ok, att - ok) if att else None
        d["pass_acc_est"], d["pass_acc_ci"] = (est[0], (est[1], est[2])) if est else (None, None)
    return data


def pane(scope, data, T, colors, hidden, show_shots=False, vis=None):
    a, b = data[0], data[1]
    rows = []
    if show_shots:
        rows.append(row(T["xg"], a["xg"], b["xg"], "dec2"))
    rows.append(row(T["possession"], a["possession"], b["possession"], "pct"))
    if show_shots:
        rows += [row(T["shots"], a["shots"], b["shots"], "int"), row(T["shots_on"], a["shots_on"], b["shots_on"], "int"),
                 row(T["big_chances"], a["big_chances"], b["big_chances"], "int")]
    rows.append(row(T["corners"], a["corners"], b["corners"], "int"))
    pass_note = ""
    if a.get("pass_acc_est") is not None and b.get("pass_acc_est") is not None:
        rows.append(row(T["passes"], a["pass_acc_est"], b["pass_acc_est"], "pct", f'≈{a["pass_acc_est"]:.0f}% ({a["pass_ok"]}/{a["pass_att"]})',
                        f'({b["pass_ok"]}/{b["pass_att"]}) ≈{b["pass_acc_est"]:.0f}%'))
        ci = lambda d: f'{d["pass_acc_ci"][0]:.0f}-{d["pass_acc_ci"][1]:.0f}%'
        pass_note = T["pass_note"].format(n=pass_calibration.judged(), m=len(pass_calibration.CALIBRATION["matches"]), ca=ci(a), cb=ci(b))
    elif a["pass_acc"] is not None and b["pass_acc"] is not None:
        rows.append(row(T["passes"], a["pass_acc"], b["pass_acc"], "pct", f'{a["pass_acc"]:.0f}% ({a["pass_ok"]}/{a["pass_att"]})',
                        f'({b["pass_ok"]}/{b["pass_att"]}) {b["pass_acc"]:.0f}%'))
    rows += [
        row(T["possession_opp"], a["possession_opp"], b["possession_opp"], "pct"),
        row(T["box_entries"], a["box_entries"], b["box_entries"], "int"),
        row(T["recoveries"], a["recoveries"], b["recoveries"], "int"),
        row(T["recoveries_opp"], a["recoveries_opp"], b["recoveries_opp"], "int"),
        row(T["losses"], a["losses"], b["losses"], "int"),
        row(T["possessions"], a["possessions"], b["possessions"], "int"),
        row(T["avg_possession"], a["avg_possession"], b["avg_possession"], "dec"),
        row(T["long_possessions"], a["long_possessions"], b["long_possessions"], "int"),
        row(T["distance"], a["distance"], b["distance"], "dec"),
    ]
    top = "".join(rows)
    if vis is not None and any(x is not None for x in (a["corners"], a["pass_acc"], a["box_entries"])):
        top += f'<p class="note" style="text-align:left;margin-top:-4px">{html.escape(T["est_note"].format(vis=vis))}</p>'
        if pass_note:
            top += f'<p class="note" style="text-align:left;margin-top:-4px">{html.escape(pass_note)}</p>'
    thirds = ""
    if a["thirds"] and b["thirds"] and None not in a["thirds"] + b["thirds"]:
        thirds = (f'<div class="panel"><h3>{T["thirds"]}</h3>' + "".join(
            row(T[k], a["thirds"][i], b["thirds"][i], "pct") for i, k in enumerate(("own_third", "middle_third", "final_third"))) + "</div>")
    shape = "".join([row(T["avg_position"], a["avg_position"], b["avg_position"], "dec"),
                     row(T["length"], a["length"], b["length"], "dec"), row(T["width"], a["width"], b["width"], "dec")])
    shape = f'<div class="panel"><h3>{T["shape"]}</h3>{shape}</div>' if shape else ""
    return (f'<div class="pane{" hidden" if hidden else ""}" data-scope="{scope}"><div class="panel"><h3>{T["top"]}</h3>{top}</div>'
            f'{thirds}{shape}</div>')


# ------------------------------------------------------------------ drawings
def load_goals(folder, match_path):
    """Goals of goals_auto.csv (detect_goals.py): [{'minute', 'recording_s', 'team', 'rough'}], minute of play if the match config is given."""
    path = folder / "goals_auto.csv"
    if not path.exists():
        return []
    starts, lens = {}, {}
    with open(path) as f:                                          # the first kick-off of every half: the clock of the match starts there
        kick = {int(r["half"]): float(r["kickoff_s"]) for r in csv.DictReader(f) if r["category"] == "start of the half"}
    if match_path:
        cfg = json.loads(Path(match_path).expanduser().read_text())
        acc = 0.0
        for i, h in enumerate(cfg["halves"], 1):
            t0 = kick.get(i, h["start_s"])
            starts[i], lens[i] = (t0, acc), h["end_s"] - t0
            acc += h["end_s"] - t0
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("category") != "goal" or not r.get("goal_s"):
                continue
            t, h = float(r["goal_s"]), int(r["half"])
            minute = int((t - starts[h][0] + starts[h][1]) // 60) + 1 if h in starts else None
            out.append({"minute": minute, "recording_s": t, "half": h, "team": int(r["scorer"]) if r.get("scorer", "") != "" else None,
                        "rough": r.get("goal_time_is_rough") == "1"})
    return sorted(out, key=lambda g: g["recording_s"])


def quality_banner(qual, T, lang):
    """The banner on top of the report: the level of trust and what is not ok (plus the standing note on the goals)."""
    items = "".join(f"<li>{html.escape(c[lang])}</li>" for c in qual["checks"] if c["status"] != "ok" or c["id"] == "info")
    return f'<div class="quality {qual["level"]}"><h3>{html.escape(T["quality_title"][qual["level"]])}</h3><ul>{items}</ul></div>'


def shows_panel(qual, T):
    """What the report shows and what it does not, with the measured accuracy (quality.MEASURED)."""
    items = "".join(f"<li>{html.escape(t.format(**qual['measured']))}</li>" for t in T["shows"])
    return f'<div class="panel"><h3>{T["shows_title"]}</h3><ul>{items}</ul></div>'


def goals_panel(goals, T, names, colors, total=None):
    if not goals:
        return ""
    rows = []
    for g in goals:
        when = T["goal_minute"].format(m=g["minute"]) if g["minute"] else f"{int(g['recording_s']) // 60:02d}:{int(g['recording_s']) % 60:02d}"
        who = (f'<span class="dot" style="background:{colors[g["team"]]}"></span>{html.escape(names[g["team"]])}' if g["team"] in (0, 1) else "?")
        rows.append(f'<div class="goal"><span class="m">{when}{"<span class=rough> ~</span>" if g["rough"] else ""}</span>{who}</div>')
    miss = f'<p class="note">{T["goals_missing"].format(n=len(goals), total=total)}</p>' if total and total > len(goals) else ""
    return f'<div class="panel"><h3>{T["goals"]}</h3><p class="note">{T["goals_auto"]}</p>{"".join(rows)}{miss}</div>'


def timeline_svg(path, T, names, colors, step=5.0, window=60.0):
    if not path.exists():
        return ""
    by = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by[int(r["half"])].append((float(r["time_s"]), int(r["possession_team"]) if r["possession_team"] != "" else -1))
    parts = []
    for half in sorted(by):
        t = np.array([x[0] for x in by[half]]); team = np.array([x[1] for x in by[half]])
        t = t - t[0]
        c0, c1 = np.cumsum(team == 0), np.cumsum(team == 1)
        grid = np.arange(window, t[-1], step)
        pts = []
        for g in grid:
            i, j = int(np.searchsorted(t, g)), int(np.searchsorted(t, g - window))
            n0, n1 = c0[min(i, len(t) - 1)] - c0[j], c1[min(i, len(t) - 1)] - c1[j]
            if n0 + n1 > 0:
                pts.append((g / 60, 100 * n0 / (n0 + n1)))
        if len(pts) < 3:
            continue
        w, h, pad_l, pad_b, pad_t = 340, 130, 26, 20, 8
        xmax = max(25.0, t[-1] / 60)
        X = lambda m: pad_l + (w - pad_l - 6) * m / xmax
        Y = lambda v: pad_t + (h - pad_t - pad_b) * (1 - v / 100)
        def area(above, col):
            xs = [X(m) for m, _ in pts]
            top = [Y(50 + max(v - 50, 0) if above else 50 - max(50 - v, 0)) for _, v in pts]
            poly = [f"{x:.1f},{y:.1f}" for x, y in zip(xs, top)] + [f"{x:.1f},{Y(50):.1f}" for x in reversed(xs)]
            return f'<polygon points="{" ".join(poly)}" fill="{col}" fill-opacity=".85"/>'
        grid_lines = "".join(f'<line x1="{pad_l}" x2="{w - 6}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#25404e" stroke-width="{1.4 if v == 50 else .8}"/>'
                             f'<text x="{pad_l - 4}" y="{Y(v) + 3:.1f}" fill="#8fa3ae" font-size="8" text-anchor="end">{v}</text>' for v in (25, 50, 75))
        ticks = "".join(f'<text x="{X(m):.1f}" y="{h - 6}" fill="#8fa3ae" font-size="8" text-anchor="middle">{m}</text>' for m in range(0, int(xmax) + 1, 5))
        parts.append(f'<div><div class="cap">{T["half"].format(n=half)}</div><svg viewBox="0 0 {w} {h}">{grid_lines}{ticks}'
                     f'{area(True, colors[0])}{area(False, colors[1])}</svg></div>')
    if not parts:
        return ""
    legend = (f'<div class="note"><span class="dot" style="background:{colors[0]}"></span> {html.escape(names[0])} &nbsp;&nbsp; '
              f'<span class="dot" style="background:{colors[1]}"></span> {html.escape(names[1])} &nbsp;·&nbsp; {T["timeline_note"]}</div>')
    return f'<div class="panel"><h3>{T["timeline"]}</h3>{legend}<div class="two">{"".join(parts)}</div></div>'


def heat_svg(path, stats, pitch, T, names, colors, nx=28, ny=16):
    if not path.exists():
        return ""
    L, W = pitch["length_m"], pitch["width_m"]
    right = stats.get("attacks_right", {})
    grids = {0: np.zeros((ny, nx)), 1: np.zeros((ny, nx))}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("team") not in ("0", "1") or not r.get("x_m") or not r.get("y_m"):
                continue
            team, half = int(r["team"]), int(r["half"])
            x, y = float(r["x_m"]), float(r["y_m"])
            if right.get(f"half_{half}") is not None and right.get(f"half_{half}") != team:
                x, y = L - x, W - y                               # every team attacks to the right
            ix, iy = int(np.clip(x / L * nx, 0, nx - 1)), int(np.clip(y / W * ny, 0, ny - 1))
            grids[team][iy, ix] += 1
    scale, pad = 9.0, 6
    w, h = L * scale + 2 * pad, W * scale + 2 * pad
    out = []
    for t in (0, 1):
        g = grids[t] / max(grids[t].max(), 1)
        cells = "".join(f'<rect x="{pad + ix * L * scale / nx:.1f}" y="{pad + iy * W * scale / ny:.1f}" width="{L * scale / nx + .3:.1f}" '
                        f'height="{W * scale / ny + .3:.1f}" fill="{colors[t]}" fill-opacity="{min(1.0, g[iy, ix] ** 0.6) * .95:.2f}"/>'
                        for iy in range(ny) for ix in range(nx) if g[iy, ix] > 0.02)
        pw = f'stroke="#e8eef2" stroke-width="1.1" fill="none"'
        lines = (f'<rect x="{pad}" y="{pad}" width="{L * scale}" height="{W * scale}" {pw}/>'
                 f'<line x1="{pad + L * scale / 2}" x2="{pad + L * scale / 2}" y1="{pad}" y2="{pad + W * scale}" {pw}/>'
                 f'<circle cx="{pad + L * scale / 2}" cy="{pad + W * scale / 2}" r="{pitch.get("centre_circle_radius_m", 4.75) * scale}" {pw}/>')
        d, pwid = pitch.get("penalty_area_depth_m", 9.0), pitch.get("penalty_area_width_m", 20.0)
        for x0, x1 in ((0, d), (L, L - d)):
            lines += (f'<rect x="{pad + min(x0, x1) * scale}" y="{pad + (W - pwid) / 2 * scale}" width="{abs(x1 - x0) * scale}" height="{pwid * scale}" {pw}/>')
        out.append(f'<div><div class="cap"><span class="dot" style="background:{colors[t]}"></span>{html.escape(names[t])}</div>'
                   f'<svg viewBox="0 0 {w:.0f} {h:.0f}"><rect width="{w:.0f}" height="{h:.0f}" rx="6" fill="#1d5a2b"/>{cells}{lines}</svg></div>')
    return f'<div class="panel"><h3>{T["heat"]}</h3><p class="note">{T["heat_note"]}</p><div class="two">{"".join(out)}</div></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder of a match")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--names", default=None, help='names of the teams, e.g. "Team A,Team B" (team 0 first)')
    ap.add_argument("--score", default=None, help="final score, team 0 first, e.g. 11-2 (without it: from the goals found automatically, goals_auto.csv)")
    ap.add_argument("--match", help="match config (half times): the goals are then shown with the minute of play")
    ap.add_argument("--date", default="", help="date of the match, e.g. 2026-09-20")
    ap.add_argument("--title", default=None, help="competition or title of the page")
    ap.add_argument("--lang", choices=["pl", "en"], default="pl")
    ap.add_argument("--style", choices=["classic", "team"], default="classic",
                    help="classic = the leader in red, the other in white (as on the portal); team = the bars in the colours of the teams")
    ap.add_argument("--shots", choices=["auto", "always", "never"], default="auto",
                    help="shots and xG: auto = only when the ball track found at least half of the goals as shots (see match_actions.py)")
    ap.add_argument("--gif", help="animation (GIF from render_overlay.py) to put in the page")
    ap.add_argument("--out")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    T = TEXT[a.lang]
    pitch = json.loads(Path(a.pitch).read_text())
    summary, stats = load_json(folder / "summary.json"), load_json(folder / "stats.json")
    if not summary or not stats:
        raise SystemExit(f"summary.json and stats.json are needed in {folder} (analyze_dual.py and match_stats.py)")
    dist = load_distance(folder / "team_distance.csv")
    actions = load_json(folder / "actions.json")
    quality = actions.get("quality", {})
    found_pct = quality.get("goals_found_percent")
    show_shots = bool(actions) and (a.shots == "always" or (a.shots == "auto" and found_pct is not None and found_pct >= 50))
    vis = quality.get("ball_visible_percent") if actions else None
    names = a.names.split(",") if a.names else [T["team"].format(n=0), T["team"].format(n=1)]
    colors = ["#2a63d6", "#c0303b"]
    cf = folder / "teams_colors.json"
    if cf.exists() and False:                                   # the kit colours are often pale; the animation uses blue and red
        c = json.loads(cf.read_text())
    goals = load_goals(folder, a.match)
    import quality as quality_module
    qpath = folder / "quality.json"
    qual = json.loads(qpath.read_text()) if qpath.exists() else quality_module.assess(folder)
    score_approx = any(c["id"] == "goals_guess" and not a.score for c in qual["checks"])
    score, score_auto = ("–", "–"), False
    if a.score:
        x, y = a.score.replace(":", "-").split("-")
        score = (x.strip(), y.strip())
    elif goals:
        score, score_auto = (str(sum(1 for g in goals if g["team"] == 0)), str(sum(1 for g in goals if g["team"] == 1))), True

    scopes = ["match", "half_1", "half_2"]
    data = {sc: collect(summary, stats, dist, actions, sc) for sc in scopes}
    panes = "".join(pane(sc, data[sc], T, colors, hidden=(sc != "match"), show_shots=show_shots, vis=vis) for sc in scopes)
    tabs = "".join(f'<button class="tab{" active" if i == 0 else ""}" data-scope="{sc}">{label}</button>' for i, (sc, label) in enumerate(zip(scopes, T["tabs"])))
    dead = sum(summary.get(k, 0) for k in ("ball_out_of_pitch_percent", "ball_still_percent", "ball_restart_percent"))
    about = T["about_text"].format(assigned=summary.get("assigned_share", 0), dead=dead)
    if actions and quality.get("goals_given"):
        about += T["est_goals"].format(given=quality["goals_given"], found=quality.get("goals_found_as_shots", 0))
    if actions and not show_shots:
        about += " " + T["shots_hidden"].format(pct=("–" if found_pct is None else f"{found_pct:.0f}"))
    banner, shows = quality_banner(qual, T, a.lang), shows_panel(qual, T)
    gif = ""
    if a.gif:
        g = Path(a.gif).expanduser()
        if not g.is_absolute() and not g.exists():
            g = folder / a.gif
        gif = (f'<div class="panel"><h3>{T["animation"]}</h3><img class="anim" alt="" src="data:image/gif;base64,'
               f'{base64.b64encode(g.read_bytes()).decode()}"/></div>')
    payload = {"teams": names, "score": list(score), "score_automatic": score_auto, "goals": goals, "date": a.date, "title": a.title or T["analysis"], "scopes": {sc: {
        f"team_{t}": {k: v for k, v in data[sc][t].items()} for t in (0, 1)} for sc in scopes},
        "quality": {"level": qual["level"], "checks": [{"id": c["id"], "status": c["status"]} for c in qual["checks"]], "score_approximate": score_approx, "possession_assigned_percent": summary.get("assigned_share"), "ball_dead_percent": dead, "actions": quality,
                    "shots_shown": show_shots}}
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    page = f"""<!DOCTYPE html>
<html lang="{a.lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(names[0])} {score[0]} - {score[1]} {html.escape(names[1])}</title><style>{CSS}</style></head>
<body class="style-{a.style}"><div class="wrap">
<div class="crumbs">{T["sport"]}<span>›</span>{html.escape(a.title or T["analysis"])}</div>
<div class="head">
  <div class="team"><div class="badge"><i style="background:{colors[0]}"></i></div>{html.escape(names[0])}</div>
  <div class="mid"><div class="date">{html.escape(a.date)}</div><div class="score">{score[0]} - {score[1]}</div><div class="status">{T["status"]}</div>{('<div class="date" style="font-size:11px;margin-top:4px;color:#b45309">' + html.escape(T["score_approx"]) + '</div>' if score_approx else '')}{('<div class="date" style="font-size:11px;margin-top:4px">' + html.escape(T["score_auto"]) + '</div>') if score_auto else ''}</div>
  <div class="team"><div class="badge"><i style="background:{colors[1]}"></i></div>{html.escape(names[1])}</div>
</div>
{banner}
<div class="tabs">{tabs}</div>
{panes}
{goals_panel(goals, T, names, colors, total=(int(score[0]) + int(score[1])) if (score[0].isdigit() and score[1].isdigit()) else None)}
{timeline_svg(folder / "possession.csv", T, names, colors)}
{heat_svg(folder / "tracks.csv", stats, pitch, T, names, colors)}
{gif}
{shows}
<div class="panel"><h3>{T["about"]}</h3><div class="about">{html.escape(about)}</div></div>
</div>
<script type="application/json" id="match-data">{payload_json}</script>
<script>{JS}</script></body></html>"""
    out = Path(a.out).expanduser() if a.out else folder / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
