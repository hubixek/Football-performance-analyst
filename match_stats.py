#!/usr/bin/env python3
"""Extended match statistics from the results of analyze_dual.py (no video needed).

Attacking direction: in every half the team whose players stand on average closer to the
left goal defends the left goal. All statistics are then computed as if each team always
attacked to the right, so both halves can be added up.

Statistics (per team, for the match and per half):
  - possession by pitch third (own third / middle / final third) and in the opponent's half,
  - team shape: average position (how high the team plays), length and width,
  - possession sequences: number, average and longest duration,
  - ball losses and recoveries, recoveries in the opponent's half (high press).
Plots:
  - stats.png: heatmaps of both teams, possession by third, where the ball was won,
    length of possession sequences, summary table,
  - possession_halves.png: ball position by team in possession, each half separately.

Usage:
  python match_stats.py ~/football/analysis/mecz1_dual --pitch pitch/pitch_6v6.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(folder):
    players = defaultdict(list)       # frame -> [(x, y, team)]
    info = {}                         # frame -> (time, half)
    with open(folder / "teams.csv") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"])
            info[fr] = (float(r["time_s"]), int(r["half"]))
            if r["class"] == "player" and r["team"] in ("0", "1"):
                players[fr].append((float(r["x_m"]), float(r["y_m"]), int(r["team"]), r.get("gk") == "1"))
    poss = []
    with open(folder / "possession.csv") as f:
        for r in csv.DictReader(f):
            poss.append({"frame": int(r["frame"]), "t": float(r["time_s"]), "half": int(r["half"]),
                         "x": float(r["ball_x"]) if r["ball_x"] else None,
                         "y": float(r["ball_y"]) if r["ball_y"] else None,
                         "team": int(r["possession_team"]) if r["possession_team"] else None})
    return players, info, poss


def attack_right(players, info):
    """(half, team) -> True if the team attacks to the right (higher x) in that half."""
    xs = defaultdict(list)
    for fr, pl in players.items():
        for x, _, t, is_gk in pl:
            if not is_gk:
                xs[(info[fr][1], t)].append(x)
    out = {}
    for half in sorted({h for h, _ in xs}):
        m0 = np.mean(xs.get((half, 0), [np.nan]))
        m1 = np.mean(xs.get((half, 1), [np.nan]))
        out[(half, 0)], out[(half, 1)] = bool(m0 < m1), bool(not m0 < m1)
    return out


def norm_x(x, half, team, direction, L):
    return x if direction[(half, team)] else L - x


def norm_xy(x, y, half, team, direction, L, W):
    return (x, y) if direction[(half, team)] else (L - x, W - y)


def sequences(poss, min_s, max_gap=0.3):
    """Possession runs of one team.

    A run ends when the other team gets the ball or nobody has possession (ball out of play,
    ball lost). Runs shorter than min_s are noise (duels, deflections) and are dropped. Two runs
    of the same team are merged only if they are separated by such dropped short runs that touch
    each other and the two runs - never across a gap without possession (max_gap seconds)."""
    raw, cur = [], None
    for p in poss:
        if cur and (p["team"] is None or p["team"] != cur["team"] or p["half"] != cur["half"]):
            raw.append(cur)
            cur = None
        if p["team"] is not None:
            if cur is None:
                cur = {"team": p["team"], "half": p["half"], "t0": p["t"], "t1": p["t"], "start": p}
            cur["t1"] = p["t"]
    if cur:
        raw.append(cur)
    merged, chain, prev = [], False, None
    for r in raw:
        if prev is None or prev["half"] != r["half"] or r["t0"] - prev["t1"] > max_gap:
            chain = False
        if r["t1"] - r["t0"] >= min_s:
            if chain and merged and merged[-1]["team"] == r["team"] and merged[-1]["half"] == r["half"]:
                merged[-1]["t1"] = r["t1"]
            else:
                merged.append(dict(r))
            chain = True
        prev = r
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="analysis folder of a match (analyze_dual.py output)")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--min-seq", type=float, default=1.0, help="shortest possession counted, s")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    pitch = json.loads(Path(a.pitch).read_text())
    L, W = pitch["length_m"], pitch["width_m"]
    players, info, poss = load(folder)
    direction = attack_right(players, info)
    halves = sorted({p["half"] for p in poss})
    seqs_all = sequences(poss, a.min_seq)

    cols = ["tab:blue", "tab:red"]
    colors_file = folder / "teams_colors.json"
    if colors_file.exists():
        c = json.loads(colors_file.read_text())
        cols = [tuple(v / 255 for v in c[f"team_{t}_bgr"][::-1]) for t in (0, 1)]

    def recoveries(seqs):
        """New possession of a team right after one of the other team (same half, < 3 s apart)."""
        return [q for prev, q in zip(seqs, seqs[1:])
                if q["team"] != prev["team"] and q["half"] == prev["half"] and q["t0"] - prev["t1"] < 3.0
                and q["start"]["x"] is not None]

    def stats_for(sel):
        out = {}
        pp = [p for p in poss if p["half"] in sel]
        seqs = [s for s in seqs_all if s["half"] in sel]
        rec_all = recoveries(seqs)
        for t in (0, 1):
            s = {}
            xs = np.array([norm_x(p["x"], p["half"], t, direction, L)
                           for p in pp if p["team"] == t and p["x"] is not None])
            if len(xs):
                s["possession_by_third_percent"] = {
                    "own": round(100 * float(np.mean(xs < L / 3)), 1),
                    "middle": round(100 * float(np.mean((xs >= L / 3) & (xs < 2 * L / 3))), 1),
                    "final": round(100 * float(np.mean(xs >= 2 * L / 3)), 1)}
                s["possession_in_opponent_half_percent"] = round(100 * float(np.mean(xs >= L / 2)), 1)
            heights, lengths, widths = [], [], []
            for fr, pl in players.items():
                if info[fr][1] not in sel:
                    continue
                pts = np.array([norm_xy(x, y, info[fr][1], t, direction, L, W)
                                for x, y, tt, is_gk in pl if tt == t and not is_gk])
                if len(pts) >= 4:
                    heights.append(pts[:, 0].mean())
                    lengths.append(np.ptp(pts[:, 0]))
                    widths.append(np.ptp(pts[:, 1]))
            if heights:
                s["average_position_m"] = round(float(np.mean(heights)), 1)
                s["team_length_m"] = round(float(np.mean(lengths)), 1)
                s["team_width_m"] = round(float(np.mean(widths)), 1)
            own = np.array([q["t1"] - q["t0"] for q in seqs if q["team"] == t])
            if len(own):
                s["possessions"] = int(len(own))
                s["possession_avg_s"] = round(float(own.mean()), 1)
                s["possession_longest_s"] = round(float(own.max()), 1)
                s["possessions_over_10s"] = int((own >= 10).sum())
            rec = [q for q in rec_all if q["team"] == t]
            s["recoveries"] = len(rec)
            s["recoveries_in_opponent_half"] = sum(
                1 for q in rec if norm_x(q["start"]["x"], q["half"], t, direction, L) >= L / 2)
            out[f"team_{t}"] = s
        for t in (0, 1):
            out[f"team_{t}"]["ball_losses"] = out[f"team_{1 - t}"]["recoveries"]
        return out

    summary = {"attacks_right": {f"half_{h}": next(t for t in (0, 1) if direction[(h, t)]) for h in halves},
               "match": stats_for(halves)}
    for h in halves:
        summary[f"half_{h}"] = stats_for([h])
    (folder / "stats.json").write_text(json.dumps(summary, indent=2))

    rows = [("possessions (>= %.0f s)" % a.min_seq, "possessions"), ("avg possession [s]", "possession_avg_s"),
            ("longest possession [s]", "possession_longest_s"), ("possessions over 10 s", "possessions_over_10s"),
            ("ball losses", "ball_losses"), ("recoveries in opp. half", "recoveries_in_opponent_half"),
            ("possession in opp. half [%]", "possession_in_opponent_half_percent"),
            ("average position [m] (no GK)", "average_position_m"), ("team length [m] (no GK)", "team_length_m"),
            ("team width [m] (no GK)", "team_width_m")]
    m = summary["match"]
    print(f"{'':30s}{'team 0':>10s}{'team 1':>10s}")
    for name, k in rows:
        print(f"{name:30s}{str(m['team_0'].get(k, '-')):>10s}{str(m['team_1'].get(k, '-')):>10s}")
    for zone in ("own", "middle", "final"):
        v = [m[f"team_{t}"].get("possession_by_third_percent", {}).get(zone, "-") for t in (0, 1)]
        print(f"{'possession in ' + zone + ' third [%]':30s}{str(v[0]):>10s}{str(v[1]):>10s}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw_pitch(ax):
        ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], "k-", lw=1)
        ax.plot([L / 2, L / 2], [0, W], "k-", lw=1)
        for x in (L / 3, 2 * L / 3):
            ax.plot([x, x], [0, W], "k:", lw=0.8)
        ax.set_xlim(-2, L + 2)
        ax.set_ylim(W + 2, -2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    for t in (0, 1):
        ax = axes[0, t]
        pts = np.array([norm_xy(x, y, info[fr][1], t, direction, L, W)
                        for fr, pl in players.items() for x, y, tt, _ in pl if tt == t])
        if len(pts):
            ax.hist2d(pts[:, 0], pts[:, 1], bins=[28, 16], range=[[0, L], [0, W]], cmap="magma")
        draw_pitch(ax)
        ax.set_title(f"Team {t}: player positions (attacking →)")
    ax = axes[1, 0]
    for t in (0, 1):
        v = m[f"team_{t}"].get("possession_by_third_percent", {})
        ax.bar(np.arange(3) + (t - 0.5) * 0.38, [v.get(z, 0) for z in ("own", "middle", "final")], 0.38,
               color=cols[t], edgecolor="k", label=f"team {t}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["own third", "middle third", "final third"])
    ax.set_ylabel("share of own possession [%]")
    ax.set_title("Where each team had the ball")
    ax.legend()
    ax = axes[1, 1]
    for q in recoveries(seqs_all):
        x, y = norm_xy(q["start"]["x"], q["start"]["y"], q["half"], q["team"], direction, L, W)
        ax.scatter(x, y, s=16, color=cols[q["team"]], edgecolors="k", linewidths=0.3)
    draw_pitch(ax)
    ax.set_title("Where the ball was won (attacking →)")
    ax = axes[2, 0]
    for t in (0, 1):
        d = [q["t1"] - q["t0"] for q in seqs_all if q["team"] == t]
        ax.hist(np.clip(d, 0, 30), bins=np.arange(0, 32, 2), alpha=0.6, color=cols[t], edgecolor="k",
                label=f"team {t}")
    ax.set_xlabel("possession length [s] (last bar = 30 s or more)")
    ax.set_ylabel("count")
    ax.set_title("Possession sequences")
    ax.legend()
    ax = axes[2, 1]
    ax.axis("off")
    cell = [[name] + [str(m[f"team_{t}"].get(k, "-")) for t in (0, 1)] for name, k in rows]
    ax.table(cellText=cell, colLabels=["", "team 0", "team 1"], loc="center", cellLoc="center").scale(1, 1.5)
    ax.set_title("Match summary")
    fig.tight_layout()
    fig.savefig(folder / "stats.png", dpi=110)

    fig, axes = plt.subplots(1, len(halves), figsize=(7 * len(halves), 5.2))
    for ax, h in zip(np.atleast_1d(axes), halves):
        for t in (0, 1):
            pts = np.array([(p["x"], p["y"]) for p in poss
                            if p["half"] == h and p["team"] == t and p["x"] is not None])
            if len(pts):
                ax.scatter(pts[::3, 0], pts[::3, 1], s=3, color=cols[t], alpha=0.5,
                           label=f"team {t} attacks {'→' if direction[(h, t)] else '←'}")
        draw_pitch(ax)
        ax.set_title(f"Half {h}: ball position by team in possession")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, markerscale=4)
    fig.tight_layout()
    fig.savefig(folder / "possession_halves.png", dpi=110)
    print(f"\nSaved {folder / 'stats.json'}, stats.png, possession_halves.png")


if __name__ == "__main__":
    main()
