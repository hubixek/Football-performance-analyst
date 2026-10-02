#!/usr/bin/env python3
"""Tidies the project: shows what is not needed, deletes only what you allow.

By default NOTHING is deleted (a dry run: every file with its size). Categories:
  junk            deleted with --apply: __pycache__, *.pyc, *.bak, *.orig, *.swp, *~, nohup.out, archives in the repository folder, provisional match
                  configs left by an interrupted process_match.py, the files of the ablation of the team steps in the analysis folders
                  (t_colour_raw*, t_kickoff_raw*, t_kickoff_ref*, t_kickoff_all*)
  backups         with --also backups: teams_default.csv (the teams of the old method, saved before the classifier)
  abandoned       with --also abandoned: the files of the runs named in --abandoned (default mecz1_auto: matches/<n>.json, calib/<n>_*.json,
                  analysis/<n>*, analysis/process_<n>.txt)
  stare-weights   with --also stare-weights: the weights (*.pt) of the old experiments in runs/detect/runs/stare (their results.csv, args.yaml
                  and plots stay as the record)
  stare           with --also stare: the whole runs/detect/runs/stare folder
  caches          with --also caches: the colour caches (*_rgb.npz) of ALL analysis folders (a re-run of the team step then reads the video again, ~20 min)
Never touched: videos, dataset_dual, the models v6/v7/v8, the calibrations and analyses of the matches you did not name, .git.
The scripts that no other script and no document mentions are only listed (maybe obsolete: look at them by hand).

  --collect-eval  copies your judgements (team_check.csv and its key) and goals_auto.csv of the analyses into eval/<analysis>/ of the repository,
                  and official_goals.json into eval/ (small files, they belong in git)
  --gitignore     appends the missing patterns (videos, weights, caches, analyses) to .gitignore

Usage:
  python tidy_project.py                                  (a dry run)
  python tidy_project.py --collect-eval --gitignore --apply
  python tidy_project.py --apply --also backups,abandoned,stare-weights
"""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALL = ("junk", "backups", "abandoned", "stare-weights", "stare", "caches")
GITIGNORE = ["__pycache__/", "*.pyc", ".pytest_cache/", "*.mp4", "*.avi", "*.pt", "*.npz", "*.tar.gz", "*.zip", "nohup.out", "analysis/", "runs/"]
JUNK_SUFFIX = (".pyc", ".bak", ".orig", ".swp", "~")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "yolo-env"}


def size_of(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def find(repo, foot, abandoned):
    """[(category, path)] of the candidates."""
    out = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for d in list(dirs):
            if d in ("__pycache__", ".pytest_cache", ".mypy_cache"):
                out.append(("junk", Path(root) / d))
                dirs.remove(d)
        for f in files:
            if f.endswith(JUNK_SUFFIX) or f == "nohup.out" or (Path(root) == repo and f.endswith((".tar.gz", ".zip"))):
                out.append(("junk", Path(root) / f))
    for p in sorted((repo / "matches").glob("*.json")) if (repo / "matches").exists() else []:
        try:
            if json.loads(p.read_text()).get("provisional"):
                out.append(("junk", p))
        except ValueError:
            pass
    analysis = foot / "analysis"
    for d in sorted(analysis.iterdir()) if analysis.exists() else []:
        if not d.is_dir():
            continue
        for pat in ("t_colour_raw*", "t_kickoff_raw*", "t_kickoff_ref*", "t_kickoff_all*"):
            out += [("junk", p) for p in d.glob(pat)]
        if (d / "teams_default.csv").exists():
            out.append(("backups", d / "teams_default.csv"))
        out += [("caches", p) for p in d.glob("*_rgb.npz")]
    for n in abandoned:
        for p in [repo / "matches" / f"{n}.json", analysis / f"process_{n}.txt", *(foot / "calib").glob(f"{n}_*.json"), *analysis.glob(f"{n}*")]:
            if p.exists():
                out.append(("abandoned", p))
    stare = foot / "runs" / "detect" / "runs" / "stare"
    if stare.exists():
        out += [("stare-weights", p) for p in stare.rglob("*.pt")]
        out.append(("stare", stare))
    seen, res = set(), []
    for cat, p in out:                                                # no path twice, no file inside a folder that is listed in the same category
        if p in seen:
            continue
        seen.add(p)
        res.append((cat, p))
    return [(c, p) for c, p in res if not any(q != p and q in p.parents and c2 == c for c2, q in res)]


def unreferenced(repo):
    """Root scripts that no other script and no document mentions."""
    scripts = sorted(p for p in repo.glob("*.py") if p.name not in ("tidy_project.py",))
    texts = {p: p.read_text(errors="ignore") for p in list(repo.glob("*.py")) + list(repo.glob("*.md")) + list((repo / "tests").glob("*.py"))}
    out = []
    for s in scripts:
        pat = re.compile(rf"\b{re.escape(s.stem)}\b")
        if not any(pat.search(t) for p, t in texts.items() if p != s):
            out.append(s)
    return out


def collect_eval(repo, foot):
    n = 0
    analysis = foot / "analysis"
    for d in sorted(analysis.iterdir()) if analysis.exists() else []:
        tc = d / "team_check" / "team_check.csv"
        if not tc.exists():
            continue
        dst = repo / "eval" / d.name
        dst.mkdir(parents=True, exist_ok=True)
        for src in (tc, d / "team_check" / ".answers_of_the_analysis.csv", d / "goals_auto.csv"):
            if src.exists():
                shutil.copy2(src, dst / src.name.lstrip("."))
                n += 1
    for name in ("official_goals.json",):
        if (repo / name).exists():
            (repo / "eval").mkdir(exist_ok=True)
            shutil.move(str(repo / name), repo / "eval" / name)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--football", default=str(Path.home() / "football"), help="the folder with videos, calib, analysis, runs")
    ap.add_argument("--repo", default=str(HERE))
    ap.add_argument("--apply", action="store_true", help="delete (without it only a list)")
    ap.add_argument("--also", default="", help="more categories to delete: " + ", ".join(ALL[1:]))
    ap.add_argument("--abandoned", default="mecz1_auto", help="runs to give up (names, comma separated)")
    ap.add_argument("--collect-eval", action="store_true")
    ap.add_argument("--gitignore", action="store_true")
    a = ap.parse_args()
    repo, foot = Path(a.repo).expanduser(), Path(a.football).expanduser()
    allowed = {"junk"} | {x.strip() for x in a.also.split(",") if x.strip()}
    bad = allowed - set(ALL)
    if bad:
        raise SystemExit(f"unknown categories: {sorted(bad)}; known: {list(ALL)}")

    if a.collect_eval:
        if a.apply:
            print(f"eval/: {collect_eval(repo, foot)} files copied into {repo / 'eval'}")
        else:
            print("eval/: would copy the judgements, goals_auto.csv and official_goals.json (with --apply)")
    if a.gitignore:
        gi = repo / ".gitignore"
        have = set(gi.read_text().split("\n")) if gi.exists() else set()
        missing = [x for x in GITIGNORE if x not in have]
        if missing and a.apply:
            with open(gi, "a") as f:
                f.write(("\n" if have and not gi.read_text().endswith("\n") else "") + "\n# added by tidy_project.py\n" + "\n".join(missing) + "\n")
        print(f".gitignore: {'added' if a.apply else 'would add'} {missing or 'nothing, all there'}")

    items = find(repo, foot, [x.strip() for x in a.abandoned.split(",") if x.strip()])
    freed = {c: 0 for c in ALL}
    print(f"\n{'category':14s} {'size':>9s}  path   ({'DELETING' if a.apply else 'dry run: nothing is deleted'})")
    for cat in ALL:
        part = [(c, p) for c, p in items if c == cat]
        if not part:
            continue
        state = "delete" if cat in allowed else f"kept (--also {cat})"
        for c, p in part:
            s = size_of(p)
            freed[cat] += s
            print(f"{cat:14s} {human(s):>9s}  {p}   [{state}]")
            if a.apply and cat in allowed:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    print()
    for cat in ALL:
        if freed[cat]:
            print(f"{cat:14s} {human(freed[cat]):>9s}  {'deleted' if (a.apply and cat in allowed) else ('to delete' if cat in allowed else 'kept')}")
    unref = unreferenced(repo)
    if unref:
        print("\nscripts that no other script and no document mentions (maybe obsolete, look at them by hand): " + ", ".join(p.name for p in unref))


if __name__ == "__main__":
    main()
