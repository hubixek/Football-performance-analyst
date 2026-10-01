#!/usr/bin/env bash
# Tidy the Football-performance-analyst repo and the data folder (~/football).
#
#   bash tidy_project.sh            dry run: only prints what would be removed
#   bash tidy_project.sh --apply    removes it
#
# Files tracked by git are removed with `git rm`, so they stay in the git history (and the old
# follow-cam version stays in the tag followcam-v1). A script is removed only when no script that
# stays imports or starts it. Only regenerable data is removed; videos, datasets, annotations,
# exports and analysis results are never touched (they are listed at the end for you to judge); the folders with
# your own judgements (analysis/*/disagreements*, analysis/*/team_check, analysis/probe) are protected on purpose.
# Old model runs are MOVED to runs/detect/runs/stare/, not deleted. Run it once and delete it afterwards.
set -u
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1
REPO="${REPO:-$HOME/football/code/Football-performance-analyst}"
DATA="${DATA:-$HOME/football}"
STASH="$DATA/runs/detect/runs/stare"
n=0
kb_total=0

size_kb() { du -sk "$1" 2>/dev/null | cut -f1; }

drop() {        # drop <path> <reason>: not tracked by git
  [ -e "$1" ] || return 0
  local kb; kb=$(size_kb "$1"); n=$((n + 1)); kb_total=$((kb_total + kb))
  printf '  %8s KB  %s   (%s)\n' "$kb" "$1" "$2"
  [ "$APPLY" = 1 ] && rm -rf -- "$1"
  return 0
}

stash() {       # stash <dir> <reason>: moved into $STASH, not deleted
  [ -e "$1" ] || return 0
  local kb; kb=$(size_kb "$1")
  printf '  %8s KB  %s -> %s/   (%s; moved, not deleted)\n' "$kb" "$1" "$STASH" "$2"
  if [ "$APPLY" = 1 ]; then mkdir -p "$STASH"; mv -- "$1" "$STASH/"; fi
  return 0
}

gitdrop() {     # gitdrop <path> <reason>: tracked files go through git rm
  [ -e "$1" ] || return 0
  if ! git ls-files --error-unmatch -- "$1" >/dev/null 2>&1; then drop "$1" "$2"; return 0; fi
  local kb; kb=$(size_kb "$1"); n=$((n + 1)); kb_total=$((kb_total + kb))
  printf '  %8s KB  %s   (%s; git rm)\n' "$kb" "$1" "$2"
  [ "$APPLY" = 1 ] && git rm -q -- "$1"
  return 0
}

cd "$REPO" 2>/dev/null || { echo "no repository at $REPO (set REPO=...)"; exit 1; }
[ "$APPLY" = 1 ] && echo "APPLYING" || echo "DRY RUN (nothing is removed; add --apply to remove)"
echo
echo "== repository: $REPO"

# ---- scripts: follow-cam version and old one-off tools
FOLLOWCAM="analyze.py detect.py ball_track.py teams.py possession.py"
OLD="pick_points.py prelabel_pitch.py clean_pose_labels.py remap_classes.py prelabel.py fix_referees.py"
if grep -qE '^\s*(from|import)\s+teams\b' dual_teams.py 2>/dev/null; then
  echo "  !! dual_teams.py still imports teams.py: copy the new dual_teams.py first. The follow-cam scripts are kept."
  CAND="$OLD"
else
  CAND="$FOLLOWCAM $OLD"
fi
KEEP=$(ls ./*.py 2>/dev/null | sed 's|^\./||' | grep -vxF -f <(echo "$CAND" | tr ' ' '\n') | tr '\n' ' ')
for f in $CAND; do
  [ -e "$f" ] || continue
  stem="${f%.py}"
  # shellcheck disable=SC2086
  ref=$(grep -lE "^\s*(from|import)\s+${stem}\b|[\"']${stem}\.py[\"']" $KEEP 2>/dev/null | head -1)
  if [ -n "$ref" ]; then
    echo "  SKIP     $f: still used by $ref"
  else
    case " $FOLLOWCAM " in
      *" $f "*) gitdrop "$f" "follow-cam version, stays in the tag followcam-v1" ;;
      *)        gitdrop "$f" "old or one-off script" ;;
    esac
  fi
done

# ---- duplicates and unused files
if [ -e pitch/pitch_6v6_measured.json ] && cmp -s pitch/pitch_6v6.json pitch/pitch_6v6_measured.json; then
  gitdrop pitch/pitch_6v6_measured.json "identical to pitch_6v6.json"
fi
if [ -f README.md ]; then
  for f in docs/*; do
    [ -f "$f" ] || continue
    grep -q "$(basename "$f")" README.md || gitdrop "$f" "not used in README.md"
  done
fi
while IFS= read -r d; do drop "$d" "python cache"; done < <(find . -name __pycache__ -type d -not -path './.git/*' 2>/dev/null)
while IFS= read -r d; do drop "$d" "notebook cache"; done < <(find . -name .ipynb_checkpoints -type d -not -path './.git/*' 2>/dev/null)

# ---- .gitignore
for pat in "__pycache__/" "*.pyc" "*.mp4" "*.zip" "*.pt" "*.npz" ".ipynb_checkpoints/"; do
  if ! grep -qxF "$pat" .gitignore 2>/dev/null; then
    echo "  .gitignore: add  $pat"
    [ "$APPLY" = 1 ] && echo "$pat" >> .gitignore
  fi
done

# ---- data folder: only what can be regenerated
echo
echo "== data: $DATA (regenerable)"
for v in v1 v2 v3 v4 v5; do drop "$DATA/runs/detect/runs/$v" "old model run (v6 and v7_dual stay)"; done
for v in v6 v7_dual; do drop "$DATA/runs/detect/runs/$v/weights/last.pt" "resume checkpoint (best.pt stays)"; done
for d in "$DATA"/runs/detect/predict* "$DATA"/runs/detect/val* "$DATA"/runs/detect/train*; do drop "$d" "ad-hoc yolo output"; done
for z in "$DATA"/frames_dual/*_prelabel.zip; do drop "$z" "already imported into CVAT (prelabel_dual.py makes it again)"; done
for c in "$DATA"/analysis/*/clips; do drop "$c" "review clips (check_possessions.py makes them again)"; done
for t in "$DATA"/videos/*_test.mp4; do drop "$t" "short test clip"; done
for l in "$DATA"/calib/*_log.txt; do drop "$l" "log of an earlier calibration attempt"; done
drop "$DATA/tmp_d" "temporary folder"

# models: experiments that did not beat v8_dual are moved aside
RUNS="$DATA/runs/detect/runs"
for d in v9_dual v9_dual-2 v9b_dual v8_nieudany; do
  stash "$RUNS/$d" "experiment that did not beat v8_dual (the model in use)"
done
for d in "$DATA"/runs/detect/chk_* "$DATA"/runs/detect/night_*; do drop "$d" "output of yolo val (made again in a minute)"; done

# dataset: sheets of one-off checks; the backup of the first dataset only if it can be rebuilt from the exports
drop "$DATA/dataset_dual/referee_candidates.jpg" "sheet of a one-off check"
for d in "$DATA"/dataset_dual/ref_check_*; do drop "$d" "sheets of a one-off check"; done
if [ -d "$DATA/dataset_dual_v7" ]; then
  if [ -f "$DATA/exports/dual_mecz2.zip" ] && [ -f "$DATA/exports/dual_mecz1.zip" ]; then
    drop "$DATA/dataset_dual_v7" "copy of the v7 dataset, made again from exports/dual_mecz2.zip and dual_mecz1.zip"
  else
    echo "  SKIP     $DATA/dataset_dual_v7: exports/dual_mecz1.zip or dual_mecz2.zip is missing, so it is the only copy"
  fi
fi

# analyses: short test runs and an unfinished run
for d in "$DATA"/analysis/mecz1_test_v7 "$DATA"/analysis/mecz1_test_v8 "$DATA"/analysis/mecz1_dual_v9 "$DATA"/analysis/mecz1_dual_v8b; do
  drop "$d" "short test run / unfinished run"
done

echo
if [ "$APPLY" = 1 ]; then
  echo "removed $n items, about $((kb_total / 1024)) MB"
  echo "next: git status --short && git commit -am 'Tidy up the repository'"
else
  echo "would remove $n items, about $((kb_total / 1024)) MB. Look through the list, then: bash tidy_project.sh --apply"
fi
echo
echo "== protected on purpose: your own judgements (they cannot be made again)"
ls -d "$DATA"/analysis/*/disagreements* "$DATA"/analysis/*/team_check "$DATA"/analysis/probe "$DATA"/dataset_dual/labels_backup_* 2>/dev/null | sed 's/^/  /'
echo
echo "== not touched, judge yourself (largest first):"
du -sh "$DATA"/* 2>/dev/null | grep -v "/code$" | sort -h | tail -12
