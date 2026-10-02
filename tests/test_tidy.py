import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import helpers


def build(root):
    """A small fake project: a repository folder and a football folder."""
    repo, foot = Path(root) / "repo", Path(root) / "football"
    for d in (repo / "matches", repo / "tests", repo / "__pycache__", foot / "analysis" / "mecz4_2dual" / "team_check", foot / "analysis" / "mecz4_2dual_auto",
              foot / "analysis" / "mecz1_auto", foot / "calib", foot / "videos", foot / "runs" / "detect" / "runs" / "stare" / "v9_dual" / "weights",
              foot / "runs" / "detect" / "runs" / "v8_dual" / "weights"):
        d.mkdir(parents=True, exist_ok=True)
    files = {
        repo / "analyze_dual.py": "import dual_teams\n", repo / "dual_teams.py": "x = 1\n", repo / "old_tool.py": "print(1)\n", repo / "README.md": "use analyze_dual.py\n",
        repo / "__pycache__" / "x.pyc": "b", repo / "notes.bak": "b", repo / "tests_pack.tar.gz": "b",
        repo / "matches" / "mecz4_2dual.json": json.dumps({"halves": []}), repo / "matches" / "mecz9.json": json.dumps({"provisional": True, "halves": []}),
        repo / "matches" / "mecz1_auto.json": json.dumps({"halves": []}),
        foot / "analysis" / "mecz4_2dual" / "teams.csv": "a", foot / "analysis" / "mecz4_2dual" / "teams_default.csv": "a",
        foot / "analysis" / "mecz4_2dual" / "t_colour_raw.csv": "a", foot / "analysis" / "mecz4_2dual" / "t_kickoff_all_rgb.npz": "a",
        foot / "analysis" / "mecz4_2dual" / "teams_rgb.npz": "a" * 100,
        foot / "analysis" / "mecz4_2dual" / "team_check" / "team_check.csv": "id,truth\n1,0\n", foot / "analysis" / "mecz4_2dual" / "team_check" / ".answers_of_the_analysis.csv": "id\n1\n",
        foot / "analysis" / "mecz4_2dual" / "goals_auto.csv": "half\n",
        foot / "analysis" / "mecz4_2dual_auto" / "report.html": "<html>", foot / "analysis" / "mecz1_auto" / "teams.csv": "a", foot / "analysis" / "process_mecz1_auto.txt": "log",
        foot / "calib" / "mecz1_auto_top.json": "{}", foot / "calib" / "mecz4_2dual_top.json": "{}", foot / "videos" / "mecz1_dual.mp4": "video",
        foot / "runs" / "detect" / "runs" / "stare" / "v9_dual" / "weights" / "best.pt": "w" * 50, foot / "runs" / "detect" / "runs" / "stare" / "v9_dual" / "results.csv": "r",
        foot / "runs" / "detect" / "runs" / "v8_dual" / "weights" / "best.pt": "w" * 50,
        repo / "official_goals.json": "{}",
    }
    for p, c in files.items():
        p.write_text(c)
    return repo, foot


def run(repo, foot, *args):
    r = subprocess.run([sys.executable, str(helpers.ROOT / "tidy_project.py"), "--repo", str(repo), "--football", str(foot), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    return r.stdout


class Tidy(unittest.TestCase):
    def test_a_dry_run_deletes_nothing_and_lists_everything(self):
        repo, foot = build(tempfile.mkdtemp())
        before = sorted(str(p) for p in Path(repo.parent).rglob("*"))
        out = run(repo, foot)
        self.assertEqual(before, sorted(str(p) for p in Path(repo.parent).rglob("*")))
        for name in ("notes.bak", "tests_pack.tar.gz", "mecz9.json", "t_colour_raw.csv", "teams_default.csv", "mecz1_auto", "best.pt"):
            self.assertIn(name, out)
        self.assertIn("old_tool", out)                                  # the script nobody mentions

    def test_apply_deletes_only_the_junk(self):
        repo, foot = build(tempfile.mkdtemp())
        run(repo, foot, "--apply")
        self.assertFalse((repo / "notes.bak").exists())
        self.assertFalse((repo / "__pycache__").exists())
        self.assertFalse((repo / "tests_pack.tar.gz").exists())
        self.assertFalse((repo / "matches" / "mecz9.json").exists())                       # the provisional config
        self.assertFalse((foot / "analysis" / "mecz4_2dual" / "t_colour_raw.csv").exists())
        self.assertFalse((foot / "analysis" / "mecz4_2dual" / "t_kickoff_all_rgb.npz").exists())
        for kept in (repo / "matches" / "mecz4_2dual.json", repo / "matches" / "mecz1_auto.json", foot / "analysis" / "mecz4_2dual" / "teams_default.csv",
                     foot / "analysis" / "mecz4_2dual" / "teams_rgb.npz", foot / "analysis" / "mecz1_auto" / "teams.csv", foot / "videos" / "mecz1_dual.mp4",
                     foot / "runs" / "detect" / "runs" / "stare" / "v9_dual" / "weights" / "best.pt", foot / "runs" / "detect" / "runs" / "v8_dual" / "weights" / "best.pt",
                     repo / "old_tool.py"):
            self.assertTrue(kept.exists(), kept)

    def test_the_other_categories_need_also(self):
        repo, foot = build(tempfile.mkdtemp())
        run(repo, foot, "--apply", "--also", "backups,abandoned,stare-weights")
        self.assertFalse((foot / "analysis" / "mecz4_2dual" / "teams_default.csv").exists())
        self.assertFalse((foot / "analysis" / "mecz1_auto").exists())
        self.assertFalse((foot / "analysis" / "process_mecz1_auto.txt").exists())
        self.assertFalse((foot / "calib" / "mecz1_auto_top.json").exists())
        self.assertFalse((repo / "matches" / "mecz1_auto.json").exists())
        self.assertFalse((foot / "runs" / "detect" / "runs" / "stare" / "v9_dual" / "weights" / "best.pt").exists())
        self.assertTrue((foot / "runs" / "detect" / "runs" / "stare" / "v9_dual" / "results.csv").exists())          # the record stays
        self.assertTrue((foot / "runs" / "detect" / "runs" / "v8_dual" / "weights" / "best.pt").exists())             # the current model stays
        self.assertTrue((foot / "analysis" / "mecz4_2dual" / "teams_rgb.npz").exists())                                # the caches need --also caches
        self.assertTrue((foot / "calib" / "mecz4_2dual_top.json").exists())

    def test_the_evaluation_data_is_copied_into_the_repository(self):
        repo, foot = build(tempfile.mkdtemp())
        run(repo, foot, "--collect-eval", "--apply")
        self.assertTrue((repo / "eval" / "mecz4_2dual" / "team_check.csv").exists())
        self.assertTrue((repo / "eval" / "mecz4_2dual" / "answers_of_the_analysis.csv").exists())
        self.assertTrue((repo / "eval" / "mecz4_2dual" / "goals_auto.csv").exists())
        self.assertTrue((repo / "eval" / "official_goals.json").exists())

    def test_gitignore_gets_only_the_missing_patterns(self):
        repo, foot = build(tempfile.mkdtemp())
        (repo / ".gitignore").write_text("*.mp4\n__pycache__/\n")
        run(repo, foot, "--gitignore", "--apply")
        lines = (repo / ".gitignore").read_text().split("\n")
        self.assertEqual(lines.count("*.mp4"), 1)
        self.assertIn("*.pt", lines)
        self.assertIn("analysis/", lines)


if __name__ == "__main__":
    unittest.main()
