from __future__ import annotations

import subprocess
import sys
import unittest
import uuid
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "update_leaderboard.py"
TMP_ROOT = REPO_ROOT / "tests"

FIXTURE_HTML = """
<html>
  <body>
    <div id="other_section">
      <table><tr><td>ignore</td></tr></table>
    </div>
    <div id="commavq_compression_challenge_table">
      <p>Intro</p>
      <table class="ranked">
        <thead><tr><th>Rank</th><th>Name</th></tr></thead>
        <tbody><tr><td>1</td><td>Alice</td></tr></tbody>
      </table>
    </div>
  </body>
</html>
""".strip()


def make_readme(path: Path, placeholder: str = "old") -> None:
    path.write_text(
        "\n".join(
            [
                "# Demo",
                "",
                "before",
                "<!-- TABLE-START -->",
                f"<table><tr><td>{placeholder}</td></tr></table>",
                "<!-- TABLE-END -->",
                "after",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_updater(tmp_root: Path, *readme_paths: Path) -> subprocess.CompletedProcess[str]:
    fixture_path = tmp_root / "leaderboard.html"
    fixture_path.write_text(FIXTURE_HTML, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--html-file",
            str(fixture_path),
            *[str(path) for path in readme_paths],
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def make_workspace() -> Path:
    workspace = TMP_ROOT / f"leaderboard_test_{uuid.uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


class UpdateLeaderboardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()

    def tearDown(self):
        if self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)

    def test_updater_replaces_table_in_both_readmes(self):
        root_readme = self.workspace / "README.md"
        compression_dir = self.workspace / "compression"
        compression_dir.mkdir()
        compression_readme = compression_dir / "README.md"
        make_readme(root_readme)
        make_readme(compression_readme)

        result = run_updater(self.workspace, root_readme, compression_readme)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Updated:", result.stdout)

        root_text = root_readme.read_text(encoding="utf-8")
        compression_text = compression_readme.read_text(encoding="utf-8")
        expected_table = '<table class="ranked">\n        <thead><tr><th>Rank</th><th>Name</th></tr></thead>\n        <tbody><tr><td>1</td><td>Alice</td></tr></tbody>\n      </table>'

        self.assertIn(expected_table, root_text)
        self.assertIn(expected_table, compression_text)
        self.assertNotIn("<td>old</td>", root_text)
        self.assertNotIn("<td>old</td>", compression_text)
        self.assertIn("before", root_text)
        self.assertIn("after", root_text)

    def test_updater_reports_no_changes_when_table_is_already_current(self):
        root_readme = self.workspace / "README.md"
        make_readme(root_readme, placeholder="stale")

        first_run = run_updater(self.workspace, root_readme)
        self.assertEqual(first_run.returncode, 0, first_run.stderr)

        second_run = run_updater(self.workspace, root_readme)
        self.assertEqual(second_run.returncode, 0, second_run.stderr)
        self.assertIn("No README changes needed.", second_run.stdout)

    def test_updater_fails_when_markers_are_missing(self):
        broken_readme = self.workspace / "README.md"
        broken_readme.write_text("# Missing markers\n", encoding="utf-8")

        result = run_updater(self.workspace, broken_readme)

        self.assertNotEqual(result.returncode, 0)
        combined_output = f"{result.stdout}\n{result.stderr}"
        self.assertIn("Missing start marker", combined_output)


if __name__ == "__main__":
    unittest.main()
