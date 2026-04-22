from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.update_leaderboard import extract_leaderboard_table, replace_marked_section


class UpdateLeaderboardUnitTests(unittest.TestCase):
    def test_extract_leaderboard_table_picks_table_in_target_div(self):
        page_html = """
        <html>
          <body>
            <table id="wrong-table"><tr><td>ignore me</td></tr></table>
            <div id="commavq_compression_challenge_table">
              <h2>Leaderboard</h2>
              <table class="ranked">
                <tr><th>Name</th></tr>
                <tr><td>Alice</td></tr>
              </table>
            </div>
          </body>
        </html>
        """

        table_html = extract_leaderboard_table(page_html)

        self.assertTrue(table_html.startswith('<table class="ranked">'))
        self.assertIn("<td>Alice</td>", table_html)
        self.assertNotIn("ignore me", table_html)

    def test_extract_leaderboard_table_raises_when_target_table_missing(self):
        page_html = """
        <html>
          <body>
            <div id="commavq_compression_challenge_table">
              <p>No table here.</p>
            </div>
          </body>
        </html>
        """

        with self.assertRaisesRegex(ValueError, "Could not find a <table>"):
            extract_leaderboard_table(page_html)

    def test_replace_marked_section_only_replaces_between_markers(self):
        original = """# Title

before
<!-- TABLE-START -->
<table><tr><td>old</td></tr></table>
<!-- TABLE-END -->
after
"""

        updated = replace_marked_section(
            original,
            '<table class="ranked"><tr><td>new</td></tr></table>',
        )

        self.assertTrue(updated.startswith("# Title"))
        self.assertIn("before", updated)
        self.assertIn("after", updated)
        self.assertNotIn("<td>old</td>", updated)
        self.assertIn('<table class="ranked"><tr><td>new</td></tr></table>', updated)
        self.assertEqual(updated.count("<!-- TABLE-START -->"), 1)
        self.assertEqual(updated.count("<!-- TABLE-END -->"), 1)

    def test_replace_marked_section_raises_on_missing_start_marker(self):
        with self.assertRaisesRegex(ValueError, "Missing start marker"):
            replace_marked_section("no markers here", "<table></table>")

    def test_replace_marked_section_raises_on_missing_end_marker(self):
        original = "<!-- TABLE-START -->\n<table></table>\n"
        with self.assertRaisesRegex(ValueError, "Missing end marker"):
            replace_marked_section(original, "<table></table>")


if __name__ == "__main__":
    unittest.main()
