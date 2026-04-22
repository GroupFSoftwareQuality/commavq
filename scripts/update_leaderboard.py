from __future__ import annotations

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen


DEFAULT_URL = "https://comma.ai/leaderboard"
DEFAULT_TARGET_ID = "commavq_compression_challenge_table"
TABLE_START_MARKER = "<!-- TABLE-START -->"
TABLE_END_MARKER = "<!-- TABLE-END -->"


class LeaderboardTableExtractor(HTMLParser):
    def __init__(self, target_div_id: str) -> None:
        super().__init__(convert_charrefs=False)
        self.target_div_id = target_div_id
        self.div_depth = 0
        self.target_div_depth: int | None = None
        self.table_depth = 0
        self.table_chunks: list[str] = []
        self.extracted_table: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        raw_tag = self.get_starttag_text()
        attr_map = dict(attrs)

        if tag == "div" and self.target_div_depth is None and attr_map.get("id") == self.target_div_id:
            self.target_div_depth = self.div_depth

        if self._inside_target_div() and self.extracted_table is None:
            if tag == "table" and self.table_depth == 0:
                self.table_depth = 1
                self.table_chunks.append(raw_tag)
                if tag == "div":
                    self.div_depth += 1
                return
            if self.table_depth > 0:
                if tag == "table":
                    self.table_depth += 1
                self.table_chunks.append(raw_tag)

        if tag == "div":
            self.div_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self.table_depth > 0:
            self.table_chunks.append(f"</{tag}>")
            if tag == "table":
                self.table_depth -= 1
                if self.table_depth == 0:
                    self.extracted_table = "".join(self.table_chunks).strip()
                    self.table_chunks = []

        if tag == "div":
            self.div_depth -= 1
            if self.target_div_depth is not None and self.div_depth == self.target_div_depth:
                self.target_div_depth = None

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.table_depth > 0:
            self.table_chunks.append(self.get_starttag_text())

    def handle_data(self, data: str) -> None:
        if self.table_depth > 0:
            self.table_chunks.append(data)

    def handle_comment(self, data: str) -> None:
        if self.table_depth > 0:
            self.table_chunks.append(f"<!--{data}-->")

    def handle_entityref(self, name: str) -> None:
        if self.table_depth > 0:
            self.table_chunks.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self.table_depth > 0:
            self.table_chunks.append(f"&#{name};")

    def handle_decl(self, decl: str) -> None:
        if self.table_depth > 0:
            self.table_chunks.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        if self.table_depth > 0:
            self.table_chunks.append(f"<?{data}>")

    def _inside_target_div(self) -> bool:
        return self.target_div_depth is not None and self.div_depth > self.target_div_depth


def extract_leaderboard_table(page_html: str, target_div_id: str = DEFAULT_TARGET_ID) -> str:
    parser = LeaderboardTableExtractor(target_div_id=target_div_id)
    parser.feed(page_html)
    parser.close()
    if parser.extracted_table is None:
        raise ValueError(f"Could not find a <table> inside div#{target_div_id}")
    return parser.extracted_table


def replace_marked_section(
    text: str,
    replacement_html: str,
    start_marker: str = TABLE_START_MARKER,
    end_marker: str = TABLE_END_MARKER,
) -> str:
    start_index = text.find(start_marker)
    if start_index == -1:
        raise ValueError(f"Missing start marker: {start_marker}")

    end_index = text.find(end_marker, start_index + len(start_marker))
    if end_index == -1:
        raise ValueError(f"Missing end marker: {end_marker}")

    newline = "\r\n" if "\r\n" in text else "\n"
    before = text[: start_index + len(start_marker)]
    after = text[end_index:]
    replacement = replacement_html.strip()
    return f"{before}{newline}{replacement}{newline}{after}"


def update_file(readme_path: Path, replacement_html: str) -> bool:
    original = readme_path.read_text(encoding="utf-8")
    updated = replace_marked_section(original, replacement_html)
    if updated == original:
        return False
    readme_path.write_text(updated, encoding="utf-8")
    return True


def fetch_page_html(url: str) -> str:
    with urlopen(url) as response:  # noqa: S310 - fixed URL or explicit CLI input
        return response.read().decode("utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the commavq leaderboard table and splice it into README files."
    )
    parser.add_argument(
        "readme_paths",
        nargs="*",
        default=["README.md", "compression/README.md"],
        help="README files to update.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Page URL to fetch when --html-file is not provided.",
    )
    parser.add_argument(
        "--html-file",
        type=Path,
        help="Use local HTML instead of fetching a remote page.",
    )
    parser.add_argument(
        "--target-div-id",
        default=DEFAULT_TARGET_ID,
        help="The div id that contains the leaderboard table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    page_html = (
        args.html_file.read_text(encoding="utf-8")
        if args.html_file is not None
        else fetch_page_html(args.url)
    )
    table_html = extract_leaderboard_table(page_html, target_div_id=args.target_div_id)

    changed_paths: list[Path] = []
    for readme_arg in args.readme_paths:
        readme_path = Path(readme_arg)
        if update_file(readme_path, table_html):
            changed_paths.append(readme_path)

    if changed_paths:
        print("Updated:", ", ".join(str(path) for path in changed_paths))
    else:
        print("No README changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
