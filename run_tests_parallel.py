from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run each pytest file under tests/ in parallel."
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Maximum number of test files to run at once.",
    )
    parser.add_argument(
        "--pattern",
        default="test_*.py",
        help="Glob pattern used to discover test files inside tests/.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed through to pytest. Prefix with -- if needed.",
    )
    return parser.parse_args()


def discover_test_files(repo_root: Path, pattern: str) -> list[Path]:
    tests_dir = repo_root / "tests"
    return sorted(path for path in tests_dir.glob(pattern) if path.is_file())


def run_one_test(repo_root: Path, test_file: Path, extra_args: list[str]) -> tuple[Path, int, str]:
    command = [sys.executable, "-m", "pytest", str(test_file), *extra_args]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    if completed.stderr:
        output = f"{output}\n{completed.stderr}".strip()
    return test_file, completed.returncode, output.strip()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    pytest_args = args.pytest_args

    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]

    if importlib.util.find_spec("pytest") is None:
        print(
            "pytest is not installed for this interpreter.\n"
            "Install it first, for example:\n"
            f'  "{sys.executable}" -m pip install pytest',
            file=sys.stderr,
        )
        return 2

    test_files = discover_test_files(repo_root, args.pattern)
    if not test_files:
        print(f'No test files found in "tests/" matching "{args.pattern}".', file=sys.stderr)
        return 2

    workers = max(1, min(args.workers, len(test_files)))
    print(f"Running {len(test_files)} test files with {workers} worker(s)...")

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(run_one_test, repo_root, test_file, pytest_args): test_file
            for test_file in test_files
        }

        for future in concurrent.futures.as_completed(future_map):
            test_file, return_code, output = future.result()
            status = "PASS" if return_code == 0 else "FAIL"
            print(f"\n[{status}] {test_file.relative_to(repo_root)}")
            if output:
                print(output)
            if return_code != 0:
                failures += 1

    if failures:
        print(f"\n{failures} test file(s) failed.")
        return 1

    print("\nAll test files passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
