#!/usr/bin/env python3
"""
Script to run type checking and linting on the codebase using uv.
This script runs pyright for type checking and ruff for linting via uv run, checking only the 'seafront' directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], name: str) -> bool:
    """Run a command and return True if it succeeded."""
    print(f"\n=== Running {name} ===")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"\n❌ {name} found issues", file=sys.stderr)
        return False

    print(f"\n✅ {name} passed")
    return True


def main() -> int:
    """Run all checks and return 0 if all passed, 1 if any failed."""
    parser = argparse.ArgumentParser(
        description="Run type checking and/or linting on the codebase."
    )
    parser.add_argument(
        "--pyright",
        choices=["on", "off"],
        default="on",
        help="Run pyright type checking (default: on)",
    )
    parser.add_argument(
        "--ruff", choices=["on", "off"], default="on", help="Run ruff linting (default: on)"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    seafront_dir = project_root / "seafront"

    success = True

    if args.pyright == "on":
        # Run pyright via uv on the entire directory
        pyright_success = run_command(
            ["uv", "run", "pyright", str(seafront_dir)], "Type checking (pyright)"
        )
        success = success and pyright_success

    if args.ruff == "on":
        # Run ruff via uv on the entire directory
        ruff_success = run_command(
            [
                "uv",
                "run",
                "ruff",
                "check",
                "--config",
                str(project_root / "pyproject.toml"),
                str(seafront_dir),
            ],
            "Linting (ruff)",
        )
        success = success and ruff_success

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
