#!/usr/bin/env python3
"""
Log search utility for seafront logs.

Searches across all log files (compressed and uncompressed) in chronological order.

Examples:
    # Show warnings and above from the last 72 hours
    uv run scripts/search_logs.py --level warning+ --since 72h

    # Show the last log message containing 'failed'
    uv run scripts/search_logs.py --pattern "failed" --last 1

    # Show only errors (not critical) from today
    uv run scripts/search_logs.py --level error --since 24h

    # Show only critical startup issues
    uv run scripts/search_logs.py --level critical --last 10

    # Show warnings and critical (skip errors)
    uv run scripts/search_logs.py --level warning,critical --since 7d

    # Show last 50 entries containing 'camera'
    uv run scripts/search_logs.py --pattern "camera" --last 50

    # Search with regex
    uv run scripts/search_logs.py --pattern "timeout|failed|error" --regex

Level filter formats:
    error           - only ERROR
    error,critical  - ERROR and CRITICAL
    warning+        - WARNING and above (WARNING, ERROR, CRITICAL)
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from seafront.config.basics import GlobalConfigHandler

# Log format: {time} | {level} | {file.path}:{line} | {message}
# Example: 2026-01-14T17:32:21.095919+0100 | DEBUG | /home/.../squid.py:1525 | message
LOG_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{4})"  # timestamp
    r" \| (\w+)"  # level
    r" \| ([^|]+)"  # file:line
    r" \| (.*)$",  # message
    re.MULTILINE,
)

# Quick pattern to extract just the timestamp from a line (for boundary checks)
TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{4})")

LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LEVEL_PRIORITY = {level: i for i, level in enumerate(LEVELS)}
LEVEL_SET = set(LEVELS)


def parse_level_filter(level_str: str) -> set[str]:
    """
    Parse a level filter string into a set of levels to include.

    Formats:
        error           -> {"ERROR"}
        error,critical  -> {"ERROR", "CRITICAL"}
        error+critical  -> {"ERROR", "CRITICAL"}
        warning+        -> {"WARNING", "ERROR", "CRITICAL"} (and above)
        info+warning    -> {"INFO", "WARNING"}

    Level names are case-insensitive.
    """
    level_str = level_str.strip().upper()

    # Check for "and above" syntax (trailing +)
    if level_str.endswith("+") and "," not in level_str and level_str.count("+") == 1:
        base_level = level_str[:-1]
        if base_level not in LEVEL_SET:
            raise ValueError(f"Unknown level: {base_level}. Valid: {', '.join(LEVELS)}")
        min_priority = LEVEL_PRIORITY[base_level]
        return {lvl for lvl, pri in LEVEL_PRIORITY.items() if pri >= min_priority}

    # Split by comma or plus
    parts = re.split(r"[,+]", level_str)
    result: set[str] = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part not in LEVEL_SET:
            raise ValueError(f"Unknown level: {part}. Valid: {', '.join(LEVELS)}")
        result.add(part)

    if not result:
        raise ValueError(f"No valid levels specified. Valid: {', '.join(LEVELS)}")

    return result


@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    location: str
    message: str
    raw_line: str

    def matches_level(self, allowed_levels: set[str]) -> bool:
        """Check if entry's level is in the allowed set."""
        return self.level in allowed_levels

    def matches_time(self, since: datetime | None, until: datetime | None) -> bool:
        """Check if entry is within the time range."""
        if since and self.timestamp < since:
            return False
        if until and self.timestamp > until:
            return False
        return True

    def matches_pattern(self, pattern: str | None, use_regex: bool) -> bool:
        """Check if entry matches the text pattern."""
        if not pattern:
            return True
        search_text = f"{self.location} | {self.message}"
        if use_regex:
            return bool(re.search(pattern, search_text, re.IGNORECASE))
        return pattern.lower() in search_text.lower()

    def format(self, color: bool = True) -> str:
        """Format the entry for display."""
        if not color:
            return self.raw_line

        # ANSI colors for different levels
        level_colors = {
            "DEBUG": "\033[90m",  # gray
            "INFO": "\033[37m",  # white
            "WARNING": "\033[33m",  # yellow
            "ERROR": "\033[31m",  # red
            "CRITICAL": "\033[35m",  # magenta
        }
        reset = "\033[0m"
        color_code = level_colors.get(self.level, "")
        return f"{color_code}{self.raw_line}{reset}"


def parse_duration(s: str) -> timedelta:
    """Parse a duration string like '72h', '7d', '30m' into a timedelta."""
    match = re.match(r"^(\d+)([hdwm])$", s.lower())
    if not match:
        raise ValueError(f"Invalid duration format: {s}. Use e.g. '72h', '7d', '30m'")

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    elif unit == "w":
        return timedelta(weeks=value)
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def parse_timestamp(timestamp_str: str) -> datetime | None:
    """Parse a timestamp string into a datetime."""
    try:
        # Handle the timezone format (+0100 -> +01:00)
        ts = timestamp_str[:-2] + ":" + timestamp_str[-2:]
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def parse_log_entry(entry: str) -> LogEntry | None:
    """Parse a log entry (possibly multiline) into a LogEntry."""
    # Match only the first line for metadata
    first_line = entry.split("\n", 1)[0]
    match = LOG_PATTERN.match(first_line.strip())
    if not match:
        return None

    timestamp_str, level, location, message = match.groups()
    timestamp = parse_timestamp(timestamp_str)
    if timestamp is None:
        return None

    # For multiline entries, include continuation lines in the message
    if "\n" in entry:
        continuation = entry.split("\n", 1)[1]
        message = message + "\n" + continuation

    return LogEntry(
        timestamp=timestamp,
        level=level.upper(),
        location=location.strip(),
        message=message,
        raw_line=entry,
    )


def extract_timestamp(line: str) -> datetime | None:
    """Quickly extract just the timestamp from a log line."""
    match = TIMESTAMP_PATTERN.match(line.strip())
    if not match:
        return None
    return parse_timestamp(match.group(1))


def get_file_time_bounds(path: Path) -> tuple[datetime | None, datetime | None]:
    """
    Get the first and last timestamp in a log file.
    Returns (first_time, last_time). Reads only the necessary lines.
    """
    first_ts: datetime | None = None
    last_ts: datetime | None = None

    if path.suffix == ".gz":
        # For gzip, we need to decompress - read first few and last few lines
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            # Get first timestamp
            for line in f:
                first_ts = extract_timestamp(line)
                if first_ts:
                    break

            # Read rest to get last timestamp (gzip doesn't support seeking from end)
            for line in f:
                ts = extract_timestamp(line)
                if ts:
                    last_ts = ts
    else:
        # For regular files, we can be smarter
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            # Get first timestamp
            for line in f:
                first_ts = extract_timestamp(line)
                if first_ts:
                    break

        # Read from end for last timestamp
        with open(path, "rb") as f:
            # Seek to end and read last chunk
            f.seek(0, 2)  # End of file
            size = f.tell()
            chunk_size = min(8192, size)
            f.seek(max(0, size - chunk_size))
            chunk = f.read().decode("utf-8", errors="replace")
            lines = chunk.split("\n")
            for line in reversed(lines):
                last_ts = extract_timestamp(line)
                if last_ts:
                    break

    return first_ts, last_ts or first_ts


def _is_log_start(line: str) -> bool:
    """Check if a line starts a new log entry (has a timestamp)."""
    return TIMESTAMP_PATTERN.match(line) is not None


def read_log_entries(path: Path) -> Iterator[str]:
    """
    Read complete log entries from a file, handling multiline messages.
    Yields each entry as a single string (with newlines for multiline messages).
    """
    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        opener = lambda: open(path, "r", encoding="utf-8", errors="replace")

    with opener() as f:
        current_entry: list[str] = []

        for line in f:
            line = line.rstrip("\n")
            if _is_log_start(line):
                # New entry - yield previous if exists
                if current_entry:
                    yield "\n".join(current_entry)
                current_entry = [line]
            elif current_entry:
                # Continuation of current entry
                current_entry.append(line)

        # Yield final entry
        if current_entry:
            yield "\n".join(current_entry)


def read_log_entries_reversed(path: Path) -> Iterator[str]:
    """
    Read complete log entries in reverse order (newest first).
    Handles multiline messages by collecting continuation lines.
    """
    if path.suffix == ".gz":
        # For gzip, decompress fully then process in reverse
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]
    else:
        # For regular files, read all lines (could optimize with chunked reading)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

    # Process lines in reverse, collecting continuations
    continuation_lines: list[str] = []

    for line in reversed(lines):
        if not line:
            continue

        if _is_log_start(line):
            # Found start of entry - yield it with any continuations
            if continuation_lines:
                # Continuations were collected in reverse, so reverse them back
                full_entry = line + "\n" + "\n".join(reversed(continuation_lines))
                continuation_lines = []
            else:
                full_entry = line
            yield full_entry
        else:
            # Continuation line - buffer it
            continuation_lines.append(line)


def get_log_files(log_dir: Path) -> list[Path]:
    """Get all log files in chronological order (oldest first)."""
    files: list[Path] = []

    # Get compressed files (already rotated, so older)
    gz_files = sorted(log_dir.glob("log.*.txt.gz"))
    files.extend(gz_files)

    # Current log file (newest)
    current = log_dir / "log.txt"
    if current.exists():
        files.append(current)

    return files


def search_logs(
    log_dir: Path,
    pattern: str | None = None,
    use_regex: bool = False,
    levels: set[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    last_n: int | None = None,
    first_n: int | None = None,
) -> list[LogEntry]:
    """
    Search all log files and return matching entries in chronological order.

    Uses lazy evaluation:
    - For --last N: reads files in reverse, stops early when enough matches found
    - For --since: skips entire files whose latest entry is before the cutoff
    """
    log_files = get_log_files(log_dir)
    levels = levels or LEVEL_SET  # Default to all levels

    if last_n is not None:
        # Reverse search: start from newest file, read backwards
        return _search_last_n(
            log_files, pattern, use_regex, levels, since, until, last_n
        )
    else:
        # Forward search with early termination
        return _search_forward(
            log_files, pattern, use_regex, levels, since, until, first_n
        )


def _search_last_n(
    log_files: list[Path],
    pattern: str | None,
    use_regex: bool,
    levels: set[str],
    since: datetime | None,
    until: datetime | None,
    last_n: int,
) -> list[LogEntry]:
    """Search for the last N matching entries, reading files in reverse."""
    # Use a deque to collect results efficiently
    results: deque[LogEntry] = deque(maxlen=last_n)

    # Process files from newest to oldest
    for log_file in reversed(log_files):
        # Quick check: can we skip this file entirely?
        if since is not None:
            first_ts, last_ts = get_file_time_bounds(log_file)
            if last_ts and last_ts < since:
                # This file's latest entry is before our cutoff, skip it
                continue

        # Read file in reverse
        for entry_text in read_log_entries_reversed(log_file):
            entry = parse_log_entry(entry_text)
            if entry is None:
                continue

            # Check time bounds - if we've gone before 'since', we can stop this file
            if since and entry.timestamp < since:
                break

            # Apply filters
            if not entry.matches_level(levels):
                continue
            if not entry.matches_time(since, until):
                continue
            if not entry.matches_pattern(pattern, use_regex):
                continue

            # Add to front (since we're reading in reverse)
            results.appendleft(entry)

            # If we have enough results, stop processing this file
            if len(results) >= last_n:
                break

        # If we have enough results, we're done with all files
        if len(results) >= last_n:
            break

    return list(results)


def _search_forward(
    log_files: list[Path],
    pattern: str | None,
    use_regex: bool,
    levels: set[str],
    since: datetime | None,
    until: datetime | None,
    first_n: int | None,
) -> list[LogEntry]:
    """Search forward through files, with optional early termination."""
    entries: list[LogEntry] = []

    for log_file in log_files:
        # Quick check: can we skip this file entirely based on time bounds?
        if since is not None or until is not None:
            first_ts, last_ts = get_file_time_bounds(log_file)

            # Skip if entire file is before 'since'
            if since and last_ts and last_ts < since:
                continue

            # Skip if entire file is after 'until'
            if until and first_ts and first_ts > until:
                continue

        for entry_text in read_log_entries(log_file):
            entry = parse_log_entry(entry_text)
            if entry is None:
                continue

            # Early exit if we've passed 'until'
            if until and entry.timestamp > until:
                break

            # Skip entries before 'since'
            if since and entry.timestamp < since:
                continue

            # Apply filters
            if not entry.matches_level(levels):
                continue
            if not entry.matches_pattern(pattern, use_regex):
                continue

            entries.append(entry)

            # Check if we have enough for first_n
            if first_n is not None and len(entries) >= first_n:
                return entries

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search seafront logs across all files in chronological order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-p", "--pattern", help="Text pattern to search for (case-insensitive)"
    )
    parser.add_argument(
        "-r", "--regex", action="store_true", help="Treat pattern as a regex"
    )
    parser.add_argument(
        "-l",
        "--level",
        help="Log levels to show. Formats: 'error' (single), 'error,critical' (multiple), 'warning+' (and above)",
    )
    parser.add_argument(
        "-s",
        "--since",
        help="Show entries since this time ago (e.g., '72h', '7d', '30m')",
    )
    parser.add_argument(
        "--last",
        type=int,
        metavar="N",
        help="Show only the last N matching entries",
    )
    parser.add_argument(
        "--first",
        type=int,
        metavar="N",
        help="Show only the first N matching entries",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    parser.add_argument(
        "--count", action="store_true", help="Only print the count of matching entries"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Log directory (default: ~/seafront/logs)",
    )

    args = parser.parse_args()

    # Show help if no filtering arguments provided
    has_filter = (
        args.pattern is not None
        or args.level is not None
        or args.since is not None
        or args.last is not None
        or args.first is not None
    )
    if not has_filter:
        parser.print_help()
        sys.exit(0)

    # Determine log directory
    log_dir = args.log_dir or (GlobalConfigHandler.home() / "logs")
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse level filter
    levels = None
    if args.level:
        try:
            levels = parse_level_filter(args.level)
        except ValueError as e:
            print(f"Invalid level filter: {e}", file=sys.stderr)
            sys.exit(1)

    # Parse time filter
    since = None
    if args.since:
        delta = parse_duration(args.since)
        since = datetime.now(timezone.utc).astimezone() - delta

    # Search
    entries = search_logs(
        log_dir=log_dir,
        pattern=args.pattern,
        use_regex=args.regex,
        levels=levels,
        since=since,
        last_n=args.last,
        first_n=args.first,
    )

    # Output
    if args.count:
        print(len(entries))
    else:
        use_color = not args.no_color and sys.stdout.isatty()
        for entry in entries:
            print(entry.format(color=use_color))

        if not entries:
            print("No matching log entries found.", file=sys.stderr)


if __name__ == "__main__":
    main()
