#!/usr/bin/env python3
"""
Inspect labels in a LeitmotifDetection JSONL file.

Prints sorted unique labels and per-label counts.

Usage
-----
    python scripts/inspect_leitmotif_labels.py path/to/leitmotif.jsonl
    python scripts/inspect_leitmotif_labels.py data/leitmotif/train.jsonl data/leitmotif/val.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def inspect(paths: list) -> None:
    counter: Counter = Counter()
    total_lines = 0

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

        with open(path, "r") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"ERROR: JSON parse error in {path}:{lineno}: {exc}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                label = obj.get("label")
                if label is None:
                    print(
                        f"WARNING: missing 'label' key in {path}:{lineno}",
                        file=sys.stderr,
                    )
                    continue

                counter[label] += 1
                total_lines += 1

    print(f"Total entries : {total_lines}")
    print(f"Unique labels : {len(counter)}")
    print()
    print(f"{'Label':<40}  {'Count':>7}")
    print("-" * 50)
    for label in sorted(counter):
        print(f"{label:<40}  {counter[label]:>7}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print sorted unique labels and per-label counts from a JSONL file."
    )
    parser.add_argument(
        "jsonl",
        nargs="+",
        help="One or more JSONL metadata files to inspect.",
    )
    args = parser.parse_args()
    inspect(args.jsonl)


if __name__ == "__main__":
    main()
