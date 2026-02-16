from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and dedupe JSONL instruction/response datasets")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seen = set()
    rows = []

    for p in args.inputs:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                key = (item.get("instruction", "").strip(), item.get("response", "").strip())
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"instruction": key[0], "response": key[1]})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    print(f"merged_rows={len(rows)}")
    print(f"output_file={args.output}")


if __name__ == "__main__":
    main()
