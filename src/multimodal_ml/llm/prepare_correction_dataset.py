from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare correction-focused dataset for adapter continuation")
    parser.add_argument("--corrections_file", type=Path, required=True)
    parser.add_argument("--base_file", type=Path, default=None)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--repeat_corrections", type=int, default=6)
    parser.add_argument("--base_sample", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for r in rows:
        key = (r.get("instruction", "").strip(), r.get("response", "").strip())
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        out.append({"instruction": key[0], "response": key[1]})
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    corrections = dedupe(load_jsonl(args.corrections_file))
    if not corrections:
        raise ValueError(f"No usable corrections found in {args.corrections_file}")

    rows = []
    for _ in range(max(args.repeat_corrections, 1)):
        rows.extend(corrections)

    base_used = 0
    if args.base_file is not None and args.base_file.exists():
        base = dedupe(load_jsonl(args.base_file))
        if base:
            random.shuffle(base)
            take = min(args.base_sample, len(base))
            rows.extend(base[:take])
            base_used = take

    random.shuffle(rows)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    print(f"corrections_unique={len(corrections)}")
    print(f"repeat_corrections={args.repeat_corrections}")
    print(f"base_used={base_used}")
    print(f"output_rows={len(rows)}")
    print(f"output_file={args.output_file}")


if __name__ == "__main__":
    main()
