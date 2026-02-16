from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic math/physics JSONL for personal LLM")
    parser.add_argument("--train_file", type=Path, default=Path("data/llm/train.jsonl"))
    parser.add_argument("--val_file", type=Path, default=Path("data/llm/val.jsonl"))
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--val_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def poly_terms() -> list[tuple[int, int]]:
    n_terms = random.randint(2, 4)
    terms: list[tuple[int, int]] = []
    for _ in range(n_terms):
        coef = random.choice([i for i in range(-9, 10) if i != 0])
        power = random.randint(0, 5)
        terms.append((coef, power))
    return terms


def format_poly(terms: list[tuple[int, int]]) -> str:
    out: list[str] = []
    for idx, (c, p) in enumerate(terms):
        sign = "-" if c < 0 else "+"
        a = abs(c)
        if p == 0:
            base = f"{a}"
        elif p == 1:
            base = "x" if a == 1 else f"{a}x"
        else:
            base = f"x^{p}" if a == 1 else f"{a}x^{p}"
        if idx == 0:
            out.append(base if c > 0 else f"-{base}")
        else:
            out.append(f" {sign} {base}")
    return "".join(out)


def derivative_terms(terms: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for c, p in terms:
        if p == 0:
            continue
        out.append((c * p, p - 1))
    return out or [(0, 0)]


def integral_terms(terms: list[tuple[int, int]]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for c, p in terms:
        new_p = p + 1
        num = c
        den = new_p
        if num % den == 0:
            out.append((str(num // den), new_p))
        else:
            out.append((f"{num}/{den}", new_p))
    return out


def format_poly_fraction(terms: list[tuple[str, int]]) -> str:
    out: list[str] = []
    for idx, (c_str, p) in enumerate(terms):
        c = c_str
        neg = c.startswith("-")
        sign = "-" if neg else "+"
        coeff = c[1:] if neg else c
        if p == 0:
            base = coeff
        elif p == 1:
            base = "x" if coeff == "1" else f"{coeff}x"
        else:
            base = f"x^{p}" if coeff == "1" else f"{coeff}x^{p}"

        if idx == 0:
            out.append(base if not neg else f"-{base}")
        else:
            out.append(f" {sign} {base}")
    return "".join(out)


def make_derivative_example() -> tuple[str, str]:
    terms = poly_terms()
    expr = format_poly(terms)
    d = format_poly(derivative_terms(terms))
    q = f"Differentiate f(x) = {expr}."
    a = f"Using the power rule term-by-term, f'(x) = {d}."
    return q, a


def make_integral_example() -> tuple[str, str]:
    terms = poly_terms()
    expr = format_poly(terms)
    integ = format_poly_fraction(integral_terms(terms))
    q = f"Integrate f(x) = {expr} with respect to x."
    a = f"Applying the reverse power rule term-by-term, integral = {integ} + C."
    return q, a


def make_limit_example() -> tuple[str, str]:
    k = random.randint(1, 12)
    q = f"Find limit of sin({k}x)/({k}x) as x approaches 0."
    a = "Use standard limit sin(u)/u -> 1 as u -> 0. Therefore the limit is 1."
    return q, a


def make_equation_example() -> tuple[str, str]:
    r1 = random.randint(-12, 12)
    r2 = random.randint(-12, 12)
    b = -(r1 + r2)
    c = r1 * r2
    if c >= 0:
        eq = f"x^2 {'+' if b >= 0 else '-'} {abs(b)}x + {c} = 0"
    else:
        eq = f"x^2 {'+' if b >= 0 else '-'} {abs(b)}x - {abs(c)} = 0"
    q = f"Solve equation {eq}."
    a = f"Factoring gives roots x = {r1} and x = {r2}."
    return q, a


def make_force_example() -> tuple[str, str]:
    m = random.choice([1, 2, 5, 10, 15, 20, 30, 50])
    acc = random.choice([0.5, 1.0, 2.0, 3.5, 4.0, 9.8, 12.0])
    f = m * acc
    q = f"A mass of {m} kg accelerates at {acc} m/s^2. Find force."
    a = f"Use F = ma. F = {m} * {acc} = {f} N."
    return q, a


def make_kinematics_example() -> tuple[str, str]:
    u = random.choice([0, 2, 5, 10, 15])
    acc = random.choice([0.5, 1.0, 2.0, 3.0, 9.8])
    t = random.choice([1, 2, 3, 4, 5, 8])
    v = u + acc * t
    s = u * t + 0.5 * acc * t * t
    q = f"Given u={u} m/s, a={acc} m/s^2, t={t} s. Find final velocity and displacement."
    a = f"Use v=u+at and s=ut+0.5at^2. v={v} m/s and s={s} m."
    return q, a


def make_example() -> tuple[str, str]:
    builders = [
        make_derivative_example,
        make_integral_example,
        make_limit_example,
        make_equation_example,
        make_force_example,
        make_kinematics_example,
    ]
    return random.choice(builders)()


def write_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n):
            instruction, response = make_example()
            obj = {"instruction": instruction, "response": response}
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    write_jsonl(args.train_file, args.train_size)
    write_jsonl(args.val_file, args.val_size)
    print(f"generated_train={args.train_file} count={args.train_size}")
    print(f"generated_val={args.val_file} count={args.val_size}")


if __name__ == "__main__":
    main()
