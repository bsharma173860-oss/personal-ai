from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hard clean math/physics dataset")
    parser.add_argument("--train_file", type=Path, default=Path("data/llm/train_hard.jsonl"))
    parser.add_argument("--val_file", type=Path, default=Path("data/llm/val_hard.jsonl"))
    parser.add_argument("--train_size", type=int, default=1200)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def sample_poly(max_degree: int = 7, min_terms: int = 4, max_terms: int = 6) -> dict[int, int]:
    poly: dict[int, int] = {}
    for _ in range(random.randint(min_terms, max_terms)):
        c = random.choice([i for i in range(-12, 13) if i != 0])
        p = random.randint(0, max_degree)
        poly[p] = poly.get(p, 0) + c
    poly = {p: c for p, c in poly.items() if c != 0}
    return poly or {0: 1}


def fmt_poly(poly: dict[int, int]) -> str:
    terms = []
    for p in sorted(poly.keys(), reverse=True):
        c = poly[p]
        sign = "-" if c < 0 else "+"
        a = abs(c)
        if p == 0:
            body = f"{a}"
        elif p == 1:
            body = "x" if a == 1 else f"{a}x"
        else:
            body = f"x^{p}" if a == 1 else f"{a}x^{p}"
        terms.append((sign, body))

    out = []
    for i, (sign, body) in enumerate(terms):
        if i == 0:
            out.append(body if sign == "+" else f"-{body}")
        else:
            out.append(f" {sign} {body}")
    return "".join(out) if out else "0"


def fmt_poly_frac(poly: dict[int, Fraction]) -> str:
    terms = []
    for p in sorted(poly.keys(), reverse=True):
        c = poly[p]
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        a = abs(c)
        coeff = str(a.numerator) if a.denominator == 1 else f"{a.numerator}/{a.denominator}"
        if p == 0:
            body = coeff
        elif p == 1:
            body = "x" if coeff == "1" else f"{coeff}x"
        else:
            body = f"x^{p}" if coeff == "1" else f"{coeff}x^{p}"
        terms.append((sign, body))

    out = []
    for i, (sign, body) in enumerate(terms):
        if i == 0:
            out.append(body if sign == "+" else f"-{body}")
        else:
            out.append(f" {sign} {body}")
    return "".join(out) if out else "0"


def derivative(poly: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for p, c in poly.items():
        if p == 0:
            continue
        out[p - 1] = out.get(p - 1, 0) + c * p
    return {p: c for p, c in out.items() if c != 0} or {0: 0}


def integral(poly: dict[int, int]) -> dict[int, Fraction]:
    out: dict[int, Fraction] = {}
    for p, c in poly.items():
        out[p + 1] = out.get(p + 1, Fraction(0, 1)) + Fraction(c, p + 1)
    return out


def make_derivative_hard() -> tuple[str, str]:
    f = sample_poly(max_degree=8)
    d = derivative(f)
    q = f"Differentiate exactly: f(x) = {fmt_poly(f)}"
    a = f"Using power rule and simplification, f'(x) = {fmt_poly(d)}."
    return q, a


def make_integral_hard() -> tuple[str, str]:
    f = sample_poly(max_degree=6)
    F = integral(f)
    q = f"Compute indefinite integral exactly: integral({fmt_poly(f)}) dx"
    a = f"Term-by-term integration gives {fmt_poly_frac(F)} + C."
    return q, a


def make_limit_hard() -> tuple[str, str]:
    k = random.randint(1, 14)
    q = f"Evaluate limit: lim_(x->0) sin({k}x)/({k}x)"
    a = "By standard limit sin(u)/u -> 1 as u->0, the value is 1."
    return q, a


def make_quadratic_hard() -> tuple[str, str]:
    r1 = random.randint(-15, 15)
    r2 = random.randint(-15, 15)
    b = -(r1 + r2)
    c = r1 * r2
    eq = f"x^2 {'+' if b >= 0 else '-'} {abs(b)}x {'+' if c >= 0 else '-'} {abs(c)} = 0"
    q = f"Solve equation exactly: {eq}"
    a = f"Roots are x = {r1} and x = {r2}."
    return q, a


def make_force_hard() -> tuple[str, str]:
    m = random.choice([0.5, 1, 2, 4, 7.5, 10, 20, 35])
    acc = random.choice([0.25, 0.5, 1.2, 3.4, 9.8, 12.5])
    F = m * acc
    q = f"Physics: mass={m} kg and acceleration={acc} m/s^2. Find force in N."
    a = f"Use F=ma. F={m}*{acc}={F} N."
    return q, a


def make_kinematics_hard() -> tuple[str, str]:
    u = random.choice([0, 1.5, 2, 5, 9])
    acc = random.choice([0.4, 0.8, 1.2, 2.5, 9.8])
    t = random.choice([1, 2, 3, 4, 6, 8])
    v = u + acc * t
    s = u * t + 0.5 * acc * t * t
    q = f"Kinematics: u={u} m/s, a={acc} m/s^2, t={t} s. Find v and s."
    a = f"Using v=u+at and s=ut+0.5at^2: v={v} m/s, s={s} m."
    return q, a


def make_example() -> tuple[str, str]:
    return random.choice(
        [
            make_derivative_hard,
            make_integral_hard,
            make_limit_hard,
            make_quadratic_hard,
            make_force_hard,
            make_kinematics_hard,
        ]
    )()


def write_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n):
            q, a = make_example()
            f.write(json.dumps({"instruction": q, "response": a}, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    write_jsonl(args.train_file, args.train_size)
    write_jsonl(args.val_file, args.val_size)
    print(f"generated_hard_train={args.train_file} count={args.train_size}")
    print(f"generated_hard_val={args.val_file} count={args.val_size}")


if __name__ == "__main__":
    main()
