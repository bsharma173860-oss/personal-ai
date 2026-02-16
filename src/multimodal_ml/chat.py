from __future__ import annotations

import argparse
import re
from pathlib import Path

from .ask import load_creator_profile, run_local_llm
from .math_physics import derivative, integral, kinematics, limit, newton_force, solve_equation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single chat-style router for calculus/physics/LLM")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--creator_profile", type=Path, default=Path("creator_profile.json"))
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    return parser.parse_args()


def _clean_expr(text: str) -> str:
    text = text.strip()
    text = text.replace("^", "**")
    return text


def _extract_after_keywords(prompt: str, keys: list[str]) -> str | None:
    low = prompt.lower()
    for key in keys:
        idx = low.find(key)
        if idx != -1:
            return prompt[idx + len(key) :].strip(" :")
    return None


def route_symbolic(prompt: str):
    low = prompt.lower().strip()

    if any(k in low for k in ["derivative", "differentiate", "d/dx"]):
        expr = _extract_after_keywords(prompt, ["derivative of", "differentiate", "d/dx"]) or prompt
        expr = _clean_expr(expr)
        return derivative(expr, "x")

    if any(k in low for k in ["integral", "integrate"]):
        expr = _extract_after_keywords(prompt, ["integral of", "integrate"]) or prompt
        expr = _clean_expr(expr)
        return integral(expr, "x")

    if "limit" in low:
        match = re.search(r"limit\s+(.+)\s+as\s+x\s*->\s*([^\s]+)", low)
        if match:
            expr = _clean_expr(match.group(1))
            point = match.group(2)
            return limit(expr, point, "x")

    if any(k in low for k in ["solve", "equation"]):
        eq = _extract_after_keywords(prompt, ["solve", "equation"]) or prompt
        eq = _clean_expr(eq)
        return solve_equation(eq, "x")

    if "force" in low and "mass" in low and ("acceleration" in low or "a=" in low):
        m = re.search(r"mass\s*=?\s*(-?\d+(?:\.\d+)?)", low)
        a = re.search(r"acceleration\s*=?\s*(-?\d+(?:\.\d+)?)", low)
        if not a:
            a = re.search(r"a\s*=\s*(-?\d+(?:\.\d+)?)", low)
        if m and a:
            return newton_force(float(m.group(1)), float(a.group(1)))

    if "kinematics" in low or ("u=" in low and "a=" in low and "t=" in low):
        u = re.search(r"u\s*=\s*(-?\d+(?:\.\d+)?)", low)
        a = re.search(r"a\s*=\s*(-?\d+(?:\.\d+)?)", low)
        t = re.search(r"t\s*=\s*(-?\d+(?:\.\d+)?)", low)
        if u and a and t:
            return kinematics(float(u.group(1)), float(a.group(1)), float(t.group(1)))

    return None


def main() -> None:
    args = parse_args()
    profile = load_creator_profile(args.creator_profile)

    if profile.get("creator_name"):
        print(f"Hello, {profile['creator_name']}.")

    routed = route_symbolic(args.prompt)
    if routed is not None:
        print(f"mode=symbolic task={routed.task}")
        print(f"input={routed.input_text}")
        print(f"result={routed.result}")
        print(f"steps={routed.steps}")
        return

    if not args.llm_model:
        raise ValueError(
            "Prompt was not matched to symbolic math/physics. Provide --llm_model to use local LLM fallback."
        )

    answer = run_local_llm(args.prompt, args.llm_model, profile, args.max_new_tokens)
    print("mode=llm task=qa")
    print(f"answer={answer}")


if __name__ == "__main__":
    main()
