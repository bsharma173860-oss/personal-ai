from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .math_physics import derivative, integral, kinematics, limit, newton_force, solve_equation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Math/Physics + optional local LLM assistant")
    parser.add_argument("--task", type=str, required=True, choices=["derivative", "integral", "limit", "solve", "physics_kinematics", "physics_force", "qa"])
    parser.add_argument("--expr", type=str, default=None)
    parser.add_argument("--equation", type=str, default=None)
    parser.add_argument("--var", type=str, default="x")
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--u", type=float, default=None)
    parser.add_argument("--a", type=float, default=None)
    parser.add_argument("--t", type=float, default=None)
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--acceleration", type=float, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--creator_profile", type=Path, default=Path("creator_profile.json"))
    parser.add_argument("--llm_model", type=str, default=None, help="Local model id/path for transformers pipeline")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    return parser.parse_args()


def load_creator_profile(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_local_llm(question: str, model_name: str, profile: dict, max_new_tokens: int) -> str:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install LLM extras first: python3 -m pip install -r requirements-llm.txt"
        ) from exc

    creator_name = profile.get("creator_name", "Creator")
    domains = ", ".join(profile.get("priority_domains", ["math", "physics"]))

    system = (
        f"You are a precise assistant for {creator_name}. "
        f"Focus on {domains}. Show formulas and concise steps."
    )
    prompt = (
        f"System: {system}\n"
        "Output format:\n"
        "Reasoning: <short steps>\n"
        "Final: <final equation/number only>\n"
        f"User: {question}\n"
        "Assistant:\n"
    )

    generator = pipeline("text-generation", model=model_name)
    out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    text = out[0]["generated_text"].split("Assistant:", maxsplit=1)[-1].strip()
    m_final = re.search(r"final:\s*(.+)", text, flags=re.IGNORECASE)
    if m_final:
        text = m_final.group(1).strip()
        text = re.split(r"(?:Question:|Answer:|Reasoning:)", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        return first_line or text
    text = re.split(r"(?:Question:)", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    text = re.sub(r"^(?:Answer:|Reasoning:)\s*", "", text, flags=re.IGNORECASE).strip()
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first_line or text or out[0]["generated_text"].split("Assistant:", maxsplit=1)[-1].strip()


def main() -> None:
    args = parse_args()
    profile = load_creator_profile(args.creator_profile)

    if profile.get("creator_name"):
        print(f"Hello, {profile['creator_name']}.")

    if args.task == "derivative":
        if not args.expr:
            raise ValueError("--expr is required for derivative")
        res = derivative(args.expr, args.var)
    elif args.task == "integral":
        if not args.expr:
            raise ValueError("--expr is required for integral")
        res = integral(args.expr, args.var)
    elif args.task == "limit":
        if not args.expr or args.point is None:
            raise ValueError("--expr and --point are required for limit")
        res = limit(args.expr, args.point, args.var)
    elif args.task == "solve":
        if not args.equation:
            raise ValueError("--equation is required for solve")
        res = solve_equation(args.equation, args.var)
    elif args.task == "physics_kinematics":
        if args.u is None or args.a is None or args.t is None:
            raise ValueError("--u --a --t are required for physics_kinematics")
        res = kinematics(args.u, args.a, args.t)
    elif args.task == "physics_force":
        if args.mass is None or args.acceleration is None:
            raise ValueError("--mass --acceleration are required for physics_force")
        res = newton_force(args.mass, args.acceleration)
    else:
        if not args.question:
            raise ValueError("--question is required for qa")
        if not args.llm_model:
            raise ValueError("--llm_model is required for qa")
        answer = run_local_llm(args.question, args.llm_model, profile, args.max_new_tokens)
        print("task=qa")
        print(f"answer={answer}")
        return

    print(f"task={res.task}")
    print(f"input={res.input_text}")
    print(f"result={res.result}")
    print(f"steps={res.steps}")


if __name__ == "__main__":
    main()
