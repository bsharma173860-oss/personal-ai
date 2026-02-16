from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


MATH_PHYSICS_PROMPTS = [
    "Solve and explain: If f(x)=x^4-3x^2+2, find critical points and classify them.",
    "Derive and explain: Use Taylor expansion up to second order for e^x near x=0.",
    "Physics reasoning: Explain why Lagrangian mechanics can simplify constrained systems.",
    "Electromagnetism: Explain Maxwell equations in intuition + formula form.",
    "Quantum: Explain why wavefunction normalization is required and how to apply it.",
    "Relativity: Explain time dilation with one numerical example.",
    "Thermodynamics: Distinguish state function vs path function with examples.",
    "Mechanics: Compare conservation of energy and momentum and when each applies.",
]

CODING_PROMPTS = [
    "Coding interview: Design an O(n) algorithm to find the first non-repeating character in a string.",
    "Coding interview: Explain two-pointer strategy with an example problem and solution sketch.",
    "Coding interview: Implement LRU cache design and explain time complexity.",
    "Coding interview: Compare BFS and DFS use-cases with edge cases.",
    "Coding interview: Explain dynamic programming state design using coin change.",
    "Coding interview: Explain binary search invariants and common bugs.",
    "Coding interview: Write approach for top-k frequent elements and compare heap vs bucket.",
    "Coding interview: Explain how to detect cycle in linked list and why it works.",
]

SYSTEM_DESIGN_PROMPTS = [
    "System design interview: Design URL shortener with scaling and datastore choices.",
    "System design interview: Design rate limiter for public API.",
    "System design interview: Design chat service with ordering and delivery guarantees.",
    "System design interview: Design notification service for high fan-out alerts.",
    "System design interview: Design metrics pipeline for 1M events/sec.",
    "System design interview: Design distributed cache and invalidation strategy.",
]

BEHAVIORAL_PROMPTS = [
    "Interview behavioral: Tell me about a time you handled conflict in a team. Give STAR structure.",
    "Interview behavioral: Describe a failure and what you changed after it. Give STAR structure.",
    "Interview behavioral: Explain how you prioritize under tight deadlines. Give framework.",
    "Interview behavioral: How do you communicate technical trade-offs to non-technical stakeholders?",
    "Interview behavioral: How do you handle ambiguity in requirements?",
]

REASONING_PROMPTS = [
    "Case reasoning: A product conversion dropped 20% after release. Provide diagnostic plan and prioritization.",
    "Case reasoning: Budget is cut by 30%. How do you re-plan roadmap with minimal customer impact?",
    "Case reasoning: Service latency doubled at peak traffic. Give investigation tree and mitigation plan.",
    "Case reasoning: You must choose between speed and reliability this quarter. Explain decision framework.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate high-difficulty exam/interview dataset")
    p.add_argument("--train_file", type=Path, default=Path("data/llm/train_exam_interview.jsonl"))
    p.add_argument("--val_file", type=Path, default=Path("data/llm/val_exam_interview.jsonl"))
    p.add_argument("--train_size", type=int, default=2500)
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--seed", type=int, default=90)
    return p.parse_args()


def make_answer(prompt: str) -> str:
    if "Coding interview" in prompt:
        reasoning = "State constraints, choose optimal data structure, provide complexity, discuss edge cases."
        final = (
            "Use a clear approach with complexity target, pseudocode-level steps, and test edge cases. "
            "Mention time/space trade-offs explicitly."
        )
    elif "System design" in prompt:
        reasoning = "Start with requirements, traffic assumptions, APIs, storage, scaling, failure modes, and observability."
        final = (
            "Answer with architecture blocks, data model, scaling plan, bottlenecks, caching strategy, and reliability controls."
        )
    elif "behavioral" in prompt.lower():
        reasoning = "Use STAR format and measurable outcome."
        final = "Answer in STAR: Situation, Task, Action, Result, plus one lesson learned."
    elif "Case reasoning" in prompt:
        reasoning = "Build hypothesis tree, rank by impact/probability, run fastest checks first."
        final = "Present a prioritized diagnostic plan, immediate mitigation, and medium-term prevention actions."
    else:
        reasoning = "Define core concept, derive or apply formula, verify result with quick sanity check."
        final = "Provide step-by-step reasoning, equations, and final validated conclusion."
    return f"Reasoning: {reasoning} Final: {final}"


def sample_prompt() -> str:
    bank = random.choice([
        MATH_PHYSICS_PROMPTS,
        CODING_PROMPTS,
        SYSTEM_DESIGN_PROMPTS,
        BEHAVIORAL_PROMPTS,
        REASONING_PROMPTS,
    ])
    return random.choice(bank)


def write_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n):
            q = sample_prompt()
            a = make_answer(q)
            f.write(json.dumps({"instruction": q, "response": a}, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    write_jsonl(args.train_file, args.train_size)
    write_jsonl(args.val_file, args.val_size)
    print(f"generated_exam_train={args.train_file} count={args.train_size}")
    print(f"generated_exam_val={args.val_file} count={args.val_size}")


if __name__ == "__main__":
    main()
