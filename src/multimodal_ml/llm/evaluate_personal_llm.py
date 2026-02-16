from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import sympy as sp
    SYMPY_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    sp = None
    SYMPY_IMPORT_ERROR = str(e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate personal LLM on JSONL test set")
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--test_file", type=Path, required=True)
    parser.add_argument("--creator_profile", type=Path, default=Path("creator_profile.json"))
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_report", type=Path, default=Path("checkpoints/personal_llm_eval.json"))
    parser.add_argument("--mistakes_jsonl", type=Path, default=None, help="Optional output JSONL for wrong samples")
    return parser.parse_args()


def load_creator_profile(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(question: str, profile: dict) -> str:
    creator = profile.get("creator_name", "Creator")
    domains = ", ".join(profile.get("priority_domains", ["math", "physics"]))
    return (
        f"You are {creator}'s personal assistant specialized in {domains}. "
        "Answer with formulas and short steps.\n"
        "Output format:\n"
        "Reasoning: <short steps>\n"
        "Final: <final equation/number only>\n"
        f"Question: {question}\n"
        "Answer:"
    )


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("**", "^")
    s = re.sub(r"\s+", " ", s)
    return s


def score_answer(pred: str, gold: str) -> float:
    p = normalize(pred)
    g = normalize(gold)

    if g in p:
        return 1.0

    g_tokens = set(re.findall(r"[a-z0-9\^\+\-\*/=.]+", g))
    p_tokens = set(re.findall(r"[a-z0-9\^\+\-\*/=.]+", p))
    if not g_tokens:
        return 0.0

    overlap = len(g_tokens & p_tokens) / len(g_tokens)
    return overlap


def _extract_expr_after_equals(text: str) -> str | None:
    if "=" not in text:
        return None
    return text.split("=", maxsplit=1)[-1].strip().rstrip(".")


def _normalize_expr(expr: str) -> str:
    return expr.replace("^", "**").strip()


def _safe_sympify(expr: str):
    if sp is None:
        return None
    try:
        return sp.sympify(_normalize_expr(expr))
    except Exception:
        return None


def math_correct(question: str, pred: str, gold: str) -> bool | None:
    if sp is None:
        return None

    q = question.lower()
    p = pred.strip()
    x = sp.Symbol("x", real=True)

    # Derivative
    if "differentiate" in q and "f(x)" in q:
        m = re.search(r"f\(x\)\s*=\s*(.+)", question, re.IGNORECASE)
        if not m:
            return None
        f_expr = _safe_sympify(m.group(1))
        if f_expr is None:
            return None
        target = sp.simplify(sp.diff(f_expr, x))
        pred_expr_txt = _extract_expr_after_equals(p) or p
        pred_expr = _safe_sympify(pred_expr_txt)
        if pred_expr is None:
            return False
        return sp.simplify(pred_expr - target) == 0

    # Integral (supports both "integral(... ) dx" and "Integrate f(x)=... with respect to x")
    if ("integral(" in q and ") dx" in q) or ("integrate" in q and "with respect to x" in q):
        m = re.search(r"integral\((.+)\)\s*dx", question, re.IGNORECASE)
        if not m:
            m = re.search(r"f\(x\)\s*=\s*(.+?)\s*(with respect to x\.?)?$", question, re.IGNORECASE)
        if not m:
            return None
        f_expr = _safe_sympify(m.group(1))
        if f_expr is None:
            return None
        target = sp.simplify(f_expr)
        pred_expr_txt = _extract_expr_after_equals(p) or p
        pred_expr_txt = pred_expr_txt.replace("+ C", "").replace("+C", "").strip()
        pred_expr = _safe_sympify(pred_expr_txt)
        if pred_expr is None:
            return False
        return sp.simplify(sp.diff(pred_expr, x) - target) == 0

    # Limit sin(kx)/(kx) at 0 (supports both compact and natural language forms)
    if ("lim_(x->0) sin(" in q) or ("limit of sin(" in q and "as x approaches 0" in q):
        pred_num = _safe_sympify(p)
        if pred_num is None:
            m = re.search(r"value\s+is\s+([^\s.]+)", p.lower())
            pred_num = _safe_sympify(m.group(1)) if m else None
        if pred_num is None:
            m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)", p)
            pred_num = _safe_sympify(m2.group(1)) if m2 else None
        return pred_num is not None and sp.simplify(pred_num - 1) == 0

    # Quadratic roots (supports both "Solve equation exactly:" and "Solve equation ...")
    if "solve equation" in q and "= 0" in q:
        if ":" in question:
            eq_txt = question.split(":", maxsplit=1)[-1].strip()
        else:
            eq_txt = re.sub(r"(?i)^solve equation\s*", "", question).strip()
            if eq_txt.endswith("."):
                eq_txt = eq_txt[:-1]
        eq_txt = eq_txt.replace("= 0", "").strip()
        poly = _safe_sympify(eq_txt)
        if poly is None:
            return None
        roots = [sp.simplify(r) for r in sp.solve(sp.Eq(poly, 0), x)]
        nums = re.findall(r"-?\d+(?:\.\d+)?", p)
        if len(nums) < 2:
            return False
        pred_roots = [sp.simplify(sp.sympify(n)) for n in nums[:2]]
        return set(roots) == set(pred_roots)

    # Force
    if "find force in n" in q and "mass=" in q and "acceleration=" in q:
        m = re.search(r"mass=([0-9.\-]+)", q)
        a = re.search(r"acceleration=([0-9.\-]+)", q)
        if not (m and a):
            return None
        target = float(m.group(1)) * float(a.group(1))
        nums = re.findall(r"-?\d+(?:\.\d+)?", p)
        if not nums:
            return False
        pred_num = float(nums[-1])
        return abs(pred_num - target) <= 1e-6

    # Kinematics
    if "kinematics:" in q and "find v and s" in q:
        mu = re.search(r"u=([0-9.\-]+)", q)
        ma = re.search(r"a=([0-9.\-]+)", q)
        mt = re.search(r"t=([0-9.\-]+)", q)
        if not (mu and ma and mt):
            return None
        u = float(mu.group(1))
        a = float(ma.group(1))
        t = float(mt.group(1))
        v = u + a * t
        s = u * t + 0.5 * a * t * t
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", p)]
        if len(nums) < 2:
            return False
        # We expect two final numbers v and s near the end.
        pv, ps = nums[-2], nums[-1]
        return abs(pv - v) <= 1e-6 and abs(ps - s) <= 1e-6

    return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    profile = load_creator_profile(args.creator_profile)

    peft_cfg = PeftConfig.from_pretrained(str(args.adapter_dir))
    base_model_name = peft_cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model.eval()

    device = pick_device()
    model.to(device)

    samples = load_jsonl(args.test_file)[: args.max_samples]

    total_score = 0.0
    full_correct = 0
    math_checked = 0
    math_correct_count = 0
    details = []
    mistakes = []

    for idx, item in enumerate(samples):
        q = item["instruction"]
        gold = item["response"]
        prompt = build_prompt(q, profile)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = text.split("Answer:", maxsplit=1)[-1].strip()

        sc = score_answer(pred, gold)
        total_score += sc
        if sc >= 0.999:
            full_correct += 1
        mc = math_correct(q, pred, gold)
        if mc is not None:
            math_checked += 1
            if mc:
                math_correct_count += 1

        wrong = sc < 0.999
        if mc is not None:
            wrong = wrong or (not mc)
        if wrong:
            mistakes.append({"instruction": q, "response": gold, "model_response": pred, "score": sc, "math_correct": mc})

        if idx < 10:
            details.append(
                {
                    "instruction": q,
                    "gold": gold,
                    "pred": pred,
                    "score": sc,
                    "math_correct": mc,
                }
            )

    n = max(len(samples), 1)
    avg_score = total_score / n
    strict_acc = full_correct / n
    math_acc = (math_correct_count / math_checked) if math_checked > 0 else None

    report = {
        "num_samples": len(samples),
        "avg_token_overlap_score": avg_score,
        "strict_exactish_accuracy": strict_acc,
        "math_checked_samples": math_checked,
        "math_correct_accuracy": math_acc,
        "mistake_count": len(mistakes),
        "sample_predictions": details,
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.mistakes_jsonl is not None:
        args.mistakes_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.mistakes_jsonl.open("w", encoding="utf-8") as f:
            for m in mistakes:
                f.write(json.dumps({"instruction": m["instruction"], "response": m["response"]}, ensure_ascii=True) + "\n")

    print(f"evaluated_samples={len(samples)}")
    print(f"avg_token_overlap_score={avg_score:.4f}")
    print(f"strict_exactish_accuracy={strict_acc:.4f}")
    if math_acc is None:
        print("math_correct_accuracy=NA (install sympy for symbolic checks)")
        if SYMPY_IMPORT_ERROR is not None:
            print(f"sympy_import_error={SYMPY_IMPORT_ERROR}")
    else:
        print(f"math_correct_accuracy={math_acc:.4f} checked={math_checked}")
    print(f"mistake_count={len(mistakes)}")
    if args.mistakes_jsonl is not None:
        print(f"mistakes_file={args.mistakes_jsonl}")
    print(f"report_file={args.output_report}")


if __name__ == "__main__":
    main()
