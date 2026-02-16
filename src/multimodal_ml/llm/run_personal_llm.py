from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run your personal fine-tuned LLM")
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--creator_profile", type=Path, default=Path("creator_profile.json"))
    parser.add_argument("--concept_kb", type=Path, default=Path("data/llm/train_concepts.jsonl"))
    parser.add_argument("--finance_kb", type=Path, default=Path("data/llm/train_business_finance.jsonl"))
    parser.add_argument(
        "--domain_mode",
        type=str,
        choices=["general", "specialized"],
        default="general",
        help="general: answer any topic, specialized: only math/physics/rocket/space",
    )
    parser.add_argument("--show_reasoning", action="store_true", help="Print reasoning and final answer separately")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    return parser.parse_args()


def load_creator_profile(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_query_text(text: str) -> str:
    q = text.lower()
    q = q.replace("blackwhole", "black hole")
    q = q.replace("blackhole", "black hole")
    q = q.replace("spacetime", "space time")
    q = q.replace("expansive", "expensive")
    return q


def is_time_sensitive_question(question: str) -> bool:
    q = normalize_query_text(question)
    markers = [
        "current affairs", "current affair", "latest", "today", "news", "breaking",
        "prime minister", "pm of", "president", "stock", "stocks", "market", "crypto",
        "bitcoin", "ethereum", "price", "election", "minister", "government update",
    ]
    return any(m in q for m in markers)


def build_prompt(question: str, profile: dict) -> str:
    creator = profile.get("creator_name", "Creator")
    domains = ", ".join(profile.get("priority_domains", ["math", "physics"]))
    base = (
        f"You are {creator}'s personal assistant. "
        f"You are strong in {domains}, and you can also answer general topics, coding, and reasoning.\n"
        "Answer with concise steps and clear final answer.\n"
        "Output format:\n"
        "Reasoning: <short steps>\n"
        "Final: <final equation/number only>\n"
    )
    if is_time_sensitive_question(question):
        base += "This question is time-sensitive. Mention that live data can change and suggest verifying with a live source.\n"
    return (
        base +
        f"Question: {question}\n"
        "Answer:"
    )


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tokenize_simple(text: str) -> set[str]:
    stop = {
        "what", "is", "the", "a", "an", "and", "or", "of", "to", "in", "for",
        "on", "with", "explain", "define", "me", "about", "tell", "difference",
        "why", "how", "every", "day", "thing", "things", "going", "so",
    }
    toks = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {t for t in toks if t not in stop}


def load_kb(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            q = item.get("instruction", "").strip()
            a = item.get("response", "").strip()
            if q and a:
                rows.append({"instruction": q, "response": a, "tokens": tokenize_simple(q)})
    return rows


def maybe_kb_answer(question: str, kb: list[dict]) -> str | None:
    # Only trigger for concept-style prompts; keep math route on model.
    ql = normalize_query_text(question)
    concept_markers = [
        "what is",
        "explain",
        "define",
        "atom",
        "molecule",
        "quantum",
        "thermodynamics",
        "rocket",
        "orbit",
        "space",
        "black hole",
        "entropy",
        "business",
        "finance",
        "market",
        "stock",
        "stocks",
        "crypto",
        "bitcoin",
        "ethereum",
        "valuation",
        "revenue",
        "profit",
        "cash flow",
        "balance sheet",
        "interest rate",
        "inflation",
    ]
    if not any(m in ql for m in concept_markers):
        return None
    if not kb:
        return None

    q_tokens = tokenize_simple(ql)
    if not q_tokens:
        return None

    best = None
    best_score = 0.0
    for row in kb:
        inter = len(q_tokens & row["tokens"])
        union = len(q_tokens | row["tokens"])
        score = (inter / union) if union else 0.0
        if score > best_score:
            best_score = score
            best = row

    if best is None or best_score < 0.18:
        return None
    # Require at least one meaningful overlapping token to avoid "what is ..." mismatches.
    overlap = q_tokens & best["tokens"]
    if len(overlap) < 1:
        return None

    ans = best["response"]
    # Reuse same cleanup logic so we return concise final answer.
    return clean_generated_answer(ans)


def maybe_kb_answer_relaxed(question: str, kb: list[dict]) -> str | None:
    if not kb:
        return None
    q_tokens = tokenize_simple(normalize_query_text(question))
    if not q_tokens:
        return None

    best = None
    best_score = 0.0
    for row in kb:
        inter = len(q_tokens & row["tokens"])
        union = len(q_tokens | row["tokens"])
        score = (inter / union) if union else 0.0
        if score > best_score:
            best_score = score
            best = row

    if best is None or best_score < 0.10:
        return None
    overlap = q_tokens & best["tokens"]
    if len(overlap) < 1:
        return None
    return clean_generated_answer(best["response"])


def is_domain_question(question: str) -> bool:
    ql = normalize_query_text(question)
    domain_markers = [
        "math", "physics", "calculus", "differentiate", "integrate", "derivative", "integral",
        "limit", "equation", "force", "kinematics", "atom", "molecule", "quantum",
        "thermodynamics", "entropy", "rocket", "orbit", "space", "black hole",
        "dark matter", "dark energy", "gravity", "newton", "relativity", "spacetime", "space time",
    ]
    return any(m in ql for m in domain_markers)


def clean_generated_answer(answer: str) -> str:
    text = answer.strip()
    # Prefer explicit final line if present.
    m_final = re.search(r"final:\s*(.+)", text, flags=re.IGNORECASE)
    if m_final:
        text = m_final.group(1).strip()
        text = re.split(r"(?:Question:|Answer:|Reasoning:)", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        return first_line or text

    # Otherwise cut only on spillover question blocks.
    text = re.split(r"(?:Question:)", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # Strip leading labels but keep content.
    text = re.sub(r"^(?:Answer:|Reasoning:)\s*", "", text, flags=re.IGNORECASE).strip()
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first_line or text or answer.strip()


def extract_reasoning_and_final(answer: str) -> tuple[str, str]:
    text = answer.strip()
    m_reason = re.search(r"reasoning:\s*(.+?)(?:\n|final:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    m_final = re.search(r"final:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)

    reasoning = m_reason.group(1).strip() if m_reason else ""
    final = clean_generated_answer(m_final.group(1).strip() if m_final else text)

    # Remove prompt spillover in reasoning too.
    if reasoning:
        reasoning = re.split(r"(?:Question:|Answer:)", reasoning, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        first_reason_line = next((ln.strip() for ln in reasoning.splitlines() if ln.strip()), "")
        reasoning = first_reason_line or reasoning

    return reasoning, final


def looks_low_quality_final(final: str) -> bool:
    f = final.strip().lower()
    if f in {"", "none", "none.", "<no change>", "n/a", "na", "10", "10."}:
        return True
    if f.startswith("none."):
        return True
    if "requires explanation rather than a single number" in f:
        return True
    if "define inflation first" in f:
        return True
    if "then explain causes and methods" in f:
        return True
    if f.startswith("define ") and "then explain" in f:
        return True
    if len(f) <= 3:
        return True
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?\.?", f):
        return True
    return False


def canned_general_fallback(question: str) -> tuple[str, str] | None:
    q = normalize_query_text(question)
    econ_markers = ["expensive", "cost of living", "inflation", "prices", "price rise", "rent", "food cost"]
    if any(m in q for m in econ_markers):
        reasoning = (
            "Break into three parts: definition, causes, and controls. "
            "Use macroeconomics factors (demand/supply, costs, policy, and expectations)."
        )
        final = (
            "Inflation is a sustained rise in overall prices. It rises due to higher production and transport costs, "
            "strong demand relative to supply, currency weakness, and supply-chain disruptions. "
            "Control methods include higher policy rates, tighter liquidity, improving supply/logistics, "
            "and reducing fiscal pressure in essential sectors."
        )
        return reasoning, final
    return None


def is_multi_part_question(question: str) -> bool:
    q = normalize_query_text(question)
    signals = [" how ", " why ", " and ", ","]
    hit_count = sum(1 for s in signals if s in f" {q} ")
    return hit_count >= 2


def main() -> None:
    args = parse_args()
    profile = load_creator_profile(args.creator_profile)
    if not args.interactive and not args.question:
        raise ValueError("Provide --question for single-run mode, or use --interactive.")

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

    if profile.get("creator_name"):
        print(f"Hello, {profile['creator_name']}.")

    concept_kb = load_kb(args.concept_kb)
    finance_kb = load_kb(args.finance_kb)
    all_kb = concept_kb + finance_kb

    def run_one(question: str) -> tuple[str, str]:
        # Prefer deterministic structured fallback for multi-part economics/business prompts.
        canned = canned_general_fallback(question)
        if canned is not None and is_multi_part_question(question):
            return canned

        kb_ans = maybe_kb_answer(question, all_kb)
        if kb_ans:
            return extract_reasoning_and_final(kb_ans)
        if args.domain_mode == "specialized" and not is_domain_question(question):
            msg = "I specialize in math, physics, rocket science, and space science. Ask within these domains."
            return ("Domain guard applied.", msg)
        prompt = build_prompt(question, profile)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = text.split("Answer:", maxsplit=1)[-1].strip()
        reasoning, final = extract_reasoning_and_final(answer)

        if looks_low_quality_final(final):
            kb_fallback = maybe_kb_answer_relaxed(question, all_kb)
            if kb_fallback:
                return ("Used knowledge-base fallback due to low-confidence generation.", kb_fallback)
            if canned is not None:
                return canned

        return reasoning, final

    if args.interactive:
        print("interactive_mode=on (type 'exit' to quit)")
        while True:
            try:
                q = input("you> ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            reasoning, final = run_one(q)
            if args.show_reasoning and reasoning:
                print(f"reasoning={reasoning}")
            print(f"answer={final}")
    else:
        reasoning, final = run_one(args.question)
        if args.show_reasoning and reasoning:
            print(f"reasoning={reasoning}")
        print(f"answer={final}")


if __name__ == "__main__":
    main()
