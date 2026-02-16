from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BASE_QA = [
    ("What is revenue?", "Revenue is total income from sales before expenses."),
    ("What is profit?", "Profit is revenue minus total costs."),
    ("Difference between gross profit and net profit?", "Gross profit is revenue minus cost of goods sold; net profit subtracts all operating, interest, and tax expenses."),
    ("What is cash flow?", "Cash flow is net movement of cash into and out of a business over time."),
    ("What is free cash flow?", "Free cash flow is operating cash flow minus capital expenditures."),
    ("What is EBITDA?", "EBITDA means earnings before interest, taxes, depreciation, and amortization."),
    ("What is balance sheet?", "A balance sheet shows assets, liabilities, and equity at a point in time."),
    ("What is income statement?", "An income statement shows revenues and expenses over a period to derive profit."),
    ("What is P/E ratio?", "P/E ratio is price per share divided by earnings per share."),
    ("What is EPS?", "EPS is earnings per share, typically net income divided by weighted average shares."),
    ("What is market capitalization?", "Market cap equals share price multiplied by total shares outstanding."),
    ("What is diversification in investing?", "Diversification spreads investments across assets to reduce concentration risk."),
    ("What is risk-adjusted return?", "Risk-adjusted return evaluates returns relative to risk taken, such as with Sharpe ratio."),
    ("What is inflation?", "Inflation is a sustained increase in general price levels over time."),
    ("How do interest rates affect stocks?", "Higher rates can reduce valuation multiples and increase discount rates; effects vary by sector."),
    ("What is compound interest?", "Compound interest means interest is earned on principal plus accumulated interest."),
    ("What is ROI?", "ROI is return on investment: (gain - cost) / cost."),
    ("What is break-even point?", "Break-even is where total revenue equals total costs and profit is zero."),
    ("What is opportunity cost?", "Opportunity cost is the value of the best alternative forgone."),
    ("What is liquidity in markets?", "Liquidity is how easily an asset can be bought or sold with low price impact."),
    ("What is volatility?", "Volatility measures the magnitude of price fluctuations over time."),
    ("What is beta in stocks?", "Beta estimates how sensitive a stock is to market movements."),
    ("What is an index fund?", "An index fund tracks a market index and offers broad exposure at low cost."),
    ("What is dollar-cost averaging?", "Dollar-cost averaging invests fixed amounts at regular intervals to reduce timing risk."),
    ("What is stop-loss?", "A stop-loss is an order to limit downside by selling when price reaches a threshold."),
    ("What is a bull market?", "A bull market is a period of generally rising prices and positive sentiment."),
    ("What is a bear market?", "A bear market is a period of prolonged declines, often defined as a 20% drop from highs."),
    ("What is blockchain?", "Blockchain is a distributed ledger where transactions are recorded in linked blocks."),
    ("What is Bitcoin?", "Bitcoin is a decentralized digital asset secured by cryptography and consensus rules."),
    ("What is Ethereum?", "Ethereum is a blockchain platform supporting smart contracts and decentralized applications."),
    ("What is stablecoin?", "A stablecoin is a crypto token designed to track a stable reference asset such as USD."),
    ("What is staking in crypto?", "Staking locks tokens in proof-of-stake networks to help secure the chain and earn rewards."),
    ("What is private key in crypto?", "A private key is a secret credential controlling ownership of crypto addresses."),
]

VARIANTS = [
    "Explain clearly: {q}",
    "Give practical answer: {q}",
    "Teach a beginner: {q}",
    "In simple words, {q}",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate business and finance reasoning dataset")
    p.add_argument("--train_file", type=Path, default=Path("data/llm/train_business_finance.jsonl"))
    p.add_argument("--val_file", type=Path, default=Path("data/llm/val_business_finance.jsonl"))
    p.add_argument("--train_size", type=int, default=3000)
    p.add_argument("--val_size", type=int, default=600)
    p.add_argument("--seed", type=int, default=77)
    return p.parse_args()


def format_reasoning_final(reasoning: str, final: str) -> str:
    return f"Reasoning: {reasoning} Final: {final}"


def sample_definition() -> tuple[str, str]:
    q, a = random.choice(BASE_QA)
    q2 = random.choice(VARIANTS).format(q=q)
    reasoning = "Start with precise definition, then give one practical interpretation."
    return q2, format_reasoning_final(reasoning, a)


def sample_inflation_case() -> tuple[str, str]:
    food = random.randint(6, 18)
    fuel = random.randint(4, 20)
    wage = random.randint(3, 12)
    demand = random.choice(["high", "moderate", "weak"])
    q = (
        f"Inflation question: food prices up {food}%, fuel up {fuel}%, wage growth {wage}%, demand is {demand}. "
        "Explain what inflation is, why it may be rising, and how policy can control it."
    )
    cause = "cost-push" if (food + fuel) > 20 else "mixed demand-pull and cost-push"
    final = (
        f"Inflation is sustained overall price rise. In this case it is mainly {cause}. "
        "Policy controls include higher interest rates, tighter liquidity, targeted supply-side improvements, "
        "and reducing bottlenecks in food/energy logistics."
    )
    reasoning = "Break into definition, current drivers, then policy tools (monetary + supply-side)."
    return q, format_reasoning_final(reasoning, final)


def sample_roi_case() -> tuple[str, str]:
    cost = random.randint(1000, 20000)
    gain = cost + random.randint(-500, 12000)
    roi = (gain - cost) / cost
    q = f"An investment cost {cost} and current value is {gain}. Compute ROI and interpret if it is good."
    final = f"ROI = (gain - cost)/cost = ({gain} - {cost})/{cost} = {roi:.4f} ({roi*100:.2f}%). Positive ROI indicates gain; compare with risk and alternatives before calling it good."
    reasoning = "Apply ROI formula, compute percentage, then interpret with risk context."
    return q, format_reasoning_final(reasoning, final)


def sample_break_even_case() -> tuple[str, str]:
    fixed = random.randint(2000, 30000)
    price = random.randint(50, 400)
    var = random.randint(10, max(11, price - 5))
    units = fixed / (price - var)
    q = f"Business case: fixed cost={fixed}, selling price per unit={price}, variable cost per unit={var}. Find break-even units and explain."
    final = f"Break-even units = fixed/(price-variable) = {fixed}/({price}-{var}) = {units:.2f}. You need to sell at least {int(units)+1} units to move into profit."
    reasoning = "Use contribution margin (price-variable cost), then divide fixed cost by that margin."
    return q, format_reasoning_final(reasoning, final)


def sample_portfolio_case() -> tuple[str, str]:
    horizon = random.choice(["3 months", "1 year", "5 years"])
    risk = random.choice(["low", "medium", "high"])
    q = f"I am a {risk}-risk investor with {horizon} horizon. Give practical portfolio strategy with risk controls."
    if risk == "low":
        alloc = "higher share in high-quality bonds/cash equivalents and smaller equity exposure"
    elif risk == "medium":
        alloc = "balanced allocation across broad equity index funds and bonds"
    else:
        alloc = "higher equity allocation with limited speculative exposure and strict risk controls"
    final = (
        f"Use {alloc}, diversify across sectors/geographies, rebalance periodically, maintain emergency cash, "
        "and set position limits/stop-loss rules for volatile assets."
    )
    reasoning = "Map horizon + risk tolerance to allocation, then add diversification and risk-control rules."
    return q, format_reasoning_final(reasoning, final)


def sample_valuation_case() -> tuple[str, str]:
    eps = random.uniform(1.0, 15.0)
    pe = random.uniform(8.0, 35.0)
    fair = eps * pe
    q = f"Quick valuation: EPS={eps:.2f}, target P/E={pe:.1f}. Estimate fair price and mention limitations."
    final = f"Estimated fair price = EPS * P/E = {eps:.2f} * {pe:.1f} = {fair:.2f}. Limitation: P/E ignores capital structure, growth quality, and cash-flow timing, so cross-check with DCF/comparables."
    reasoning = "Use simple relative valuation then state model limitations."
    return q, format_reasoning_final(reasoning, final)


def sample_crypto_risk_case() -> tuple[str, str]:
    q = "I want to invest in crypto. Explain major risks and a safer approach for beginners."
    final = (
        "Major risks: high volatility, regulatory uncertainty, custody/security failure, and liquidity shocks. "
        "Safer approach: small allocation, diversified exposure, reputable custody, strict position sizing, and avoid leverage."
    )
    reasoning = "List risk categories first, then provide practical mitigation steps."
    return q, format_reasoning_final(reasoning, final)


def sample_item() -> tuple[str, str]:
    builders = [
        sample_definition,
        sample_inflation_case,
        sample_roi_case,
        sample_break_even_case,
        sample_portfolio_case,
        sample_valuation_case,
        sample_crypto_risk_case,
    ]
    return random.choice(builders)()


def write_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n):
            q, a = sample_item()
            f.write(json.dumps({"instruction": q, "response": a}, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    write_jsonl(args.train_file, args.train_size)
    write_jsonl(args.val_file, args.val_size)
    print(f"generated_bizfin_train={args.train_file} count={args.train_size}")
    print(f"generated_bizfin_val={args.val_file} count={args.val_size}")


if __name__ == "__main__":
    main()
