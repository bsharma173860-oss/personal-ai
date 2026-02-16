from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp


@dataclass
class MathResult:
    task: str
    input_text: str
    result: Any
    steps: str


def _symbol(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


def derivative(expr: str, var: str = "x") -> MathResult:
    x = _symbol(var)
    f = sp.sympify(expr)
    df = sp.diff(f, x)
    return MathResult(
        task="derivative",
        input_text=f"d/d{var} ({expr})",
        result=sp.simplify(df),
        steps="Parsed expression, applied symbolic differentiation, simplified output.",
    )


def integral(expr: str, var: str = "x") -> MathResult:
    x = _symbol(var)
    f = sp.sympify(expr)
    F = sp.integrate(f, x)
    return MathResult(
        task="integral",
        input_text=f"integral ({expr}) d{var}",
        result=sp.simplify(F),
        steps="Parsed expression, applied symbolic integration, simplified output.",
    )


def limit(expr: str, point: str, var: str = "x") -> MathResult:
    x = _symbol(var)
    f = sp.sympify(expr)
    p = sp.sympify(point)
    lim = sp.limit(f, x, p)
    return MathResult(
        task="limit",
        input_text=f"limit {expr} as {var}->{point}",
        result=sp.simplify(lim),
        steps="Parsed expression and point, evaluated symbolic limit.",
    )


def solve_equation(equation: str, var: str = "x") -> MathResult:
    x = _symbol(var)
    if "=" in equation:
        left, right = equation.split("=", maxsplit=1)
        eq = sp.Eq(sp.sympify(left), sp.sympify(right))
    else:
        eq = sp.Eq(sp.sympify(equation), 0)

    sol = sp.solve(eq, x)
    return MathResult(
        task="solve",
        input_text=f"solve {equation} for {var}",
        result=sol,
        steps="Converted to symbolic equation and solved for target variable.",
    )


def kinematics(u: float, a: float, t: float) -> MathResult:
    v = u + a * t
    s = u * t + 0.5 * a * t * t
    return MathResult(
        task="physics_kinematics",
        input_text=f"u={u}, a={a}, t={t}",
        result={"v": v, "s": s},
        steps="Used v=u+at and s=ut+0.5at^2.",
    )


def newton_force(mass: float, acceleration: float) -> MathResult:
    force = mass * acceleration
    return MathResult(
        task="physics_force",
        input_text=f"m={mass}, a={acceleration}",
        result={"F": force},
        steps="Used F=ma.",
    )
