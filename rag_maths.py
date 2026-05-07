"""
RAG pipeline for the Maths competition.
Calculator and symbolic evaluator, our tools they are.
"""

import re
import math
from typing import Optional


_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "half": 0.5, "quarter": 0.25, "third": 1/3, "eighth": 0.125,
}


def _word_to_num(text: str) -> str:
    """Replace English word-numbers with digits."""
    for word, val in _WORD_NUMBERS.items():
        text = re.sub(rf"\b{word}\b", str(val), text, flags=re.I)
    return text


def _eval_expr(expr: str) -> Optional[float]:
    """
    Safely evaluate a mathematical expression.
    Only math functions allowed — dangerous code, execute we do not.
    """
    safe_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_names["abs"] = abs
    try:
        result = eval(expr, {"__builtins__": {}}, safe_names)  # noqa: S307
        return float(result)
    except Exception:
        return None


def rag_maths(question_text: str) -> str:
    """
    Extract and compute mathematical expressions from the question.
    Handles percentages, roots, powers, factorials, combinations, word numbers,
    and falls back to sympy for general symbolic evaluation.
    """
    results = []
    q = _word_to_num(question_text)

    # Percentage: "X% of Y" or "X percent of Y"
    for pct, total in re.findall(r"(\d+\.?\d*)\s*(?:%|percent)\s*of\s*(\d+\.?\d*)", q, re.I):
        val = float(pct) / 100 * float(total)
        results.append(f"{pct}% of {total} = {val}")

    # Square root: "sqrt(X)", "square root of X", "√X"
    for n in re.findall(r"(?:sqrt\s*\(|square\s+root\s+of|√)\s*(\d+\.?\d*)\)?", q, re.I):
        results.append(f"sqrt({n}) = {math.sqrt(float(n)):.6f}")

    # Power: "X^Y", "X**Y", "X to the power of Y"
    for base, exp in re.findall(r"(\d+\.?\d*)\s*(?:\^|\*\*|to\s+the\s+power\s+of)\s*(\d+\.?\d*)", q, re.I):
        results.append(f"{base}^{exp} = {float(base) ** float(exp)}")

    # Factorial: "X!" or "factorial of X"
    for n in re.findall(r"(\d+)\s*!", q):
        results.append(f"{n}! = {math.factorial(int(n))}")
    for n in re.findall(r"factorial\s+of\s+(\d+)", q, re.I):
        results.append(f"{n}! = {math.factorial(int(n))}")

    # Combinations nCr: "n choose r", "C(n,r)", "nCr"
    for n, r in re.findall(r"(\d+)\s*[Cc](?:hoose|r)?\s*(\d+)", q):
        results.append(f"C({n},{r}) = {math.comb(int(n), int(r))}")
    for n, r in re.findall(r"[Cc]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", q):
        results.append(f"C({n},{r}) = {math.comb(int(n), int(r))}")

    # Inline arithmetic: e.g. "3 + 4 * 2"
    for expr in re.findall(r"(\d+\.?\d*(?:\s*[\+\-\*\/]\s*\d+\.?\d*)+)", q):
        val = _eval_expr(expr.replace(" ", ""))
        if val is not None:
            results.append(f"{expr.strip()} = {val}")

    # sympy fallback for general symbolic expressions
    if not results:
        try:
            import sympy
            cleaned = re.sub(r"[^0-9\+\-\*\/\^\(\)\.\s]", " ", q).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if re.search(r"\d", cleaned):
                val = sympy.sympify(cleaned.replace("^", "**"))
                results.append(f"sympy({cleaned}) = {float(val):.6f}")
        except Exception:
            pass

    if not results:
        return "No direct computation extracted. Reason carefully, you must."
    return "Computed results: " + "; ".join(results)
