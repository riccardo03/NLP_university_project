"""
RAG pipeline for Mathematics — v2.
Key improvement: option texts used to detect answer type + numeric hints.
"""

import re, math
from rag_shared import (wiki_fetch, wiki_search, ddg_search,
                        best_paragraphs, extract_keywords, _STOP_WORDS)

_ARTICLE_REF_RE = re.compile(
    r"\b(according to (the )?(article|text|passage|paragraph|excerpt))\b", re.I
)
_MATH_STOP = _STOP_WORDS | {
    "calculate","compute","evaluate","solve","determine","express","give",
    "find","show","prove","formula","value","result","answer","following",
    "statement","true","false","always","never","sometimes",
}
_WORD_NUMBERS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
    "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
    "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
    "nineteen":19,"twenty":20,"half":0.5,"quarter":0.25,"third":1/3,
}

_CONCEPT_MAP = {
    r"\bpythagor":               "Pythagorean theorem",
    r"\bfibonacci":              "Fibonacci sequence",
    r"\bprime\b":               "prime number mathematics",
    r"\bfactori":                "factorial mathematics",
    r"\bpermutation":            "permutation combination",
    r"\bcombination\b":         "combination binomial coefficient",
    r"\bderivative|\bdifferentiat": "derivative calculus",
    r"\bintegral|\bintegrat":   "integral calculus",
    r"\bmatrix|\bmatrices":     "matrix mathematics",
    r"\beigenvalue|\beigenvector": "eigenvalue eigenvector",
    r"\bbayes":                  "Bayes theorem probability",
    r"\bstandard deviation":     "standard deviation statistics",
    r"\bmean\b.*\bmedian|\bmedian\b.*\bmean": "mean median mode statistics",
    r"\bprobability":            "probability theory",
    r"\btrigonometr|\bsin\b|\bcos\b|\btan\b": "trigonometry",
    r"\blogarithm|\blog\b|\bln\b": "logarithm mathematics",
    r"\bquadratic":              "quadratic equation formula",
    r"\bgeometric series":       "geometric series sum",
    r"\barithmetic series":      "arithmetic series sum",
    r"\barea.*circle|circle.*area": "area circle pi formula",
    r"\bvolume.*sphere|sphere.*volume": "volume sphere formula",
    r"\bgroup\b.*\babelian|\babelian": "abelian group abstract algebra",
    r"\bgroup\b.*\bidentity":  "group theory identity element",
    r"\bmodular|\bmod\b":      "modular arithmetic",
    r"\bgcd|greatest common":    "greatest common divisor",
    r"\blcm|least common multiple": "least common multiple",
    r"\bvector\b":              "vector mathematics",
    r"\bmatrix\b":              "matrix linear algebra",
    r"\bdeterminant":            "determinant matrix",
    r"\beigenvalue":             "eigenvalue linear algebra",
    r"\bgraph theory":           "graph theory mathematics",
    r"\btopology":               "topology mathematics",
    r"\bnumber theory":          "number theory",
    r"\bset theory":             "set theory mathematics",
    r"\bbinomial":               "binomial theorem",
    r"\bpoisson":                "Poisson distribution",
    r"\bnormal distribution":    "normal distribution statistics",
}


def _word_to_num(text: str) -> str:
    for word, val in _WORD_NUMBERS.items():
        text = re.sub(rf"\b{word}\b", str(val), text, flags=re.I)
    return text


def _eval_safe(expr: str):
    safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe["abs"] = abs
    try:
        return float(eval(expr, {"__builtins__": {}}, safe))
    except Exception:
        return None


def _compute_inline(question: str) -> str:
    results = []
    q = _word_to_num(question)
    for pct, total in re.findall(r"(\d+\.?\d*)\s*(?:%|percent)\s*of\s*(\d+\.?\d*)", q, re.I):
        results.append(f"{pct}% of {total} = {float(pct)/100*float(total):.4g}")
    for n in re.findall(r"(?:sqrt\s*\(|square\s+root\s+of|√)\s*(\d+\.?\d*)\)?", q, re.I):
        results.append(f"sqrt({n}) = {math.sqrt(float(n)):.6f}")
    for base, exp in re.findall(r"(\d+\.?\d*)\s*(?:\^|\*\*|to the power of)\s*(\d+\.?\d*)", q, re.I):
        results.append(f"{base}^{exp} = {float(base)**float(exp):.4g}")
    for n in re.findall(r"(\d+)\s*!", q):
        if int(n) <= 20: results.append(f"{n}! = {math.factorial(int(n))}")
    for n, r in re.findall(r"(\d+)\s*[Cc](?:hoose|r)?\s*(\d+)", q):
        results.append(f"C({n},{r}) = {math.comb(int(n), int(r))}")
    for n, r in re.findall(r"P\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", q):
        results.append(f"P({n},{r}) = {math.perm(int(n), int(r))}")
    for expr in re.findall(r"(\d+\.?\d*(?:\s*[\+\-\*\/]\s*\d+\.?\d*)+)", q):
        val = _eval_safe(expr.replace(" ",""))
        if val is not None: results.append(f"{expr.strip()} = {val:.4g}")
    return "Computed: " + "; ".join(results) if results else ""


def _options_numeric_hint(option_texts: list) -> str:
    """Extract numeric values from options to guide computation."""
    nums = []
    for opt in option_texts:
        found = re.findall(r"-?\d+\.?\d*", opt)
        nums.extend(found)
    if nums:
        return "Possible answer values: " + ", ".join(nums[:8])
    return ""


def _concept_query(question: str, option_texts: list) -> str:
    q_lower = question.lower()
    for pattern, concept in _CONCEPT_MAP.items():
        if re.search(pattern, q_lower): return concept
    # Try option texts for concept clues
    for opt in option_texts:
        opt_lower = opt.lower()
        for pattern, concept in _CONCEPT_MAP.items():
            if re.search(pattern, opt_lower): return concept
    kws = [w for w in extract_keywords(question, min_len=5) if w not in _MATH_STOP]
    return " ".join(kws[:3]) if kws else ""


def _is_pure_arithmetic(question: str) -> bool:
    patterns = [
        r"what is \d", r"calculate|compute|evaluate|simplify",
        r"\d+\s*[\+\-\*\/\^]\s*\d+", r"\d+\s*%\s*of\s*\d+",
        r"square root of \d", r"\d+\s*!",
    ]
    return any(re.search(p, question.lower()) for p in patterns)


def rag_maths(question_text: str, option_texts: list = None,
              generate_answer_fn=None) -> str:
    """Maths RAG v2: inline computation + concept lookup + option numeric hints."""
    if _ARTICLE_REF_RE.search(question_text): return ""
    print("  [RAG-Maths] Pipeline started...")
    option_texts = option_texts or []

    computed      = _compute_inline(question_text)
    numeric_hint  = _options_numeric_hint(option_texts)
    if computed: print(f"  [RAG-Maths] Computed: {computed}")
    if numeric_hint: print(f"  [RAG-Maths] Hint: {numeric_hint}")

    if _is_pure_arithmetic(question_text) and computed:
        return (computed + "\n" + numeric_hint).strip()

    concept_q = _concept_query(question_text, option_texts)
    if not concept_q:
        return (computed + "\n" + numeric_hint).strip() if (computed or numeric_hint) else ""

    print(f"  [RAG-Maths] Concept: {concept_q!r}")
    text = wiki_fetch(concept_q, max_chars=3000)
    if not text or len(text.strip()) < 100:
        titles = wiki_search(concept_q, max_results=2)
        for t in titles:
            text = wiki_fetch(t, max_chars=3000)
            if text and len(text.strip()) >= 100: break

    wiki_ctx = ""
    if text:
        wiki_ctx = best_paragraphs(text, question_text, top_n=2, min_len=60)
        print(f"  [RAG-Maths] Wiki hit ({len(wiki_ctx)} chars).")
    if not wiki_ctx:
        snips = ddg_search(concept_q, max_results=2, timeout=3)
        wiki_ctx = "\n\n".join(snips[:2])

    parts = [p for p in [computed, numeric_hint, wiki_ctx] if p.strip()]
    result = "\n\n".join(parts)
    print(f"  [RAG-Maths] Done. Context: {len(result)} chars.")
    return result[:3000]
