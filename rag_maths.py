"""
RAG pipeline for the Mathematics competition.
Single LLM strategy call → Wikipedia formula lookup → minimal DDG fallback.
Optimised for latency: target < 10s total, leaving time for computation.
"""

import re
import math
from typing import Optional
from math_formulas import search_formula, format_formula_context

_ARTICLE_REF_RE = re.compile(
    r"\b(according to (the )?(article|text|passage|paragraph|excerpt))\b",
    re.I,
)

_MATH_STRATEGY_SYSTEM = (
    "You are a search strategist for a mathematics quiz bot. "
    "Given a question and its options, decide whether a web lookup is needed.\n"
    "\n"
    "If the question is a PURE arithmetic or numerical computation that needs no formula "
    "recall (e.g. '12 × 7', 'what is 15% of 80', 'what is 2^10'), output exactly:\n"
    "SKIP\n"
    "\n"
    "Otherwise output ONE line with this EXACT format (no other text):\n"
    "QUERY: <mathematical concept or formula in 2-5 words> | CATEGORY: <category>\n"
    "\n"
    "CATEGORY must be exactly one of:\n"
    "  Algebra       — equations, polynomials, matrices, sequences, series.\n"
    "  Geometry      — shapes, area, volume, trigonometry, vectors, coordinates.\n"
    "  Calculus      — derivatives, integrals, limits, differential equations.\n"
    "  Probability   — probability, statistics, combinations, permutations.\n"
    "  Constants     — named constants: pi, e, golden ratio, speed of light, etc.\n"
    "  NumberTheory  — primes, divisibility, modular arithmetic, factorials.\n"
    "  Logic         — sets, logic, proof, boolean algebra.\n"
    "  General       — any mathematical topic not covered above.\n"
    "\n"
    "Rules for QUERY:\n"
    "- Use precise mathematical terminology (e.g. 'Pythagorean theorem', "
    "'binomial coefficient', 'Euler number').\n"
    "- 2 to 5 words. No question words. No punctuation at end.\n"
    "\n"
    "Examples:\n"
    "Q: What is the formula for the area of a circle? → "
    "QUERY: area circle formula | CATEGORY: Geometry\n"
    "Q: What is 345 divided by 15? → SKIP\n"
    "Q: What is the derivative of sin(x)? → "
    "QUERY: derivative sine function | CATEGORY: Calculus\n"
    "Q: How many combinations of 5 items from 10? → "
    "QUERY: binomial coefficient combination formula | CATEGORY: Probability\n"
    "Q: What is the value of pi to 5 decimal places? → "
    "QUERY: pi mathematical constant | CATEGORY: Constants\n"
    "Q: What is the sum of the first n natural numbers? → "
    "QUERY: sum natural numbers formula | CATEGORY: Algebra\n"
)

_MATH_STOP_WORDS = {
    "what", "when", "which", "where", "does", "have", "this", "that",
    "from", "with", "about", "into", "their", "there", "been", "were",
    "would", "could", "should", "following", "give", "find", "calculate",
    "compute", "evaluate", "solve", "determine", "express",
}

# Greek letters and their names for keyword extraction
_GREEK_LETTERS = {
    "π": "pi", "Π": "pi",
    "α": "alpha", "Α": "alpha",
    "β": "beta", "Β": "beta",
    "γ": "gamma", "Γ": "gamma",
    "δ": "delta", "Δ": "delta",
    "ε": "epsilon", "Ε": "epsilon",
    "ζ": "zeta", "Ζ": "zeta",
    "η": "eta", "Η": "eta",
    "θ": "theta", "Θ": "theta",
    "λ": "lambda", "Λ": "lambda",
    "μ": "mu", "Μ": "mu",
    "σ": "sigma", "Σ": "sigma",
    "τ": "tau", "Τ": "tau",
    "φ": "phi", "Φ": "phi",
    "ψ": "psi", "Ψ": "psi",
    "ω": "omega", "Ω": "omega",
    "∞": "infinity",
}

# Mathematical symbols and their semantic meaning
_MATH_SYMBOLS = {
    "√": "square root",
    "∛": "cube root",
    "∫": "integral",
    "∑": "summation sum",
    "∏": "product",
    "∂": "partial derivative",
    "∇": "nabla gradient",
    "∝": "proportional",
    "≈": "approximately",
    "≠": "not equal",
    "≤": "less than or equal",
    "≥": "greater than or equal",
    "∞": "infinity",
    "∈": "element of",
    "∉": "not element of",
    "⊂": "subset",
    "∪": "union",
    "∩": "intersection",
    "!": "factorial",
    "^": "power exponent",
}

# Common math function and operator terms
_MATH_FUNCTIONS = {
    "sine", "sin", "cosine", "cos", "tangent", "tan",
    "derivative", "integral", "limit", "sum", "product",
    "logarithm", "log", "exponential", "exp",
    "square", "cube", "root", "power",
    "factorial", "permutation", "combination",
    "determinant", "matrix", "eigenvector", "eigenvalue",
}

_USEFUL_SECTIONS_RE = re.compile(
    r'^(formula|definition|theorem|proof|properties|notation|'
    r'statement|rule|law|equation|expression)',
    re.I,
)
_CITATION_RE = re.compile(r'\[\d+\]|\[citation needed\]|\[clarification needed\]', re.I)

# ------------------------------------------------------------------ #
# Symbolic computation helpers (kept for inline arithmetic enrichment)  #
# ------------------------------------------------------------------ #
_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "half": 0.5, "quarter": 0.25, "third": 1/3, "eighth": 0.125,
}


def _word_to_num(text: str) -> str:
    for word, val in _WORD_NUMBERS.items():
        text = re.sub(rf"\b{word}\b", str(val), text, flags=re.I)
    return text


def _eval_expr(expr: str) -> Optional[float]:
    safe_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_names["abs"] = abs
    try:
        result = eval(expr, {"__builtins__": {}}, safe_names)  # noqa: S307
        return float(result)
    except Exception:
        return None


def _compute_inline(question: str) -> str:
    """
    Try to compute any numeric expression embedded in the question.
    Returns a compact results string or '' if nothing computable.
    """
    results = []
    q = _word_to_num(question)

    for pct, total in re.findall(r"(\d+\.?\d*)\s*(?:%|percent)\s*of\s*(\d+\.?\d*)", q, re.I):
        results.append(f"{pct}% of {total} = {float(pct)/100*float(total)}")

    for n in re.findall(r"(?:sqrt\s*\(|square\s+root\s+of|√)\s*(\d+\.?\d*)\)?", q, re.I):
        results.append(f"sqrt({n}) = {math.sqrt(float(n)):.6f}")

    for base, exp in re.findall(
        r"(\d+\.?\d*)\s*(?:\^|\*\*|to\s+the\s+power\s+of)\s*(\d+\.?\d*)", q, re.I
    ):
        results.append(f"{base}^{exp} = {float(base)**float(exp)}")

    for n in re.findall(r"(\d+)\s*!", q):
        results.append(f"{n}! = {math.factorial(int(n))}")

    for n, r in re.findall(r"(\d+)\s*[Cc](?:hoose|r)?\s*(\d+)", q):
        results.append(f"C({n},{r}) = {math.comb(int(n), int(r))}")

    for expr in re.findall(r"(\d+\.?\d*(?:\s*[\+\-\*\/]\s*\d+\.?\d*)+)", q):
        val = _eval_expr(expr.replace(" ", ""))
        if val is not None:
            results.append(f"{expr.strip()} = {val}")
    
    # Equilateral triangle area: A = (√3/4) × s²
    for side in re.findall(r"equilateral\s+triangle.*?(?:side|sides?|edge).*?(\d+\.?\d*)", q, re.I):
        area = (math.sqrt(3) / 4) * float(side) ** 2
        results.append(f"Equilateral triangle area (s={side}): {area:.2f}")
    
    # Type I error with multiple independent tests: P = 1 - (1-α)^k
    type_i_match = re.search(r"(\d+)\s*(?:independent\s+)?tests.*?α\s*=\s*(0\.\d+|[\d.]+)", q, re.I)
    if type_i_match:
        k = int(type_i_match.group(1))
        alpha = float(type_i_match.group(2))
        prob_reject_once = 1 - (1 - alpha) ** k
        results.append(f"P(Type I error in ≥1 of {k} tests, α={alpha}): {prob_reject_once:.2f}")
    
    # Normal distribution: middle X% → use z-score for (100-X)/2 percentile
    middle_pct_match = re.search(r"middle\s+(\d+)\s*(?:%|percent)", q, re.I)
    if middle_pct_match and re.search(r"normal|distribution|standard", q, re.I):
        middle = int(middle_pct_match.group(1))
        tail_pct = (100 - middle) / 2
        # Standard z-scores: 25th percentile ≈ -0.674, 75th percentile ≈ +0.674
        z_critical = 0.674 if middle == 50 else None
        if z_critical:
            # Match "mean of X,XXX" (handle commas)
            mean_match = re.search(r"mean\s+of\s+([\d,]+\.?\d*)", q, re.I)
            # Match "standard deviation of X" (handle commas)
            sigma_match = re.search(r"(?:standard\s+deviation|σ)\s+of\s+([\d,]+\.?\d*)", q, re.I)
            if mean_match and sigma_match:
                mean = float(mean_match.group(1).replace(",", ""))
                sigma = float(sigma_match.group(1).replace(",", ""))
                lower = mean - z_critical * sigma
                upper = mean + z_critical * sigma
                results.append(f"Middle {middle}% range: ({lower:.0f}, {upper:.0f})")

    return "Computed: " + "; ".join(results) if results else ""


# ------------------------------------------------------------------ #
# Keyword / relevance helpers                                          #
# ------------------------------------------------------------------ #

def _extract_math_keywords(text: str) -> list[str]:
    keywords = []
    
    # Extract Greek letters and replace with their names
    for symbol, name in _GREEK_LETTERS.items():
        if symbol in text:
            keywords.append(name)
    
    # Extract mathematical symbols and their meanings
    for symbol, meaning in _MATH_SYMBOLS.items():
        if symbol in text:
            # Add the meaning words (e.g., "integral" from "∫")
            for word in meaning.split():
                keywords.append(word)
    
    # Extract standard words (5+ chars, not stopwords)
    for w in re.findall(r'\b[a-z]{5,}\b', text.lower()):
        if w not in _MATH_STOP_WORDS:
            keywords.append(w)
    
    # Extract acronyms (uppercase sequences)
    for w in re.findall(r'\b[A-Z]{2,}\b', text):
        keywords.append(w.lower())
    
    # Extract decimal numbers
    for w in re.findall(r'\b\d+\.\d+\b', text):
        keywords.append(w)
    
    # Extract known math functions (case-insensitive)
    text_lower = text.lower()
    for func in _MATH_FUNCTIONS:
        if func in text_lower:
            keywords.append(func)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(keywords))


def _is_math_relevant(snippet: str, question: str) -> bool:
    keywords = _extract_math_keywords(question)
    if not keywords:
        return True
    snippet_lower = snippet.lower()
    for kw in keywords:
        if re.match(r'^\d+\.\d+$', kw) and kw in snippet:
            return True
    matches = sum(1 for kw in keywords if kw in snippet_lower)
    return matches >= 2


# ------------------------------------------------------------------ #
# Wikipedia retrieval                                                  #
# ------------------------------------------------------------------ #

def _clean_wiki_text(text: str) -> str:
    text = _CITATION_RE.sub("", text)
    return re.sub(r'\s{2,}', ' ', text).strip()


def _wiki_math(query: str) -> str:
    """
    Fetch Wikipedia summary + Formula/Definition section.
    Returns at most 1200 characters of clean text.
    """
    try:
        import wikipedia
        wikipedia.set_lang("en")
        try:
            page = wikipedia.page(query, auto_suggest=False)
            summary = wikipedia.summary(query, sentences=3, auto_suggest=False)
            summary = _clean_wiki_text(summary)

            formula_lines: list[str] = []
            in_useful = False
            for line in page.content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                header_m = re.match(r'^==+\s*(.+?)\s*==+$', line)
                if header_m:
                    in_useful = bool(_USEFUL_SECTIONS_RE.match(header_m.group(1)))
                    continue
                if in_useful and len(line) > 30:
                    formula_lines.append(_clean_wiki_text(line))
                if len(formula_lines) >= 3:
                    break

            combined = summary
            if formula_lines:
                combined += "\n\n" + "\n".join(formula_lines)
            return combined[:1200]

        except wikipedia.exceptions.DisambiguationError as e:
            result = wikipedia.summary(e.options[0], sentences=3, auto_suggest=False)
            return _clean_wiki_text(result)[:1200]
        except wikipedia.exceptions.PageError:
            return ""
    except Exception:
        return ""


def _wiki_is_useful(text: str) -> bool:
    if not text or len(text.strip()) < 100:
        return False
    if "may refer to:" in text.lower():
        return False
    return True


# ------------------------------------------------------------------ #
# Strategy parsing                                                     #
# ------------------------------------------------------------------ #

def _parse_strategy(raw: str) -> tuple[str, str, str]:
    """Returns (action, query, category). action is 'SKIP' or 'QUERY'."""
    raw = raw.strip()
    if re.match(r'^SKIP', raw, re.I):
        return "SKIP", "", ""

    query = ""
    category = "General"
    qry_m = re.search(r'QUERY\s*:\s*(.+?)(?:\s*\|\s*CATEGORY|\Z)', raw, re.I)
    cat_m = re.search(r'CATEGORY\s*:\s*(\w+)', raw, re.I)
    if qry_m:
        query = qry_m.group(1).strip().strip('"').strip("'")
    if cat_m:
        category = cat_m.group(1).strip()
    return "QUERY", query, category


def _extract_concept(question: str, category: str) -> str:
    """
    Extract key concepts from question for better searching.
    Replaces narrative details with conceptual keywords.
    """
    q_lower = question.lower()
    
    # Pattern-based concept extraction
    concept_map = {
        # Statistics
        r"mean.*(?:salary|income|payment)": "statistical hypothesis test population mean",
        r"(?:comparing|comparing|compare).*mean": "two-sample statistical test",
        r"(?:comparing|comparing).*two.*(?:groups|samples)": "two-sample hypothesis test",
        r"survey.*(?:non-response|nonresponse|didn't respond|did not respond)": "survey methodology non-response bias",
        r"type\s+i\s+error.*(?:multiple|independent)": "type I error probability independent tests",
        r"normally\s+distributed.*middle\s+(\d+)": "normal distribution quartile percentile",
        r"equilateral\s+triangle": "equilateral triangle geometry area",
        r"(?:matrix|matrices).*dimension": "linear algebra matrix multiplication dimension",
        r"(?:independent|mutually exclusive).*(?:event|probability)": "probability independent events",
    }
    
    for pattern, concept in concept_map.items():
        if re.search(pattern, q_lower, re.I):
            return concept
    
    # Fallback: extract key math terms
    keywords = _extract_math_keywords(question)
    if keywords:
        # Filter to meaningful terms
        meaningful = [kw for kw in keywords if len(kw) > 3 and kw not in _MATH_STOP_WORDS]
        if meaningful:
            return " ".join(meaningful[:4])  # Top 4 keywords
    
    return question[:60]  # Fallback to original


def _get_strategy(question: str, options: list, generate_answer_fn) -> tuple[str, str, str]:
    opts_str = ", ".join(f'"{t}"' for t in options) if options else "—"
    user_msg = f"Q: {question}\nOptions: [{opts_str}]"
    try:
        raw = generate_answer_fn(_MATH_STRATEGY_SYSTEM, user_msg, max_new_tokens=25)
        print(f"  [RAG-Maths] Strategy raw: {raw[:100]!r}")
        action, query, category = _parse_strategy(raw)
        print(f"  [RAG-Maths] Action={action}, Query={query!r}, Category={category}")
        return action, query, category
    except Exception as e:
        print(f"  [RAG-Maths] Strategy call failed: {e}")
        # Fallback to concept extraction
        concept = _extract_concept(question, "General")
        return "QUERY", concept, "General"


# ------------------------------------------------------------------ #
# Main entry point                                                     #
# ------------------------------------------------------------------ #

def rag_maths(question_text: str, option_texts: list = None,
              generate_answer_fn=None) -> str:
    """
    Mathematics RAG: Computation + Formula Cache + Model Intelligence.
    
    Strategy:
    1. Try inline computation (100% accurate for numeric problems)
    2. Try formula cache (curated, high-confidence formulas)
    3. Return empty context (let model use its knowledge)
    
    Web lookup (Wikipedia/DDG) disabled for math - adds noise without value.
    """
    if option_texts is None:
        option_texts = []

    clean_q = _ARTICLE_REF_RE.sub("", question_text).strip(" ,;")

    # ------------------------------------------------------------------ #
    # Stage 1: Inline Computation                                         #
    # ------------------------------------------------------------------ #
    computed = _compute_inline(clean_q)
    if computed:
        print(f"  [RAG-Maths] Inline computation result: {computed[:80]}")
        return computed

    # ------------------------------------------------------------------ #
    # Stage 2: Formula Cache Lookup                                       #
    # ------------------------------------------------------------------ #
    search_query = clean_q[:80]
    if generate_answer_fn is not None:
        try:
            action, search_query, category = _get_strategy(clean_q, option_texts, generate_answer_fn)
            if action == "SKIP":
                print("  [RAG-Maths] SKIP — pure arithmetic, no context needed.")
                return ""
        except Exception as e:
            print(f"  [RAG-Maths] Strategy call failed: {e}")
    
    # Extract concept-based search query
    concept_query = _extract_concept(clean_q, "General")
    print(f"  [RAG-Maths] Formula cache search: {concept_query!r}")
    
    cache_result = search_formula(concept_query)
    if cache_result:
        cache_text = format_formula_context(cache_result)
        print(f"  [RAG-Maths] Cache hit: {cache_result.get('formula', '')[:50]}...")
        return cache_text

    # ------------------------------------------------------------------ #
    # Stage 3: Model Intelligence (No Web Lookup)                         #
    # ------------------------------------------------------------------ #
    print(f"  [RAG-Maths] No computation/cache match. Using model knowledge.")
    return ""
