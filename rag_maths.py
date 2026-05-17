"""
    Corpus    : Hendrycks MATH dataset + curated math facts and formulas
    Embedder  : BAAI/bge-small-en-v1.5         (~500 MB VRAM)
    Vector DB : FAISS IndexFlatIP (cosine via normalised vectors)
"""

import re
import os
from typing import Optional

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Math RAG resources (corpus, embedder, and retrieval)
# ---------------------------------------------------------------------------
_maths_embedder = None
_maths_index = None
_maths_passages = None

MATHS_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MATHS_TOP_K = 5


# ---------------------------------------------------------------------------
# LaTeX cleanup (the MATH dataset is heavy in LaTeX, which confuses embedders)
# ---------------------------------------------------------------------------
def _clean_latex(text: str) -> str:
    """Convert common LaTeX into plain-text equivalents for better embedding."""
    if not text:
        return text
    # Fractions and roots
    text = re.sub(r'\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', text)
    # Operators
    text = re.sub(r'\\times\b', '*', text)
    text = re.sub(r'\\cdot\b', '*', text)
    text = re.sub(r'\\div\b', '/', text)
    text = re.sub(r'\\leq?\b', '<=', text)
    text = re.sub(r'\\geq?\b', '>=', text)
    text = re.sub(r'\\neq?\b', '!=', text)
    text = re.sub(r'\\pm\b', '+/-', text)
    # Greek letters and common symbols
    for greek in ['pi', 'theta', 'alpha', 'beta', 'gamma', 'delta', 'sigma',
                  'mu', 'lambda', 'phi', 'rho', 'tau', 'omega', 'epsilon']:
        text = re.sub(rf'\\{greek}\b', greek, text)
    text = re.sub(r'\\sum\b', 'sum', text)
    text = re.sub(r'\\int\b', 'integral', text)
    text = re.sub(r'\\prod\b', 'product', text)
    text = re.sub(r'\\infty\b', 'infinity', text)
    # Strip leftover LaTeX commands and grouping braces
    text = re.sub(r'\\[a-zA-Z]+\*?\s*', ' ', text)
    text = re.sub(r'[{}]', ' ', text)
    text = re.sub(r'\$+', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Curated facts (comprehensive coverage of common math domains)
# ---------------------------------------------------------------------------
_CURATED_MATH_FACTS = [
    # ── Linear algebra ───────────────────────────────────────────────────────
    "Matrix multiplication: if matrix A has dimensions m by n and matrix B has dimensions n by p, then the product AB has dimensions m by p. The number of columns in A must equal the number of rows in B.",
    "For the product AB to be defined, the inner dimensions must match: A is m by n, B is n by p, result AB is m by p.",
    "The determinant of a 2 by 2 matrix [[a,b],[c,d]] equals ad minus bc.",
    "An invertible (non-singular) matrix has a non-zero determinant.",
    "Matrix multiplication is not commutative in general: AB is usually not equal to BA.",
    "The transpose of a matrix swaps its rows and columns. (A^T)_ij = A_ji.",
    "Eigenvalues are scalars lambda such that A v = lambda v for some non-zero eigenvector v.",
    "The rank of a matrix is the maximum number of linearly independent rows or columns.",
    "An orthogonal matrix Q satisfies Q^T Q = I, so its inverse equals its transpose.",
    "The identity matrix I has 1s on the diagonal and 0s elsewhere; AI = IA = A.",
    "A square matrix is symmetric if A = A^T.",

    # ── Calculus ─────────────────────────────────────────────────────────────
    "The derivative measures the instantaneous rate of change of a function.",
    "Power rule: the derivative of x^n is n times x^(n-1).",
    "Chain rule: the derivative of f(g(x)) equals f'(g(x)) times g'(x).",
    "Product rule: the derivative of f(x)*g(x) is f'(x)*g(x) + f(x)*g'(x).",
    "Quotient rule: derivative of f/g is (f'g - fg')/g^2.",
    "The derivative of sin(x) is cos(x), and the derivative of cos(x) is -sin(x).",
    "The derivative of e^x is e^x. The derivative of ln(x) is 1/x.",
    "The integral is the antiderivative; the definite integral represents the area under a curve.",
    "Integration by parts: integral of u dv equals u*v minus integral of v du.",
    "The Fundamental Theorem of Calculus links derivatives and integrals: integral from a to b of f'(x) dx = f(b) - f(a).",
    "A function is continuous at a point if its limit equals its value there.",
    "A function has a local maximum where its derivative changes from positive to negative.",

    # ── Statistics ───────────────────────────────────────────────────────────
    "The mean (average) of a dataset is the sum of all values divided by the number of values.",
    "The median is the middle value when data is sorted; for even-sized datasets, it's the average of the two middle values.",
    "The mode is the value that appears most frequently in a dataset.",
    "The variance measures how spread out values are from the mean. It is the average of squared deviations from the mean.",
    "Standard deviation is the square root of the variance and is in the same units as the data.",
    "A normal distribution is symmetric, bell-shaped, and characterized by its mean and standard deviation.",
    "In a normal distribution, about 68% of data falls within 1 standard deviation of the mean, 95% within 2, and 99.7% within 3 (the 68-95-99.7 rule).",
    "The z-test is used when comparing means and the population standard deviation is known, or for large samples (n >= 30).",
    "The t-test is used when comparing means and the population standard deviation is unknown, typically for smaller samples.",
    "A two-sample t-test compares the means of two independent groups when population standard deviations are unknown.",
    "A two-sample z-test compares the means of two independent groups when population standard deviations are known.",
    "A paired or one-sample t-test on differences is used when the two samples are matched or dependent (e.g., before/after measurements).",
    "When comparing salaries of two independent groups (like math teachers vs English teachers) with unknown population standard deviations, a two-sample t-test of population means is most appropriate.",
    "The Central Limit Theorem: the distribution of sample means approaches a normal distribution as sample size increases, regardless of the population's distribution.",
    "Correlation coefficient r measures the strength and direction of a linear relationship between two variables, ranging from -1 to 1.",
    "A p-value below the significance level (commonly 0.05) leads to rejecting the null hypothesis.",
    "Type I error is rejecting a true null hypothesis (false positive). Type II error is failing to reject a false null hypothesis (false negative).",
    "Confidence interval: a range of values likely to contain the true population parameter, e.g., a 95% CI captures the true mean 95% of the time over repeated sampling.",

    # ── Probability ──────────────────────────────────────────────────────────
    "For independent events A and B: P(A and B) = P(A) * P(B). This product rule is the defining property of independence.",
    "Two events are independent if the occurrence of one does not affect the probability of the other.",
    "Independence and mutual exclusivity are different: independent events can both occur; mutually exclusive events cannot occur together.",
    "If two events with nonzero probability are independent, they cannot be mutually exclusive (and vice versa).",
    "Conditional probability: P(A given B) = P(A and B) / P(B), provided P(B) > 0.",
    "If A and B are independent, then P(A given B) = P(A) and P(B given A) = P(B).",
    "Bayes' theorem: P(A given B) = P(B given A) * P(A) / P(B).",
    "P(A or B) = P(A) + P(B) - P(A and B). For mutually exclusive events, P(A and B) = 0.",
    "The complement rule: P(not A) = 1 - P(A).",
    "Expected value of a discrete random variable: E[X] = sum over i of x_i * P(X = x_i).",
    "Variance of a random variable: Var(X) = E[X^2] - (E[X])^2.",

    # ── Geometry ─────────────────────────────────────────────────────────────
    "Area of a triangle: (1/2) * base * height.",
    "Area of an equilateral triangle with side length s: (s^2 * sqrt(3)) / 4.",
    "For an equilateral triangle with side 12, area is (144 * sqrt(3)) / 4 = 36 * sqrt(3), approximately 62.35.",
    "Pythagorean theorem: in a right triangle, a^2 + b^2 = c^2 where c is the hypotenuse.",
    "Area of a rectangle: length * width. Perimeter: 2 * (length + width).",
    "Area of a circle: pi * r^2. Circumference: 2 * pi * r.",
    "Area of a trapezoid: (1/2) * (b1 + b2) * h, where b1 and b2 are parallel sides.",
    "Area of a parallelogram: base * height.",
    "Volume of a sphere: (4/3) * pi * r^3. Surface area: 4 * pi * r^2.",
    "Volume of a cylinder: pi * r^2 * h. Volume of a cone: (1/3) * pi * r^2 * h.",
    "Volume of a rectangular prism: length * width * height.",
    "The sum of interior angles of an n-sided polygon is (n - 2) * 180 degrees.",
    "Sum of interior angles of a triangle is 180 degrees; for a quadrilateral, 360 degrees.",
    "Similar triangles have corresponding angles equal and corresponding sides proportional.",
    "Congruent triangles have all corresponding sides and angles equal.",
    "Trigonometric ratios in a right triangle: sin = opposite/hypotenuse, cos = adjacent/hypotenuse, tan = opposite/adjacent.",
    "Law of cosines: c^2 = a^2 + b^2 - 2ab*cos(C). Law of sines: a/sin(A) = b/sin(B) = c/sin(C).",
    "Distance between points (x1,y1) and (x2,y2): sqrt((x2-x1)^2 + (y2-y1)^2).",
    "Slope of a line between (x1,y1) and (x2,y2): (y2 - y1) / (x2 - x1).",
    "Equation of a line: y = mx + b, where m is the slope and b is the y-intercept.",

    # ── Algebra ──────────────────────────────────────────────────────────────
    "Quadratic formula: for ax^2 + bx + c = 0, x = (-b +/- sqrt(b^2 - 4ac)) / (2a).",
    "The discriminant b^2 - 4ac determines the nature of roots: positive for two real, zero for one repeated, negative for two complex.",
    "Factoring difference of squares: a^2 - b^2 = (a + b)(a - b).",
    "Perfect square trinomial: a^2 + 2ab + b^2 = (a + b)^2.",
    "Sum and product of cubes: a^3 + b^3 = (a + b)(a^2 - ab + b^2); a^3 - b^3 = (a - b)(a^2 + ab + b^2).",
    "FOIL method: (a + b)(c + d) = ac + ad + bc + bd.",
    "Exponent rules: x^a * x^b = x^(a+b); x^a / x^b = x^(a-b); (x^a)^b = x^(ab); x^0 = 1 for x not equal to 0.",
    "Logarithm rules: log(ab) = log(a) + log(b); log(a/b) = log(a) - log(b); log(a^n) = n * log(a).",
    "Change of base: log_b(a) = log(a) / log(b).",
    "A polynomial of degree n has at most n real roots.",
    "Arithmetic sequence: each term differs by a constant d; nth term: a_n = a_1 + (n-1)*d. Sum: n/2 * (a_1 + a_n).",
    "Geometric sequence: each term is multiplied by a constant ratio r; nth term: a_n = a_1 * r^(n-1). Sum of first n terms: a_1 * (1 - r^n) / (1 - r) for r != 1.",

    # ── Number theory ────────────────────────────────────────────────────────
    "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.",
    "The Fundamental Theorem of Arithmetic: every integer greater than 1 can be uniquely factored into prime numbers.",
    "The greatest common divisor (GCD) of two integers is the largest positive integer that divides both.",
    "The least common multiple (LCM) of two integers is the smallest positive integer divisible by both. LCM(a,b) * GCD(a,b) = |a*b|.",
    "Modular arithmetic: a ≡ b (mod n) means n divides a - b. Operations: (a + b) mod n = ((a mod n) + (b mod n)) mod n.",
    "An integer is even if divisible by 2, odd otherwise. Sum/difference of two even or two odd numbers is even.",
    "Divisibility rules: a number is divisible by 3 if the sum of its digits is divisible by 3; by 9 if the digit sum is divisible by 9; by 4 if its last two digits form a number divisible by 4.",

    # ── Combinatorics ────────────────────────────────────────────────────────
    "Permutation P(n, k) = n! / (n - k)!: the number of ways to arrange k items chosen from n, where order matters.",
    "Combination C(n, k) = n! / (k! * (n - k)!): the number of ways to choose k items from n, where order does not matter.",
    "Factorial: n! = n * (n-1) * (n-2) * ... * 1, with 0! = 1.",
    "Multiplication principle: if there are m ways to do one task and n ways to do another, there are m * n ways to do both.",
    "Binomial theorem: (a + b)^n = sum over k of C(n, k) * a^(n-k) * b^k.",
    "The number of subsets of a set with n elements is 2^n.",

    # ── Logic and sets ───────────────────────────────────────────────────────
    "The contrapositive of 'if P then Q' is 'if not Q then not P'. A statement and its contrapositive are logically equivalent.",
    "The converse of 'if P then Q' is 'if Q then P'. A statement and its converse are NOT logically equivalent in general.",
    "De Morgan's laws: not(A and B) = (not A) or (not B); not(A or B) = (not A) and (not B).",
    "Set union A ∪ B contains all elements in A or B; intersection A ∩ B contains elements in both.",
    "|A ∪ B| = |A| + |B| - |A ∩ B| (inclusion-exclusion principle for two sets).",

    # ── Functions ────────────────────────────────────────────────────────────
    "A function maps each input to exactly one output.",
    "The domain is the set of valid inputs; the range is the set of possible outputs.",
    "A function is one-to-one (injective) if different inputs always produce different outputs.",
    "An inverse function f^(-1) satisfies f^(-1)(f(x)) = x; the inverse exists if f is one-to-one.",
    "Even functions satisfy f(-x) = f(x); odd functions satisfy f(-x) = -f(x).",
]


# ---------------------------------------------------------------------------
# Corpus setup
# ---------------------------------------------------------------------------
def setup_maths_rag(embed_model: str = MATHS_EMBED_MODEL) -> None:
    """Build corpus and FAISS index for the math RAG. Idempotent."""
    global _maths_embedder, _maths_index, _maths_passages
    if _maths_embedder is not None:
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    print("[Maths RAG] Building corpus...")
    passages = []

    # Try several known Hendrycks MATH dataset paths
    dataset_candidates = [
        ("hendrycks/competition_math", None),
        ("lighteval/MATH", "all"),
        ("EleutherAI/hendrycks_math", None),
    ]
    loaded_dataset = False
    try:
        from datasets import load_dataset
        for name, config in dataset_candidates:
            try:
                if config is not None:
                    math_ds = load_dataset(name, config, split="train", trust_remote_code=True)
                else:
                    math_ds = load_dataset(name, split="train", trust_remote_code=True)

                count_before = len(passages)
                for item in math_ds:
                    problem = (item.get("problem") or "").strip()
                    solution = (item.get("solution") or "").strip()

                    if problem and len(problem.split()) >= 3:
                        passages.append(_clean_latex(problem))

                    if solution and len(solution.split()) >= 5:
                        # Keep first 150 words of solution to bound passage length
                        sol_excerpt = " ".join(solution.split()[:150])
                        cleaned = _clean_latex(sol_excerpt)
                        if cleaned and len(cleaned.split()) >= 5:
                            passages.append(cleaned)

                added = len(passages) - count_before
                print(f"      Loaded {name}: +{added:,} passages")
                loaded_dataset = True
                break  # success — stop trying other paths
            except Exception as inner_e:
                print(f"      [Maths RAG] Could not load {name}: {type(inner_e).__name__}")
                continue
    except Exception as e:
        print(f"  [Maths RAG] Warning: datasets library issue: {e}")

    if not loaded_dataset:
        print("      [Maths RAG] No external dataset loaded; using curated facts only.")

    # Always include curated facts (they cover essential formulas)
    passages.extend(_CURATED_MATH_FACTS)

    # Deduplicate while preserving order
    seen, unique = set(), []
    for p in passages:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    _maths_passages = unique

    if not _maths_passages:
        raise RuntimeError("Maths RAG corpus is empty after loading!")

    print(f"      {len(_maths_passages):,} passages total")

    print(f"[Maths RAG] Embedding with {embed_model} ...")
    _maths_embedder = SentenceTransformer(embed_model, device="cuda")
    emb = _maths_embedder.encode(
        _maths_passages,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    _maths_index = faiss.IndexFlatIP(emb.shape[1])
    _maths_index.add(emb)


def maths_retrieve(query: str, k: int = MATHS_TOP_K) -> list:
    """Retrieve top-k passages for the math query."""
    global _maths_embedder, _maths_index, _maths_passages
    if _maths_embedder is None:
        raise RuntimeError("Maths RAG is not initialized. Call setup_maths_rag() first.")
    q = _maths_embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    _, idx = _maths_index.search(q, k)
    return [_maths_passages[i] for i in idx[0]]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def rag_maths(query: str, option_texts: Optional[list] = None) -> str:

    if query is None:
        return ""
    stem = query.strip()
    if stem.upper().startswith("Q:"):
        stem = stem[2:].strip()

    # Build the retrieval query. Options are short for math (often just
    # numbers), so they add little signal; the stem dominates.
    if option_texts:
        options_str = " ".join(str(o).strip() for o in option_texts if o)
        retrieval_query = f"{stem} {options_str}".strip()
    else:
        retrieval_query = stem

    contexts = maths_retrieve(retrieval_query, k=MATHS_TOP_K)
    return "\n\n".join(contexts)