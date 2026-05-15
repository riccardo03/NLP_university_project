"""
Mathematical formula cache for quick lookup.
Covers high-frequency quiz topics across algebra, geometry, calculus, probability.
"""

# High-value formulas indexed by category and keywords
_FORMULA_CACHE = {
    # Algebra
    "quadratic formula": {
        "formula": "x = (-b ± √(b²-4ac)) / 2a",
        "context": "Solves quadratic equations ax² + bx + c = 0",
        "category": "Algebra",
        "keywords": ["quadratic", "equation", "roots", "solutions"],
    },
    "sum arithmetic sequence": {
        "formula": "S_n = n(a₁ + aₙ) / 2 = n(2a₁ + (n-1)d) / 2",
        "context": "Sum of arithmetic progression with first term a₁, common difference d, n terms",
        "category": "Algebra",
        "keywords": ["arithmetic", "series", "sum", "sequence", "progression"],
    },
    "sum geometric sequence": {
        "formula": "S_n = a(1 - rⁿ) / (1 - r)  [r ≠ 1]",
        "context": "Sum of geometric series with first term a, common ratio r, n terms",
        "category": "Algebra",
        "keywords": ["geometric", "series", "sum", "sequence", "ratio"],
    },
    "binomial expansion": {
        "formula": "(a + b)ⁿ = Σ C(n,k) aⁿ⁻ᵏ bᵏ",
        "context": "Binomial theorem expansion; C(n,k) is binomial coefficient",
        "category": "Algebra",
        "keywords": ["binomial", "expansion", "theorem", "coefficient"],
    },
    
    # Geometry
    "pythagorean theorem": {
        "formula": "a² + b² = c²",
        "context": "Right triangle with legs a, b and hypotenuse c",
        "category": "Geometry",
        "keywords": ["pythagorean", "right triangle", "hypotenuse", "legs"],
    },
    "circle area": {
        "formula": "A = πr²",
        "context": "Area of circle with radius r",
        "category": "Geometry",
        "keywords": ["circle", "area", "radius"],
    },
    "circle circumference": {
        "formula": "C = 2πr = πd",
        "context": "Circumference of circle with radius r or diameter d",
        "category": "Geometry",
        "keywords": ["circle", "circumference", "perimeter", "radius", "diameter"],
    },
    "sphere volume": {
        "formula": "V = (4/3)πr³",
        "context": "Volume of sphere with radius r",
        "category": "Geometry",
        "keywords": ["sphere", "volume", "radius"],
    },
    "sphere surface area": {
        "formula": "A = 4πr²",
        "context": "Surface area of sphere with radius r",
        "category": "Geometry",
        "keywords": ["sphere", "surface area", "radius"],
    },
    "cylinder volume": {
        "formula": "V = πr²h",
        "context": "Volume of cylinder with radius r and height h",
        "category": "Geometry",
        "keywords": ["cylinder", "volume", "radius", "height"],
    },
    "cone volume": {
        "formula": "V = (1/3)πr²h",
        "context": "Volume of cone with radius r and height h",
        "category": "Geometry",
        "keywords": ["cone", "volume", "radius", "height"],
    },
    
    # Trigonometry
    "sin cos tan": {
        "formula": "sin(θ) = opposite/hypotenuse, cos(θ) = adjacent/hypotenuse, tan(θ) = opposite/adjacent",
        "context": "Basic trigonometric ratios in right triangle",
        "category": "Geometry",
        "keywords": ["sine", "cosine", "tangent", "trigonometric", "ratio"],
    },
    "law of sines": {
        "formula": "a/sin(A) = b/sin(B) = c/sin(C)",
        "context": "Relates sides and angles in any triangle",
        "category": "Geometry",
        "keywords": ["law", "sine", "triangle", "sides", "angles"],
    },
    
    # Calculus
    "power rule derivative": {
        "formula": "d/dx(xⁿ) = n·xⁿ⁻¹",
        "context": "Derivative of power function",
        "category": "Calculus",
        "keywords": ["derivative", "power", "rule"],
    },
    "chain rule": {
        "formula": "d/dx[f(g(x))] = f'(g(x)) · g'(x)",
        "context": "Derivative of composite function",
        "category": "Calculus",
        "keywords": ["derivative", "chain", "rule", "composite"],
    },
    "product rule": {
        "formula": "d/dx[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)",
        "context": "Derivative of product of functions",
        "category": "Calculus",
        "keywords": ["derivative", "product", "rule"],
    },
    
    # Probability & Statistics
    "combination": {
        "formula": "C(n,k) = n! / (k!(n-k)!)",
        "context": "Number of ways to choose k items from n items (order doesn't matter)",
        "category": "Probability",
        "keywords": ["combination", "binomial", "coefficient", "choose"],
    },
    "permutation": {
        "formula": "P(n,k) = n! / (n-k)!",
        "context": "Number of ways to arrange k items from n items (order matters)",
        "category": "Probability",
        "keywords": ["permutation", "arrangement", "ordered"],
    },
    "factorial": {
        "formula": "n! = n × (n-1) × (n-2) × ... × 1",
        "context": "Product of all positive integers ≤ n; 0! = 1",
        "category": "Probability",
        "keywords": ["factorial", "product"],
    },
    
    # Constants
    "golden ratio": {
        "formula": "φ = (1 + √5) / 2 ≈ 1.618",
        "context": "Appears in geometry, nature, art",
        "category": "Constants",
        "keywords": ["golden", "ratio", "phi"],
    },
    "euler number": {
        "formula": "e ≈ 2.71828",
        "context": "Base of natural logarithm",
        "category": "Constants",
        "keywords": ["euler", "e", "natural", "logarithm"],
    },
    "pi": {
        "formula": "π ≈ 3.14159",
        "context": "Ratio of circle circumference to diameter",
        "category": "Constants",
        "keywords": ["pi", "circle", "constant"],
    },
    
    # Statistics & Hypothesis Testing
    "type I error multiple tests": {
        "formula": "P(reject H₀ at least once) = 1 - (1-α)ᵏ",
        "context": "Probability of Type I error in k independent tests with significance level α. For α=0.05, k=10: P ≈ 1-(0.95)¹⁰ ≈ 0.40",
        "category": "Probability",
        "keywords": ["type I", "error", "multiple", "tests", "significance", "independent"],
    },
    "quartile range": {
        "formula": "IQR = Q₃ - Q₁; Middle 50% ≈ mean ± 0.674σ (for normal distribution)",
        "context": "Interquartile range (IQR) contains middle 50% of data. For normal distribution, Q₁≈μ-0.674σ, Q₃≈μ+0.674σ",
        "category": "Probability",
        "keywords": ["quartile", "IQR", "interquartile", "middle 50%", "Q1", "Q3"],
    },
    "z-score": {
        "formula": "z = (x - μ) / σ",
        "context": "Standardized score showing how many standard deviations x is from mean μ. Used in normal distribution",
        "category": "Probability",
        "keywords": ["z-score", "standard", "normal", "distribution"],
    },
    "two-sample t-test": {
        "formula": "t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)",
        "context": "Tests if means of two independent samples are significantly different. Use when population std dev unknown and sample size small (n<30)",
        "category": "Probability",
        "keywords": ["t-test", "two-sample", "independent", "means", "comparison"],
    },
    "two-sample z-test": {
        "formula": "z = (x̄₁ - x̄₂) / √(σ₁²/n₁ + σ₂²/n₂)",
        "context": "Tests if means of two independent samples are significantly different. Use when population std dev is known or sample size large (n≥30)",
        "category": "Probability",
        "keywords": ["z-test", "two-sample", "independent", "means", "large sample"],
    },
    "survey response bias": {
        "formula": "Use observed sample size (n) not original sample size for analysis when non-response occurs",
        "context": "When non-response occurs in surveys, use the actual number of respondents as sample size (n=88, not 120) to avoid bias",
        "category": "Probability",
        "keywords": ["survey", "non-response", "bias", "sample size", "respondents"],
    },
}


def search_formula(query: str) -> dict | None:
    """
    Search formula cache by query keywords.
    Returns matching formula dict or None.
    
    Only returns HIGH-CONFIDENCE matches to avoid false positives.
    """
    query_lower = query.lower()
    
    # Exact match first
    if query_lower in _FORMULA_CACHE:
        return _FORMULA_CACHE[query_lower]
    
    # Keyword matching with HIGHER confidence threshold
    best_match = None
    best_score = 0
    
    for name, data in _FORMULA_CACHE.items():
        score = 0
        keywords = data.get("keywords", [])
        
        # Match against formula name (strong signal)
        name_words = name.lower().split()
        matching_words = sum(1 for word in query_lower.split() if word in name_words)
        score += matching_words * 5  # Weighted: name matches are strong
        
        # Match against keywords (medium signal)
        for kw in keywords:
            if kw in query_lower:
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = data
    
    # Only return if GOOD confidence (score >= 3, not >= 1)
    # This prevents random matches like "euler number" for "green balls"
    # but still allows legitimate formula lookups
    return best_match if best_score >= 3 else None


def get_formula_by_category(category: str) -> list[dict]:
    """Get all formulas in a specific category."""
    return [
        data for data in _FORMULA_CACHE.values()
        if data.get("category", "").lower() == category.lower()
    ]


def format_formula_context(formula_data: dict) -> str:
    """Format formula data into readable context string."""
    if not formula_data:
        return ""
    
    formula = formula_data.get("formula", "")
    context = formula_data.get("context", "")
    return f"Formula: {formula}\n{context}" if formula else ""
