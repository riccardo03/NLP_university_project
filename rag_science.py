"""
RAG pipeline for the Science & Nature competition.
Single LLM strategy call → Wikipedia primary → DuckDuckGo fallback → keyword reranking.
"""

import re

_ARTICLE_REF_RE = re.compile(
    r"\b(according to (the )?(article|text|passage|paragraph|excerpt))\b",
    re.I,
)

_SCIENCE_STRATEGY_SYSTEM = (
    "You are a search strategist for a science quiz bot. "
    "Given a question, decide the retrieval action and output a search query.\n"
    "\n"
    "Output ONE line with this EXACT format (no other text):\n"
    "DECISION: <ACTION> | QUERY: <3-6 technical keywords>\n"
    "\n"
    "ACTION must be exactly one of:\n"
    "  SEARCH     — specific scientific fact, experiment result, historical discovery, "
    "named constant, organism, chemical compound, law, or person.\n"
    "  PARAMETRIC — general definition or well-known fact the model already knows "
    "(e.g. 'What is the speed of light?', 'What is photosynthesis?').\n"
    "  SKIP       — question refers to a specific text/article/passage not available online.\n"
    "\n"
    "Rules for QUERY:\n"
    "- Use technical terminology: chemical names, scientific names, proper nouns, units.\n"
    "- 3 to 6 words maximum. No punctuation at end. No filler words.\n"
    "- Even for PARAMETRIC and SKIP, provide a best-effort query.\n"
    "\n"
    "Examples:\n"
    "Q: What is the atomic number of gold? → DECISION: SEARCH | QUERY: gold atomic number element\n"
    "Q: What is the process of photosynthesis? → DECISION: PARAMETRIC | QUERY: photosynthesis process chlorophyll\n"
    "Q: According to the passage, what did Darwin observe? → DECISION: SKIP | QUERY: Darwin natural selection observation\n"
    "Q: What is the boiling point of water at 1 atm? → DECISION: PARAMETRIC | QUERY: water boiling point 100 celsius\n"
    "Q: Which enzyme catalyses the first step of glycolysis? → DECISION: SEARCH | QUERY: glycolysis first enzyme hexokinase\n"
    "Q: Who discovered penicillin? → DECISION: SEARCH | QUERY: penicillin discovery Alexander Fleming\n"
)

_SCIENCE_STOP_WORDS = {
    "what", "when", "which", "where", "does", "have", "this", "that",
    "from", "with", "about", "into", "their", "there", "been", "were",
    "would", "could", "should", "according", "following", "describe",
}

_SPAM_DOMAINS = re.compile(
    r'(quora\.com|reddit\.com|yahoo\.com|answers\.com|wikihow\.com'
    r'|stackexchange\.com|chegg\.com|coursehero\.com|pinterest\.com)',
    re.I,
)
_PREFERRED_DOMAINS = re.compile(
    r'(\.edu|\.gov|wikipedia\.org|britannica\.com|nature\.com'
    r'|sciencedirect\.com|pubmed\.ncbi\.nlm\.nih\.gov|newscientist\.com'
    r'|scientificamerican\.com|khanacademy\.org)',
    re.I,
)

# Patterns for scientific entities that should trigger exact matching
_NUMBER_RE = re.compile(r'\b\d+\.?\d*\b')
_CHEM_FORMULA_RE = re.compile(r'\b([A-Z][a-z]?\d*){1,6}\b')


def _extract_science_keywords(text: str) -> list[str]:
    keywords = []
    # Words longer than 5 letters (lowercase)
    for w in re.findall(r'\b[a-z]{6,}\b', text.lower()):
        if w not in _SCIENCE_STOP_WORDS:
            keywords.append(w)
    # ALL-CAPS tokens (acronyms, element symbols, units)
    for w in re.findall(r'\b[A-Z]{2,}\b', text):
        keywords.append(w.lower())
    # 4-digit years
    for w in re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', text):
        keywords.append(w)
    # Numeric constants and measurements (e.g. 9.81, 273.15, 6.022)
    for w in re.findall(r'\b\d+\.\d+\b', text):
        keywords.append(w)
    return list(dict.fromkeys(keywords))


def _is_relevant_science(snippet: str, question: str) -> bool:
    """
    Relevance check tuned for scientific precision.
    A snippet qualifies if it satisfies ANY of:
      1. Contains a numeric constant/measurement from the question.
      2. Contains a chemical formula or element symbol from the question.
      3. Shares at least 2 long (>5 chars) scientific keywords with the question.
    """
    snippet_lower = snippet.lower()

    # 1. Numeric constants exact match
    question_numbers = _NUMBER_RE.findall(question)
    for num in question_numbers:
        if len(num) >= 2 and num in snippet:
            return True

    # 2. Chemical formula / element symbol match
    chem_tokens = _CHEM_FORMULA_RE.findall(question)
    for token in chem_tokens:
        if isinstance(token, str) and len(token) >= 2 and token in snippet:
            return True

    # 3. Shared long scientific keywords
    kws = _extract_science_keywords(question)
    if not kws:
        return True
    matches = sum(1 for kw in kws if kw in snippet_lower)
    return matches >= 2


def _wiki_science(query: str) -> str:
    """
    Fetch a Wikipedia page; returns '' on failure.
    Limits output to summary + first 2 dense paragraphs, max 2200 chars.
    """
    try:
        import wikipedia
        wikipedia.set_lang("en")
        try:
            page = wikipedia.page(query, auto_suggest=False)
            summary = wikipedia.summary(query, sentences=3, auto_suggest=False)

            # Extract only substantive paragraphs, skip section headers and stubs
            skip_sections = re.compile(
                r'^(voci correlate|see also|references|bibliography|notes|'
                r'external links|further reading|== )', re.I
            )
            paragraphs = []
            for p in page.content.split("\n"):
                p = p.strip()
                if len(p) > 120 and not skip_sections.match(p):
                    paragraphs.append(p)
                if len(paragraphs) >= 2:
                    break

            combined = summary
            if paragraphs:
                combined += "\n\n" + "\n\n".join(paragraphs)
            return combined[:2200]

        except wikipedia.exceptions.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=3, auto_suggest=False)
        except wikipedia.exceptions.PageError:
            return ""
    except Exception:
        return ""


def _wiki_is_useful(text: str) -> bool:
    if not text or len(text.strip()) < 200:
        return False
    if "may refer to:" in text.lower():
        return False
    return True


def _parse_strategy(raw: str) -> tuple[str, str]:
    """
    Parse 'DECISION: <ACTION> | QUERY: <value>' from LLM output.
    Returns (decision, query). Defaults to ('SEARCH', '') on failure.
    """
    raw = raw.strip()
    decision = "SEARCH"
    query = ""

    dec_m = re.search(r'DECISION\s*:\s*(SEARCH|PARAMETRIC|SKIP)', raw, re.I)
    qry_m = re.search(r'QUERY\s*:\s*(.+)', raw, re.I)

    if dec_m:
        decision = dec_m.group(1).upper()
    if qry_m:
        query = qry_m.group(1).strip().strip('"').strip("'")

    return decision, query


def _get_strategy(question: str, generate_answer_fn) -> tuple[str, str]:
    """Single LLM call returning (decision, ddg_query)."""
    try:
        raw = generate_answer_fn(
            _SCIENCE_STRATEGY_SYSTEM,
            f"Q: {question}",
            max_new_tokens=30,
        )
        print(f"  [RAG-Science] Strategy raw: {raw[:120]!r}")
        decision, query = _parse_strategy(raw)
        print(f"  [RAG-Science] Decision={decision}, Query={query!r}")
        return decision, query
    except Exception as e:
        print(f"  [RAG-Science] Strategy call failed: {e}")
        return "SEARCH", question[:80]


def _ddg_science(query: str, num_results: int = 3) -> list[str]:
    """DuckDuckGo search filtered to trusted scientific domains."""
    snippets = []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results + 4, timeout=4):
                url = r.get("href", "")
                body = r.get("body", "")
                title = r.get("title", "")

                if not body or len(body.strip()) < 80:
                    continue
                if _SPAM_DOMAINS.search(url):
                    continue

                # Prefer trusted domains but don't exclude others entirely
                text = f"[{title}] {body}" if title else body
                if _PREFERRED_DOMAINS.search(url):
                    snippets.insert(0, text)
                else:
                    snippets.append(text)

                if len(snippets) >= num_results:
                    break
    except Exception as exc:
        print(f"  [RAG-Science] DDG failed: {exc}")
    return snippets


def rag_science(question_text: str, option_texts: list = None,
                generate_answer_fn=None) -> str:
    """
    Science RAG: single LLM strategy → Wikipedia primary → DDG fallback → keyword filter.
    """
    if option_texts is None:
        option_texts = []

    # Strip article-reference preambles before deciding
    clean_q = _ARTICLE_REF_RE.sub("", question_text).strip(" ,;")
    if clean_q != question_text:
        print(f"  [RAG-Science] Article-reference stripped.")

    # ------------------------------------------------------------------ #
    # Stage 1: single fused LLM call                                      #
    # ------------------------------------------------------------------ #
    if generate_answer_fn is not None:
        decision, search_query = _get_strategy(clean_q, generate_answer_fn)
    else:
        decision, search_query = "SEARCH", clean_q[:80]

    if decision == "SKIP":
        print("  [RAG-Science] SKIP decision — article-reference question, no search.")
        return ""

    if not search_query or len(search_query) < 3:
        search_query = clean_q[:80]

    # For PARAMETRIC, still search — the model's answer benefits from confirmation
    print(f"  [RAG-Science] Searching: {search_query!r} (decision={decision})")

    # ------------------------------------------------------------------ #
    # Stage 2: Wikipedia primary                                          #
    # ------------------------------------------------------------------ #
    wiki_text = _wiki_science(search_query)
    wiki_useful = _wiki_is_useful(wiki_text)

    snippets: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            snippets.append(text)

    if wiki_useful and len(wiki_text.strip()) >= 400:
        print(f"  [RAG-Science] Wikipedia hit ({len(wiki_text)} chars).")
        _add(wiki_text)
    else:
        if wiki_text:
            print(f"  [RAG-Science] Wikipedia too short ({len(wiki_text)} chars), trying DDG.")
            _add(wiki_text)  # keep as weak signal / fail-safe
        else:
            print(f"  [RAG-Science] Wikipedia miss, falling back to DDG.")

        ddg_snippets = _ddg_science(search_query, num_results=3)
        for s in ddg_snippets:
            _add(s)

    if not snippets:
        return ""

    # ------------------------------------------------------------------ #
    # Stage 3: relevance filter with fail-safe                            #
    # ------------------------------------------------------------------ #
    relevant = [s for s in snippets if _is_relevant_science(s, clean_q)]
    if relevant:
        context_snippets = relevant
    else:
        # fail-safe: never return empty when we have encyclopedic content
        print(f"  [RAG-Science] No relevant snippets — fail-safe: returning best-effort.")
        if wiki_useful and wiki_text:
            context_snippets = [wiki_text]
        else:
            context_snippets = snippets[:2]

    return "\n\n".join(context_snippets)[:3000]
