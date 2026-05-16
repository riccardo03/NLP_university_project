"""
Pipeline:
  1. Estrae i soggetti dalla domanda (titoli citati, nomi propri).
  2. Per ogni opzione costruisce una query: "<opzione> <soggetto>".
  3. Cerca su Wikipedia (soggetto principale) + DuckDuckGo (per ogni opzione).
  4. Assegna un punteggio a ogni opzione misurando co-occorrenza opzione/soggetto.
  5. Restituisce un contesto strutturato all'LLM.
  6. Se nessuna ricerca produce evidenze → stringa vuota → fallback sul LLM.
"""

import re
import concurrent.futures
import urllib.parse
import requests

# ── Costanti ──────────────────────────────────────────────────────────────────

_WIKI_UA = "QuizBot/1.0 (research)"
_TIMEOUT = 4

_STOP_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "when", "where", "why", "how", "does", "do", "did",
    "has", "have", "had", "will", "would", "could", "should", "can", "may",
    "this", "that", "these", "those", "their", "there", "according", "following",
    "describes", "describe", "best", "most", "called", "named", "own",
    "film", "movie", "song", "show", "album", "band", "role", "character",
    "single", "track", "series", "actor", "actress", "director", "article",
}

_RELATION_VERBS = frozenset({
    "starred", "stars", "starring", "played", "plays", "playing",
    "appeared", "appears", "performed", "performs", "voiced", "voices",
    "portrayed", "portrays", "directed", "directs", "wrote", "writes",
    "written", "produced", "produces", "created", "creates", "recorded",
    "records", "released", "releases", "hosted", "hosts", "featured",
    "features", "sang", "sings", "won", "wins", "nominated", "known",
    "describes", "describe", "called", "named", "is", "was",
})

_QUOTED_RE        = re.compile(r"""['"\u2018\u2019\u201C\u201D]([\w][\w\s,.\-&!]{1,58}?)['"\u2018\u2019\u201C\u201D]""")
_PROPER_MULTI_RE  = re.compile(r'\b[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+(?:\s+[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+)+\b')
_PROPER_SINGLE_RE = re.compile(r'^[A-ZÀ-Ý][a-zA-ZÀ-ÿ]{2,}$')
_TOKEN_RE         = re.compile(r"[a-zA-ZÀ-ÿ0-9$!&]+")
_CITE_RE          = re.compile(r"\[\d+\]")
_SECTION_HEADER   = re.compile(r"^=+\s*[^=]+\s*=+$")

# ── Tokenizzazione & keyword ──────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _keywords(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) >= 3 and t not in _STOP_WORDS}

def _clean_query_text(text: str) -> str:
    kept = [w for w in text.split()
            if w.lower().rstrip(".,!?:;'\"") not in _STOP_WORDS]
    return " ".join(kept) if kept else text

# ── Estrazione soggetti ───────────────────────────────────────────────────────

def _extract_subjects(question: str) -> list[str]:
    subjects: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        s = raw.strip()
        sl = s.lower()
        if s and sl not in seen and sl not in _STOP_WORDS:
            seen.add(sl)
            subjects.append(s)

    # 1. Titoli tra virgolette
    for q in _QUOTED_RE.findall(question):
        _add(q)

    # 2. Nomi propri multi-parola 
    for m in _PROPER_MULTI_RE.findall(question):
        _add(m)

    # 3. Nomi propri singoli
    words = question.split()
    for i, w in enumerate(words):
        if i == 0:
            continue
        clean = re.sub(r"[^\w]+$", "", w)
        if _PROPER_SINGLE_RE.match(clean) and clean.lower() not in seen:
            if not any(clean in s.split() for s in subjects):
                _add(clean)

    return subjects


# ── Ricerca: Wikipedia ────────────────────────────────────────────────────────

def _wiki_lookup(query: str) -> str:
    """OpenSearch → primo titolo non-ambiguo → estratto pulito."""
    try:
        url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=opensearch&search={urllib.parse.quote(query)}&limit=3&format=json"
        )
        r = requests.get(url, headers={"User-Agent": _WIKI_UA}, timeout=_TIMEOUT)
        if r.status_code != 200:
            return ""
        candidates = r.json()[1]
        if not candidates:
            return ""
        title = next(
            (c for c in candidates if "disambiguation" not in c.lower()),
            candidates[0],
        )
        url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&exintro=false&explaintext=true"
            f"&titles={urllib.parse.quote(title)}&format=json"
        )
        r = requests.get(url, headers={"User-Agent": _WIKI_UA}, timeout=_TIMEOUT)
        if r.status_code != 200:
            return ""
        pages = r.json()["query"]["pages"]
        text = pages[next(iter(pages))].get("extract", "")
        text = _CITE_RE.sub("", text)
        return text if "may refer to:" not in text.lower() else ""
    except Exception:
        return ""

def _wiki_relevant_passages(wiki_text: str, question: str,
                            max_chars: int = 1500) -> str:

    if not wiki_text:
        return ""
 
    paragraphs = [
        p.strip() for p in re.split(r"\n+", wiki_text)
        if len(p.strip()) > 50 and not _SECTION_HEADER.match(p.strip())
    ]
    if not paragraphs:
        return wiki_text[:max_chars]
 
    q_kws = _keywords(question)
    if not q_kws:
        return paragraphs[0][:max_chars]
 
    intro = paragraphs[0]
    rest  = paragraphs[1:]
 
    scored = [(len(q_kws & set(_tokenize(p))), p) for p in rest]
    scored.sort(key=lambda x: -x[0])
 
    out: list[str] = [intro]
    budget = max_chars - len(intro)
    for score, p in scored:
        if score < 2 or budget <= 100:
            break
        snippet = p if len(p) <= budget else p[:budget].rsplit(" ", 1)[0] + "…"
        out.append(snippet)
        budget -= len(snippet) + 2
 
    return "\n\n".join(out)


# ── Ricerca: DuckDuckGo ───────────────────────────────────────────────────────

def _ddg_lookup(query: str, max_results: int = 2) -> list[str]:
    """Lista di snippet (titolo + body) per la query data."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            out = []
            for r in ddgs.text(query, max_results=max_results, timeout=_TIMEOUT):
                body  = r.get("body", "")
                title = r.get("title", "")
                if body and len(body) >= 60:
                    out.append(f"{title}. {body}" if title else body)
            return out
    except Exception as e:
        print(f"  [RAG] DDG error: {e}")
        return []


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_option(option: str, subjects: list[str], snippets: list[str]) -> float:

    opt_kws = _keywords(option)
    if not opt_kws:
        return 0.0

    subj_kws: set[str] = set()
    for s in subjects:
        subj_kws |= _keywords(s)

    total = 0.0
    for snip in snippets:
        if not snip:
            continue
        words = _tokenize(snip)
        wset  = set(words)

        opt_hits = opt_kws & wset
        if not opt_hits:
            continue
        if subj_kws and not (subj_kws & wset):
            continue

        coverage  = len(opt_hits) / len(opt_kws)       
        verb_bonus = min(sum(1 for w in words if w in _RELATION_VERBS) * 0.25, 1.0)
        total += coverage + verb_bonus

    return total


# ── Pipeline principale ───────────────────────────────────────────────────────

def rag_entertainment(query: str, num_results: int = 3,
                      generate_answer_fn=None, option_texts: list = None) -> str:

    subjects = _extract_subjects(query)
    print(f"  [RAG] Subjects: {subjects or '(none)'}")

    if subjects:
        main_term = subjects[0]
    else:
        kws = [w for w in _tokenize(query) if len(w) >= 4 and w not in _STOP_WORDS]
        main_term = " ".join(kws[:4]) if kws else query[:60]

    subj_str = " ".join(subjects[:2]) if subjects else main_term

    if not option_texts:
        wiki_full = _wiki_lookup(main_term)
        wiki = _wiki_relevant_passages(wiki_full, query, max_chars=1200)
        ddg  = _ddg_lookup(subj_str, num_results)
        return "\n\n".join(([wiki] if wiki else []) + ddg)[:1500]
 
    n_opts = min(len(option_texts), 4)

    cand_queries = [
        f"{option_texts[i].strip()[:50]} {subj_str}".strip()[:90]
        for i in range(n_opts)
    ]

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_opts + 2)
    wiki_fut    = pool.submit(_wiki_lookup, main_term)
    general_fut = pool.submit(_ddg_lookup, f"{subj_str} {query[:50]}".strip()[:90],
                              num_results)
    opt_futs    = [pool.submit(_ddg_lookup, q, 2) for q in cand_queries]

    def _safe(fut, default):
        try:
            return fut.result(timeout=_TIMEOUT + 1)
        except Exception:
            return default

    wiki_full    = _safe(wiki_fut, "")
    general_snip = _safe(general_fut, [])
    opt_snips    = [_safe(f, []) for f in opt_futs]
    pool.shutdown(wait=False)

    wiki_text = _wiki_relevant_passages(wiki_full, query, max_chars=1400)

    shared: list[str] = []
    seen_key: set[str] = set()
    if wiki_text:
        shared.append(wiki_text)
        seen_key.add(wiki_text[:120])
    for s in general_snip:
        k = s[:120]
        if k not in seen_key:
            shared.append(s)
            seen_key.add(k)

    # Scoring per ogni opzione 
    scores: dict[int, float] = {}
    for i in range(n_opts):
        scores[i] = _score_option(option_texts[i], subjects,
                                  opt_snips[i] + shared)

    ranked = sorted(range(n_opts), key=lambda i: scores[i], reverse=True)
    print(f"  [RAG] Scores: { {i: round(scores[i], 1) for i in ranked} }")

    # ── Fallback su conoscenza LLM se nessuna evidenza ────────────────────────
    if not shared and all(s == 0.0 for s in scores.values()):
        print("  [RAG] No evidence → LLM fallback")
        return ""

    parts: list[str] = []
    if wiki_text:
        parts.append(f"WIKIPEDIA (key passages):\n{wiki_text}")
 
    # Marca esplicitamente il vincitore: contrasta l'allucinazione del LLM
    top_idx = ranked[0]
    top_score = scores[top_idx]
    has_clear_winner = top_score > 0 and (
        len(ranked) < 2 or top_score >= scores[ranked[1]] * 1.5
    )

    for i in ranked:
        marker = " ★ STRONGEST EVIDENCE" if (i == top_idx and has_clear_winner) else ""
        label  = f"[{i}] {option_texts[i]} (score {scores[i]:.1f}){marker}"
        if opt_snips[i]:
            parts.append(f"{label}:\n{opt_snips[i][0][:350]}")
        else:
            parts.append(f"{label}: (no specific evidence)")
 
    return "\n\n".join(parts)[:2500]