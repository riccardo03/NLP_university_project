"""Entertainment RAG: GLiNER entity extraction → semantic queries → Wikipedia + DDG → BM25 vote."""

import re
import concurrent.futures
import urllib.parse
import requests
from dataclasses import dataclass
from functools import lru_cache
from rank_bm25 import BM25Okapi

_WIKI_UA = "QuizBot/1.0 (research)"
_TIMEOUT = 4

_STOP_WORDS_BASE: frozenset[str] = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "when", "where", "why", "how", "does", "do", "did",
    "has", "have", "had", "will", "would", "could", "should", "can", "may",
    "this", "that", "these", "those", "their", "there", "according", "following",
    "describes", "describe", "best", "most", "called", "named", "own", "article",
})
_DOMAIN_TERMS: frozenset[str] = frozenset({
    "film", "movie", "song", "show", "album", "band", "role",
    "character", "single", "track", "series", "actor", "actress", "director",
})
_STOP_WORDS:       frozenset[str] = _STOP_WORDS_BASE | _DOMAIN_TERMS
_STOP_WORDS_QUERY: frozenset[str] = _STOP_WORDS_BASE

_GLINER_MODEL_NAME = "urchade/gliner_medium-v2.1"
_GLINER_LABELS = [
    "movie", "film", "TV show", "TV series", "person", "actor", "musician",
    "director", "band", "music group", "album", "song", "family member",
    "historical person", "character",
]
_GLINER_LABEL_PRIORITY: dict[str, int] = {
    "movie": 0, "film": 0, "TV show": 0, "TV series": 0,
    "album": 1, "song": 1,
    "person": 2, "actor": 2, "musician": 2, "director": 2,
    "band": 2, "music group": 2, "family member": 2, "historical person": 2,
    "character": 3,
}
_TITLE_LABELS  = frozenset({"movie", "film", "TV show", "TV series"})
_PERSON_LABELS = frozenset({
    "person", "actor", "musician", "director", "band",
    "music group", "family member", "historical person",
})

_QUOTED_RE        = re.compile(r"""['\"''""]([\w][\w\s,.\-&!]{1,58}?)['\"''""]""")
_PROPER_MULTI_RE  = re.compile(r'\b[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+(?:\s+[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+)+\b')
_PROPER_SINGLE_RE = re.compile(r'^[A-ZÀ-Ý][a-zA-ZÀ-ÿ]{2,}$')
_TOKEN_RE         = re.compile(r"[a-zA-ZÀ-ÿ0-9$!&]+")
_CITE_RE          = re.compile(r"\[\d+\]")
_SECTION_HEADER   = re.compile(r"^=+\s*[^=]+\s*=+$")
_NOT_EXCEPT_RE    = re.compile(r'\b(NOT|EXCEPT|least|never|false|incorrect|wrong)\b', re.IGNORECASE)
_BIO_KEYWORDS_RE  = re.compile(
    r"\b(father|mother|parent|sibling|brother|sister|wife|husband|"
    r"relation|relationship|childhood|born|raised|family|married|"
    r"early life|upbringing|ancestry)\b", re.IGNORECASE,
)

# Tokens carrying negation; contracted forms like "didn't" → ["didn","t"] are excluded.
_NEGATION_TOKENS: frozenset[str] = frozenset({
    "not", "never", "no", "nor", "neither",
    "denied", "failed", "refused", "unrelated", "unconnected",
})
_VOTE_CONFIDENCE_MIN: float = 0.5

_gliner_model: object = None


@dataclass(frozen=True)
class WeightedQuery:
    text:     str
    weight:   float
    strategy: str


def setup_entertainment_rag() -> None:
    global _gliner_model
    print("  [RAG-Entertainment] Loading GLiNER model...")
    try:
        from gliner import GLiNER
        _gliner_model = GLiNER.from_pretrained(_GLINER_MODEL_NAME)
        print("  [RAG-Entertainment] GLiNER ready.")
    except Exception as e:
        print(f"  [RAG-Entertainment] GLiNER unavailable: {e}")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _keywords(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) >= 3 and t not in _STOP_WORDS}


def _clean_query_text(text: str) -> str:
    kept = [w for w in text.split() if w.lower().rstrip(".,!?:;'\"") not in _STOP_WORDS_QUERY]
    return " ".join(kept) if kept else text


def _extract_subjects_regex(question: str) -> list[str]:
    seen: set[str] = set()
    out:  list[str] = []

    def _add(s: str) -> None:
        s = s.strip()
        if s and s.lower() not in seen and s.lower() not in _STOP_WORDS:
            seen.add(s.lower()); out.append(s)

    for q in _QUOTED_RE.findall(question): _add(q)
    for m in _PROPER_MULTI_RE.findall(question): _add(m)
    for w in question.split()[1:]:
        c = re.sub(r"[^\w]+$", "", w)
        if _PROPER_SINGLE_RE.match(c) and c.lower() not in seen and not any(c in s.split() for s in out):
            _add(c)
    return out


def _extract_subjects_gliner(question: str) -> list[tuple[str, str]]:
    if _gliner_model is None:
        return []
    try:
        entities = _gliner_model.predict_entities(question, _GLINER_LABELS, threshold=0.45)
        # Lower threshold for person-type labels to catch family members.
        person_labels = ["person", "actor", "musician", "director", "family member", "historical person"]
        secondary     = _gliner_model.predict_entities(question, person_labels, threshold=0.30)
        primary_texts = {e["text"].strip().lower() for e in entities}
        entities     += [e for e in secondary if e["text"].strip().lower() not in primary_texts]
        entities.sort(key=lambda e: (_GLINER_LABEL_PRIORITY.get(e["label"], 99), e["start"]))
        seen: set[str] = set()
        result: list[tuple[str, str]] = []
        for e in entities:
            text, tl = e["text"].strip(), e["text"].strip().lower()
            if text and tl not in seen and tl not in _STOP_WORDS:
                seen.add(tl); result.append((text, e["label"]))
        return result
    except Exception:
        return []


def _extract_option_entity(option: str) -> str:
    m = _QUOTED_RE.search(option)
    if m:
        return m.group(1).strip()
    m = re.search(r'\b(19|20)\d{2}s?\b', option)
    if m:
        return m.group(0)
    m = _PROPER_MULTI_RE.search(option)
    if m:
        return m.group(0)
    return ""


def _pick_main_term(labeled: list[tuple[str, str]]) -> str:
    persons = [t for t, l in labeled if l in _PERSON_LABELS]
    titles  = [t for t, l in labeled if l in _TITLE_LABELS]
    # Prefer person as anchor when there are multiple titles (disambiguation).
    if persons and len(titles) > 1:
        return persons[0]
    for preferred in (_TITLE_LABELS, _PERSON_LABELS):
        for text, label in labeled:
            if label in preferred:
                return text
    return labeled[0][0] if labeled else ""


def _pick_wiki_anchors(labeled: list[tuple[str, str]]) -> list[str]:
    titles  = [t for t, l in labeled if l in _TITLE_LABELS]
    persons = [t for t, l in labeled if l in _PERSON_LABELS]
    anchors = (titles[:1] + persons[:1]) or ([labeled[0][0]] if labeled else [])
    if len(anchors) < 2:
        for text, _ in labeled:
            if text not in anchors:
                anchors.append(text); break
    return anchors[:2]


@lru_cache(maxsize=64)
def _wiki_lookup(query: str) -> str:
    try:
        ua, q = {"User-Agent": _WIKI_UA}, urllib.parse.quote(query)
        r = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={q}&srlimit=3&srnamespace=0&format=json",
            headers=ua, timeout=_TIMEOUT,
        )
        if r.status_code != 200: return ""
        results = r.json().get("query", {}).get("search", [])
        if not results: return ""
        candidates = [item["title"] for item in results]
        title = next((c for c in candidates if "disambiguation" not in c.lower()), candidates[0])
        r = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            f"&exintro=false&explaintext=true&exchars=8000"
            f"&titles={urllib.parse.quote(title)}&format=json",
            headers=ua, timeout=_TIMEOUT + 2,
        )
        if r.status_code != 200: return ""
        text = next(iter(r.json()["query"]["pages"].values())).get("extract", "")
        text = _CITE_RE.sub("", text)
        return text if "may refer to:" not in text.lower() else ""
    except Exception:
        return ""


def _wiki_relevant_passages(wiki_text: str, question: str, max_chars: int = 1500) -> str:
    if not wiki_text:
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n+", wiki_text)
                  if len(p.strip()) > 50 and not _SECTION_HEADER.match(p.strip())]
    if not paragraphs:
        return wiki_text[:max_chars]
    q_kws = _keywords(question)
    if not q_kws:
        return paragraphs[0][:max_chars]
    intro  = paragraphs[0]
    scored = sorted(((len(q_kws & set(_tokenize(p))), p) for p in paragraphs[1:]),
                    key=lambda x: x[0], reverse=True)
    out, budget = [intro], max_chars - len(intro)
    for score, p in scored:
        if score < 2 or budget <= 100:
            break
        snippet = p if len(p) <= budget else p[:budget].rsplit(" ", 1)[0] + "..."
        out.append(snippet); budget -= len(snippet) + 2
    return "\n\n".join(out)


def _fetch_multi_wiki(anchors: list[str], question: str, max_chars: int = 600) -> str:
    if not anchors:
        return ""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        raw = dict(zip(anchors, ex.map(_wiki_lookup, anchors)))
    parts = [
        f"[WIKI: {a}]\n{p}"
        for a in anchors
        if (p := _wiki_relevant_passages(raw.get(a, ""), question, max_chars=max_chars))
    ]
    return "\n\n".join(parts)


@lru_cache(maxsize=128)
def _ddg_lookup(query: str, max_results: int = 2) -> tuple[str, ...]:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return tuple(
                f"{r['title']}. {r['body']}" if r.get("title") else r["body"]
                for r in ddgs.text(query, max_results=max_results, timeout=_TIMEOUT)
                if len(r.get("body", "")) >= 60
            )
    except Exception as e:
        print(f"  [RAG] DDG error: {e}")
        return ()


def _bm25_rank(corpus: list[str], query: str, top_k: int = 3) -> list[tuple[float, str]]:
    if not corpus:
        return []
    bm25   = BM25Okapi([_tokenize(d) for d in corpus])
    scores = bm25.get_scores(_tokenize(query))
    return sorted(zip(scores, corpus), reverse=True)[:top_k]


def _snippet_polarity(snippet: str, option: str) -> float:
    tokens, opt_kws = _tokenize(snippet), _keywords(option)
    for i, tok in enumerate(tokens):
        if tok in opt_kws and any(w in _NEGATION_TOKENS for w in tokens[max(0, i - 8): i + 8]):
            return -1.0
    return 1.0


def _extract_bio_keywords(question: str) -> str:
    parts = list(dict.fromkeys(_BIO_KEYWORDS_RE.findall(question) + _PROPER_MULTI_RE.findall(question)))
    return " ".join(parts[:4])


def _detect_question_type(question: str) -> str:
    return "biography" if _BIO_KEYWORDS_RE.search(question) else "factual"


def _build_queries(
    question: str,
    option_texts: list[str],
    labeled: list[tuple[str, str]],
    option_entities: list[str],
    subj_str: str,
    main_term: str,
    question_type: str,
) -> list[WeightedQuery]:
    queries:     list[WeightedQuery] = []
    clean_q      = _clean_query_text(question)
    all_entities = " ".join(t for t, _ in labeled[:3])

    queries.append(WeightedQuery(f"{main_term} {clean_q}"[:120], 1.2, "entity_question"))
    if all_entities and all_entities != main_term:
        queries.append(WeightedQuery(all_entities[:120], 1.0, "all_entities"))

    if question_type == "biography":
        bio_kws = _extract_bio_keywords(question)
        if bio_kws:
            queries.append(WeightedQuery(f"{main_term} {bio_kws}"[:120], 1.4, "biography_keywords"))
        secondary = [t for t, _ in labeled if t != main_term]
        if secondary:
            queries.append(WeightedQuery(f"{secondary[0]} {main_term}"[:120], 1.3, "secondary_entity"))

    # Per-option: GLiNER entity from each option → targeted DDG query for that option only.
    # Options with no extractable entity (e.g. "Indifferent and distant") are skipped.
    for idx, opt in enumerate(option_texts):
        ent = option_entities[idx] if idx < len(option_entities) else ""
        if ent:
            queries.append(WeightedQuery(
                f"{main_term} {ent}"[:120],
                1.1 if question_type == "biography" else 1.0,
                f"entity_option_{idx}",
            ))
        elif question_type == "factual" and idx < 2:
            opt_clean = _clean_query_text(opt)
            if opt_clean:
                queries.append(WeightedQuery(f"{main_term} {opt_clean}"[:120], 0.8, f"text_option_{idx}"))

    return queries


def _vote(
    option_texts: list[str],
    global_snips: tuple[str, ...],
    subj_str: str,
    question: str,
) -> list[float]:
    votes    = [0.0] * len(option_texts)
    subj_kws = _keywords(subj_str)
    pool     = list(global_snips)
    if not pool:
        return votes
    for i, opt_text in enumerate(option_texts):
        for score, snippet in _bm25_rank(pool, f"{question} {opt_text}", top_k=3):
            subj_hits = len(subj_kws & set(_tokenize(snippet)))
            if subj_hits:
                votes[i] += score * subj_hits * _snippet_polarity(snippet, opt_text)
    return votes


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    window   = text[:max_chars]
    last_end = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    return window[:last_end + 1] if last_end > max_chars // 2 else window.rsplit(" ", 1)[0] + "..."


def _build_llm_prompt(
    question: str,
    option_texts: list[str],
    wiki_text: str,
    global_snips: tuple[str, ...],
    votes: list[float],
    is_negative: bool,
    low_confidence: bool = False,
) -> str:
    lines: list[str] = []
    n           = len(option_texts)
    sorted_by_vote = sorted(range(n), key=lambda i: votes[i], reverse=True)
    rank_map       = {idx: rank for rank, idx in enumerate(sorted_by_vote)}

    if is_negative:
        lines.append("[NOT/EXCEPT: pick the option with the LEAST supporting evidence]\n")

    if wiki_text:
        lines.append(f"WIKIPEDIA:\n{wiki_text}\n")

    if global_snips:
        top = _bm25_rank(list(global_snips), question, top_k=2)
        if top:
            lines.append("WEB CONTEXT:")
            for _, s in top:
                lines.append(f"  {_truncate_at_sentence(s, 200)}")
            lines.append("")

    lines.append(f"QUESTION: {question}\n")
    lines.append("OPTIONS:")
    for i, opt in enumerate(option_texts):
        if not low_confidence:
            rank = rank_map[i]
            tag = "  [retrieval: strong]" if rank == 0 else (
                  "  [retrieval: weak]"   if rank == n - 1 else "")
        else:
            tag = ""
        lines.append(f"[{i}] {opt}{tag}")

    lines.append("\nAnswer (0/1/2/3):")
    return "\n".join(lines)


def rag_entertainment(
    query: str,
    num_results: int = 3,
    option_texts: list[str] | None = None,
) -> str:
    labeled = _extract_subjects_gliner(query)
    if labeled:
        subjects  = [text for text, _ in labeled]
        main_term = _pick_main_term(labeled)
        print(f"  [ENT] GLiNER entities: {labeled}")
    else:
        subjects  = _extract_subjects_regex(query)
        main_term = subjects[0] if subjects else ""
        print(f"  [ENT] regex subjects: {subjects}")

    if not main_term:
        kws       = [w for w in _tokenize(query) if len(w) >= 4 and w not in _STOP_WORDS]
        main_term = " ".join(kws[:4]) if kws else query[:60]

    subj_str = " ".join(subjects[:2]) if subjects else main_term
    anchors  = _pick_wiki_anchors(labeled) if labeled else [main_term]
    print(f"  [ENT] main_term={main_term!r}  subj_str={subj_str!r}  anchors={anchors}")

    if not option_texts:
        wiki = _fetch_multi_wiki(anchors, query, max_chars=1200)
        ddg  = _ddg_lookup(subj_str, num_results)
        return "\n\n".join(filter(None, [wiki, *ddg]))[:1500]

    n_opts          = min(len(option_texts), 4)
    question_type   = _detect_question_type(query)
    option_entities = [_extract_option_entity(opt) for opt in option_texts[:n_opts]]
    print(f"  [ENT] option_entities={list(zip(option_texts[:n_opts], option_entities))}")
    queries = _build_queries(
        query, list(option_texts[:n_opts]), labeled, option_entities, subj_str, main_term, question_type
    )
    print(f"  [ENT] question_type={question_type!r}  queries={[(q.strategy, q.text) for q in queries]}")

    def _safe(fut, default):
        try:   return fut.result(timeout=_TIMEOUT + 1)
        except Exception: return default

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries) + 1) as pool:
        wiki_fut    = pool.submit(_fetch_multi_wiki, anchors, query, 1200)
        query_futs  = [pool.submit(_ddg_lookup, q.text, 2) for q in queries]
        wiki_text   = _safe(wiki_fut, "")
        query_snips = [_safe(f, ()) for f in query_futs]

    seen_keys: set[str] = set()
    global_snips_list: list[str] = []
    for snips in query_snips:
        for s in snips:
            k = s[:120]
            if k not in seen_keys:
                seen_keys.add(k)
                global_snips_list.append(s)

    global_snips = tuple(global_snips_list)
    print(f"  [ENT] wiki chars={len(wiki_text)}  global_snips={len(global_snips)}")

    has_title_entity = any(label in _TITLE_LABELS for _, label in labeled)
    if not has_title_entity and not wiki_text:
        print("  [ENT] Low confidence -> LLM fallback")
        return ""

    votes          = _vote(list(option_texts[:n_opts]), global_snips, subj_str, query)
    is_negative    = bool(_NOT_EXCEPT_RE.search(query))
    max_vote       = max(votes) if votes else 0.0
    low_confidence = max_vote < _VOTE_CONFIDENCE_MIN
    winner         = (min if is_negative else max)(range(n_opts), key=lambda i: votes[i])
    print(f"  [ENT] votes={votes}  is_negative={is_negative}  winner=[{winner}]  low_confidence={low_confidence}")

    return _build_llm_prompt(
        question=query, option_texts=list(option_texts[:n_opts]),
        wiki_text=wiki_text, global_snips=global_snips,
        votes=votes, is_negative=is_negative, low_confidence=low_confidence,
    )
