import re
import warnings

import wikipedia

wikipedia.set_user_agent("PoliMillionaireBot_NLP_Project/3.0 (tuo_nome@mail.polimi.it)")
wikipedia.set_lang("en")
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

STOPWORDS = {
    "what", "how", "which", "in", "according", "why", "who", "where", "when",
    "is", "are", "was", "were", "did", "do", "does", "the", "a", "an", "of",
    "on", "at", "to", "for", "with", "by", "about", "and", "or", "it", "that",
    "this", "following", "best", "describes", "describe", "term", "used",
    "significance", "primary", "not", "purpose", "given", "information",
    "provided", "contribute", "image", "true", "false", "statement", "refer",
    "refers", "called", "known", "means", "choose", "between", "often",
    "accompanied", "reason", "decline", "use", "approach", "idea", "various",
    "claims", "claimed", "differ", "terms", "influence", "from", "cause", "connection",
}


def clean_option_prefix(opt: str) -> str:
    return re.sub(r"^(\[[0-9]\]|[A-D][\.:\)]|[0-9][\.:\)])\s+", "", opt, flags=re.IGNORECASE).strip()


def get_smart_queries(question: str, option_texts: list) -> tuple:
    quoted_terms = re.findall(r"['\"]([^'\"]+)['\"]", question)

    clean_q = re.sub(r"(?i)according to the (passage|article),?\s*", "", question)
    clean_q = re.sub(r"'s\b", "", clean_q)
    clean_q = re.sub(r"[^\w\s/]", " ", clean_q)
    all_keywords = [w for w in clean_q.split() if w.lower() not in STOPWORDS]

    shield_words = []
    for i, w in enumerate(all_keywords):
        if w[0].isupper() or w[0].isdigit():
            if w not in shield_words:
                shield_words.append(w)
            if i + 1 < len(all_keywords):
                next_w = all_keywords[i + 1]
                if not (next_w[0].isupper() or next_w[0].isdigit()) and next_w not in shield_words:
                    shield_words.append(next_w)

    context_kws = " ".join(shield_words[:2]) if shield_words else " ".join(all_keywords[:2])

    queries = []
    if quoted_terms:
        primary = quoted_terms[0]
        context_anchor = shield_words[0] if shield_words else (all_keywords[0] if all_keywords else "")
        queries = [f"{primary} {context_anchor}".strip(), primary]
    else:
        primary_ctx = context_kws
        for opt in option_texts:
            clean_opt = clean_option_prefix(opt)
            opt_words = clean_opt.split()
            if len(opt_words) <= 3:
                queries.append(clean_opt)
                if primary_ctx:
                    queries.append(f"{clean_opt} {primary_ctx}")
            else:
                long_kws = [w for w in opt_words if len(w) >= 4 and w.lower() not in STOPWORDS]
                if long_kws and primary_ctx:
                    queries.append(f"{' '.join(long_kws[:2])} {primary_ctx}")
        if primary_ctx:
            queries.append(primary_ctx)

    seen = set()
    unique_queries = []
    for q in queries:
        if q and q.lower() not in seen:
            seen.add(q.lower())
            unique_queries.append(q)

    return unique_queries, all_keywords


def extract_deep_details(full_text: str, option_texts: list, q_keywords: list) -> str:
    if not option_texts:
        return ""
    paragraphs = full_text.split("\n")
    q_words = {w.lower() for w in q_keywords if len(w) >= 3 and w.lower() not in STOPWORDS}
    best_paragraphs = []

    for opt in option_texts:
        clean_opt = clean_option_prefix(opt)
        opt_words = {w.lower() for w in clean_opt.split() if len(w) >= 2 and w.lower() not in STOPWORDS}
        if not opt_words:
            continue

        best_p, max_score = "", 0
        for p in paragraphs[1:]:
            if len(p) < 60:
                continue
            p_lower = p.lower()
            opt_score = sum(1 for w in opt_words if w in p_lower)
            if opt_score > 0:
                q_score = sum(1 for w in q_words if w in p_lower)
                total = (opt_score * 5 + q_score * 3) if q_score > 0 else opt_score
                if total > max_score:
                    max_score, best_p = total, p

        if best_p and best_p not in best_paragraphs:
            best_paragraphs.append(best_p)

    return " ... ".join(best_paragraphs)[:3500]


def _process_wiki_page(title: str, option_texts: list, keywords: list, pages_processed: set):
    if title in pages_processed:
        return None
    try:
        page = wikipedia.page(title, auto_suggest=False)
        pages_processed.add(title)

        sentences = re.split(r"(?<=[.!?]) +", page.summary)
        summary = " ".join(sentences[:3])
        if not summary.endswith("."):
            summary += "."

        detail = extract_deep_details(page.content, option_texts, keywords)

        hint = ""
        for opt in option_texts:
            clean_opt = clean_option_prefix(opt)
            if clean_opt and len(clean_opt) > 3 and (
                clean_opt.lower() in title.lower() or title.lower() in clean_opt.lower()
            ):
                hint = f" [CRITICAL HINT: This page matches Option '{clean_opt}']"
                break

        result = f"[{title}]{hint} SUMMARY: {summary}"
        if detail:
            result += f" | DEEP DETAIL: ...{detail}..."
        return result
    except Exception:
        return None


def _wikipedia_rag(question_text: str, option_texts: list) -> str:
    queries, keywords = get_smart_queries(question_text, option_texts)
    snippets = []
    pages_processed = set()

    try:
        for query in queries:
            if len(snippets) >= 5:
                break
            for title in wikipedia.search(query, results=2):
                if len(snippets) >= 5:
                    break
                try:
                    snippet = _process_wiki_page(title, option_texts, keywords, pages_processed)
                    if snippet:
                        snippets.append(snippet)
                        break
                except wikipedia.exceptions.DisambiguationError as e:
                    try:
                        snippet = _process_wiki_page(e.options[0], option_texts, keywords, pages_processed)
                        if snippet:
                            snippets.append(snippet)
                            break
                    except Exception:
                        pass
                except Exception:
                    pass
    except Exception as e:
        print(f"  [RAG-History] Wikipedia error: {e}")

    return "\n\n".join(snippets)


def rag_history(question_text: str, option_texts: list = None) -> str:
    if option_texts is None:
        option_texts = []

    # DuckDuckGo (primary)
    try:
        from ddgs import DDGS
        queries, _ = get_smart_queries(question_text, option_texts)
        query = queries[0]
        snippets = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                snippets.append(f"[WEB SNIPPET: {r.get('title', '')}] {r.get('body', '')}")
        if snippets:
            return "\n\n".join(snippets)
    except Exception:
        pass

    # Wikipedia fallback
    return _wikipedia_rag(question_text, option_texts)
