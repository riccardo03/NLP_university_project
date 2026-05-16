import re
import wikipedia
import warnings

# Try to import DDGS, otherwise we will just use Wikipedia
try:
    from ddgs import DDGS
except ImportError:
    pass

wikipedia.set_user_agent("PoliMillionaireBot_NLP_Project/3.0 (tuo_nome@mail.polimi.it)")
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
wikipedia.set_lang("en")

# --- GLOBAL STOPWORDS ---
STOPWORDS = {"what", "how", "which", "in", "according", "why", "who", "where", "when", 
             "is", "are", "was", "were", "did", "do", "does", "the", "a", "an", "of", 
             "on", "at", "to", "for", "with", "by", "about", "and", "or", "it", "that", 
             "this", "following", "best", "describes", "describe", "term", "used", 
             "significance", "primary", "not", "purpose", "given", "information", 
             "provided", "contribute", "image", "true", "false", "statement", "refer", 
             "refers", "called", "known", "means", "choose", "between", "often", 
             "accompanied", "reason", "decline", "use", "approach", "idea", "various", 
             "claims", "claimed", "differ", "terms", "influence", "from", "cause", "connection"}

# ==========================================
# ENGINE 2: WIKIPEDIA FALLBACK (Our V2.1)
# ==========================================
def clean_option_prefix(opt: str) -> str:
    return re.sub(r'^(\[[0-9]\]|[A-D][\.:\)]|[0-9][\.:\)])\s+', '', opt, flags=re.IGNORECASE).strip()

def get_smart_queries(question: str, option_texts: list) -> tuple:
    queries = []
    quoted_terms = re.findall(r"['\"]([^'\"]+)['\"]", question)
    
    clean_q = re.sub(r'(?i)according to the (passage|article),?\s*', '', question)
    clean_q = re.sub(r"'s\b", "", clean_q) 
    clean_q = re.sub(r'[^\w\s/]', ' ', clean_q)
    
    all_keywords = [w for w in clean_q.split() if w.lower() not in STOPWORDS]
    
    shield_words = []
    for i, w in enumerate(all_keywords):
        if w[0].isupper() or w[0].isdigit():
            if w not in shield_words:
                shield_words.append(w)
            if i + 1 < len(all_keywords):
                next_w = all_keywords[i+1]
                if not (next_w[0].isupper() or next_w[0].isdigit()) and next_w not in shield_words:
                    shield_words.append(next_w)

    if not shield_words:
        context_kws = " ".join(all_keywords[:2])
    else:
        context_kws = " ".join(shield_words[:2])

    primary_ctx = quoted_terms[0] if quoted_terms else context_kws
        
    if option_texts:
        for opt in option_texts:
            clean_opt = clean_option_prefix(opt)
            opt_words = clean_opt.split()
            
            if len(opt_words) <= 3:
                queries.append(clean_opt)
                if primary_ctx:
                    queries.append(f"{clean_opt} {primary_ctx}")
            else:
                long_opt_kws = [w for w in opt_words if len(w) >= 4 and w.lower() not in STOPWORDS]
                if long_opt_kws and primary_ctx:
                    queries.append(f"{' '.join(long_opt_kws[:2])} {primary_ctx}")

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
    if not option_texts: return ""
    paragraphs = full_text.split('\n')
    q_words = set(w.lower() for w in q_keywords if len(w) >= 3 and w.lower() not in STOPWORDS)
    best_paragraphs = []
    
    for opt in option_texts:
        clean_opt = clean_option_prefix(opt)
        opt_words = set(w.lower() for w in clean_opt.split() if len(w) >= 2 and w.lower() not in STOPWORDS)
        if not opt_words: continue
            
        best_p_for_opt = ""
        max_score = 0
        
        for p in paragraphs[1:]:
            if len(p) < 60: continue
            p_lower = p.lower()
            opt_score = sum(1 for w in opt_words if w in p_lower)
            if opt_score > 0:
                q_score = sum(1 for w in q_words if w in p_lower)
                total_score = (opt_score * 5) + (q_score * 3) if q_score > 0 else opt_score 
                if total_score > max_score:
                    max_score = total_score
                    best_p_for_opt = p
                    
        if best_p_for_opt and best_p_for_opt not in best_paragraphs:
            best_paragraphs.append(best_p_for_opt)
            
    return " ... ".join(best_paragraphs)[:3500]

def process_wiki_page(title: str, option_texts: list, keywords: list, pages_processed: set) -> str:
    if title in pages_processed: return None
    try:
        page = wikipedia.page(title, auto_suggest=False)
        pages_processed.add(title)
        summary_sentences = re.split(r'(?<=[.!?]) +', page.summary)
        summary_text = ' '.join(summary_sentences[:3]) + ('.' if not summary_sentences[:3][-1].endswith('.') else '')
        detail_text = extract_deep_details(page.content, option_texts, keywords)
        
        exact_match_hint = ""
        if option_texts:
            for opt in option_texts:
                clean_opt = clean_option_prefix(opt)
                if clean_opt and len(clean_opt) > 3 and (clean_opt.lower() in title.lower() or title.lower() in clean_opt.lower()):
                    exact_match_hint = f" [CRITICAL HINT: This page matches Option '{clean_opt}']"
                    
        return f"[{title}]{exact_match_hint} SUMMARY: {summary_text}" + (f" | DEEP DETAIL: ...{detail_text}..." if detail_text else "")
    except Exception:
        return None

def wikipedia_rag(question_text: str, option_texts: list) -> str:
    """Plan B: High-precision Wikipedia search"""
    queries, keywords = get_smart_queries(question_text, option_texts)
    context_snippets = []
    pages_processed = set()
    
    try:
        for query in queries:
            if len(context_snippets) >= 5: break
            search_results = wikipedia.search(query, results=2)
            for title in search_results:
                if len(context_snippets) >= 5: break
                try:
                    snippet = process_wiki_page(title, option_texts, keywords, pages_processed)
                    if snippet:
                        context_snippets.append(snippet)
                        break
                except wikipedia.exceptions.DisambiguationError as e:
                    try:
                        snippet = process_wiki_page(e.options[0], option_texts, keywords, pages_processed)
                        if snippet:
                            context_snippets.append(snippet)
                            break
                    except Exception: pass
                except Exception: pass
        return "\n\n".join(context_snippets)
    except Exception as e:
        print(f"  [Wiki-Fallback] Error: {e}")
        return ""

# ==========================================
# ENGINE 1: DUCKDUCKGO (Primary RAG)
# ==========================================
def rag_history(question_text: str, option_texts: list = None) -> str:
    """
    MASTER RAG: Test the semantic power of DuckDuckGo. 
    If it fails due to Rate Limit, instantly fallback to Wikipedia.
    """
    if option_texts is None: option_texts = []
    print(f"  [RAG-History] Processing: {question_text[:50]}...")
    
    # 1. DUCKDUCKGO ATTEMPT
    try:
        from ddgs import DDGS
        
        # Clean the question from misleading phrases
        clean_q = re.sub(r'(?i)according to the (passage|article),?\s*', '', question_text).strip()
        clean_opts = [clean_option_prefix(opt) for opt in option_texts]
        
        # Create a powerful Query by joining the question and options
        ddg_query = f"{clean_q} {' '.join(clean_opts)}"
        
        # Limit to 200 characters to avoid angering the search engine
        ddg_query = ddg_query[:200]
        
        snippets = []
        with DDGS() as ddgs:
            # Ask for 3 results: fast and almost never rate-limited
            results = ddgs.text(ddg_query, max_results=3)
            for r in results:
                snippets.append(f"[WEB SNIPPET: {r.get('title', '')}] {r.get('body', '')}")
                
        if snippets:
            print("  [RAG-History] Answer found on DuckDuckGo!")
            return "\n\n".join(snippets)
            
    except Exception as e:
        # If there is a RateLimitException or DDG is not found, the script DOES NOT stop!
        print(f"  [RAG-History] Warning: DuckDuckGo blocked or unavailable. Switching to Wikipedia...")
        
    # 2. WIKIPEDIA FALLBACK
    return wikipedia_rag(question_text, option_texts)