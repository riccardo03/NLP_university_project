import time
import wikipedia
import concurrent.futures
import re
import torch

# Set Wikipedia language for history queries
wikipedia.set_lang("en")
wikipedia.set_user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

def _generate_history_queries(question_text: str) -> list:
    """Uses the LLM to extract highly targeted queries."""
    
    # DELAYED IMPORT: Avoids Circular Import
    import millionaire_bot as bot
    
    system_prompt = (
        "You are an expert ancient history researcher. Extract the core entities. "
        "CRITICAL RULE: If the question contains an ambiguous term, you MUST infer and append the relevant ancient civilization to the search query. "
        "Example: 'The City' -> 'Ancient Athens'. 'The Republic' -> 'Roman Republic'. "
        "Return a comma-separated list of exactly 3 concise Wikipedia search queries. "
        "Output ONLY the comma-separated list."
    )
    try:
        raw = bot.generate_answer(system_prompt, question_text, max_new_tokens=60)

        torch.cuda.empty_cache()

        queries = [q.strip() for q in raw.split(",") if q.strip()]
        return queries if queries else [question_text]
    except Exception as e:
        print(f"  [RAG-History V4] Query gen error: {e}")
        return [question_text]

def _extract_best_paragraphs(content: str, question_text: str, top_n: int = 3) -> str:
    """Lightweight local re-ranker: scores paragraphs based on keyword overlap."""
    # Split text into meaningful paragraphs
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100]
    
    # Extract unique core words from the question (ignoring common stopwords)
    q_words = set(re.findall(r'\w+', question_text.lower()))
    stopwords = {"what", "how", "why", "the", "in", "of", "and", "a", "an", "to", "is", "for", "does", "did", "concept", "term"}
    q_words = q_words - stopwords
    
    scored_paragraphs = []
    for p in paragraphs:
        p_lower = p.lower()
        # Count how many question keywords appear in this paragraph
        score = sum(1 for w in q_words if w in p_lower)
        scored_paragraphs.append((score, p))
        
    # Sort by score descending
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    
    # Keep the top matching paragraphs
    best = [p for score, p in scored_paragraphs[:top_n] if score > 0]
    
    # Fallback to the introduction if no exact keyword overlap is found
    if not best:
        return "\n\n".join(paragraphs[:top_n])
    
    return "\n\n".join(best)

def _fetch_wikipedia(query: str, question_text: str) -> str:
    """Downloads the full page and extracts the most relevant paragraphs."""
    try:
        results = wikipedia.search(query, results=1)
        if not results:
            return ""
        
        try:
            page = wikipedia.page(results[0], auto_suggest=False)
            # Send the full content to the smart extractor
            return _extract_best_paragraphs(page.content, question_text)
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"  [RAG-History V4] Disambiguation caught. Trying: {e.options[0]}")
            fallback_page = wikipedia.page(e.options[0], auto_suggest=False)
            return _extract_best_paragraphs(fallback_page.content, question_text)
            
    except Exception as e:
        print(f"  [RAG-History] Fetch error for '{query}': {type(e).__name__} - {e}")
        return ""

def rag_history(question_text: str) -> str:
    """
    RAG Pipeline for History: Deep Scoring & Paragraph Extraction
    """
    print("  [RAG-History] Pipeline started...")
    
    t0 = time.time()
    queries = _generate_history_queries(question_text)
    print(f"  [RAG-History] Targeted queries: {queries} (in {time.time()-t0:.1f}s)")
    
    t1 = time.time()
    snippets = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # We now pass both the query and the original question to the fetcher
        futures = {executor.submit(_fetch_wikipedia, q, question_text): q for q in queries}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                snippets.append(res)
                
    print(f"  [RAG-History V4] Retrieval completed (in {time.time()-t1:.1f}s)")
    
    # Context Aggregation
    context = "\n\n...\n\n".join(snippets[:3])
    
    # Final safeguard for token length
    return context[:8000]