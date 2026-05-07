"""
RAG pipeline for the Ancient History & Politics competition.
Wikipedia, the encyclopedia of all knowledge it is.
"""


def rag_history(query: str, sentences: int = 5) -> str:
    """
    Fetch Wikipedia summary for historical context.
    The encyclopedia of all knowledge, Wikipedia is.
    """
    try:
        import wikipedia

        # Wikipedia hard-limits queries to ~300 chars; truncate at word boundary we must
        if len(query) > 280:
            query = query[:280].rsplit(' ', 1)[0]

        wikipedia.set_lang("en")
        try:
            summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            # Among the options, the first we choose
            try:
                summary = wikipedia.summary(e.options[0], sentences=sentences)
                return summary
            except Exception:
                return ""
        except wikipedia.exceptions.PageError:
            # Not found, empty context we return
            return ""
    except Exception as exc:
        print(f"  [RAG-History] Failed, it has: {exc}")
        return ""
