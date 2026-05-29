import multiprocessing
import time
import re
import math

try:
    import sympy as _sp
    from sympy import symbols as _sym, N as _spN, sympify as _spS, solve as _spsolve, Eq as _spEq
    _SYMPY_OK = True
except ImportError:
    _SYMPY_OK = False

try:
    from fractions import Fraction as _Frac
    from collections import deque as _deque
    _FRAC_OK = True
except ImportError:
    _FRAC_OK = False

try:
    import wikipedia
    wikipedia.set_user_agent("PoliMillionaireBot_NLP_Project/3.0")
    wikipedia.set_lang("en")
    _WIKI_OK = True
except ImportError:
    _WIKI_OK = False

_llm_callback = None


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_maths_rag(llm_callback=None):
    global _llm_callback
    _llm_callback = llm_callback
    print("  [RAG-Maths] Inizializzato: Fast-Path + Text Guard + PAL + Wiki RAG.")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _parse_opt_num(text: str) -> float | None:
    t = re.sub(r'\[retrieval:[^\]]*\]', '', text).strip()
    t = t.replace('π', 'pi')
    # Fix: √(expr) → sqrt(expr), then √x → sqrt(x)
    t = re.sub(r'√\(([^)]+)\)', r'sqrt(\1)', t)
    t = re.sub(r'√(\w+(?:\.\d+)?)', r'sqrt(\1)', t)
    try:
        if _SYMPY_OK:
            return float(_spN(_spS(t.replace('^', '**')), 10))
    except Exception:
        pass
    nums = re.findall(r'-?\d+(?:\.\d+)?', t)
    try:
        return float(nums[0]) if nums else None
    except ValueError:
        return None


def _build_hints(question: str) -> str:
    q = question.lower()
    lines = []
    if any(k in q for k in ['probabilit', 'bayes']):
        lines.append("Bayes: P(D|+) = P(+|D)·P(D) / P(+)")
    if any(k in q for k in ['irriducib', 'campo', 'field', 'z_']):
        lines.append("Z_n[x]/(p(x)) is a field iff p(x) is irreducible over Z_n")
    if any(k in q for k in ['ordine', 'order', 'z_', 'zn']):
        lines.append("Max order in ℤ_m × ℤ_n = lcm(m, n)")
    if any(k in q for k in ['domain', 'dominio', 'denominatore']):
        lines.append("For fractions like 1/(1 + 1/x), ensure NO denominator is zero at any step: x != 0, 1+1/x != 0, etc.")
    return "Math context:\n" + "\n".join(f"- {l}" for l in lines) if lines else ""


# ══════════════════════════════════════════════════════════════════════════════
# FAST-PATH INLINE
# ══════════════════════════════════════════════════════════════════════════════

def _fast_path_inline(q: str, options: list) -> tuple | None:
    m = re.search(r'(\d+[\.,]?\d*)\s*%\s*of\s+the\s+variation', q, re.I)
    if m:
        r2 = float(m.group(1).replace(',', '.')) / 100
        r  = math.sqrt(r2)
        for i, opt in enumerate(options):
            for num_str in re.findall(r'0\.\d+', opt):
                try:
                    if abs(float(num_str) - r) < 0.01:
                        return i, f"r = {r:.4f}"
                except ValueError:
                    pass

    m2 = re.search(r'how many (\w+).*?(\d+[\.,]?\d*)\s+(\w+)', q, re.I)
    if m2 and _FRAC_OK:
        relations = re.findall(r'(\d+[\.,]?\d*)\s+(\w+)\s*=\s*(\d+[\.,]?\d*)\s+(\w+)', q)
        if len(relations) >= 2:
            try:
                target_to   = m2.group(1).lower()
                target_n    = _Frac(m2.group(2).replace(',', '.'))
                target_from = m2.group(3).lower()
                ratios = {}
                for n1, u1, n2, u2 in relations:
                    f1, f2 = _Frac(n1.replace(',', '.')), _Frac(n2.replace(',', '.'))
                    ratios[(u1.lower(), u2.lower())] = f2 / f1
                    ratios[(u2.lower(), u1.lower())] = f1 / f2
                dq, vis, conv = _deque([(target_from, _Frac(1))]), {target_from}, None
                while dq:
                    unit, rate = dq.popleft()
                    if unit == target_to:
                        conv = rate
                        break
                    for (u1, u2), rt in ratios.items():
                        if u1 == unit and u2 not in vis:
                            vis.add(u2)
                            dq.append((u2, rate * rt))
                if conv is not None:
                    res = float(target_n * conv)
                    for i, opt in enumerate(options):
                        n_opt = _parse_opt_num(opt)
                        if n_opt is not None and abs(n_opt - res) < 1e-4:
                            return i, f"Conversione: {res}"
            except Exception:
                pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SANDBOX
# ══════════════════════════════════════════════════════════════════════════════

def _worker_exec(code_str: str, queue: multiprocessing.Queue):
    import math as _m, sys as _sys, io as _io, random as _rand
    env = {"math": _m, "random": _rand, "result": None}
    try:
        import sympy as sp
        env.update({"sp": sp, "sympy": sp})
    except ImportError:
        pass
    try:
        from scipy import stats
        import numpy as np
        env.update({"stats": stats, "np": np, "numpy": np})
    except ImportError:
        pass

    old, _sys.stdout = _sys.stdout, _io.StringIO()
    try:
        exec(code_str, env)
        out = _sys.stdout.getvalue().strip()
        queue.put({"status": "ok", "output": out})
    except Exception as e:
        queue.put({"status": "error", "output": str(e)})
    finally:
        _sys.stdout = old


def _execute_code(code: str, timeout: float = 5.0) -> str:
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker_exec, args=(code, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return f"Error: Timeout ({timeout}s)"
    if not q.empty():
        res = q.get()
        return res["output"] if res["status"] == "ok" else f"Error: {res['output']}"
    return "Error: No output returned"


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL VERIFY (Sostituzione)
# ══════════════════════════════════════════════════════════════════════════════

def _verify_option_worker(args: tuple) -> dict:
    question, option_text, option_idx = args
    import re, math
    q = (question
         .replace('–', '-').replace('—', '-')
         .replace('−', '-').replace('×', '*').replace('÷', '/'))
    opt_clean = re.sub(r'\[retrieval:[^\]]*\]', '', option_text).strip()
    opt_num = None
    try:
        from sympy import N, sympify
        opt_num = float(N(sympify(opt_clean.replace('^', '**')), 10))
    except Exception:
        nums = re.findall(r'-?\d+(?:\.\d+)?', opt_clean)
        if nums:
            try:
                opt_num = float(nums[0])
            except ValueError:
                pass

    eq_m = re.search(r'([0-9x()\s\^*+\-.]+)\s*=\s*([0-9x()\s\^*+\-.]+)', q)
    if eq_m and opt_num is not None:
        lhs_raw, rhs_raw = eq_m.group(1).strip(), eq_m.group(2).strip()
        if 'x' in lhs_raw and 'x' not in rhs_raw:
            safe_chars = set('0123456789x()+*-. ')
            lhs_s = re.sub(r'\^', '**', lhs_raw)
            rhs_s = re.sub(r'\^', '**', rhs_raw)
            if all(c in safe_chars for c in lhs_s) and all(c in safe_chars for c in rhs_s):
                try:
                    from sympy import symbols, N as SN, sympify as sy
                    x = symbols('x')
                    lhs_val = float(SN(sy(lhs_s).subs(x, opt_num)))
                    rhs_val = float(SN(sy(rhs_s)))
                    if abs(lhs_val - rhs_val) < 1e-6:
                        return {"idx": option_idx, "verified": True,  "confidence": 1.0, "note": f"x={opt_num} ok"}
                    else:
                        return {"idx": option_idx, "verified": False, "confidence": 0.9, "note": f"x={opt_num} no"}
                except Exception:
                    pass

    if opt_num is not None and re.search(r'\bprime\b|\bprimo\b', question, re.I):
        try:
            from sympy import isprime
            n    = int(round(opt_num))
            is_p = isprime(n)
            ver  = not is_p if re.search(r'\bnot prime\b|\bcomposite\b', question, re.I) else is_p
            return {"idx": option_idx, "verified": ver, "confidence": 0.8, "note": f"{n} is prime: {is_p}"}
        except Exception:
            pass

    return {"idx": option_idx, "verified": None, "confidence": 0.0, "note": "none"}


def _verify_option_target(args: tuple, queue: multiprocessing.Queue):
    try:
        queue.put(_verify_option_worker(args))
    except Exception as e:
        queue.put({"idx": args[2], "verified": None, "confidence": 0.0, "note": str(e)})


def _parallel_verify(question: str, options: list, timeout: float = 2.5) -> tuple | None:
    if not options:
        return None
    shared_q = multiprocessing.Queue()
    procs = [
        multiprocessing.Process(target=_verify_option_target, args=((question, opt, i), shared_q))
        for i, opt in enumerate(options)
    ]
    for p in procs:
        p.start()
    deadline = time.time() + timeout
    results  = []
    for _ in range(len(procs)):
        wait = max(0.0, deadline - time.time())
        if wait == 0:
            break
        try:
            results.append(shared_q.get(timeout=wait))
        except Exception:
            break
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(0.3)
    if not results:
        return None
    verified_true  = [r for r in results if r.get("verified") is True]
    verified_false = [r for r in results if r.get("verified") is False and r.get("confidence", 0) >= 0.85]
    if len(verified_true) == 1:
        return verified_true[0]["idx"], verified_true[0]["note"]
    false_idxs = {r["idx"] for r in verified_false}
    remaining  = set(range(len(options))) - false_idxs
    if len(remaining) == 1 and len(verified_false) >= len(options) - 1:
        return next(iter(remaining)), "Eliminazione"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PAL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _run_pal_pipeline(question: str, options: list, remaining_time: float) -> str:
    print(f"  [RAG-Maths PAL] Generating pure Python script...")

    sys_prompt = (
        "You are an elite Python code generator. Write a Python script to compute the exact answer to the math problem.\n"
        "RULES:\n"
        "1. Output ONLY valid, runnable Python code. NO MARKDOWN. NO EXPLANATIONS.\n"
        "2. You MUST print() the final numerical or boolean result.\n"
        "3. Use `sympy` for algebra and calculus. Use `scipy.stats` for statistics.\n"
        "4. For modular arithmetic or finding specific integers, use a simple `for` loop (brute-force). Do NOT use sympy.Eq with modulo.\n"
    )

    opt_str    = " | ".join(f"[{i}] {opt}" for i, opt in enumerate(options))
    usr_prompt = f"Problem:\n{question}\n\nOptions:\n{opt_str}"

    messages = [
        {"role": "system",    "content": sys_prompt},
        {"role": "user",      "content": "Problem:\nWhat is 12% of 50?\nOptions:\n[0] 5 | [1] 6 | [2] 7"},
        {"role": "assistant", "content": "print(0.12 * 50)"},
        {"role": "user",      "content": usr_prompt},
    ]

    try:
        code_resp = _llm_callback(messages, max_new_tokens=150)

        code_str = code_resp.strip()
        # Strip markdown code fences
        code_str = re.sub(r'^\x60{3}(?:python)?\s*\n', '', code_str, flags=re.IGNORECASE)
        code_str = re.sub(r'\n\x60{3}\s*$', '', code_str).strip()

        print(f"  [DEBUG PAL] Script da eseguire:\n{code_str[:120]}...\n")

        sandbox_timeout = min(4.0, remaining_time - 3.0)
        if sandbox_timeout <= 0:
            return ""

        stdout = _execute_code(code_str, timeout=sandbox_timeout)
        print(f"  [RAG-Maths PAL] Output Sandbox: {stdout.strip()[:100]}")

        if not stdout.startswith("Error:"):
            print(f"  [RAG-Maths PAL] Matching output con opzioni...")
            match_sys = (
                "You are a strict numerical parser. Compare the 'Script Output' to the 'Options'.\n"
                "Find the option that mathematically matches the output (e.g., 20.4 matches 20.4, 0.416 matches 5/12).\n"
                "Output ONLY a single line: ANSWER: X (where X is 0, 1, 2, or 3)."
            )
            match_usr  = f"Options:\n{opt_str}\n\nScript Output:\n{stdout}\n\nWhich option mathematically matches the script output? ANSWER: X"
            match_msgs = [{"role": "system", "content": match_sys}, {"role": "user", "content": match_usr}]

            match_resp = _llm_callback(match_msgs, max_new_tokens=15)
            m_ans = re.search(r'ANSWER\s*:\s*([0-3])', match_resp, re.IGNORECASE)

            if m_ans:
                ans_id = int(m_ans.group(1))
                print(f"  [RAG-Maths PAL] Successo! Risolto con opzione [{ans_id}]")
                return f"DIRECT_ANSWER:{ans_id}\n[PAL Pipeline Output] {stdout.strip()}"

    except Exception as e:
        print(f"  [RAG-Maths PAL] Errore pipeline: {e}")

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# GATEKEEPER & TEXT GUARD
# ══════════════════════════════════════════════════════════════════════════════

def _are_options_texty(options: list) -> bool:
    """Returns True if options look like prose sentences (blocks PAL execution)."""
    if not options:
        return False
    text_opts = 0
    for opt in options:
        clean_opt = re.sub(r'\[retrieval:[^\]]*\]', '', opt)
        if len(re.findall(r'[a-zA-Z]{3,}', clean_opt)) > 3:
            text_opts += 1
    return text_opts >= 2


def _is_abstract_theory(question: str, options: list) -> bool:
    if _llm_callback is None:
        return False
    sys_prompt = (
        "You are a strict routing classifier for a math answering system.\n"
        "Determine if the question is primarily about ABSTRACT THEORY, DEFINITIONS (e.g., 'simple random sample', 'bias'), or ACADEMIC PROOFS.\n"
        "If the question involves explicit calculations, numbers, evaluating polynomials, matrices, probability, or rates, it is NOT theory.\n"
        "Answer ONLY with exactly one word: 'YES' (if theory/definition) or 'NO' (if calculation)."
    )
    usr_prompt = f"Question: {question}\nOptions: {' | '.join(options)}"
    messages   = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}]
    try:
        resp = _llm_callback(messages, max_new_tokens=5).strip().upper()
        return "YES" in resp
    except Exception:
        return False


def _fetch_academic_theory(question: str) -> str:
    if not _WIKI_OK:
        return ""
    print(f"  [RAG-Maths Theory] Ricerca su Wikipedia in corso...")
    clean_q = re.sub(r'(?i)according to the (passage|article),?\s*', '', question).strip()

    stopwords = {
        "what", "is", "the", "a", "an", "of", "in", "to", "for", "with",
        "on", "by", "are", "which", "following", "statement", "true", "false",
        "find", "value",
    }
    words = [w for w in re.findall(r'\b[A-Za-z]+\b', clean_q)
             if w.lower() not in stopwords and len(w) > 2]
    query = " ".join(words[:3])

    if not query:
        return ""

    try:
        results = wikipedia.search(query, results=1)
        if results:
            page    = wikipedia.page(results[0], auto_suggest=False)
            summary = " ".join(page.summary.split('.')[:3]) + "."
            print(f"  [RAG-Maths Theory] Trovato: {page.title}")
            return f"Mathematical Theory Context ({page.title}):\n{summary}"
    except Exception as e:
        print(f"  [RAG-Maths Theory] Nessun risultato esatto su Wiki: {e}")
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def rag_maths(question_text: str, option_texts: list = None, time_budget: float = 8.0) -> str:
    start   = time.time()
    options = option_texts or []

    # 1. Fast-path inline
    try:
        result = _fast_path_inline(question_text, options)
        if result is not None:
            print(f"  [RAG-Maths] Fast-path: opzione {result[0]}")
            return f"DIRECT_ANSWER:{result[0]}\n[Computed] {result[1]}"
    except Exception:
        pass

    # 2. Parallel verify
    rem = time_budget - (time.time() - start)
    if rem > 1.0 and options:
        res_par = _parallel_verify(question_text, options, timeout=min(2.5, rem - 0.5))
        if res_par is not None:
            print(f"  [RAG-Maths] Verifica parallela: opzione {res_par[0]}")
            return f"DIRECT_ANSWER:{res_par[0]}\n[Verified] {res_par[1]}"

    # 3. Text Guard & LLM Gatekeeper
    rem = time_budget - (time.time() - start)
    if _llm_callback is not None and rem > 3.0:
        if _are_options_texty(options) or _is_abstract_theory(question_text, options):
            # 4a. Theory path — Wikipedia RAG
            theory_data = _fetch_academic_theory(question_text)
            if theory_data:
                return theory_data
        else:
            # 4b. Calculation path — PAL pipeline
            pal_result = _run_pal_pipeline(question_text, options, remaining_time=rem)
            if pal_result.startswith("DIRECT_ANSWER:"):
                return pal_result

    # 5. Fallback hints
    return _build_hints(question_text)
