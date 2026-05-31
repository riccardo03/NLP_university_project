"""
rag_maths.py  —  Tool Calling + Fast-Path
==========================================
Pipeline (in priority order, with LLM call count):
  1. Fast-path inline     regex / fraction conversion           0 LLM
  2. Parallel verify      SymPy substitution (ThreadPool)       0 LLM
  3. Tool calling path    Qwen native tool use                  1 LLM
     └─ WIKI: routing     Wikipedia lookup                      0 extra
     └─ numeric match     direct comparison                     0 extra
     └─ context fallback  result passed to main LLM             0 extra
Time budget (passed from play_game via get_context):
  - Fast paths:      ≤2.5 s
  - Tool calling:    ≤(budget − 2 s)
  - Reserve:         ≥2 s (main LLM overhead)
"""
import json
import math
import multiprocessing
import os
import re
import time
from collections import deque as _deque
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout
from fractions import Fraction as _Frac
from urllib.error import HTTPError as _HTTPError
from urllib.parse import urlencode as _urlencode
from urllib.request import urlopen as _urlopen

# ── Optional dependencies ──────────────────────────────────────────────────
try:
    import sympy as _sp
    from sympy import N as _spN, sympify as _spS
    _SYMPY_OK = True
except ImportError:
    _SYMPY_OK = False

try:
    import wikipedia
    wikipedia.set_user_agent("PoliMillionaireBot_NLP_Project/4.0")
    wikipedia.set_lang("en")
    _WIKI_OK = True
except ImportError:
    _WIKI_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════
_llm_callback    = None   # _pipe_call from millionaire_bot (lazy ref fallback)
_tokenizer_ref   = None   # Qwen tokenizer  (apply_chat_template with tools=)
_model_ref       = None   # Qwen model      (model.generate)
_wolfram_app_id  = None   # WolframAlpha Short Answers API key
_wolfram_unavail = False  # Set True on first HTTP 401 — removes tool from list

# ══════════════════════════════════════════════════════════════════════════════
# LAZY-FETCH HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _get_refs() -> "tuple":
    """
    Return (tokenizer, model, llm_callback).
    Priority: explicit setup_maths_rag() call > lazy import from millionaire_bot.
    Survives hot-reloads: if the module globals were reset to None (e.g. after
    importlib.reload(rag_math)), the references are recovered directly from the
    already-loaded millionaire_bot module without any extra setup call.
    """
    tok = _tokenizer_ref
    mod = _model_ref
    cb  = _llm_callback
    if tok is None or mod is None:
        try:
            import millionaire_bot as _bot
            if tok is None and getattr(_bot, '_tokenizer', None) is not None:
                tok = _bot._tokenizer
            if mod is None and getattr(_bot, '_model', None) is not None:
                mod = _bot._model
            if cb is None and callable(getattr(_bot, '_pipe_call', None)):
                cb = _bot._pipe_call
        except Exception:
            pass
    return tok, mod, cb

# ══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════
MATH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "wolfram_query",
            "description": (
                "Query WolframAlpha for accurate mathematical answers. "
                "Best for: algebra, calculus, number theory, combinatorics, geometry, "
                "symbolic computation, and closed-form evaluation. "
                "Do NOT use for probability distributions, statistical tests, or "
                "numerical simulations — use python_execute with scipy.stats instead. "
                "CRITICAL: use SHORT mathematical notation, not long sentences. "
                "Good: '10/(2/5)', 'log_8(2)', 'diff e^x - c*x', 'lcm(2,3,5)'. "
                "Bad: 'what is the multiplicative inverse of -i in the group...'. "
                "For group/ring element computations, prefer python_execute instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Compact mathematical expression or short query. "
                            "Examples: '10/(2/5)', 'integrate 2x from 0 to 3', "
                            "'solve a* b = a^b - ab, 2* x = 22 for x', "
                            "'maximum order element S_10', 'log base 8 of 2'"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": (
                "Execute Python code to solve numerical math problems. "
                "PREFER for: probability distributions (norm, binom, t, chi2 from scipy.stats), "
                "confidence intervals, group/ring element computations (brute-force mod arithmetic), "
                "matrix operations, combinatorics, simulation, numerical evaluation. "
                "Also use when the answer requires enumerating cases (e.g. finding identity element, "
                "inverses, or checking group axioms). "
                "scipy.stats, numpy (np), sympy (sp), math, fractions.Fraction are available. "
                "The code MUST print() the final result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Valid Python code that prints the result. "
                            "Available: math, sympy (as sp), numpy (as np), "
                            "scipy.stats (as stats), fractions.Fraction."
                        )
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sympy_solve",
            "description": (
                "Solve equations or evaluate expressions symbolically with SymPy. "
                "Use for: 'find x such that...', derivatives, integrals, limits, "
                "polynomial roots, simplification. "
                "Do NOT use for purely theoretical/definition questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "SymPy-compatible expression. Use ** for powers (not ^), "
                            "sp.sin/cos/exp/log for functions, sp.pi for pi. "
                            "For equations write: 'x**2 - 4' (=0 implied)."
                        )
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["solve", "simplify", "integrate", "diff", "limit", "evaluate"],
                        "description": "Operation to perform."
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable symbol (default: 'x')."
                    }
                },
                "required": ["expression", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_lookup",
            "description": (
                "Look up mathematical definitions, named theorems, or theoretical properties. "
                "Use for: conceptual questions about definitions (binomial model conditions, "
                "t-test assumptions, control group meaning, simulation schemes, z-score interpretation), "
                "named theorems, or qualitative 'which of the following is true/correct' questions "
                "where no numerical computation is needed. "
                "Do NOT use for questions involving computation or numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "2-4 word search query for the mathematical concept."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════
def setup_maths_rag(llm_callback=None, tokenizer=None, model=None, wolfram_app_id=None):
    """
    Initialise the maths RAG module.
    Args:
        llm_callback  : _pipe_call from millionaire_bot (kept for lazy ref resolution)
        tokenizer     : Qwen tokenizer for apply_chat_template(tools=...)
        model         : Qwen model for model.generate (tool calling forward pass)
        wolfram_app_id: WolframAlpha Short Answers API key (optional;
                        falls back to env var WOLFRAM_APP_ID)
    """
    global _llm_callback, _tokenizer_ref, _model_ref, _wolfram_app_id
    _llm_callback   = llm_callback
    _tokenizer_ref  = tokenizer
    _model_ref      = model
    _wolfram_app_id = (wolfram_app_id
                       or os.environ.get("WOLFRAM_APP_ID")
                       or _wolfram_app_id)
    wolfram = "Wolfram ✓" if _wolfram_app_id else "Wolfram ✗"
    print(f"  [RAG-Maths] Initialised (Tool Calling + Fast-Path + SymPy + {wolfram}).")

# ══════════════════════════════════════════════════════════════════════════════
# WIKIPEDIA RAG
# ══════════════════════════════════════════════════════════════════════════════
_MATH_VOCAB = {
    "theorem", "axiom", "lemma", "corollary", "distribution", "probability",
    "integral", "derivative", "matrix", "eigenvalue", "series", "limit",
    "polynomial", "convergent", "confidence", "variance", "hypothesis",
    "prime", "modular", "combinat", "permutation", "function", "equation",
    "algebra", "calculus", "statistic", "normal", "binomial", "vector",
    "determinant", "linear", "field", "group", "ring", "interval",
    # analysis / topology
    "continuous", "continuity", "compact", "differentiable", "differentiability",
    "topology", "homeomorphism", "metric", "convergence", "bounded",
    # abstract algebra
    "injective", "surjective", "bijective", "isomorphism", "homomorphism",
    "subgroup", "coset", "ideal", "quotient", "kernel", "image",
}

def _fetch_academic_theory_by_query(query: str) -> str:
    """Wikipedia lookup with a pre-built query string (no stopword processing)."""
    if not _WIKI_OK or not query.strip():
        return ""
    query = query.strip()[:80]
    print(f"  [RAG-Maths Wiki] Searching: '{query}'")
    try:
        results = wikipedia.search(query, results=2)
        for title in results:
            try:
                page    = wikipedia.page(title, auto_suggest=False)
                summary = " ".join(page.summary.split(".")[:4]) + "."
                print(f"  [RAG-Maths Wiki] Found: {page.title}")
                return f"Mathematical Theory Context ({page.title}):\n{summary}"
            except Exception:
                continue
    except Exception as e:
        print(f"  [RAG-Maths Wiki] Error: {e}")
    return ""



def _fetch_academic_theory(question: str) -> str:
    if not _WIKI_OK:
        return ""
    q_lower = question.lower()
    if not any(w in q_lower for w in _MATH_VOCAB):
        print("  [RAG-Maths Wiki] Skipped — no math vocabulary detected")
        return ""
    stopwords = {
        "what", "is", "the", "a", "an", "of", "in", "to", "for", "with",
        "on", "by", "are", "which", "following", "statement", "true",
        "false", "find", "value", "given", "consider", "let",
        # conjunctions / conditionals that pollute math queries
        "and", "or", "if", "then", "not", "such", "that", "where",
        "when", "there", "its", "any", "all", "has", "some", "each",
        "from", "can", "was", "will", "must", "does", "only", "one",
        "two", "three", "four", "five", "six", "also", "but", "both",
    }
    clean_q = re.sub(r'(?i)according to the (passage|article),?\s*', '', question).strip()
    words = [
        w for w in re.findall(r'\b[A-Za-z]+\b', clean_q)
        if w.lower() not in stopwords and len(w) > 2
    ]
    query = " ".join(words[:4])
    if not query:
        return ""
    print(f"  [RAG-Maths Wiki] Searching: '{query}'")
    try:
        results = wikipedia.search(query, results=2)
        for title in results:
            try:
                page    = wikipedia.page(title, auto_suggest=False)
                summary = " ".join(page.summary.split(".")[:4]) + "."
                print(f"  [RAG-Maths Wiki] Found: {page.title}")
                return f"Mathematical Theory Context ({page.title}):\n{summary}"
            except Exception:
                continue
    except Exception as e:
        print(f"  [RAG-Maths Wiki] Error: {e}")
    return ""

# ══════════════════════════════════════════════════════════════════════════════
# SANDBOX  (multiprocessing — required for untrusted code isolation)
# ══════════════════════════════════════════════════════════════════════════════
def _worker_exec(code_str: str, queue: multiprocessing.Queue):
    import math as _m, sys as _sys, io as _io, ast as _ast
    env = {"math": _m, "result": None}
    try:
        import sympy as sp
        env.update({"sp": sp, "sympy": sp})
    except ImportError:
        pass
    try:
        import numpy as np
        from scipy import stats
        env.update({"np": np, "numpy": np, "stats": stats})
    except ImportError:
        pass
    try:
        from fractions import Fraction
        env["Fraction"] = Fraction
    except ImportError:
        pass
    old_stdout, _sys.stdout = _sys.stdout, _io.StringIO()
    try:
        # ── Jupyter-style auto-print of last bare expression ──────────────
        _compiled = code_str
        try:
            tree = _ast.parse(code_str)
            if tree.body and isinstance(tree.body[-1], _ast.Expr):
                last_val = tree.body[-1].value
                # Skip if already a print/display call
                is_print = (isinstance(last_val, _ast.Call) and
                            isinstance(last_val.func, _ast.Name) and
                            last_val.func.id in ('print', 'display'))
                if not is_print:
                    tree.body[-1].value = _ast.Call(
                        func=_ast.Name(id='print', ctx=_ast.Load()),
                        args=[last_val], keywords=[]
                    )
                    _ast.fix_missing_locations(tree)
                    _compiled = compile(tree, '<auto>', 'exec')
        except SyntaxError:
            pass
        exec(_compiled, env)
        out = _sys.stdout.getvalue().strip()
        # ── Fallback: check common result variable names ───────────────────
        if not out:
            for _var in ('result', 'ans', 'answer', 'output',
                         'p', 'prob', 'n', 'val', 'value'):
                _v = env.get(_var)
                if _v is not None:
                    _s = str(_v)
                    if _s not in ('None', ''):
                        out = _s
                        break
        queue.put({"status": "ok", "output": out})
    except Exception as e:
        queue.put({"status": "error", "output": str(e)})
    finally:
        _sys.stdout = old_stdout

def _execute_code(code: str, timeout: float = 5.0) -> str:
    if not code.strip():
        return "Error: empty code"
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker_exec, args=(code, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return f"Error: Timeout ({timeout:.1f}s)"
    if not q.empty():
        res = q.get()
        if res["status"] == "ok":
            return res["output"]
        err = res["output"]
        # NameError hint: suggest scipy/sympy equivalent for hallucinated functions
        name_m = re.search(r"name '([^']+)' is not defined", err)
        if name_m:
            bad = name_m.group(1)
            return (
                f"Error: '{bad}' is not defined. "
                f"Available: scipy.stats (norm, binom, t, chi2), "
                f"sympy as sp, numpy as np, math, fractions.Fraction."
            )
        return f"Error: {err}"
    return "Error: no output"

# ══════════════════════════════════════════════════════════════════════════════
# TOOL CALLING
# ══════════════════════════════════════════════════════════════════════════════
def _generate_with_tools(messages: list, max_new_tokens: int = 180,
                          tools: list = None) -> "tuple[str, dict | None]":
    """
    Single forward pass using Qwen's native tool calling.
    Returns:
        (clean_text, tool_call_dict)  — tool_call_dict is None if no tool was called
        ("", None)                    — on any error (triggers fallback)
    """
    tok, mod, _ = _get_refs()
    if tok is None or mod is None:
        return "", None
    if tools is None:
        tools = MATH_TOOLS
    try:
        import torch
        prompt = tok.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        # Use first parameter device — safe with device_map="auto" (multi-device)
        device = next(mod.parameters()).device
        inputs = tok([prompt], return_tensors="pt").to(device)
        with torch.no_grad():
            out = mod.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        raw = tok.decode(new_tokens, skip_special_tokens=False)
        tool_call = _parse_tool_call(raw)
        # Strip <tool_call>...</tool_call> blocks and Qwen special tokens
        clean = re.sub(r'<tool_call>.*?</tool_call>', '', raw, flags=re.DOTALL)
        clean = re.sub(r'<\|[^|]+\|>', '', clean).strip()
        return clean, tool_call
    except Exception as e:
        print(f"  [Tool gen] Error: {e}")
        return "", None

def _parse_tool_call(text: str) -> "dict | None":
    """
    Robust parser for Qwen's tool call output.
    Handles both <tool_call>{...}</tool_call> and raw JSON.
    """
    # Primary format: <tool_call>{"name":..., "arguments":...}</tool_call>
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: raw JSON object with "name" + "arguments" keys
    pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}'
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None

def _execute_sympy_tool(expression: str, operation: str, variable: str = "x",
                         timeout: float = 4.0) -> str:
    """Build a SymPy Python script from structured tool args and run it in the sandbox."""
    expr = (expression
            .replace('^', '**')
            .replace('π', 'sp.pi')
            .replace('∞', 'sp.oo')
            .replace('√', 'sp.sqrt'))
    # Translate math function names to their sp.* equivalents
    for _fn, _sp in [('exp', 'sp.exp'), ('sqrt', 'sp.sqrt'), ('ln', 'sp.log'),
                     ('log', 'sp.log'), ('sin', 'sp.sin'), ('cos', 'sp.cos'),
                     ('tan', 'sp.tan'), ('abs', 'sp.Abs')]:
        expr = re.sub(rf'\b{_fn}\b', _sp, expr)
    # Standalone 'e' (Euler's number) → sp.E  (must run after exp replacement)
    expr = re.sub(r'\bsp\.exp\b', 'sp.exp', expr)   # protect already-replaced
    expr = re.sub(r'(?<!\w)e(?!\w)', 'sp.E', expr)
    # Auto-declare any remaining free symbols (single/short letters that aren't
    # the main variable, sp, or known constants)
    _reserved = {'sp', 'x', 'y', 'z', variable, 'E', 'I', 'pi', 'oo', 'True', 'False'}
    _candidates = set(re.findall(r'\b([a-df-wyzA-Z][a-zA-Z0-9_]*)\b', expr))
    _extra_syms = sorted(s for s in _candidates
                         if s not in _reserved and not s.startswith('sp.'))
    _sym_decls  = "\n".join(f"{s} = sp.Symbol('{s}')" for s in _extra_syms)
    expr = _normalize_sympy_expr(expr)
    op_map = {
        "solve":     f"sp.solve({expr}, {variable})",
        "simplify":  f"sp.simplify({expr})",
        "integrate": f"sp.integrate({expr}, {variable})",
        "diff":      f"sp.diff({expr}, {variable})",
        "limit":     f"sp.limit({expr}, {variable}, sp.oo)",
        "evaluate":  f"sp.N({expr}, 15)",
    }
    code = (
        f"import sympy as sp\n"
        f"{variable} = sp.Symbol('{variable}')\n"
        f"{_sym_decls}\n"
        f"result = {op_map.get(operation, 'sp.N(' + expr + ', 15)')}\n"
        f"print(result)"
    )
    return _execute_code(code, timeout=timeout)

def _execute_wolfram(query: str, timeout: float = 4.0) -> str:
    """
    Call WolframAlpha Short Answers API.
    Returns a plain-text answer string, or an Error: string on failure.
    """
    app_id = _wolfram_app_id or os.environ.get("WOLFRAM_APP_ID", "")
    if not app_id:
        return "Error: WolframAlpha App ID not configured."
    # Normalise equations: move all terms to left side (x^2+12y+57=-y^2-10x → x^2+y^2+10x+12y+57=0)
    # This avoids Wolfram 501 on rearranged equations
    _q = query
    if '=' in _q and not _q.strip().startswith('solve'):
        try:
            parts = _q.split('=', 1)
            lhs, rhs = parts[0].strip(), parts[1].strip()
            if rhs and rhs != '0' and _SYMPY_OK:
                def _impl_mul(s):
                    return re.sub(r'(\d)([a-zA-Z])', r'\1*\2',
                                  s.replace('^', '**'))
                _lhs = _sp.sympify(_impl_mul(lhs))
                _rhs = _sp.sympify(_impl_mul(rhs))
                _std = _sp.expand(_lhs - _rhs)
                _q   = f"{_std} = 0"
        except Exception:
            pass   # keep original query on any parse failure
    masked = app_id[:4] + "-" + "*" * (len(app_id) - 5) if len(app_id) > 5 else "***"
    print(f"  [Wolfram] key={masked}  query={_q[:60]}")
    try:
        params = _urlencode({"i": _q, "appid": app_id})
        url    = f"https://api.wolframalpha.com/v1/result?{params}"
        with _urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8").strip()
    except _HTTPError as e:
        if e.code == 401:
            global _wolfram_unavail
            _wolfram_unavail = True
            return (
                f"Error: HTTP 401 — App ID '{masked}' rejected. "
                "WolframAlpha disabled for this session. "
                "Call rag_math.setup_wolfram('CORRECT-KEY') to re-enable."
            )
        if e.code == 501:
            return "Error: WolframAlpha could not interpret the query."
        return f"Error: HTTP {e.code}"
    except Exception as e:
        return f"Error: {e}"


def setup_wolfram(app_id: str) -> None:
    """
    Convenience helper — set the WolframAlpha App ID at any time without
    reloading the model.  Call this from the notebook after bot.load_model().

    Usage:
        import rag_math
        rag_math.setup_wolfram("XXXXX-XXXXXXXXXX")
    """
    global _wolfram_app_id, _wolfram_unavail
    _wolfram_app_id  = app_id.strip()
    _wolfram_unavail = False   # reset any previous 401 ban
    print(f"  [Wolfram] App ID set ({_wolfram_app_id[:4]}-{'*'*(len(_wolfram_app_id)-5)})")


def test_wolfram(query: str = "2 + 2") -> None:
    """
    Quick connectivity test.  Run from the notebook to verify the key works.

    Usage:
        import rag_math
        rag_math.test_wolfram()          # default query: "2 + 2"
        rag_math.test_wolfram("log_8 2") # custom query
    """
    result = _execute_wolfram(query, timeout=8.0)
    ok = not result.startswith("Error:")
    print(f"  [Wolfram test] {'✅' if ok else '❌'}  query='{query}'  →  {result}")


def _ensure_print(code: str) -> str:
    """
    If the generated code has no print() call, append one for the last
    assigned variable or the last bare expression.
    Qwen 7B frequently omits print() despite being instructed to include it.
    """
    if "print(" in code:
        return code
    lines = [l for l in code.strip().split("\n") if l.strip()]
    # Walk backwards to find the last assignment or expression
    for line in reversed(lines):
        line_s = line.strip()
        m = re.match(r'^(\w+)\s*=', line_s)
        if m and not line_s.startswith("#"):
            return code + f"\nprint({m.group(1)})"
        # Bare expression (not assignment, not import/def/class)
        if (line_s and
                not line_s.startswith(("#", "import", "from", "def ", "class ",
                                       "if ", "for ", "while ", "return", "else",
                                       "elif", "try", "except", "with"))):
            return code + f"\nprint({line_s})"
    return code


def _normalize_sympy_expr(expr: str) -> str:
    """
    Replace bare math function names with their sp.* equivalents so
    SymPy sandbox code doesn't fail with 'name X is not defined'.
    E.g.: exp(x) → sp.exp(x),  log(c) → sp.log(c)
    """
    for fn in ("exp", "log", "ln", "sin", "cos", "tan",
               "asin", "acos", "atan", "sqrt", "Abs", "factorial"):
        # Only replace if not already prefixed with 'sp.' or 'math.'
        expr = re.sub(rf'(?<!\.)\b{fn}\b(?!\s*\.)', f"sp.{fn}", expr)
    expr = re.sub(r'(?<!sp\.)\bpi\b', "sp.pi", expr)
    expr = re.sub(r'(?<!sp\.)\boo\b',  "sp.oo",  expr)
    return expr


def _execute_tool(tool_call: dict, tool_timeout: float = 4.0) -> str:
    """Dispatch a validated tool call to the appropriate local executor."""
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})
    if name == "wolfram_query":
        return _execute_wolfram(args.get("query", ""), timeout=tool_timeout)
    elif name == "python_execute":
        code = args.get("code", "").replace("^", "**")
        code = _ensure_print(code)
        return _execute_code(code, timeout=tool_timeout) if code else "Error: empty code"
    elif name == "sympy_solve":
        return _execute_sympy_tool(
            expression=args.get("expression", ""),
            operation=args.get("operation", "evaluate"),
            variable=args.get("variable", "x"),
            timeout=tool_timeout,
        )
    elif name == "wikipedia_lookup":
        return _fetch_academic_theory(args.get("query", ""))
    return f"Error: unknown tool '{name}'"

def _match_range_opt(computed: float, opt_text: str) -> bool:
    """
    Return True if `computed` falls within the range described by opt_text.
    Handles: 'between X and Y', 'greater than X', 'less than X',
             'at least X', 'at most X'.
    """
    t = re.sub(r'\[retrieval:[^\]]*\]', '', opt_text).lower()
    m = re.search(r'between\s+([\d.]+)\s+and\s+([\d.]+)', t)
    if m:
        return float(m.group(1)) <= computed <= float(m.group(2))
    m = re.search(r'greater than\s+([\d.]+)', t)
    if m:
        return computed > float(m.group(1))
    m = re.search(r'less than\s+([\d.]+)', t)
    if m:
        return computed < float(m.group(1))
    m = re.search(r'at least\s+([\d.]+)', t)
    if m:
        return computed >= float(m.group(1))
    m = re.search(r'at most\s+([\d.]+)', t)
    if m:
        return computed <= float(m.group(1))
    return False


def _try_functional_range_match(sympy_result: str, options: list) -> "int | None":
    """
    For symbolic results like '3*sin(x) + 5' or '-2*cos(x) + 1',
    compute the periodic range [min, max] via SymPy and match against
    options formatted as 'a ≤ y ≤ b' (function range questions).
    Returns matching option index or None.
    """
    if not _SYMPY_OK:
        return None
    if not re.search(r'\b(sin|cos)\b', sympy_result):
        return None
    try:
        x = _sp.Symbol('x')
        expr = _spS(sympy_result.replace('^', '**'))
        period_interval = _sp.Interval(0, 2 * _sp.pi)
        mn = float(_spN(_sp.calculus.util.minimum(expr, x, period_interval)))
        mx = float(_spN(_sp.calculus.util.maximum(expr, x, period_interval)))
        print(f"  [Func range] [{mn:.3f}, {mx:.3f}]")
        for i, opt in enumerate(options):
            # Normalise Unicode minus/dash to ASCII minus
            t = re.sub(r'[–−‒—]', '-', opt)
            # Format: "a ≤ y ≤ b" or "a <= y <= b" or "a ≤ y ≤ b"
            m = re.search(
                r'(-?\d+(?:\.\d+)?)\s*[≤<]=?\s*\w\s*[≤<]=?\s*(-?\d+(?:\.\d+)?)', t
            )
            if m:
                lo, hi = float(m.group(1)), float(m.group(2))
                if abs(lo - mn) < 0.15 and abs(hi - mx) < 0.15:
                    return i
    except Exception:
        pass
    return None


def _match_output_to_option(stdout: str, options: list) -> "int | None":
    """
    Direct numerical comparison between sandbox output and option values.
    Replaces an extra LLM call — pure numeric comparison.
    Returns the index of the matching option, or None if no match found.
    """
    if not stdout or stdout.startswith("Error:") or not options:
        return None
    # Use the last non-empty output line (most likely to be the final result)
    last_line = next(
        (ln.strip() for ln in reversed(stdout.strip().split('\n')) if ln.strip()),
        ""
    )
    computed = None
    # Try SymPy evaluation (handles fractions, sqrt, expressions like "2*sqrt(2)")
    if _SYMPY_OK:
        try:
            computed = float(_spN(_spS(last_line.replace('^', '**')), 15))
        except Exception:
            pass
    # Fallback: extract last number from line — ONLY if no variables present
    # (prevents "10 x + 20" from yielding 20 and false-matching numeric options)
    if computed is None and not re.search(r'[a-zA-Z]', last_line):
        nums = re.findall(r'-?\d+(?:\.\d+)?', last_line)
        if nums:
            try:
                computed = float(nums[-1])
            except ValueError:
                pass
    # Handle list of solutions: "[1, -2, 3]" — try each solution against options
    if computed is None and _SYMPY_OK:
        list_m = re.search(r'\[([^\]]+)\]', stdout)
        if list_m:
            try:
                sols = [float(_spN(_spS(x.strip()))) for x in list_m.group(1).split(',')]
                for sol in sols:
                    for i, opt in enumerate(options):
                        oval = _parse_opt_num(opt)
                        if oval is not None and abs(oval - sol) < max(1e-6, abs(sol) * 1e-4):
                            return i
            except Exception:
                pass
        return None

    if computed is None:
        return None

    # Find the closest option within tolerance
    best_idx, best_err = None, float('inf')
    for i, opt in enumerate(options):
        oval = _parse_opt_num(opt)
        if oval is None:
            continue
        tol  = max(1e-5, abs(computed) * 1e-4)
        diff = abs(oval - computed)
        if diff < tol and diff < best_err:
            best_idx, best_err = i, diff

    if best_idx is not None:
        return best_idx

    # Range-type options ("between X and Y", "greater than X", …)
    range_hits = [i for i, opt in enumerate(options)
                  if _match_range_opt(computed, opt)]
    if len(range_hits) == 1:
        return range_hits[0]

    return None

def _run_tool_calling_path(question: str, options: list,
                            time_budget: float) -> str:
    """
    Tool calling path: 1 LLM call → local tool execution → numeric match.
    Best case  : 1 LLM call + local exec + numeric match  → DIRECT_ANSWER: X
    Worst case : 1 LLM call + local exec                  → context for main LLM
    Failure    : ""                                        → caller uses fallback
    """
    tok, mod, _ = _get_refs()
    if tok is None or mod is None or time_budget < 5.0:
        return ""
    t_start = time.time()

    # Build active tool list — exclude wolfram if key missing or 401 seen
    _has_wolfram = bool(
        not _wolfram_unavail and
        (_wolfram_app_id or os.environ.get("WOLFRAM_APP_ID"))
    )
    active_tools = [
        t for t in MATH_TOOLS
        if _has_wolfram or t["function"]["name"] != "wolfram_query"
    ]

    opt_str = "\n".join(f"[{i}] {opt}" for i, opt in enumerate(options))
    messages = [
        {
            "role": "system",
            "content": (
                "You are a mathematical problem solver. Choose exactly ONE action:\n\n"
                "① If the question is about a definition, theorem, or theoretical property "
                "(e.g. True/False statements, named theorems, axioms, statistical concepts): "
                "output 'WIKI: <2-4 word Wikipedia query>' — no tools.\n"
                "   Examples: 'WIKI: normal subgroup product', "
                "'WIKI: binomial distribution conditions', "
                "'WIKI: central limit theorem'\n\n"
                "② If the question requires computation: call exactly ONE tool.\n"
                "   - wolfram_query : algebra, calculus, geometry (SHORT math notation only)\n"
                "   - python_execute: probability, statistics, group/ring enumeration, "
                "numerical evaluation (scipy.stats, numpy, sympy available)\n"
                "   - sympy_solve   : symbolic differentiation, integration, equation solving\n\n"
                "Never answer directly. Never explain. Just output WIKI: ... or call a tool."
            )
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nOptions:\n{opt_str}"
        }
    ]
    # Single forward pass — model outputs WIKI: or calls a tool
    response, tool_call = _generate_with_tools(messages, max_new_tokens=160,
                                               tools=active_tools)
    print(f"  [Tool path] tool={'None' if tool_call is None else tool_call.get('name', '?')}")

    # ── WIKI: routing — LLM chose theory lookup ───────────────────────────
    if tool_call is None:
        wiki_m = re.match(r'WIKI\s*:\s*(.+)', response.strip(), re.I)
        if wiki_m:
            wiki_query = wiki_m.group(1).strip().split('\n')[0][:80]
            print(f"  [Tool path] Wikipedia routing: '{wiki_query}'")

            # Dual-statement: search per statement with LLM-generated queries
            has_dual = bool(re.search(r'\bStatement\s+[12]\b', question, re.I))
            if has_dual:
                stmt_parts = re.findall(
                    r'Statement\s+\d+\s*[|.:]\s*(.+?)(?=Statement\s+\d+\s*[|.:]|$)',
                    question, flags=re.I | re.DOTALL
                )
                if len(stmt_parts) >= 2:
                    results, seen = [], set()
                    for stmt in stmt_parts[:2]:
                        # Use the part after the statement separator as query seed
                        q = re.sub(r'\s+', ' ', stmt.strip())[:80]
                        theory = _fetch_academic_theory_by_query(q)
                        if theory:
                            title_m = re.match(r'Mathematical Theory Context \(([^)]+)\)', theory)
                            key = title_m.group(1) if title_m else theory[:40]
                            if key not in seen:
                                seen.add(key)
                                results.append(theory)
                    if results:
                        return "\n\n".join(results)

            # Single theory question — use the LLM-generated query directly
            theory = _fetch_academic_theory_by_query(wiki_query)
            return theory if theory else ""

        # No tool and no WIKI → fall through to main LLM
        print("  [Tool path] No tool called — falling through to main LLM")
        return ""

    # ── Execute the tool locally ──────────────────────────────────────────
    remaining = time_budget - (time.time() - t_start)
    if remaining < 1.0:
        return ""
    tool_result = _execute_tool(tool_call, tool_timeout=min(remaining - 1.0, 4.5))
    tool_name   = tool_call.get("name", "tool")
    print(f"  [Tool path] {tool_name} → {tool_result[:80]}")

    if tool_result.startswith("Error:") or not tool_result.strip():
        print(f"  [Tool path] {tool_name} → no usable output — falling through")
        return ""

    # ── Numeric match — zero extra LLM calls ─────────────────────────────
    direct = _match_output_to_option(tool_result, options)
    if direct is not None:
        print(f"  [Tool path] Numeric match → [{direct}]")
        return f"DIRECT_ANSWER: {direct}\n[{tool_name}] {tool_result[:150]}"

    # ── Functional range match for trig symbolic outputs ──────────────────
    if tool_name in ("sympy_solve", "wolfram_query"):
        fr = _try_functional_range_match(tool_result, options)
        if fr is not None:
            print(f"  [Tool path] Functional range match → [{fr}]")
            return f"DIRECT_ANSWER: {fr}\n[{tool_name}] {tool_result[:150]}"

    # ── Sympy returned symbolic (LambertW, RootOf…) → retry with Wolfram ─
    _symbolic_markers = ('LambertW', 'RootOf', 'CRootOf', 'Integral',
                         'AccumBounds', 'zoo', 'nan')
    if (tool_name == "sympy_solve" and
            any(m in tool_result for m in _symbolic_markers) and
            _has_wolfram):
        print(f"  [Tool path] sympy symbolic — retrying with wolfram_query")
        # Build a compact Wolfram query — strip LaTeX markup
        _wq = question
        _wq = re.sub(r'\\star\b', '*', _wq)
        _wq = re.sub(r'\\cdot\b', '*', _wq)
        _wq = re.sub(r'\\times\b', '*', _wq)
        _wq = re.sub(r'\\[a-zA-Z]+', ' ', _wq)   # remaining LaTeX cmds
        _wq = re.sub(r'[\$\{\}]', '', _wq)
        _wq = re.sub(r'\s+', ' ', _wq).strip()[:120]
        wolfram_result = _execute_wolfram(f"solve: {_wq}", timeout=6.0)
        print(f"  [Tool path] wolfram fallback → {wolfram_result[:80]}")
        if not wolfram_result.startswith("Error:"):
            direct2 = _match_output_to_option(wolfram_result, options)
            if direct2 is not None:
                print(f"  [Tool path] Wolfram numeric match → [{direct2}]")
                return f"DIRECT_ANSWER: {direct2}\n[wolfram] {wolfram_result[:150]}"
            return f"[wolfram]\n{wolfram_result}"

    # ── Return tool result as enriched context for the main LLM ──────────
    return f"[{tool_name} result]\n{tool_result}"

# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL VERIFY  (ThreadPoolExecutor replaces multiprocessing for SymPy checks)
# ══════════════════════════════════════════════════════════════════════════════
def _verify_option_worker(args: tuple) -> dict:
    """
    Check a single option via SymPy substitution or primality test.
    Pure computation: safe to run in threads (no exec, no untrusted code).
    """
    question, option_text, option_idx = args
    q = (question
         .replace('\u2013', '-').replace('\u2014', '-')
         .replace('\u2212', '-').replace('\u00d7', '*').replace('\u00f7', '/'))
    opt_clean = re.sub(r'\[retrieval:[^\]]*\]', '', option_text).strip()
    opt_num = None
    if _SYMPY_OK:
        try:
            opt_num = float(_spN(_spS(opt_clean.replace('^', '**')), 10))
        except Exception:
            pass
    if opt_num is None:
        nums = re.findall(r'-?\d+(?:\.\d+)?', opt_clean)
        if nums:
            try:
                opt_num = float(nums[0])
            except ValueError:
                pass

    # Equation substitution check: f(x) = c
    eq_m = re.search(r'([0-9x()\s\^*+\-.]+)\s*=\s*([0-9x()\s\^*+\-.]+)', q)
    if eq_m and opt_num is not None:
        lhs_raw, rhs_raw = eq_m.group(1).strip(), eq_m.group(2).strip()
        if 'x' in lhs_raw and 'x' not in rhs_raw:
            safe  = set('0123456789x()+*-. ')
            lhs_s = lhs_raw.replace('^', '**')
            rhs_s = rhs_raw.replace('^', '**')
            if (all(c in safe for c in lhs_s) and
                    all(c in safe for c in rhs_s) and _SYMPY_OK):
                try:
                    x  = _sp.Symbol('x')
                    lv = float(_spN(_spS(lhs_s).subs(x, opt_num)))
                    rv = float(_spN(_spS(rhs_s)))
                    if abs(lv - rv) < 1e-6:
                        return {"idx": option_idx, "verified": True,
                                "confidence": 1.0, "note": f"x={opt_num} ✓"}
                    else:
                        return {"idx": option_idx, "verified": False,
                                "confidence": 0.9, "note": f"x={opt_num} ✗"}
                except Exception:
                    pass

    # Primality check
    if (opt_num is not None and
            re.search(r'\bprime\b|\bprimo\b', question, re.I) and _SYMPY_OK):
        try:
            n    = int(round(opt_num))
            is_p = _sp.isprime(n)
            ver  = (not is_p) if re.search(r'\bnot prime\b|\bcomposite\b',
                                            question, re.I) else is_p
            return {"idx": option_idx, "verified": ver,
                    "confidence": 0.8, "note": f"{n} prime={is_p}"}
        except Exception:
            pass

    return {"idx": option_idx, "verified": None, "confidence": 0.0, "note": "—"}

def _parallel_verify(question: str, options: list,
                      timeout: float = 2.5) -> "tuple | None":
    """
    Run _verify_option_worker for each option in parallel using threads.
    Threads are safe here: only SymPy, no untrusted code execution.
    Much lower cold-start cost than multiprocessing.Process.
    """
    if not options:
        return None
    results = []
    with ThreadPoolExecutor(max_workers=min(len(options), 4)) as ex:
        future_map = {
            ex.submit(_verify_option_worker, (question, opt, i)): i
            for i, opt in enumerate(options)
        }
        try:
            for fut in as_completed(future_map, timeout=timeout):
                try:
                    results.append(fut.result(timeout=0.2))
                except Exception:
                    pass
        except _FuturesTimeout:
            pass

    if not results:
        return None
    verified_true  = [r for r in results if r.get("verified") is True]
    verified_false = [r for r in results if
                      r.get("verified") is False and r.get("confidence", 0) >= 0.85]
    if len(verified_true) == 1:
        # Require at least one other option verified False — prevents coincidental
        # matches (e.g. an equation in the question accidentally satisfied by an
        # option value) from triggering a wrong DIRECT_ANSWER.
        if len(verified_false) >= 1:
            return verified_true[0]["idx"], verified_true[0]["note"]
        return None  # single True without eliminations → not trustworthy
    false_idxs = {r["idx"] for r in verified_false}
    remaining  = set(range(len(options))) - false_idxs
    if len(remaining) == 1 and len(verified_false) >= len(options) - 1:
        return next(iter(remaining)), "elimination"
    return None

# ══════════════════════════════════════════════════════════════════════════════
# FAST-PATH INLINE
# ══════════════════════════════════════════════════════════════════════════════
def _parse_opt_num(text: str) -> "float | None":
    t = re.sub(r'\[retrieval:[^\]]*\]', '', text).strip()
    t = t.replace('π', 'pi').replace('√', 'sqrt(')
    if _SYMPY_OK:
        try:
            return float(_spN(_spS(t.replace('^', '**')), 10))
        except Exception:
            pass
    nums = re.findall(r'-?\d+(?:\.\d+)?', t)
    try:
        return float(nums[0]) if nums else None
    except ValueError:
        return None

def _fast_path_inline(q: str, options: list) -> "tuple | None":
    """Deterministic regex-based solver for common patterns. Zero LLM calls."""
    # Pattern: "X% of the variation" → r = sqrt(X/100)
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

    # Pattern: unit conversion chains
    m2 = re.search(r'how many (\w+).*?(\d+[\.,]?\d*)\s+(\w+)', q, re.I)
    if m2:
        relations = re.findall(
            r'(\d+[\.,]?\d*)\s+(\w+)\s*=\s*(\d+[\.,]?\d*)\s+(\w+)', q)
        if len(relations) >= 2:
            try:
                target_to   = m2.group(1).lower()
                target_n    = _Frac(m2.group(2).replace(',', '.'))
                target_from = m2.group(3).lower()
                ratios = {}
                for n1, u1, n2, u2 in relations:
                    f1 = _Frac(n1.replace(',', '.'))
                    f2 = _Frac(n2.replace(',', '.'))
                    ratios[(u1.lower(), u2.lower())] = f2 / f1
                    ratios[(u2.lower(), u1.lower())] = f1 / f2
                dq  = _deque([(target_from, _Frac(1))])
                vis = {target_from}
                conv = None
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
                            return i, f"Conversion: {res}"
            except Exception:
                pass
    return None

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def rag_maths(question_text: str, option_texts: list = None,
              time_budget: float = 20.0) -> str:
    """
    Main RAG entry point for mathematics questions.
    Args:
        question_text : full question text
        option_texts  : list of 4 option strings
        time_budget   : seconds available for this call
                        (caller should pass: time_remaining - LLM_reserve)
    Returns:
        "DIRECT_ANSWER: X\\n..."  if answer is certain (main LLM reads and echoes it)
        context string            if enriched context was found
        ""                        if nothing useful found
    """
    start   = time.time()
    elapsed = lambda: time.time() - start
    options = option_texts or []

    tok, mod, _ = _get_refs()
    print(f"  [RAG-Maths] State: "
          f"tokenizer={'SET' if tok else 'NONE'} | "
          f"budget={time_budget:.1f}s")

    # ── 1. Fast deterministic paths  (≤2.5 s, 0 LLM calls) ───────────────
    try:
        result = _fast_path_inline(question_text, options)
        if result is not None:
            print(f"  [RAG-Maths] Fast-path → [{result[0]}]  ({elapsed():.2f}s)")
            return f"DIRECT_ANSWER: {result[0]}\n[Computed] {result[1]}"
    except Exception:
        pass

    if options and elapsed() < 2.5:
        verify_budget = min(2.0, time_budget * 0.1)
        res_par = _parallel_verify(question_text, options, timeout=verify_budget)
        if res_par is not None:
            print(f"  [RAG-Maths] Parallel verify → [{res_par[0]}]  ({elapsed():.2f}s)")
            return f"DIRECT_ANSWER: {res_par[0]}\n[Verified] {res_par[1]}"

    remaining = time_budget - elapsed()

    # ── 2. Tool calling path: theory routing (WIKI:) + computation ────────
    if remaining > 5.0 and tok is not None:
        result = _run_tool_calling_path(
            question_text, options,
            time_budget=remaining - 2.0,
        )
        if result:
            return result

    return ""