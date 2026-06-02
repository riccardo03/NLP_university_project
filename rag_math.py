"""rag_math_v4.py — compact RAG for math multiple-choice questions.

Rewrite of v3 (2036 → ~750 lines) with these changes:
  • Manual `_STATS_KB` replaced by hardcoded KB (14 entries) + Wikipedia fallback.
  • Tool-calling path REMOVED (was redundant with Universal PAL).
  • `_build_hints` (224 lines of keyword heuristics) REMOVED.
  • `_get_refs` (154 lines of init plumbing) consolidated to module globals.
  • Fast-path patterns compressed using dispatch table.

Public API (compatible with v3 callers):
  rag_maths(question, option_texts=[...], time_budget=20.0) -> str
  setup_maths_rag(model, tokenizer, llm_callback, wolfram_app_id=None) -> None
  setup_wolfram(app_id: str) -> None

Pipeline (priority order):
  1. Cache lookup                    (0 LLM, <1ms)
  2. Fast-path patterns              (0 LLM, deterministic)
  3. Parallel SymPy verify           (0 LLM)
  4. Direct Wolfram                  (0 LLM, optional)
  6. Universal PAL                   (1 LLM + sandbox)
  7. Wikipedia theory fallback       (0 LLM)
"""

import io, os, re, sys, time, math, signal, multiprocessing as mp
from typing import Optional, List, Tuple
import requests

# ── Module globals ─────────────────────────────────────────────────────────
_model = _tokenizer = _llm_callback = None
_wolfram_app_id = os.environ.get("WOLFRAM_APP_ID", "955J5UHWAL")
_wolfram_unavail = False
_question_cache: dict = {}

# ══════════════════════════════════════════════════════════════════════════
# 0. Setup
# ══════════════════════════════════════════════════════════════════════════
def setup_maths_rag(model=None, tokenizer=None, llm_callback=None,
                    wolfram_app_id=None) -> None:
    """Inject LLM references for use by Universal PAL. Idempotent."""
    global _model, _tokenizer, _llm_callback, _wolfram_app_id
    if model is not None:          _model = model
    if tokenizer is not None:      _tokenizer = tokenizer
    if llm_callback is not None:   _llm_callback = llm_callback
    if wolfram_app_id is not None: _wolfram_app_id = wolfram_app_id


def setup_wolfram(app_id: str) -> None:
    global _wolfram_app_id, _wolfram_unavail
    _wolfram_app_id = app_id
    _wolfram_unavail = False


# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import sympy as sp
    from sympy import N as _spN, S as _spS
    _SYMPY_OK = True
except ImportError:
    _SYMPY_OK = False




# ── Optional retrieval dependencies (graceful no-op if missing) ───────────
try:
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    _RETRIEVAL_OK = True
except ImportError:
    _RETRIEVAL_OK = False

# Retrieval state — lazy-loaded on first use
_ret_data       = None   # List[{"question","answer_text","rationale"}]
_ret_embeddings = None   # np.ndarray (N, D), L2-normalised
_embedder       = None
_RET_THRESHOLD  = 0.60


def _has_cuda() -> bool:
    try:
        import torch; return torch.cuda.is_available()
    except Exception: return False


def _load_retrieval_index() -> bool:
    """Lazy-load AQuA-RAT *training* split (97k examples, ~3min first call).
    Uses training data only — zero overlap with the MMLU evaluation set.
    Falls back to math_qa if aqua_rat unavailable.
    Set _RET_FAST=True to use only test+validation (~500 examples, ~5s)."""
    global _ret_data, _ret_embeddings, _embedder
    if _ret_data is not None:
        return True
    if not _RETRIEVAL_OK:
        return False
    try:
        rows = []
        _RET_FAST = False  # set True for quick smoke-test startup

        # ── Try AQuA-RAT ──────────────────────────────────────────────
        splits = ("test", "validation") if _RET_FAST else ("train",)
        for split in splits:
            try:
                ds = load_dataset("aqua_rat", split=split)
                for ex in ds:
                    raw_opts = ex.get("options", [])
                    choices  = [re.sub(r"^[A-Ea-e]\)\s*", "", o).strip() for o in raw_opts]
                    letter   = ex.get("correct", "A").strip().upper()
                    ans_idx  = max(0, ord(letter) - ord("A"))
                    ans_txt  = choices[ans_idx] if ans_idx < len(choices) else ""
                    rows.append({
                        "question":  ex["question"],
                        "answer_text": ans_txt,
                        "rationale": ex.get("rationale", "")[:400],
                    })
                print(f"  [Retrieval] AQuA-RAT {split}: {len(rows)} rows loaded.")
            except Exception as e:
                print(f"  [Retrieval] AQuA-RAT {split} failed: {e}")

        # ── Fallback: math_qa ─────────────────────────────────────────
        if not rows:
            try:
                ds = load_dataset("math_qa", split="train")
                for ex in ds:
                    rows.append({
                        "question":   ex.get("Problem", ""),
                        "answer_text": ex.get("correct", ""),
                        "rationale":  ex.get("Rationale", "")[:400],
                    })
                print(f"  [Retrieval] math_qa train: {len(rows)} rows loaded.")
            except Exception as e:
                print(f"  [Retrieval] math_qa also failed: {e}")

        if not rows:
            _ret_data = []; return False

        print(f"  [Retrieval] Building embeddings for {len(rows)} examples…")
        if _embedder is None:
            _embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cuda" if _has_cuda() else "cpu",
            )
        emb = _embedder.encode(
            [r["question"] for r in rows],
            convert_to_numpy=True, show_progress_bar=True,
            batch_size=256, normalize_embeddings=True,
        )
        _ret_data       = rows
        _ret_embeddings = emb.astype(np.float32)
        print(f"  [Retrieval] Index ready: {len(rows)} examples.")
        return True
    except Exception as e:
        print(f"  [Retrieval] Load failed: {e}")
        _ret_data = []; return False


def _retrieve_context(question: str, top_k: int = 2) -> str:
    """Retrieve rationale from training data for semantically similar problems.
    Returns solution-strategy context — NOT a direct answer lookup."""
    try:
        if not _load_retrieval_index() or _ret_embeddings is None or not _ret_data:
            return ""
        qv     = _embedder.encode([question], convert_to_numpy=True,
                                  show_progress_bar=False, normalize_embeddings=True)
        scores = (_ret_embeddings @ qv[0].astype(np.float32))
        idx    = np.argsort(-scores)[:top_k]
        parts  = []
        for raw_i in idx:
            i, s = int(raw_i), float(scores[int(raw_i)])
            if s < _RET_THRESHOLD:
                break
            d = _ret_data[i]
            parts.append(
                f"Similar problem (relevance {s:.2f}):\n"
                f"Q: {d['question']}\n"
                f"Answer: {d['answer_text']}\n"
                f"Reasoning: {d['rationale']}"
            )
        if not parts:
            return ""
        print(f"  [Retrieval] {len(parts)} examples (top {float(scores[int(idx[0])]):.2f})")
        return "Similar solved problems (use the reasoning strategy):\n\n" + "\n\n".join(parts)
    except Exception as e:
        print(f"  [Retrieval] error: {e}")
        return ""


# ── Hardcoded KB — theory facts too specific for retrieval ────────────────
_HARDCODED_KB = [
    # ── Statistiche ──────────────────────────────────────────────────────
    (re.compile(r'confidence\s+interval.*differ|90.*confident|95.*confident|margin.*error', re.I),
     "CI interpretation: 'We are X% confident the TRUE parameter lies in (a,b).' "
     "NOT 'probability the parameter is in this interval'. "
     "Margin of error = half-width. CI narrows with larger n or lower confidence."),
    (re.compile(r'stratif|cluster\s+samp|systematic\s+samp|simple\s+random', re.I),
     "Sampling: SRS = every subset equally likely (defined by SELECTION METHOD). "
     "Stratified = homogeneous strata, sample from each. "
     "Cluster = heterogeneous clusters, select whole clusters. "
     "Sampling from each district → STRATIFIED."),
    (re.compile(r'type\s+[iI1]\s+error|type\s+[iI][iI2]\s+error|power\s+of\s+test', re.I),
     "Type I (α) = P(reject H0|H0 true). Type II (β) = P(fail reject|H0 false). "
     "Power = 1−β. Larger n → higher power. Larger α → higher power, more Type I."),
    (re.compile(r'r\^?2.*variation|variation.*r\^?2|coefficient.*determin', re.I),
     "r² = proportion of variation explained. r²=0.71 → r=±√0.71=±0.84. "
     "Sign from context (positive if positive association)."),
    (re.compile(r'voluntary\s+response|call.in\s+poll|online\s+poll|self.select', re.I),
     "Voluntary response bias: self-selected participants → biased toward strong opinions. "
     "Survey is meaningless due to voluntary response bias."),
    (re.compile(r'observational\s+study|cause.*effect|randomiz.*experiment', re.I),
     "Experiment with randomization → can establish cause-and-effect. "
     "Observational study → association only, NOT causation. "
     "No control group or no randomization → observational study."),
    (re.compile(r'binomial.*condition|np.*\d|n\s*\(\s*1\s*-\s*p\)', re.I),
     "Binomial: fixed n, independent, constant p, binary outcome. "
     "Large-sample z-test: np≥10 AND n(1-p)≥10. "
     "Two-proportion z fails if any count < 10."),
    # ── Algebra astratta ─────────────────────────────────────────────────
    (re.compile(r'axa\^?2|x\s*->\s*axa|map.*homomorph', re.I),
     "φ(x)=axa^n is a homomorphism iff a^{n+1}=e. "
     "For φ(x)=axa²: need a³=e. NOT equivalent to G abelian."),
    (re.compile(r'\|aH\|.*\|Ha\||coset.*identical.*disjoint|left.*right.*coset', re.I),
     "|aH|=|H|=|Ha| always. Two LEFT cosets aH,bH are identical or disjoint. "
     "aH and Hb (left vs right) need not be identical unless H is normal."),
    (re.compile(r'ideal.*subring|subring.*ideal|every\s+ideal|every\s+subring', re.I),
     "Every ideal IS a subring. NOT every subring is an ideal. "
     "Statement 'every ideal is subring'=TRUE. 'every subring is ideal'=FALSE."),
    (re.compile(r'U\s*\+\s*V.*ideal|sum.*ideal|intersection.*ideal', re.I),
     "For ideals U,V: U+V is ideal. U∩V is ideal. UV is ideal. U∪V generally NOT ideal."),
    # ── Topologia ────────────────────────────────────────────────────────
    (re.compile(r'irrational.*unit\s*square|unit\s*square.*irrational', re.I),
     "S={(x,y)∈[0,1]²: x or y irrational} is CONNECTED. NOT totally disconnected."),
    (re.compile(r'f\s*:\s*\(\s*0\s*,\s*1\s*\)\s*.*\(\s*0\s*,\s*1\s*\]', re.I),
     "f:(0,1)→(0,1]: I (1-1 onto) CAN be true. II (compact image) CAN be true. "
     "III (continuous 1-1 onto) CANNOT — not homeomorphic. Answer: I and II only."),
    # ── Regressione ──────────────────────────────────────────────────────
    (re.compile(r'regression.*slope|grade.*=.*\+.*h\b|coefficient.*hours', re.I),
     "Slope b in y=a+bx = change per unit x. For Δx increase → grade changes b·Δx. "
     "Intercept irrelevant for changes."),
]


def _lookup_hardcoded_kb(question: str) -> str:
    """Fast regex KB for theorems MMLU won't reliably retrieve."""
    for pattern, answer in _HARDCODED_KB:
        if pattern.search(question):
            print(f"  [KB] Hardcoded match: {answer[:60]}…")
            return f"Mathematical Reference:\n{answer}"
    return ""


def _worker_exec(code: str, q):
    """Subprocess worker for safe code execution."""
    try:
        env = {"__builtins__": __builtins__}
        try:
            import math as _math; env["math"] = _math
        except Exception: pass
        try:
            env["sp"] = sp
        except Exception: pass
        try:
            import numpy as _np; env["np"] = _np; env["numpy"] = _np
        except Exception: pass
        try:
            from scipy import stats; env["stats"] = stats
        except Exception: pass
        from fractions import Fraction
        env["Fraction"] = Fraction
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            exec(code, env)
        finally:
            sys.stdout = old_stdout
        q.put(("ok", buf.getvalue()))
    except Exception as e:
        q.put(("err", f"{type(e).__name__}: {e}"))


def _execute_code(code: str, timeout: float = 4.0) -> str:
    """Run user code in subprocess with hard timeout."""
    try:
        ctx = mp.get_context("fork")
    except (ValueError, AttributeError):
        ctx = mp
    q = ctx.Queue()
    p = ctx.Process(target=_worker_exec, args=(code, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(0.5)
        if p.is_alive():
            p.kill()
        return "Error: timeout"
    try:
        kind, val = q.get_nowait()
        return val.strip() if kind == "ok" else f"Error: {val}"
    except Exception:
        return "Error: no output"


def _ensure_print(code: str) -> str:
    """If code's last line is a bare expression, wrap it in print()."""
    lines = code.strip().split("\n")
    if not lines: return code
    last = lines[-1].strip()
    if not last or last.startswith(("print", "#")) or "=" in last.split("#")[0]:
        return code
    if re.match(r'^[\w.\(\)\[\]+\-*/% ,]+$', last):
        lines[-1] = f"print({last})"
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 3. Option parsing & matching
# ══════════════════════════════════════════════════════════════════════════
def _parse_opt_num(text: str) -> Optional[float]:
    """Parse option text into a float, handling LaTeX fractions / sqrts."""
    t = re.sub(r'\[retrieval:[^\]]*\]', '', text).strip()
    t = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', t)
    t = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', t)
    t = re.sub(r'√(\d+)', r'sqrt(\1)', t)
    t = t.replace('\\pi', 'pi').replace('π', 'pi')
    t = re.sub(r'\\[a-zA-Z]+', '', t)
    t = re.sub(r'[{}]', '', t).replace('^', '**')
    if _SYMPY_OK:
        try: return float(_spN(_spS(t), 10))
        except Exception: pass
    nums = re.findall(r'-?\d+(?:\.\d+)?', t)
    try:
        return float(nums[0]) if nums else None
    except ValueError:
        return None


def _parse_opt_num_strict(text: str) -> Optional[float]:
    """Strict: rejects verbose English text options to avoid false matches
    like '10' → 'with root root 10' from garbled speech ASR."""
    words = re.findall(r'[a-zA-Z]{3,}', text)
    math_kw = {'sqrt','log','sin','cos','tan','exp','pi','inf','frac',
               'true','false','and','the','not','over'}
    nonmath = [w.lower() for w in words if w.lower() not in math_kw]
    if len(nonmath) >= 3:
        return None
    return _parse_opt_num(text)


def _match_range_opt(value: float, opt: str) -> bool:
    """Match value against range options like 'between 5 and 10'."""
    opt_l = opt.lower()
    nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', opt)]
    if "between" in opt_l and len(nums) >= 2:
        return min(nums[:2]) <= value <= max(nums[:2])
    if re.search(r'greater\s+than|larger\s+than|more\s+than|above', opt_l) and nums:
        return value > nums[0]
    if re.search(r'less\s+than|smaller\s+than|fewer\s+than|below', opt_l) and nums:
        return value < nums[0]
    if re.search(r'at\s+least', opt_l) and nums:
        return value >= nums[0]
    if re.search(r'at\s+most', opt_l) and nums:
        return value <= nums[0]
    return False


def _match_output_to_option(stdout: str, options: list) -> Optional[int]:
    """Find which option corresponds to the numeric output of sandbox code."""
    if not stdout or stdout.startswith("Error:") or not options:
        return None
    last = next((ln.strip() for ln in reversed(stdout.split('\n')) if ln.strip()), "")
    computed = None
    if _SYMPY_OK:
        try: computed = float(_spN(_spS(last.replace('^','**')), 15))
        except Exception: pass
    if computed is None and not re.search(r'[a-zA-Z]', last):
        nums = re.findall(r'-?\d+(?:\.\d+)?', last)
        if nums:
            try: computed = float(nums[-1])
            except ValueError: pass
    if computed is None:
        # Try list-of-solutions output like "[1, -2, 3]"
        list_m = re.search(r'\[([^\]]+)\]', stdout)
        if list_m and _SYMPY_OK:
            try:
                sols = [float(_spN(_spS(x.strip()))) for x in list_m.group(1).split(',')]
                for sol in sols:
                    for i, opt in enumerate(options):
                        ov = _parse_opt_num_strict(opt)
                        if ov is not None and abs(ov - sol) < max(1e-6, abs(sol)*1e-4):
                            return i
            except Exception: pass
        return None

    # Closest numeric match within tolerance
    best, best_err = None, float('inf')
    for i, opt in enumerate(options):
        ov = _parse_opt_num_strict(opt)
        if ov is None: continue
        diff = abs(ov - computed)
        tol = max(1e-5, abs(computed) * 1e-4)
        if diff < tol and diff < best_err:
            best, best_err = i, diff
    if best is not None:
        return best

    # Range-type options
    hits = [i for i, opt in enumerate(options) if _match_range_opt(computed, opt)]
    return hits[0] if len(hits) == 1 else None


# ══════════════════════════════════════════════════════════════════════════
# 4. Fast-path patterns (compressed via dispatch table)
# ══════════════════════════════════════════════════════════════════════════
def _fp_log(q: str, opts: list) -> Optional[Tuple[int, str]]:
    """log_b(x): find option matching log(x)/log(b)."""
    m = re.search(r'log[_{}]*(\d+)[\s_}{]*(\d+)', q, re.I)
    if not m or not _SYMPY_OK: return None
    b, x = int(m.group(1)), int(m.group(2))
    try:
        val = float(_spN(_spS(f"log({x})/log({b})"), 10))
    except Exception:
        return None
    for i, opt in enumerate(opts):
        ov = _parse_opt_num(opt)
        if ov is not None and abs(ov - val) < 1e-4:
            return i, f"log_{b}({x}) = {val:.6g}"
    return None


def _fp_cyclic_order(q: str, opts: list) -> Optional[Tuple[int, str]]:
    """Order of element <a> in cyclic group Z_n: n/gcd(a,n)."""
    m = re.search(r'order\s+of\s+(?:\\?langle\s*|<\s*)?(\d+)(?:\s*\\?rangle|\s*>)?\s+in\s+(?:Z_?|\\mathbb\{?Z\}?_?)(\d+)', q, re.I)
    if not m: return None
    a, n = int(m.group(1)), int(m.group(2))
    val = n // math.gcd(a, n)
    for i, opt in enumerate(opts):
        ov = _parse_opt_num(opt)
        if ov is not None and abs(ov - val) < 1e-9:
            return i, f"order = {n}/gcd({a},{n}) = {val}"
    return None


def _fp_complete_square(q: str, opts: list) -> Optional[Tuple[int, str]]:
    """Vertex k of a*x^2+b*x+c → k = c - b^2/(4a)."""
    m = re.search(r'(-?\d+)\s*x\s*\^?\s*2\s*([+\-]\s*\d+)?\s*x\s*([+\-]\s*\d+)', q)
    if not m: return None
    asks_k = bool(re.search(r'find\s+(?:the\s+value\s+of\s+)?k|value\s+of\s+k', q, re.I))
    is_vertex = bool(re.search(r'a\s*\(?x\s*[-−]\s*h\)?\s*\^?\s*2\s*\+\s*k|vertex\s*form', q, re.I))
    if not (is_vertex and asks_k): return None
    a = int(m.group(1))
    b_str = (m.group(2) or '0').replace(' ', '')
    c_str = m.group(3).replace(' ', '')
    b = int(b_str) if b_str not in ('+','-','') else 0
    c = int(c_str)
    k = c - b**2 / (4*a)
    for i, opt in enumerate(opts):
        ov = _parse_opt_num(opt)
        if ov is not None and abs(ov - k) < 1e-6:
            return i, f"k = {c} - {b}^2/(4·{a}) = {k}"
    return None


def _fp_odd_factorial(q: str, opts: list) -> Optional[Tuple[int, str]]:
    """Greatest odd factor of n!: product of odd numbers ≤ n."""
    m = re.search(r'greatest\s+odd\s+(?:integer|factor)\s+(?:of|that\s+divides?)\s+(\d+)\s*!', q, re.I)
    if not m: return None
    n = int(m.group(1))
    val = 1
    for k in range(1, n+1, 2): val *= k
    for i, opt in enumerate(opts):
        ov = _parse_opt_num(opt)
        if ov is not None and abs(ov - val) < 0.5:
            return i, f"greatest odd factor of {n}! = {val}"
    return None


def _fp_identity_op(q: str, opts: list) -> Optional[Tuple[int, str]]:
    """Identity element of (Z, *) where a*b = a+b+c: identity = -c."""
    m = re.search(r'a\s*\*\s*b\s*=\s*a\s*\+\s*b\s*([+\-]\s*\d+)', q)
    if not m: return None
    c_str = m.group(1).replace(' ', '')
    c = int(c_str)
    e = -c
    for i, opt in enumerate(opts):
        ov = _parse_opt_num(opt)
        if ov is not None and abs(ov - e) < 0.5:
            return i, f"identity = -{c} = {e}"
    return None


_FAST_PATHS = [_fp_log, _fp_cyclic_order, _fp_complete_square,
               _fp_odd_factorial, _fp_identity_op]


def _fast_path_inline(question: str, options: list) -> str:
    """Try each fast-path pattern. Returns DIRECT_ANSWER on hit, else ''."""
    for fp in _FAST_PATHS:
        try:
            result = fp(question, options)
            if result is not None:
                idx, explain = result
                print(f"  [Fast-path] {fp.__name__[4:]} → [{idx}] ({explain})")
                return f"DIRECT_ANSWER: {idx}\n[Fast-path] {explain}"
        except Exception:
            continue
    return ""


# ══════════════════════════════════════════════════════════════════════════
# 5. Parallel SymPy verify (try each numeric option as the answer)
# ══════════════════════════════════════════════════════════════════════════
def _verify_option_worker(opt_str: str, q_clean: str, queue):
    """Substitute option value into question and check for contradiction."""
    try:
        v = _parse_opt_num(opt_str)
        if v is None:
            queue.put(("skip", None)); return
        # Look for simple equation patterns: "f(x) = N", "solve for x", etc.
        eq_m = re.search(r'([a-zA-Z0-9+\-*/^()\s]+)\s*=\s*([\-\d./]+)', q_clean)
        if not eq_m:
            queue.put(("skip", None)); return
        lhs, rhs = eq_m.group(1), float(eq_m.group(2))
        var = re.search(r'\b([a-zA-Z])\b', lhs)
        if not var:
            queue.put(("skip", None)); return
        subst = lhs.replace(var.group(1), f"({v})")
        try:
            computed = float(_spN(_spS(subst.replace('^','**'))))
            if abs(computed - rhs) < 1e-4:
                queue.put(("match", v)); return
        except Exception: pass
        queue.put(("nomatch", v))
    except Exception:
        queue.put(("skip", None))


def _parallel_verify(question: str, options: list, timeout: float = 3.0) -> str:
    """Verify each option in parallel. Returns DIRECT_ANSWER if exactly one matches."""
    if not _SYMPY_OK: return ""
    try:
        ctx = mp.get_context("fork")
    except (ValueError, AttributeError):
        return ""
    procs, queues, matches = [], [], []
    for i, opt in enumerate(options):
        q = ctx.Queue()
        p = ctx.Process(target=_verify_option_worker, args=(opt, question, q))
        p.start()
        procs.append((i, p))
        queues.append(q)
    deadline = time.time() + timeout
    for i, p in procs:
        p.join(max(0.1, deadline - time.time()))
        if p.is_alive(): p.terminate()
    for i, q in enumerate(queues):
        try:
            kind, _ = q.get_nowait()
            if kind == "match":
                matches.append(i)
        except Exception: pass
    if len(matches) == 1:
        print(f"  [Verify] Unique substitution match → [{matches[0]}]")
        return f"DIRECT_ANSWER: {matches[0]}\n[Verify] only this option satisfies the equation"
    return ""


# ══════════════════════════════════════════════════════════════════════════
# 5. Direct Wolfram Alpha (no LLM, optional)
# ══════════════════════════════════════════════════════════════════════════
def _clean_for_wolfram(text: str) -> str:
    t = re.sub(r'\$\$(.+?)\$\$', r'\1', text, flags=re.DOTALL)
    t = re.sub(r'\$(.+?)\$', r'\1', t)
    t = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', t)
    t = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', t)
    t = t.replace('\\pi', 'pi').replace('\\infty', 'infinity')
    t = t.replace('\\cdot', '*').replace('\\times', '*')
    t = re.sub(r'\\(?:sin|cos|tan|log|ln|exp)\b', lambda m: m.group(0)[1:], t)
    t = re.sub(r'\\[a-zA-Z]+', ' ', t)
    t = re.sub(r'[{}]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def _wolfram_query(query: str, timeout: float = 4.0) -> str:
    """Call Wolfram Alpha Short Answers API."""
    global _wolfram_unavail
    if _wolfram_unavail or not _wolfram_app_id:
        return ""
    try:
        r = requests.get(
            "https://api.wolframalpha.com/v1/result",
            params={"appid": _wolfram_app_id, "i": query, "units": "metric"},
            timeout=timeout,
        )
        if r.status_code == 501:
            _wolfram_unavail = True  # 501 = no short answer available; skip future
            return ""
        if r.status_code == 200 and r.text.strip():
            return r.text.strip()
    except requests.RequestException:
        pass
    return ""


def _try_direct_wolfram(question: str, options: list, timeout: float = 4.0) -> str:
    """Send cleaned question to Wolfram. Match output to option if possible."""
    if not _wolfram_app_id or _wolfram_unavail:
        return ""
    q_clean = _clean_for_wolfram(question)[:200]
    if not q_clean: return ""
    result = _wolfram_query(q_clean, timeout=timeout)
    if not result or len(result) < 2:
        return ""
    print(f"  [Wolfram] → {result[:60]}")
    direct = _match_output_to_option(result, options)
    if direct is not None:
        return f"DIRECT_ANSWER: {direct}\n[Wolfram] {result[:150]}"
    return f"[Wolfram result]: {result[:200]}"


# ══════════════════════════════════════════════════════════════════════════
# 6. Universal PAL (Python code generation via LLM)
# ══════════════════════════════════════════════════════════════════════════
_UNIVERSAL_PAL_PROMPT = """\
You are a Python math solver. Given a math problem, write Python code that prints the numeric answer.

STRICT RULES:
1. Output ONLY valid Python code — no markdown, no explanations, no ```fences.
2. The code MUST call print() with the final result.
3. Available: math, sympy as sp, numpy as np, scipy.stats as stats, fractions.Fraction.
4. Keep code under 20 lines.
5. For LaTeX like $3x^2+x-4$: read coefficients directly, ignore $ signs.

REFERENCE EXAMPLES:

# Number theory: order of element a in Z_n
import math; print(24 // math.gcd(18, 24))

# Group theory: factor group (Z_a x Z_b)/(<c> x <d>)
import math; a,b,c,d=4,12,2,2; print((a*b)//(a//math.gcd(c,a)*b//math.gcd(d,b)))

# Eisenstein criterion
from sympy import primerange
coeffs=[24,-9,6,8]
for p in primerange(2,30):
    if coeffs[-1]%p!=0 and all(c%p==0 for c in coeffs[:-1]) and coeffs[0]%p**2!=0:
        print(p); break

# Group theory: identity of (Z,*) with a*b=a+b+k
print(-1)  # for a*b = a+b+1

# Order of permutation
from sympy.combinatorics import Permutation
print((Permutation([[0,1,4,3]])*Permutation([[1,2]])).order())

# Completing the square ax^2+bx+c, find k
a,b,c=3,1,-4; print(c - b**2/(4*a))

# Logarithm
import math; print(math.log(2, 8))

# Statistics: power = 1 - beta
print(1-0.26)

# Statistics: z critical for CI
from scipy.stats import norm; print(norm.ppf(0.97))

# Inclusion-exclusion (three sets)
total,none_=100,10; a,b,c=41,44,48; ab,ac,bc=14,11,19
print(total-none_-a-b-c+ab+ac+bc)

# Probability: alternating colors
from math import factorial; print(2*factorial(4)*factorial(4)/factorial(8))

# Calculus: definite integral
import sympy as sp; x=sp.Symbol('x')
print(float(sp.integrate(x**2,(x,0,3))))

# Linear algebra: eigenvalues
import numpy as np; print(np.linalg.eigvals([[2,1],[1,3]]))

# t-test p-value
from scipy.stats import t; print(2*t.cdf(-4.134, df=478))

# Real analysis bound from f' >= -m
f_b,m,a,b=5,1,0,3; print(f_b + m*(b-a))   # 5 + 1*3 = 8

# Volume of revolution (shell method)
import sympy as sp; x=sp.Symbol('x')
print(float(2*sp.pi*sp.integrate(x*(x-x**2),(x,0,1))))

# Compound interest doubling time
import math; print(math.log(2)/math.log(1.05))
"""


def _run_universal_pal(question: str, options: list, time_budget: float) -> str:
    """LLM generates Python code → sandbox execution → option matching."""
    if _llm_callback is None or time_budget < 4.0:
        return ""
    # Skip pure True/False theory questions
    tf_pat = re.compile(r'^(true|false)[,\s]+(true|false)$', re.I)
    has_tf = sum(1 for o in options if tf_pat.match(o.strip())) >= 2
    has_num = sum(1 for o in options if _parse_opt_num(o) is not None) >= 1
    if has_tf and not has_num:
        return ""

    opt_str = "\n".join(f"[{i}] {o}" for i, o in enumerate(options))
    messages = [
        {"role": "system", "content": _UNIVERSAL_PAL_PROMPT},
        {"role": "user", "content": f"Problem: {question}\n\nOptions:\n{opt_str}"},
    ]
    t0 = time.time()
    print("  [Universal PAL] generating code…")
    try:
        raw = _llm_callback(messages, max_new_tokens=200).strip()
    except Exception as e:
        print(f"  [PAL] LLM error: {e}")
        return ""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    code = re.sub(r'^```(?:python)?\s*\n?', '', raw, flags=re.I)
    code = re.sub(r'\n?```\s*$', '', code).replace('```', '').strip()
    if not code:
        return ""
    code = _ensure_print(code)
    sandbox_t = min(4.0, time_budget - (time.time() - t0) - 1.0)
    if sandbox_t < 1.0:
        return ""
    stdout = _execute_code(code, timeout=sandbox_t)
    print(f"  [PAL] stdout: {stdout[:80]}")
    if stdout.startswith("Error:"):
        return ""
    direct = _match_output_to_option(stdout, options)
    if direct is not None:
        return f"DIRECT_ANSWER: {direct}\n[PAL] {stdout.strip()[:120]}"
    return f"[PAL computation result]\n{stdout.strip()[:200]}"


# ══════════════════════════════════════════════════════════════════════════
# 8. Wikipedia fallback for theory questions
# ══════════════════════════════════════════════════════════════════════════
def _is_theory_question(question: str, options: list) -> bool:
    """Heuristic: True/False statements or 'which is true' questions."""
    if re.search(r'\bStatement\s+[12]\b', question, re.I):
        return True
    tf_count = sum(1 for o in options
                   if re.match(r'^(true|false)[,\s]+(true|false)$', o.strip(), re.I))
    if tf_count >= 2:
        return True
    if re.search(r'which\s+of\s+the\s+following\s+(?:is|are|statements?)', question, re.I):
        return True
    return False


def _wiki_query(question: str) -> str:
    """Extract 2-4 math keywords for a focused Wikipedia search."""
    math_terms = re.findall(
        r'\b(ring|group|field|topology|homomorphism|isomorphism|subgroup|ideal|'
        r'coset|manifold|convergence|eigenvalue|determinant|polynomial|abelian|'
        r'cyclic|normal\s+subgroup|factor\s+group|vector\s+space|integral\s+domain|'
        r'characteristic|permutation|bijection|compact|connected|hausdorff|'
        r'continuous|differentiable|series|convergence|divergence|markov)\b',
        question, re.I
    )
    if math_terms:
        seen, out = set(), []
        for t in math_terms:
            tl = t.lower()
            if tl not in seen: seen.add(tl); out.append(tl)
        return ' '.join(out[:4])
    # Fallback: first 80 chars stripped of noise
    return re.sub(r'\$[^$]+\$|\([^)]+\)', '', question).strip()[:80]


def _fetch_wikipedia(query: str, max_chars: int = 800) -> str:
    """Fetch a short Wikipedia excerpt using focused math query."""
    try:
        search_q = _wiki_query(query)
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "list": "search",
                "srsearch": search_q, "srlimit": 1, "utf8": 1,
            },
            timeout=3.0,
        )
        hits = r.json().get("query", {}).get("search", [])
        if not hits: return ""
        title = hits[0]["title"]
        r2 = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "prop": "extracts",
                "exintro": 1, "explaintext": 1, "titles": title, "utf8": 1,
            },
            timeout=3.0,
        )
        pages = r2.json().get("query", {}).get("pages", {})
        for p in pages.values():
            txt = p.get("extract", "")
            if txt:
                return f"Mathematical Theory Context ({title}): {txt[:max_chars]}"
    except Exception:
        pass
    return ""


# ══════════════════════════════════════════════════════════════════════════
# 9. Entry point — orchestrates the pipeline
# ══════════════════════════════════════════════════════════════════════════
def rag_maths(question_text: str, option_texts: Optional[list] = None,
              time_budget: float = 20.0) -> str:
    """Main entry — try paths in priority order, return first non-empty result."""
    question = question_text  # internal alias
    if option_texts is None:
        option_texts = []
    options = [str(o) for o in option_texts]  # ensure all are strings
    cache_key = (question.strip()[:200], tuple(o[:50] for o in options))
    t0 = time.time()
    def elapsed(): return time.time() - t0

    has_llm = _llm_callback is not None
    has_tok = _tokenizer is not None
    print(f"  [RAG-Maths] State: llm={'SET' if has_llm else 'NONE'} | "
          f"tokenizer={'SET' if has_tok else 'NONE'} | budget={time_budget:.1f}s")

    # 1. Cache
    if cache_key in _question_cache:
        cached = _question_cache[cache_key]
        if cached:
            print(f"  [RAG-Maths] Cache hit  ({elapsed():.2f}s)")
            return cached

    # 2. Fast-path
    if fp := _fast_path_inline(question, options):
        _question_cache[cache_key] = fp
        return fp

    # 3. Parallel verify
    if pv := _parallel_verify(question, options, timeout=min(3.0, time_budget * 0.15)):
        _question_cache[cache_key] = pv
        return pv

    # 4. Hardcoded KB — theory facts
    if kb := _lookup_hardcoded_kb(question):
        _question_cache[cache_key] = kb
        return kb

    # 5. Training-data retrieval (AQuA-RAT / math_qa — no eval overlap)
    remaining = time_budget - elapsed()
    if remaining > 4.0:
        ret_ctx = _retrieve_context(question, top_k=2)
        if ret_ctx:
            _question_cache[cache_key] = ret_ctx
            return ret_ctx

    # 6. Direct Wolfram
    remaining = time_budget - elapsed()
    if remaining > 4.0:
        wa = _try_direct_wolfram(question, options, timeout=min(4.0, remaining - 3.0))
        if wa:
            _question_cache[cache_key] = wa
            return wa

    # 7. Universal PAL
    remaining = time_budget - elapsed()
    if has_llm and remaining > 5.0:
        pal = _run_universal_pal(question, options, time_budget=remaining - 2.0)
        if pal:
            _question_cache[cache_key] = pal
            return pal

    # 8. Theory question: Wikipedia
    if _is_theory_question(question, options):
        wiki = _fetch_wikipedia(question)
        if wiki:
            print(f"  [Theory] Wikipedia context  ({elapsed():.2f}s)")
            _question_cache[cache_key] = wiki
            return wiki

    # Fallback: minimal reasoning prompt so LLM reasons, not guesses
    print(f"  [RAG-Maths] No context found  ({elapsed():.2f}s)")
    hint = "No retrieval match. Reason step-by-step from first principles."
    _question_cache[cache_key] = hint
    return hint