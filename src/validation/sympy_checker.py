"""SymPy-based equivalence checker for AMPS wrong-answer validation.

AMPS answers are in LaTeX format (e.g. $\frac{\sqrt{77}-\sqrt{23}}{\sqrt{10}+\sqrt{34}}$).
Uses sympy.parsing.latex.parse_latex for single-expression answers and falls back to
parse_error for multi-part answers (disjunctions, multi-variable systems, structured
outputs like conic-section classifications) where symbolic comparison is intractable.

Classifications:
  "equivalent"     : SymPy proves the two expressions are equal
  "not_equivalent" : SymPy parses both and they differ
  "parse_error"    : multi-part, structured, or unparseable answer — needs manual review
"""

from __future__ import annotations

import re


# LaTeX constructs that signal a multi-part or structured answer we won't try to parse.
_MULTIPART_PATTERNS = [
    r'\\lor',           # disjunction  x=a \lor x=b
    r'\\land',          # conjunction
    r'\\begin\{',       # matrix / array / environment
    r'\\text\{',        # embedded prose
    r'\n',              # multi-line (conic sections etc.)
    r'Classification',  # structured conic output
    r'Foci',
    r'Asymptotes',
    r'Eccentricity',
    r'Center',
    r'Equation:',
]

_MULTIPART_RE = re.compile('|'.join(_MULTIPART_PATTERNS), re.IGNORECASE)


def _strip_latex_delimiters(s: str) -> str:
    """Strip surrounding $...$ or $$...$$ and cleanup display hints."""
    s = s.strip()
    s = re.sub(r'\$\$(.+?)\$\$', r'\1', s, flags=re.DOTALL)
    s = re.sub(r'\$(.+?)\$', r'\1', s, flags=re.DOTALL)
    # \left / \right are only sizing hints — remove them
    s = re.sub(r'\\left\s*', '', s)
    s = re.sub(r'\\right\s*', '', s)
    return s.strip()


def _extract_rhs(expr: str) -> str:
    """For 'var = expr' patterns, return just the RHS."""
    # e.g. "x=\frac{143}{211}" -> "\frac{143}{211}"
    m = re.match(r'^[a-zA-Z]\s*=\s*(.+)$', expr.strip())
    return m.group(1).strip() if m else expr


def _try_parse_latex(expr_str: str):
    """Parse a LaTeX math string with sympy. Returns sympy expr or None."""
    try:
        from sympy.parsing.latex import parse_latex
        cleaned = _strip_latex_delimiters(expr_str)
        cleaned = _extract_rhs(cleaned)
        return parse_latex(cleaned)
    except Exception:
        return None


def _try_parse_sympify(expr_str: str):
    """Fallback: try sympy.sympify on plain-text expressions."""
    try:
        import sympy
        cleaned = expr_str.strip().replace('^', '**').replace('$', '')
        cleaned = _extract_rhs(cleaned)
        return sympy.sympify(cleaned, evaluate=True)
    except Exception:
        return None


def _try_parse(expr_str: str):
    result = _try_parse_latex(expr_str)
    if result is None:
        result = _try_parse_sympify(expr_str)
    return result


def check_equivalence(correct: str, proposed: str) -> dict:
    """Check whether proposed is mathematically equivalent to correct.

    Returns a dict with keys:
      status : "equivalent" | "not_equivalent" | "parse_error"
      detail : human-readable explanation
    """
    if not correct.strip() or not proposed.strip():
        return {"status": "parse_error", "detail": "empty string"}

    # Multi-part or structured answers — skip symbolic comparison
    for s in (correct, proposed):
        if _MULTIPART_RE.search(s):
            return {"status": "parse_error",
                    "detail": f"multi-part or structured answer — manual review needed"}

    # Handle multi-variable answers: "$x=a$, $y=b$"
    # Extract all $...$ tokens; if there's more than one, it's multi-part
    dollar_blocks = re.findall(r'\$[^$]+\$', correct)
    if len(dollar_blocks) > 1:
        return {"status": "parse_error",
                "detail": "multi-variable answer (multiple $...$ blocks)"}

    expr_correct = _try_parse(correct)
    expr_proposed = _try_parse(proposed)

    if expr_correct is None:
        return {"status": "parse_error",
                "detail": f"could not parse correct answer: {correct!r}"}
    if expr_proposed is None:
        return {"status": "parse_error",
                "detail": f"could not parse proposed answer: {proposed!r}"}

    try:
        import sympy
        diff = sympy.simplify(expr_correct - expr_proposed)
        if diff == 0:
            return {"status": "equivalent",
                    "detail": f"simplify(correct - proposed) == 0"}

        # Numerical check at a test point (catches cases where simplify is slow/inconclusive)
        free_syms = expr_correct.free_symbols | expr_proposed.free_symbols
        try:
            subs = {s: sympy.Rational(7, 3) for s in free_syms}
            val_c = complex(expr_correct.subs(subs).evalf())
            val_p = complex(expr_proposed.subs(subs).evalf())
            if abs(val_c - val_p) < 1e-9:
                return {"status": "equivalent",
                        "detail": f"numerically equal at test point (simplify returned {diff})"}
        except Exception:
            pass

        # Pure numeric (no free symbols)
        try:
            fc = float(expr_correct.evalf())
            fp = float(expr_proposed.evalf())
            if abs(fc - fp) < 1e-9:
                return {"status": "equivalent",
                        "detail": f"numerically equal: {fc} ≈ {fp}"}
        except Exception:
            pass

        return {"status": "not_equivalent",
                "detail": f"simplify difference = {diff}"}

    except Exception as e:
        return {"status": "parse_error", "detail": f"sympy error during comparison: {e}"}
