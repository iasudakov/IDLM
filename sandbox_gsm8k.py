"""GSM8K / TinyGSM answer grading by executing the generated program.

Vendored **verbatim** from the S-FLM repository
(https://github.com/jdeschena/s-flm, `sandbox_gsm8k.py`, itself adapted from
https://github.com/JaeyeonKim01/PUMA) so that `mode=gsm8k_eval` reproduces the
exact number the IDLM README points at -- see "Conditional TinyGSM Evaluation",
which prescribes evaluating our released TinyGSM checkpoints "together with the
TinyGSM evaluation code from the S-FLM repository". Keeping the grader
bit-for-bit identical is the whole point: a re-implementation would silently
move the accuracy through its extraction heuristics rather than through the
model, and the comparison against the published baselines would stop being
apples-to-apples. Do not "clean up" anything above the `IDLM additions` banner,
including the 4-space indentation that differs from the rest of this repo.

`evaluate_tinygsm_sample` below is the one addition, and it is additive: the
TinyGSM *training* split's ground truth is a Python program rather than a
`#### <number>` string, so validation-time accuracy has to execute the
reference solution to learn what the right answer even is. The GSM8K test path
never calls it.
"""

# Adapted from https://github.com/JaeyeonKim01/PUMA
import signal
import contextlib
import io
import re
import warnings
import math
import numpy as np


def _safe_exec_no_timer(code: str):
    """
    Executes code in a restricted environment (no timeout here).
    Timeout should be applied by wrapping the whole evaluate step with _time_limit().
    """
    

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum,
        "len": len, "range": range, "enumerate": enumerate,
        "int": int, "float": float, "str": str, "bool": bool,
        "round": round,
        "print": print,
    }

    def _limited_import(name, globals=None, locals=None, 
                        fromlist=(), level=0):
        if name == "math":
            return __import__(name, globals, locals, fromlist, 
                              level)
        raise ImportError(f"Import blocked: {name}")

    safe_builtins["__import__"] = _limited_import
    import math as _math
    ns = {"__builtins__": safe_builtins, "math": _math}

    # Catch output from executing code
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                exec(code, ns, ns)
    return ns


@contextlib.contextmanager
def _time_limit(timeout_s: float):
  """
  Hard wall-clock time limit using SIGALRM/ITIMER_REAL (POSIX).
  Note: works only in the main thread of the process.
  """
  has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
  old_handler = None
  if has_alarm:
      old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
      signal.setitimer(signal.ITIMER_REAL, timeout_s)
  try:
      yield
  finally:
      if has_alarm:
          signal.setitimer(signal.ITIMER_REAL, 0)
          signal.signal(signal.SIGALRM, old_handler)


class _Timeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _Timeout()


def _extract_code(text):
    """
    Heuristics:
      - If the sample contains a triple-backtick code block, extract that block.
      - Else, start from first 'def ' if present.
      - If parsing fails because of trailing junk, cut at the failing line.
    """
    # If the model wrapped the function in Markdown, prefer that block.
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, 
                      flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)

    # If there is explanatory text before the function, drop it.
    i = text.find("def ")
    if i != -1:
        text = text[i:]

    text = text.strip()
    if not text:
        return text

    # Best case: the extracted text is already valid Python.
    try:
        compile(text, "<sample>", "exec")
        return text
    except SyntaxError as err:
        # If parsing fails immediately, there is no useful prefix to recover.
        if err.lineno is None or err.lineno <= 1:
            return text
        error_lineno = err.lineno

    # Otherwise, keep the valid prefix before the failing line.
    lines = text.splitlines()
    candidate = "\n".join(lines[:error_lineno - 1]).strip()
    if not candidate:
        return text

    try:
        compile(candidate, "<sample>", "exec")
        return candidate
    except SyntaxError:
        return text


def _to_number(x):
    """
    Normalize return values to int or float where possible.
    """
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return None
        xf = float(x)
        if abs(xf - round(xf)) < 1e-6:
            return int(round(xf))
        return xf
    if isinstance(x, str):
        m = re.search(r"[-+]?\$?\d[\d,]*\.?\d*", x)
        if not m:
            return None
        s = m.group(0).replace(",", "").replace("$", "")
        if s.count(".") == 1:
            f = float(s)
            if abs(f - round(f)) < 1e-6:
                return int(round(f))
            return f
        return int(s)
    # tuples/lists etc -> not supported for GSM8K scoring
    return None


def _extract_gold_answer(answer):
    match = re.search(r"####\s*(.+)", answer)
    if match is not None:
        return _to_number(match.group(1))
    return _to_number(answer)


def _extract_text_answer(text: str):
    if not text:
        return None

    patterns = [
        r"(?:final answer|answer|result)\s*(?:is|=|:)\s*([-+]?\$?\d[\d,]*\.?\d*)",
        r"####\s*([-+]?\$?\d[\d,]*\.?\d*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return _to_number(matches[-1])

    matches = re.findall(r"[-+]?\$?\d[\d,]*\.?\d*", text)
    if matches:
        return _to_number(matches[-1])
    return None


def _numbers_equal(pred, gold):
    if pred is None or gold is None:
        return False
    if isinstance(pred, float) or isinstance(gold, float):
        return abs(float(pred) - float(gold)) <= 1e-3
    return int(pred) == int(gold)


def evaluate_samples(sample, answer, timeout_s):
    """
    sample: model output (string)
    answer: GSM8K answer (string)
    """
    code = _extract_code(sample)

    try:
        with _time_limit(timeout_s):
            ns = _safe_exec_no_timer(code)

            fn = ns.get("simple_math_problem", None)
            if fn is None:
                return False

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with contextlib.redirect_stderr(stderr):
                    out = fn()

    except (_Timeout, Exception):
        return False

    gold = _extract_gold_answer(answer)
    out = _to_number(out)
    if out is None:
        out = _extract_text_answer(stdout.getvalue())
    return _numbers_equal(out, gold)


# --------------------------------------------------------------------------
# IDLM additions
# --------------------------------------------------------------------------
# Everything below is IDLM's, written in this repo's style. It reuses the
# extraction/exec/comparison primitives above so a TinyGSM validation score and
# a GSM8K test score are graded by the same rules.


def run_solution(code, timeout_s):
  """Execute one candidate program and return the number it produced.

  Returns None when the program does not parse, blows the time limit, raises,
  defines no `simple_math_problem`, or returns something that is not a number
  (a tuple, a string with no digits, NaN). None means "no answer", which is
  never equal to anything -- `_numbers_equal` short-circuits on it.
  """
  code = _extract_code(code)
  try:
    with _time_limit(timeout_s):
      ns = _safe_exec_no_timer(code)
      fn = ns.get('simple_math_problem', None)
      if fn is None:
        return None
      stdout = io.StringIO()
      with contextlib.redirect_stdout(stdout):
        with contextlib.redirect_stderr(io.StringIO()):
          out = fn()
  except (_Timeout, Exception):
    return None
  number = _to_number(out)
  if number is None:
    number = _extract_text_answer(stdout.getvalue())
  return number


def evaluate_tinygsm_sample(sample, gold_code, timeout_s):
  """Grade a sample whose ground truth is itself a Python program.

  Returns ``(correct, gradable)``. `gradable` is False when the *reference*
  program failed to produce a number, in which case the example says nothing
  about the model and the caller should drop it from the denominator rather
  than score it wrong -- TinyGSM's programs are synthetic and a small fraction
  do not run under the restricted builtins this sandbox allows.
  """
  gold = run_solution(gold_code, timeout_s)
  if gold is None:
    return False, False
  return _numbers_equal(run_solution(sample, timeout_s), gold), True
