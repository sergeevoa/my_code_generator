"""
Trace-Augmented Debugging: AST-based code instrumenter.

Public API:
    instrument(source)                    → instrumented source str | None
    extract_and_compress_trace(output)    → compact trace str | None

How it works:
  instrument() rewrites every function body in the source, inserting
  __trace_log__() calls after variable assignments, at the start of each
  loop iteration, and before simple-name returns.  Top-level test/assertion
  statements are wrapped in try/finally so the trace is always flushed to
  stdout even when an assertion fails.

  The sandbox output then contains:
      __TRACE_START__
      L3 count=0
      L5 c='H'
      ...
      __TRACE_END__

  extract_and_compress_trace() finds these markers, deduplicates repetitive
  lines, and truncates long runs so the model receives a compact observation.

Limitations (intentional for HumanEval scope):
  - Generator functions (yield) are not instrumented to avoid semantic change.
  - Only simple-name returns (return var) are logged; complex expressions
    are left as-is to avoid double evaluation / side effects.
  - Nested function definitions are not recursively instrumented.
"""

import ast
import textwrap
from typing import List, Optional

# ── Limits ────────────────────────────────────────────────────────────────────

# Max trace events kept in memory (tail-based: deque evicts oldest on overflow)
_TRACE_MAX_EVENTS = 200
# Max repr() length per variable value
_TRACE_MAX_VALUE_LEN = 100

# ── Trace header injected at the top of every instrumented code block ─────────
# Uses deque(maxlen=N) so only the LAST N events are retained — the tail of
# execution is where the bug symptom actually appears.
# Names prefixed with __ to minimise clashes with user code.

_TRACE_HEADER = """\
from collections import deque as __deque__
__trace_events__ = __deque__(maxlen=200)

def __trace_log__(lineno, name, value):
    try:
        v = repr(value)
        if len(v) > 100:
            v = v[:97] + '...'
    except Exception:
        v = '<repr_error>'
    __trace_events__.append('L' + str(lineno) + ' ' + name + '=' + v)

"""


# ── AST helpers ───────────────────────────────────────────────────────────────

def _names_from_targets(targets: list) -> List[str]:
    names: List[str] = []
    for t in targets:
        names.extend(_names_from_node(t))
    return names


def _names_from_node(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        result: List[str] = []
        for elt in node.elts:
            result.extend(_names_from_node(elt))
        return result
    return []


def _name_from_node(node: ast.AST) -> Optional[str]:
    return node.id if isinstance(node, ast.Name) else None


def _make_log_call(lineno: int, name: str) -> ast.Expr:
    """Return an AST node for:  __trace_log__(lineno, 'name', name)"""
    return ast.Expr(value=ast.Call(
        func=ast.Name(id="__trace_log__", ctx=ast.Load()),
        args=[
            ast.Constant(value=lineno),
            ast.Constant(value=name),
            ast.Name(id=name, ctx=ast.Load()),
        ],
        keywords=[],
    ))


def _contains_attribute(node: ast.AST) -> bool:
    """Return True if *node* or any descendant is an ast.Attribute."""
    return any(isinstance(n, ast.Attribute) for n in ast.walk(node))


def _attr_path_str(node: ast.AST) -> str:
    """Return a display string for an attribute chain, e.g. 'node.successor'."""
    if isinstance(node, ast.Attribute):
        return f"{_attr_path_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return "<expr>"


def _make_attr_error_guard(lineno: int, body: list) -> ast.Try:
    """
    Wrap *body* statements in:
        try:
            <body>
        except AttributeError as __ae{lineno}__:
            __trace_log__(lineno, 'AttributeError', str(__ae{lineno}__))
            raise
    """
    exc_var = f"__ae{lineno}__"
    handler = ast.ExceptHandler(
        type=ast.Name(id="AttributeError", ctx=ast.Load()),
        name=exc_var,
        body=[
            ast.Expr(value=ast.Call(
                func=ast.Name(id="__trace_log__", ctx=ast.Load()),
                args=[
                    ast.Constant(value=lineno),
                    ast.Constant(value="AttributeError"),
                    ast.Call(
                        func=ast.Name(id="str", ctx=ast.Load()),
                        args=[ast.Name(id=exc_var, ctx=ast.Load())],
                        keywords=[],
                    ),
                ],
                keywords=[],
            )),
            ast.Raise(),
        ],
    )
    node = ast.Try(body=body, handlers=[handler], orelse=[], finalbody=[])
    ast.copy_location(node, body[0])
    ast.fix_missing_locations(node)
    return node


def _has_yield(node: ast.AST) -> bool:
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))


# ── Statement-level instrumentation ──────────────────────────────────────────

def _instrument_stmts(stmts: list) -> list:
    """
    Walk a statement list and insert __trace_log__ calls.

    Recurses into For / While / If bodies.
    Does NOT recurse into nested FunctionDef / AsyncFunctionDef / ClassDef
    (to keep noise low and avoid double-instrumenting).
    """
    result: list = []

    for stmt in stmts:
        # ── Simple assignment: x = expr ───────────────────────────────────
        if isinstance(stmt, ast.Assign):
            log_stmts = [
                _make_log_call(stmt.lineno, name)
                for name in _names_from_targets(stmt.targets)
            ]
            if _contains_attribute(stmt.value):
                result.append(_make_attr_error_guard(stmt.lineno, [stmt] + log_stmts))
            else:
                result.append(stmt)
                result.extend(log_stmts)

        # ── Augmented assignment: x += expr ──────────────────────────────
        elif isinstance(stmt, ast.AugAssign):
            result.append(stmt)
            name = _name_from_node(stmt.target)
            if name:
                result.append(_make_log_call(stmt.lineno, name))

        # ── Annotated assignment: x: T = expr ────────────────────────────
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            result.append(stmt)
            name = _name_from_node(stmt.target)
            if name:
                result.append(_make_log_call(stmt.lineno, name))

        # ── Return with a simple name: return var ─────────────────────────
        # Only log when value is a bare Name to avoid double-evaluating
        # expressions with side effects.
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None and isinstance(stmt.value, ast.Name):
                result.append(_make_log_call(stmt.lineno, stmt.value.id))
            result.append(stmt)

        # ── For loop: log the loop variable at each iteration start ───────
        elif isinstance(stmt, ast.For):
            iter_logs: list = []
            name = _name_from_node(stmt.target)
            if name:
                iter_logs = [_make_log_call(stmt.lineno, name)]

            instrumented_body = iter_logs + _instrument_stmts(stmt.body)

            if isinstance(stmt.iter, ast.Attribute):
                # Guard the attribute access before iteration begins
                tmp = f"__it{stmt.lineno}__"
                tmp_assign = ast.Assign(
                    targets=[ast.Name(id=tmp, ctx=ast.Store())],
                    value=stmt.iter,
                    lineno=stmt.lineno,
                    col_offset=stmt.col_offset,
                )
                tmp_log = ast.Expr(value=ast.Call(
                    func=ast.Name(id="__trace_log__", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=stmt.lineno),
                        ast.Constant(value=_attr_path_str(stmt.iter)),
                        ast.Name(id=tmp, ctx=ast.Load()),
                    ],
                    keywords=[],
                ))
                ast.fix_missing_locations(tmp_assign)
                ast.fix_missing_locations(tmp_log)
                result.append(_make_attr_error_guard(stmt.lineno, [tmp_assign, tmp_log]))
                new_iter: ast.expr = ast.Name(id=tmp, ctx=ast.Load())
                ast.copy_location(new_iter, stmt.iter)
            else:
                new_iter = stmt.iter

            new_for = ast.For(
                target=stmt.target,
                iter=new_iter,
                body=instrumented_body,
                orelse=_instrument_stmts(stmt.orelse) if stmt.orelse else [],
            )
            ast.copy_location(new_for, stmt)
            ast.fix_missing_locations(new_for)
            result.append(new_for)

        # ── While loop: recurse into body ─────────────────────────────────
        elif isinstance(stmt, ast.While):
            new_while = ast.While(
                test=stmt.test,
                body=_instrument_stmts(stmt.body),
                orelse=_instrument_stmts(stmt.orelse) if stmt.orelse else [],
            )
            ast.copy_location(new_while, stmt)
            result.append(new_while)

        # ── If branch: recurse into both branches ─────────────────────────
        elif isinstance(stmt, ast.If):
            new_if = ast.If(
                test=stmt.test,
                body=_instrument_stmts(stmt.body),
                orelse=_instrument_stmts(stmt.orelse) if stmt.orelse else [],
            )
            ast.copy_location(new_if, stmt)
            result.append(new_if)

        # ── Nested definitions: pass through unchanged ────────────────────
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.append(stmt)

        # ── Everything else unchanged ─────────────────────────────────────
        else:
            result.append(stmt)

    return result


# ── Public entry point ────────────────────────────────────────────────────────

def instrument(source_code: str) -> Optional[str]:
    """
    Instrument Python source for trace-augmented debugging.

    Returns the instrumented source string, or None if parsing fails.

    Output structure:
        [trace header: __trace_log__ definition]
        [instrumented function definitions]
        try:
            [original test/assertion code]
        finally:
            print('__TRACE_START__')
            for __ev__ in __trace_events__:
                print(__ev__)
            print('__TRACE_END__')
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None

    if not tree.body:
        return None

    # Split module body into definitions (functions / classes / imports)
    # and executable test statements (asserts, calls, prints, etc.)
    def_stmts: list = []
    test_stmts: list = []
    for stmt in tree.body:
        if isinstance(stmt, (
            ast.FunctionDef, ast.AsyncFunctionDef,
            ast.ClassDef, ast.Import, ast.ImportFrom,
        )):
            def_stmts.append(stmt)
        else:
            test_stmts.append(stmt)

    # Instrument function definitions (skip generators — their semantics would
    # change if we introduced temp variables around yield expressions)
    for stmt in def_stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _has_yield(stmt):
                stmt.body = _instrument_stmts(stmt.body)

    # Unparse instrumented definitions
    if def_stmts:
        def_module = ast.Module(body=def_stmts, type_ignores=[])
        ast.fix_missing_locations(def_module)
        defs_code = ast.unparse(def_module)
    else:
        defs_code = ""

    # Wrap test statements in try/finally so trace is always flushed
    if test_stmts:
        test_module = ast.Module(body=test_stmts, type_ignores=[])
        ast.fix_missing_locations(test_module)
        raw_test = ast.unparse(test_module)
        trace_flush = (
            "    print('__TRACE_START__')\n"
            "    for __ev__ in __trace_events__:\n"
            "        print(__ev__)\n"
            "    print('__TRACE_END__')"
        )
        wrapped = (
            "try:\n"
            + textwrap.indent(raw_test, "    ")
            + "\nfinally:\n"
            + trace_flush
            + "\n"
        )
    else:
        wrapped = (
            "print('__TRACE_START__')\n"
            "for __ev__ in __trace_events__:\n"
            "    print(__ev__)\n"
            "print('__TRACE_END__')\n"
        )

    parts = [_TRACE_HEADER]
    if defs_code:
        parts.append(defs_code)
        parts.append("")
    parts.append(wrapped)
    return "\n".join(parts)


# ── Trace extraction and compression ─────────────────────────────────────────

def extract_and_compress_trace(output: str, max_shown: int = 60) -> Optional[str]:
    """
    Extract the trace from sandbox stdout and return a compact string.

    Steps:
      1. Find __TRACE_START__ / __TRACE_END__ markers.
      2. Collapse consecutive identical lines (repetitive loop iterations).
      3. Truncate head+tail if still over max_shown, noting how many were omitted.

    Returns None if markers are absent or trace is empty.
    """
    start_marker = "__TRACE_START__"
    end_marker   = "__TRACE_END__"

    s = output.find(start_marker)
    e = output.find(end_marker)
    if s == -1 or e == -1 or e <= s:
        return None

    raw_lines = [
        ln.strip()
        for ln in output[s + len(start_marker):e].split("\n")
        if ln.strip()
    ]
    if not raw_lines:
        return None

    # Collapse runs of identical lines
    compressed: List[str] = []
    prev: Optional[str] = None
    run = 0
    for line in raw_lines:
        if line == prev:
            run += 1
        else:
            if prev is not None:
                compressed.append(prev if run == 1 else f"{prev}  (×{run})")
            prev = line
            run = 1
    if prev is not None:
        compressed.append(prev if run == 1 else f"{prev}  (×{run})")

    # Truncate if needed, keeping head and tail
    if len(compressed) > max_shown:
        half    = max_shown // 2
        skipped = len(compressed) - max_shown
        compressed = (
            compressed[:half]
            + [f"  ... [{skipped} events omitted] ..."]
            + compressed[len(compressed) - (max_shown - half):]
        )

    return "\n".join(compressed)
