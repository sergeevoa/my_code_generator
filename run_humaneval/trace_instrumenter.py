# Thin re-export — the canonical source lives in src/trace_instrumenter.py,
# which is on sys.path for this package (see __init__.py).
from trace_instrumenter import instrument, extract_and_compress_trace  # noqa: F401
