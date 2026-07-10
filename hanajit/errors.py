class UnsupportedError(Exception):
    """Raised when code falls outside the compilable subset.

    The @jit decorator catches this and falls back to the CPython
    interpreter, preserving full Python-ecosystem compatibility.
    """
