from __future__ import annotations

from pydantic import PrivateAttr
import builtins
import contextlib
import io
import logging
import re
import sys
from typing import Any, Dict, Optional

import pandas as pd
from langchain_experimental.tools.python.tool import PythonAstREPLTool


class CustomPythonAstREPLTool(PythonAstREPLTool):
    """
    Executes Python code and captures stdout/stderr.
    Designed for running LLM-generated code that produces df_corrected.

    Inject variables/functions into the execution globals via set_runtime(...).
    """
    __name__ = "Custom_Python_AST_REPL"

    _exec_globals: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _runtime: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Base globals: builtins + common libs you allow
        self._exec_globals = {"__builtins__": builtins}
        # Optional: preload pandas in the environment
        self._exec_globals["pd"] = pd

        # Store runtime injections separately so you can reset them per call
        self._runtime = {}

    def set_runtime(self, *, curated_md: Optional[str] = None,
                    markdown_to_dataframe=None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Set runtime objects that the executed code can access.
        Call this before _run.
        """
        self._runtime = {}
        if curated_md is not None:
            self._runtime["curated_md"] = curated_md
        if markdown_to_dataframe is not None:
            self._runtime["markdown_to_dataframe"] = markdown_to_dataframe
        if extra:
            self._runtime.update(extra)

    @staticmethod
    def _extract_code(query: str) -> str:
        # Prefer fenced code blocks, but accept raw code too
        code_match = re.search(r"```(?:python)?\s*(.*?)```", query, re.DOTALL | re.IGNORECASE)
        code = code_match.group(1) if code_match else query

        # Remove trailing tool artifacts like "Observation"
        code = re.sub(r"\n?Observation\s*$", "", code.strip(), flags=re.IGNORECASE)
        return code.strip()

    def _fresh_globals(self) -> Dict[str, Any]:
        """
        Build globals for this run: base + runtime.
        """
        g = dict(self._exec_globals)  # shallow copy
        g.update(self._runtime)
        return g

    def _run(self, query: str, run_manager=None) -> str:
        code = self._extract_code(query)

        output_capture = io.StringIO()

        # Prepare run-specific globals
        g = self._fresh_globals()

        # Capture stdout/stderr including logging
        root_logger = logging.getLogger()
        handler_streams: Dict[logging.Handler, Any] = {}
        try:
            # Redirect root logger handler streams that write to stdout/stderr
            for h in root_logger.handlers:
                if hasattr(h, "stream"):
                    handler_streams[h] = h.stream
                    if h.stream in (sys.stdout, sys.stderr):
                        h.stream = output_capture

            with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
                # Execute the code
                exec(code, g, g)

                # Pull results
                df_corrected = g.get("df_corrected", None)
                corrected_md = g.get("corrected_md", None)

                # Emit a concise execution report
                if df_corrected is not None:
                    if not hasattr(df_corrected, "shape"):
                        raise TypeError("df_corrected exists but is not a pandas DataFrame-like object.")
                    print(f"[OK] df_corrected shape: {df_corrected.shape}", file=output_capture)
                    # Optional: show first few rows as a sanity check
                    try:
                        print(df_corrected.head(3).to_string(index=False), file=output_capture)
                    except Exception:
                        pass
                else:
                    print("[WARN] df_corrected not found in executed code.", file=output_capture)

                if corrected_md is not None:
                    # Don’t print the whole thing unless you want it
                    print(f"[OK] corrected_md length: {len(str(corrected_md))}", file=output_capture)

                # Persist updated globals (optional)
                # If you want state across runs, keep g merged back
                self._exec_globals.update({k: v for k, v in g.items() if k not in ("__builtins__",)})

        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {e}\n\nCaptured output:\n{output_capture.getvalue()}"
        finally:
            # Restore logger handler streams
            for h, stream in handler_streams.items():
                try:
                    h.stream = stream
                except Exception:
                    # Best-effort restore; ignore if handler disappeared or is immutable
                    pass

        out = output_capture.getvalue().strip()
        return out if out else "Execution completed without output."
