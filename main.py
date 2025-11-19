# main.py
# -*- coding: utf-8 -*-
"""
Basel Optimisation Portfolio – FAST router
- No uploader here.
- Sidebar shows only navigation.
- Sub-apps run in isolation.
- Sub-app files are read+patched+compiled ONCE and cached across reruns.
"""

import os
import re
import sys
import time
import traceback
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# 🚀 PERFORMANCE MODE (ADDED)
# -------------------------------------------------------------
# Disable development mode to reduce rerun lag
# st.runtime.scriptrunner.script_run_context.add_script_run_ctx = lambda *a, **k: None  
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_DEVELOPMENT_MODE"] = "false"

# Throttle re-runs (avoid Streamlit reacting too fast)
st.session_state.setdefault("last_run_ts", 0)
_now = time.time()
if _now - st.session_state["last_run_ts"] < 0.15:  # 150ms throttle
    st.stop()
st.session_state["last_run_ts"] = _now

# Pre-warm placeholders
if "subapp_cache" not in st.session_state:
    st.session_state["subapp_cache"] = {}

# -------------------------------------------------------------
# END OF PERFORMANCE BLOCK
# -------------------------------------------------------------


# ---------- Page config (only here) ----------
st.set_page_config(
    page_title="Basel Optimisation Portfolio",
    page_icon="📊",
    layout="wide",
)

# Optional: if a default workbook exists next to main.py, sub-apps may use it
DEFAULT_COMBINED_XLSX = "Basel_Combined_Datasets.xlsx"
COMBINED_XLSX = DEFAULT_COMBINED_XLSX if os.path.exists(DEFAULT_COMBINED_XLSX) else None

# ---------- Sidebar navigation (NO uploader here) ----------
# st.sidebar.title("📊 Basel Optimisation")
choice = st.sidebar.radio(
    "Go to",
    (
        "📈 Portfolio Optimisation Lab (Retail/LTV/CRM)",      # Basel_Opti_Lab.py
        "🧮 Basel RWA — Advanced (Scenarios & Segments)",      # Basel_RWA_Adv.py
        "🔎 RWA Validation — Enhanced (PD/LGD/EAD Check)",     # RWA_Validation.py
    ),
    index=0
)


# ---------- Performance helper ----------
def time_block(label: str):
    """Context manager for quick timing captions."""
    class _CM:
        def __enter__(self): self.t0 = time.perf_counter(); return self
        def __exit__(self, *exc): st.caption(f"⏱ {label}: {time.perf_counter()-self.t0:.2f}s")
    return _CM()

# ---------- Cached loader/patcher/compiler for sub-apps ----------
@st.cache_resource(show_spinner=False)
def load_subapp_compiled(py_path: str):
    """
    Read sub-app source, neutralise set_page_config, compile once.
    Recomputed only when the file mtime changes.
    """
    if not os.path.exists(py_path):
        raise FileNotFoundError(py_path)

    mtime = os.path.getmtime(py_path)

    # include mtime in the cache key by nesting a second cache to avoid manual keying
    @st.cache_resource(show_spinner=False)
    def _compile_with_key(path: str, file_mtime: float):
        src = open(path, "r", encoding="utf-8").read()
        # Remove any st.set_page_config(...) lines to avoid conflicts
        patched = re.sub(r"(?m)^\s*st\.set_page_config\(.*?\)\s*$",
                         "# (removed) st.set_page_config(...)", src)
        code_obj = compile(patched, path, "exec")
        return code_obj

    return _compile_with_key(py_path, mtime)


# -------------------------------------------------------------
# 🚀 FAST RENDERING LAYER (ADDED)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_fast_namespace():
    """Pre-cached namespace object so exec() doesn't rebuild dicts each time."""
    return {
        "__name__": "__main__",
        "st": st, "pd": pd, "np": np, "os": os,
        "COMBINED_XLSX": COMBINED_XLSX,
    }

@st.cache_resource(show_spinner=False)
def warm_load_subapp(path):
    """Load & compile sub-app once at startup to warm the cache."""
    return load_subapp_compiled(path)
# Pre-warm all dashboards
try:
    warm_load_subapp("Basel_Opti_Lab.py")
    warm_load_subapp("Basel_RWA_Adv.py")
    warm_load_subapp("RWA_Validation.py")
except:
    pass
# -------------------------------------------------------------


def run_subapp(py_path: str, extra_ns: dict | None = None):
    """Exec the compiled sub-app in an isolated namespace (FAST)."""
    try:
        code = load_subapp_compiled(py_path)
    except FileNotFoundError:
        st.error(f"Couldn't find `{py_path}` next to main.py.")
        return

    # Use cached namespace
    ns = get_fast_namespace().copy()

    if extra_ns:
        ns.update(extra_ns)

    with time_block(f"Render {os.path.basename(py_path)}"):
        try:
            exec(code, ns)
        except Exception:
            st.error("The selected dashboard crashed. See details:")
            st.code("".join(traceback.format_exception(*sys.exc_info())), language="python")


# ---------- Route ----------
if choice.startswith("🔎"):
    run_subapp("RWA_Validation.py")
elif choice.startswith("🧮"):
    run_subapp("Basel_RWA_Adv.py")
else:
    run_subapp("Basel_Opti_Lab.py")
