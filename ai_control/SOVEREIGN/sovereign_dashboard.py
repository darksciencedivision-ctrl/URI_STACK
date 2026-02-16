# C:\ai_control\SOVEREIGN\sovereign_dashboard.py
# Streamlit UI ONLY. No orchestration logic. Writes prompt + job.json, displays artifacts/logs.

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ----------------------------
# Canonical paths (file bus)
# ----------------------------
AI_ROOT = Path(r"C:\ai_control")

PRAXIS_DIR = AI_ROOT / "PRAXIS"
ORCH_DIR = AI_ROOT / "ORCHESTRA"
SOV_DIR = AI_ROOT / "SOVEREIGN"

PRAXIS_INBOX = PRAXIS_DIR / "inbox"
PRAXIS_OUTBOX = PRAXIS_DIR / "outbox"
PRAXIS_LOGS = PRAXIS_DIR / "logs"

ORCH_INBOX = ORCH_DIR / "inbox"
ORCH_OUTBOX = ORCH_DIR / "outbox"
ORCH_LOGS = ORCH_DIR / "logs"
ORCH_ARCHIVE = ORCH_DIR / "archive"

SOV_LOGS = SOV_DIR / "logs"
SOV_UI_OUTBOX = SOV_DIR / "ui_outbox"

PROMPT_PATH = PRAXIS_INBOX / "user_prompt.txt"
JOB_PATH = ORCH_INBOX / "job.json"

ORCH_REDUCER_PATH = ORCH_OUTBOX / "reducer_output.json"
ORCH_RESPONSE_MD = ORCH_OUTBOX / "response_to_user.md"
ORCH_RAW_PATH = ORCH_OUTBOX / "raw_model_output.txt"

PRAXIS_CANON_PATH = PRAXIS_OUTBOX / "canonical_state.json"

ORCH_LOG = ORCH_LOGS / "orchestra.log"
PRAXIS_LOG = PRAXIS_LOGS / "praxis.log"

# ----------------------------
# Defaults (LOCKED to Qwen 32B)
# ----------------------------
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:32b-instruct"  # <- central default you asked for
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 512

# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs() -> None:
    for d in [
        PRAXIS_INBOX, PRAXIS_OUTBOX, PRAXIS_LOGS,
        ORCH_INBOX, ORCH_OUTBOX, ORCH_LOGS, ORCH_ARCHIVE,
        SOV_LOGS, SOV_UI_OUTBOX
    ]:
        d.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding, newline="\n")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    tmp.replace(path)


def tail_text(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as e:
        return f"[read_error] {e}"


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def get_last_orch_model() -> str:
    data = read_json(ORCH_REDUCER_PATH)
    if isinstance(data, dict):
        # If your ORCHESTRA includes model in reducer_output someday, it will show here.
        # For now, infer from artifacts or leave blank.
        return data.get("model", "") or ""
    return ""


# ----------------------------
# UI
# ----------------------------
ensure_dirs()

st.set_page_config(page_title="SOVEREIGN Dashboard", layout="wide")

st.title("SOVEREIGN ⇄ ORCHESTRA ⇄ PRAXIS")
st.caption("Offline, file-driven. UI writes artifacts; ORCHESTRA runs jobs; PRAXIS ingests canonical state.")

# Sidebar: execution settings
with st.sidebar:
    st.subheader("Execution Settings")

    ollama_host = st.text_input("Ollama Host", value=DEFAULT_OLLAMA_HOST)

    # IMPORTANT: default is Qwen 32B
    selected_model = st.text_input("Model", value=DEFAULT_MODEL, help="This is written into job.json as options.model")

    temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(DEFAULT_TEMPERATURE), step=0.05)
    top_p = st.number_input("Top-p", min_value=0.0, max_value=1.0, value=float(DEFAULT_TOP_P), step=0.05)
    max_new_tokens = st.number_input("Max New Tokens", min_value=16, max_value=8192, value=int(DEFAULT_MAX_NEW_TOKENS), step=16)

    strict_json = st.checkbox("strict_json", value=False, help="If your ORCHESTRA supports it, it can enforce JSON-only output.")
    mode = st.selectbox("Mode", ["normal", "keys_only"], index=0, help="keys_only should be deterministic if ORCHESTRA supports it.")
    task_label = st.text_input("Task label", value="job")

    st.divider()

    # Cycle ID controls
    auto_cycle_id = st.checkbox("Auto-generate cycle_id", value=True)
    if auto_cycle_id:
        cycle_id = f"cycle_ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.code(cycle_id, language="text")
    else:
        cycle_id = st.text_input("cycle_id", value="cycle_ui_custom_0001")

# Main: tabs
tab_run, tab_outputs, tab_logs = st.tabs(["Run", "Outputs", "Logs"])

with tab_run:
    st.subheader("Prompt")
    default_prompt = "Return ONLY the reducer schema keys-only JSON skeleton. No extra keys."
    prompt_text = st.text_area("Write prompt to PRAXIS inbox", value=default_prompt, height=220)

    colA, colB, colC = st.columns([1, 1, 2])

    with colA:
        submit = st.button("Submit Job", type="primary")

    with colB:
        refresh = st.button("Refresh View")

    with colC:
        st.markdown("**Job path:**")
        st.code(str(JOB_PATH), language="text")
        st.markdown("**Prompt path:**")
        st.code(str(PROMPT_PATH), language="text")

    if submit:
        # 1) Write prompt
        atomic_write_text(PROMPT_PATH, prompt_text)

        # 2) Write job.json (THIS is what makes the UI selector actually control ORCHESTRA)
        job = {
            "cycle_id": cycle_id,
            "timestamp_utc": utc_now_iso(),
            "user_prompt_path": str(PROMPT_PATH),
            "options": {
                "ollama_host": ollama_host,
                "model": selected_model,  # <- the whole point: UI controls central model per job
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_new_tokens": int(max_new_tokens),
                "strict_json": bool(strict_json),
                "mode": str(mode),
                "task": str(task_label),
            },
        }

        atomic_write_json(JOB_PATH, job)

        # UI bookkeeping
        atomic_write_json(
            SOV_UI_OUTBOX / "last_run.json",
            {
                "cycle_id": cycle_id,
                "timestamp_utc": job["timestamp_utc"],
                "model": selected_model,
                "job_path": str(JOB_PATH),
                "prompt_path": str(PROMPT_PATH),
            },
        )

        st.success(f"Job submitted: {cycle_id}  |  model={selected_model}")

    st.divider()

    st.markdown("### Quick Status")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("job.json present", "YES" if JOB_PATH.exists() else "NO")
    with c2:
        st.metric("reducer_output.json", "YES" if ORCH_REDUCER_PATH.exists() else "NO")
    with c3:
        st.metric("response_to_user.md", "YES" if ORCH_RESPONSE_MD.exists() else "NO")
    with c4:
        st.metric("canonical_state.json", "YES" if PRAXIS_CANON_PATH.exists() else "NO")

with tab_outputs:
    st.subheader("ORCHESTRA Outputs")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**response_to_user.md**")
        if ORCH_RESPONSE_MD.exists():
            st.text_area("response_to_user.md", ORCH_RESPONSE_MD.read_text(encoding="utf-8", errors="replace"), height=360)
            # Save a copy for UI convenience
            atomic_write_text(SOV_UI_OUTBOX / "last_response.md", ORCH_RESPONSE_MD.read_text(encoding="utf-8", errors="replace"))
        else:
            st.info("No response_to_user.md yet.")

    with col2:
        st.markdown("**reducer_output.json**")
        reducer = read_json(ORCH_REDUCER_PATH)
        if reducer is not None:
            st.json(reducer)
        else:
            if ORCH_REDUCER_PATH.exists():
                st.warning("reducer_output.json exists but is not valid JSON.")
                st.text_area("raw reducer_output.json", ORCH_REDUCER_PATH.read_text(encoding="utf-8", errors="replace"), height=360)
            else:
                st.info("No reducer_output.json yet.")

    st.divider()
    st.subheader("PRAXIS Canonical State")
    canon = read_json(PRAXIS_CANON_PATH)
    if canon is not None:
        st.json(canon)
    else:
        if PRAXIS_CANON_PATH.exists():
            st.warning("canonical_state.json exists but is not valid JSON.")
            st.text_area("raw canonical_state.json", PRAXIS_CANON_PATH.read_text(encoding="utf-8", errors="replace"), height=260)
        else:
            st.info("No canonical_state.json yet.")

    st.divider()
    st.subheader("Raw model trace")
    if ORCH_RAW_PATH.exists():
        st.text_area("raw_model_output.txt", ORCH_RAW_PATH.read_text(encoding="utf-8", errors="replace"), height=220)
    else:
        st.info("No raw_model_output.txt yet.")

with tab_logs:
    st.subheader("Live-ish Logs (tail)")
    colL, colR = st.columns(2)

    with colL:
        st.markdown("**ORCHESTRA log (tail)**")
        st.text_area("orchestra.log", tail_text(ORCH_LOG, 160), height=420)

    with colR:
        st.markdown("**PRAXIS log (tail)**")
        st.text_area("praxis.log", tail_text(PRAXIS_LOG, 160), height=420)

    st.caption("If logs look stale after edits, restart Streamlit only (not the whole stack).")

# optional auto-refresh
if refresh:
    st.rerun()

