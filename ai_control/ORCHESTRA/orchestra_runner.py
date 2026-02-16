import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import urllib.request


# ===== Paths (locked root) =====
ROOT = Path(r"C:\ai_control\ORCHESTRA")
INBOX = ROOT / "inbox"
OUTBOX = ROOT / "outbox"
ARCHIVE = ROOT / "archive"
LOGS = ROOT / "logs"
DATA = ROOT / "data"
STOP_FILE = ROOT / "STOP"

JOB_FILE = INBOX / "job.json"
LOG_FILE = LOGS / "orchestra.log"

DEFAULT_SYSTEM_PROMPT = ROOT / "system_prompt.txt"

PRAXIS_CANONICAL = Path(r"C:\ai_control\PRAXIS\outbox\canonical_state.json")

CONV_DB = DATA / "conversations.db"

# ===== Reducer required keys (locked) =====
REDUCER_REQUIRED_KEYS = [
    "cycle_id",
    "timestamp_utc",
    "task",
    "summary",
    "final_answer",
    "artifacts_written",
    "decision_log",
    "warnings",
    "next_actions",
    "schema_version",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def ensure_dirs():
    INBOX.mkdir(parents=True, exist_ok=True)
    OUTBOX.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.write_text("", encoding="utf-8")
    if not DEFAULT_SYSTEM_PROMPT.exists():
        DEFAULT_SYSTEM_PROMPT.write_text("You are ORCHESTRA. Be precise.\n", encoding="utf-8")
    init_conversation_db(CONV_DB)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def write_json_atomic(path: Path, obj) -> None:
    write_text_atomic(path, json.dumps(obj, indent=2))


def stable_read_json(path: Path, stable_ms: int = 250):
    if not path.exists():
        return None
    s1 = path.stat().st_size
    time.sleep(stable_ms / 1000.0)
    if not path.exists():
        return None
    s2 = path.stat().st_size
    if s1 != s2:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def http_post_json(url: str, payload: dict, timeout: int = 180):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)


def call_ollama_chat(ollama_host: str, model: str, messages: list, temperature: float, top_p: float, max_new_tokens: int, timeout_sec: int = 300):
    url = f"{ollama_host}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_new_tokens,
        },
    }
    return http_post_json(url, payload, timeout=timeout_sec)


def compat_job_shims(job: dict) -> dict:
    """
    Locked contract + additive optional fields.
    Compat shims must not break existing jobs.
    """
    job = dict(job or {})
    opts = dict(job.get("options") or {})

    # compat shims
    if "model_id" in opts and "model" not in opts:
        opts["model"] = opts["model_id"]
    if "max_tokens" in opts and "max_new_tokens" not in opts:
        opts["max_new_tokens"] = opts["max_tokens"]
    if "timeout" in opts and "timeout_sec" not in opts:
        opts["timeout_sec"] = opts["timeout"]

    job["options"] = opts
    return job


def init_conversation_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conv ON messages(conversation_id, idx);")
        conn.commit()
    finally:
        conn.close()


def conv_next_idx(conn, conversation_id: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM messages WHERE conversation_id = ?;", (conversation_id,))
    return int(cur.fetchone()[0])


def conv_append(conversation_id: str, role: str, content: str):
    init_conversation_db(CONV_DB)
    conn = sqlite3.connect(CONV_DB)
    try:
        idx = conv_next_idx(conn, conversation_id)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages(conversation_id, idx, role, content, timestamp_utc) VALUES (?, ?, ?, ?, ?);",
            (conversation_id, idx, role, content, utc_now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def conv_load(conversation_id: str, max_chars_budget: int):
    """
    Load recent messages backwards until budget is used.
    Budget is chars-based estimate: tokens ~= chars/4.
    """
    if not CONV_DB.exists():
        return []

    conn = sqlite3.connect(CONV_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content, timestamp_utc FROM messages WHERE conversation_id = ? ORDER BY idx DESC LIMIT 400;",
            (conversation_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    out_rev = []
    used = 0
    for role, content, ts in rows:
        block = f"{role}:{content}"
        if used + len(block) > max_chars_budget:
            break
        out_rev.append({"role": role, "content": content})
        used += len(block)

    return list(reversed(out_rev))


def read_system_prompt(path_opt: str | None) -> str:
    p = Path(path_opt) if path_opt else DEFAULT_SYSTEM_PROMPT
    if not p.exists():
        p = DEFAULT_SYSTEM_PROMPT
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return "You are ORCHESTRA."


def load_praxis_memory_block(prompt_text: str, max_items_each: int = 6) -> str:
    """
    Minimal semantic retrieval: keyword overlap scoring (no embeddings, no dependencies).
    """
    if not PRAXIS_CANONICAL.exists():
        return ""

    try:
        canon = json.loads(PRAXIS_CANONICAL.read_text(encoding="utf-8"))
    except Exception:
        return ""

    mem = canon.get("memory") or {}
    observations = list(mem.get("observations") or [])
    beliefs = list(mem.get("beliefs") or [])
    knowledge = list(mem.get("knowledge") or [])

    # Tokenize prompt
    words = set(re.findall(r"[A-Za-z0-9_]{3,}", (prompt_text or "").lower()))
    if not words:
        return ""

    def score_text(t: str) -> int:
        tw = set(re.findall(r"[A-Za-z0-9_]{3,}", (t or "").lower()))
        return len(words.intersection(tw))

    def pick(items, key_text):
        scored = []
        for it in items:
            txt = str(it.get(key_text, ""))
            s = score_text(txt)
            if s > 0:
                scored.append((s, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:max_items_each]]

    obs_pick = pick(observations, "text")
    bel_pick = pick(beliefs, "text")
    kn_pick = pick(knowledge, "text")

    if not (obs_pick or bel_pick or kn_pick):
        return ""

    def fmt_obs(x): return f"- ({x.get('timestamp_utc','')}) {x.get('text','')}"
    def fmt_bel(x): return f"- ({x.get('timestamp_utc','')}) [{x.get('confidence',0.5)}] {x.get('text','')}"
    def fmt_kn(x): return f"- ({x.get('timestamp_utc','')}) [{x.get('topic','')}] {x.get('text','')}"

    block = "[ENTITY_MEMORY]\n"
    if obs_pick:
        block += "Observations:\n" + "\n".join(fmt_obs(x) for x in obs_pick) + "\n"
    if bel_pick:
        block += "Beliefs:\n" + "\n".join(fmt_bel(x) for x in bel_pick) + "\n"
    if kn_pick:
        block += "Knowledge:\n" + "\n".join(fmt_kn(x) for x in kn_pick) + "\n"
    block += "[/ENTITY_MEMORY]"
    return block


TAG_SELF_OBS = re.compile(r"\[SELF_OBSERVE:\s*(.*?)\]", re.DOTALL)
TAG_BELIEF = re.compile(r"\[BELIEF:\s*(.*?)\]", re.DOTALL)
TAG_KNOWLEDGE = re.compile(r"\[KNOWLEDGE:\s*(.*?)\]", re.DOTALL)
TAG_RRRQ = re.compile(r"\[RRR_QUERY:\s*(.*?)\]", re.DOTALL)


def parse_tags(text: str):
    def clean_list(matches):
        out = []
        for m in matches:
            t = " ".join(str(m).strip().split())
            if t:
                out.append(t)
        return out

    return {
        "self_observations": clean_list(TAG_SELF_OBS.findall(text or "")),
        "beliefs": clean_list(TAG_BELIEF.findall(text or "")),
        "knowledge": clean_list(TAG_KNOWLEDGE.findall(text or "")),
        "rrr_queries": clean_list(TAG_RRRQ.findall(text or "")),
    }


def coerce_reducer(model_text: str, cycle_id: str, task_hint: str, warnings_out: list):
    obj = None
    stripped = (model_text or "").strip()

    try:
        obj = json.loads(stripped)
    except Exception:
        try:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(stripped[start:end + 1])
        except Exception:
            obj = None

    if not isinstance(obj, dict):
        warnings_out.append("model_output_not_json; synthesized reducer_output.json")
        obj = {}

    coerced = {
        "cycle_id": str(obj.get("cycle_id") or cycle_id),
        "timestamp_utc": str(obj.get("timestamp_utc") or utc_now_iso()),
        "task": str(obj.get("task") or task_hint or "unknown_task"),
        "summary": str(obj.get("summary") or ""),
        "final_answer": str(obj.get("final_answer") or ""),
        "artifacts_written": list(obj.get("artifacts_written") or []),
        "decision_log": list(obj.get("decision_log") or []),
        "warnings": list(obj.get("warnings") or []),
        "next_actions": list(obj.get("next_actions") or []),
        "schema_version": "1.0",
    }

    for w in warnings_out:
        if w not in coerced["warnings"]:
            coerced["warnings"].append(w)

    return coerced


def reducer_system_prompt(strict_json: bool) -> str:
    base = (
        "You are ORCHESTRA Reducer for an RRR pipeline. This is NOT Redux.\n"
        "Return JSON for Reducer Output Contract Schema v1.0 with required keys.\n\n"
        "Required keys:\n"
        "{\n"
        '  "cycle_id": "",\n'
        '  "timestamp_utc": "",\n'
        '  "task": "",\n'
        '  "summary": "",\n'
        '  "final_answer": "",\n'
        '  "artifacts_written": [],\n'
        '  "decision_log": [],\n'
        '  "warnings": [],\n'
        '  "next_actions": [],\n'
        '  "schema_version": "1.0"\n'
        "}\n"
    )
    if strict_json:
        base += "Output ONLY JSON. No markdown. No fences."
    return base


@dataclass
class RRRConfig:
    reasoner_model: str = "qwen2.5:32b-instruct"
    critic_model: str = "deepseek-r1:14b"
    reducer_model: str = "qwen2.5:14b-instruct"
    max_turns: int = 6
    reasoner_system_prompt_path: str | None = None
    critic_system_prompt_path: str | None = None
    reducer_system_prompt_path: str | None = None


def rrr_config_from_job(job: dict) -> RRRConfig:
    rc = dict(job.get("rrr_config") or {})
    cfg = RRRConfig()
    # additive keys only; missing means defaults
    cfg.reasoner_model = str(rc.get("reasoner_model", cfg.reasoner_model))
    cfg.critic_model = str(rc.get("critic_model", cfg.critic_model))
    cfg.reducer_model = str(rc.get("reducer_model", cfg.reducer_model))
    cfg.max_turns = int(rc.get("max_turns", cfg.max_turns))
    cfg.reasoner_system_prompt_path = rc.get("reasoner_system_prompt_path", None)
    cfg.critic_system_prompt_path = rc.get("critic_system_prompt_path", None)
    cfg.reducer_system_prompt_path = rc.get("reducer_system_prompt_path", None)
    return cfg


def build_messages(system_prompt: str, memory_block: str, conv_history: list, user_prompt: str):
    sys = system_prompt.strip()
    if memory_block:
        sys = f"{sys}\n\n{memory_block}"
    msgs = [{"role": "system", "content": sys}]
    msgs.extend(conv_history or [])
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def run_single(job: dict, prompt_text: str, memory_block: str):
    opts = job["options"]
    ollama_host = opts.get("ollama_host") or "http://localhost:11434"
    model = opts.get("model") or "qwen2.5:32b-instruct"
    temperature = float(opts.get("temperature", 0.2))
    top_p = float(opts.get("top_p", 0.9))
    max_new_tokens = int(opts.get("max_new_tokens", 512))
    context_window = int(opts.get("context_window", 8192))
    timeout = int(opts.get("timeout_sec", 300))

    conv_id = job.get("conversation_id", None)

    sys_prompt = read_system_prompt(job.get("system_prompt_path", None))
    sys_chars = len(sys_prompt) + len(memory_block)
    total_char_budget = int(context_window * 4)
    char_budget = max(0, int((total_char_budget - sys_chars) * 0.7))

    history = conv_load(str(conv_id), char_budget) if conv_id else []

    messages = build_messages(sys_prompt, memory_block, history, prompt_text)
    resp = call_ollama_chat(ollama_host, model, messages, temperature, top_p, max_new_tokens, timeout)
    text = (resp.get("message") or {}).get("content") or json.dumps(resp, indent=2)

    if conv_id:
        conv_append(str(conv_id), "user", prompt_text)
        conv_append(str(conv_id), "assistant", text)

    return text, {"model": model, "conversation_id": conv_id, "messages": messages}


def run_rrr(job: dict, prompt_text: str, memory_block: str):
    opts = job["options"]
    ollama_host = opts.get("ollama_host") or "http://localhost:11434"
    timeout = int(opts.get("timeout_sec", 300))

    rrr = rrr_config_from_job(job)

    reasoner_sys = read_system_prompt(rrr.reasoner_system_prompt_path or job.get("system_prompt_path", None))
    critic_sys = read_system_prompt(rrr.critic_system_prompt_path or job.get("system_prompt_path", None))

    if memory_block:
        reasoner_sys = f"{reasoner_sys}\n\n{memory_block}"
        critic_sys = f"{critic_sys}\n\n{memory_block}"

    dialogue = []

    # Accumulated message histories for each agent
    reasoner_msgs = [
        {"role": "system", "content": reasoner_sys},
        {"role": "user", "content": prompt_text},
    ]
    r0 = call_ollama_chat(ollama_host, rrr.reasoner_model, reasoner_msgs, 0.2, 0.9, 768, timeout)
    reasoner_text = (r0.get("message") or {}).get("content") or ""
    reasoner_msgs.append({"role": "assistant", "content": reasoner_text})
    dialogue.append({"turn": 0, "speaker": "reasoner", "model": rrr.reasoner_model, "text": reasoner_text})

    critic_msgs = [
        {"role": "system", "content": critic_sys},
        {"role": "user", "content": f"The user asked:\n{prompt_text}\n\nThe reasoner answered:\n{reasoner_text}\n\nCritique it. Find gaps, errors, weak reasoning, missing rigor."},
    ]
    c0 = call_ollama_chat(ollama_host, rrr.critic_model, critic_msgs, 0.2, 0.9, 768, timeout)
    critic_text = (c0.get("message") or {}).get("content") or ""
    critic_msgs.append({"role": "assistant", "content": critic_text})
    dialogue.append({"turn": 0, "speaker": "critic", "model": rrr.critic_model, "text": critic_text})

    for t in range(1, max(1, rrr.max_turns)):
        # Reasoner sees critic's challenge in its own conversation history
        reasoner_msgs.append({"role": "user", "content": f"The critic responds:\n{critic_text}\n\nAddress their points. Revise where warranted. Defend where correct."})
        try:
            rr = call_ollama_chat(ollama_host, rrr.reasoner_model, reasoner_msgs, 0.2, 0.9, 768, timeout)
            reasoner_text = (rr.get("message") or {}).get("content") or ""
        except Exception as e:
            log(f"[RRR] reasoner failed turn {t}: {e}")
            dialogue.append({"turn": t, "speaker": "reasoner", "model": rrr.reasoner_model, "text": f"[ERROR: {e}]"})
            break
        reasoner_msgs.append({"role": "assistant", "content": reasoner_text})
        dialogue.append({"turn": t, "speaker": "reasoner", "model": rrr.reasoner_model, "text": reasoner_text})

        # Critic sees reasoner's revision in its own conversation history
        critic_msgs.append({"role": "user", "content": f"The reasoner revised:\n{reasoner_text}\n\nCritique remaining issues only. If satisfied, say what is now strong."})
        try:
            cc = call_ollama_chat(ollama_host, rrr.critic_model, critic_msgs, 0.2, 0.9, 768, timeout)
            critic_text = (cc.get("message") or {}).get("content") or ""
        except Exception as e:
            log(f"[RRR] critic failed turn {t}: {e}")
            dialogue.append({"turn": t, "speaker": "critic", "model": rrr.critic_model, "text": f"[ERROR: {e}]"})
            break
        critic_msgs.append({"role": "assistant", "content": critic_text})
        dialogue.append({"turn": t, "speaker": "critic", "model": rrr.critic_model, "text": critic_text})

        if "satisfied" in critic_text.lower() or "no remaining issues" in critic_text.lower():
            break

    # Reducer produces canonical answer from full transcript
    transcript = "\n\n".join([f"[{d['speaker'].upper()}:{d['model']}] {d['text']}" for d in dialogue])
    reducer_msgs = [
        {"role": "system", "content": reducer_system_prompt(True)},
        {"role": "user", "content": f"User prompt:\n{prompt_text}\n\nRRR Transcript:\n{transcript}\n\nReturn reducer JSON."},
    ]
    try:
        rd = call_ollama_chat(ollama_host, rrr.reducer_model, reducer_msgs, 0.0, 1.0, 700, timeout)
        reducer_text = (rd.get("message") or {}).get("content") or ""
    except Exception as e:
        log(f"[RRR] reducer failed: {e}")
        reducer_text = ""

    return reducer_text, dialogue, {"reasoner_model": rrr.reasoner_model, "critic_model": rrr.critic_model, "reducer_model": rrr.reducer_model}


def run_reflection(job: dict, reducer_obj: dict, memory_block: str):
    opts = job["options"]
    ollama_host = opts.get("ollama_host") or "http://localhost:11434"
    model = opts.get("model") or "qwen2.5:32b-instruct"
    context_window = int(opts.get("context_window", 8192))

    sys = read_system_prompt(job.get("system_prompt_path", None))

    prompt = (
        "Review the following cycle output and your own performance.\n\n"
        "[CYCLE_OUTPUT]\n"
        f"Summary:\n{reducer_obj.get('summary','')}\n\n"
        f"Final Answer:\n{reducer_obj.get('final_answer','')}\n\n"
        f"Warnings:\n{json.dumps(reducer_obj.get('warnings',[]), indent=2)}\n"
        "[/CYCLE_OUTPUT]\n\n"
    )
    if memory_block:
        prompt += f"{memory_block}\n\n"

    prompt += (
        "Analyze:\n"
        "1. What worked well?\n"
        "2. What failed or was weak?\n"
        "3. What patterns do you notice in your behavior?\n"
        "4. Record new observations with [SELF_OBSERVE: ...]\n"
        "5. Record new/updated beliefs with [BELIEF: ...]\n"
        "6. Record knowledge worth persisting with [KNOWLEDGE: ...]\n"
    )

    # Reflection is single-shot (not conversation history)
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]
    rr = call_ollama_chat(ollama_host, model, messages, 0.2, 0.9, 700)
    text = (rr.get("message") or {}).get("content") or json.dumps(rr, indent=2)
    return text


def archive_job(job: dict) -> Path:
    cycle_id = job.get("cycle_id") or "cycle_unknown"
    ts = job.get("timestamp_utc") or utc_now_iso()
    safe_ts = ts.replace(":", "").replace("-", "").replace("Z", "Z")
    name = f"job_{safe_ts}_{cycle_id}.json"
    fp = ARCHIVE / name
    write_json_atomic(fp, job)
    return fp


def main():
    ensure_dirs()
    log("[START] ORCHESTRA runner online")

    while True:
        if STOP_FILE.exists():
            log("[STOP] STOP file present, exiting cleanly")
            break

        if JOB_FILE.exists():
            try:
                job = stable_read_json(JOB_FILE)
                if job is None:
                    time.sleep(0.2)
                    continue

                job = compat_job_shims(job)

                # Archive immediately
                arch = archive_job(job)
                log(f"[ARCHIVE] {arch.name}")

                # Delete job file to prevent re-processing
                try:
                    JOB_FILE.unlink(missing_ok=True)
                except Exception:
                    pass

                cycle_id = job.get("cycle_id") or f"cycle_{int(time.time())}"
                prompt_path = Path(job.get("user_prompt_path") or "")
                if not prompt_path.exists():
                    raise RuntimeError(f"user_prompt_path not found: {prompt_path}")
                prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace").strip()

                mode = str(job.get("mode") or "single").lower()
                if mode not in ("single", "rrr"):
                    mode = "single"

                # Memory injection
                memory_block = load_praxis_memory_block(prompt_text)

                # === Main execution ===
                raw_text = ""
                rrr_dialogue = None
                rrr_meta = {}
                messages = []

                if mode == "single":
                    raw_text, meta = run_single(job, prompt_text, memory_block)
                    rrr_meta = meta
                    messages = list(meta.get("messages") or [])
                else:
                    reducer_model_text, dialogue, meta = run_rrr(job, prompt_text, memory_block)
                    # In RRR, "raw" is reducer-model draft, but we still write raw_model_output.txt for observability.
                    raw_text = reducer_model_text
                    rrr_dialogue = dialogue
                    rrr_meta = meta

                # Write raw output
                raw_out = OUTBOX / "raw_model_output.txt"
                write_text_atomic(raw_out, raw_text)

                # If RRR used, write dialogue artifacts
                if rrr_dialogue is not None:
                    dj = OUTBOX / "rrr_dialogue.json"
                    write_json_atomic(dj, {"cycle_id": cycle_id, "timestamp_utc": utc_now_iso(), "dialogue": rrr_dialogue, "meta": rrr_meta})
                    dm = OUTBOX / "rrr_dialogue.md"
                    md = []
                    md.append(f"# RRR Dialogue\n\ncycle_id: {cycle_id}\n\n")
                    for d in rrr_dialogue:
                        md.append(f"## Turn {d['turn']} · {d['speaker']} · {d['model']}\n\n{d['text']}\n")
                    write_text_atomic(dm, "\n".join(md))

                # === Reducer enforcement (programmatic) ===
                reducer_model_text = raw_text

                warnings_out = []
                if "```" in (reducer_model_text or ""):
                    warnings_out.append("strict_json_violated; model added code fences")

                task_hint = (prompt_text.splitlines()[0][:120] if prompt_text else "user_prompt")
                reducer_obj = coerce_reducer(reducer_model_text, cycle_id=cycle_id, task_hint=task_hint, warnings_out=warnings_out)

                # Tag parsing (from final_answer primarily)
                tags_main = parse_tags(reducer_obj.get("final_answer", "") + "\n" + raw_text)

                # Artifacts list (truth)
                artifacts = [
                    str(raw_out),
                    str(OUTBOX / "reducer_output.json"),
                    str(OUTBOX / "response_to_user.md"),
                ]
                if rrr_dialogue is not None:
                    artifacts.extend([str(OUTBOX / "rrr_dialogue.json"), str(OUTBOX / "rrr_dialogue.md")])

                reducer_obj["artifacts_written"] = artifacts

                # decision log addendum
                opts = job["options"]
                reducer_obj["decision_log"] = [
                    f"mode={mode}",
                    f"cycle_id={cycle_id}",
                    f"model={opts.get('model') or 'qwen2.5:32b-instruct'}",
                    f"ollama_host={opts.get('ollama_host') or 'http://localhost:11434'}",
                ] + list(reducer_obj.get("decision_log") or [])

                # Ensure schema_version locked
                reducer_obj["schema_version"] = "1.0"

                # Additive fields (allowed; PRAXIS accepts extras)
                if job.get("conversation_id", None):
                    reducer_obj["conversation_id"] = str(job.get("conversation_id"))
                reducer_obj["mode"] = mode
                reducer_obj["parsed_tags"] = tags_main

                reducer_obj["message_count"] = len(messages) if mode == "single" else len(rrr_dialogue or []) * 2

                # Human response markdown
                final_answer = reducer_obj.get("final_answer") or raw_text
                md_out = OUTBOX / "response_to_user.md"
                md = f"# Response\n\n{final_answer}\n\n---\n\n## Summary\n\n{reducer_obj.get('summary','')}\n"
                write_text_atomic(md_out, md)

                # Reflection pass (optional)
                reflection_tags = {"self_observations": [], "beliefs": [], "knowledge": [], "rrr_queries": []}
                reflection_text = ""
                if bool(job.get("reflect", False)):
                    reflection_text = run_reflection(job, reducer_obj, memory_block)
                    reflection_tags = parse_tags(reflection_text)
                    write_text_atomic(OUTBOX / "reflection_output.txt", reflection_text)
                    reducer_obj["reflection"] = {
                        "enabled": True,
                        "parsed_tags": reflection_tags,
                    }
                    reducer_obj["artifacts_written"].append(str(OUTBOX / "reflection_output.txt"))

                    # merge tags into main tags for PRAxis ingestion convenience
                    reducer_obj["parsed_tags"] = {
                        "self_observations": list(dict.fromkeys(tags_main["self_observations"] + reflection_tags["self_observations"])),
                        "beliefs": list(dict.fromkeys(tags_main["beliefs"] + reflection_tags["beliefs"])),
                        "knowledge": list(dict.fromkeys(tags_main["knowledge"] + reflection_tags["knowledge"])),
                        "rrr_queries": list(dict.fromkeys(tags_main["rrr_queries"] + reflection_tags["rrr_queries"])),
                    }
                else:
                    # Ensure artifact exists? No. Only if reflect is enabled.
                    pass

                # Write reducer output atomically
                reducer_path = OUTBOX / "reducer_output.json"
                write_json_atomic(reducer_path, reducer_obj)

                log(f"[JOB] done cycle_id={cycle_id} mode={mode} reflect={bool(job.get('reflect', False))}")

            except Exception as e:
                log(f"[ERROR] processing job.json -> {e}")

        time.sleep(0.35)

    log("[EXIT] ORCHESTRA runner exited")


if __name__ == "__main__":
    main()

