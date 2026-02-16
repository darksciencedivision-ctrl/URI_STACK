import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\ai_control\PRAXIS")
INBOX = ROOT / "inbox"
OUTBOX = ROOT / "outbox"
LOGS = ROOT / "logs"
ARCHIVE = ROOT / "archive"
STOP_FILE = ROOT / "STOP"

LOG_FILE = LOGS / "praxis.log"

ORCH_OUTBOX = Path(r"C:\ai_control\ORCHESTRA\outbox")
REDUCER_FILE = ORCH_OUTBOX / "reducer_output.json"

CANON_FILE = OUTBOX / "canonical_state.json"

# Reducer required keys (locked)
REQUIRED_KEYS = [
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

DEFAULT_CYCLES_CAP = 500
DEFAULT_SHARDS_CAP = 50


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def write_json_atomic(path: Path, obj) -> None:
    write_text_atomic(path, json.dumps(obj, indent=2))


def stable_read_json(path: Path, stable_ms=250):
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


def ensure_dirs():
    INBOX.mkdir(parents=True, exist_ok=True)
    OUTBOX.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    if not LOG_FILE.exists():
        LOG_FILE.write_text("", encoding="utf-8")

    if not CANON_FILE.exists():
        init = {
            "cycles": [],
            "memory_shards": [],
            "memory": {
                "observations": [],
                "beliefs": [],
                "knowledge": [],
                "entity_state": {
                    "name": "Praxis",
                    "conversation_count": 0,
                    "total_messages": 0,
                    "created_utc": utc_now_iso(),
                    "last_active_utc": utc_now_iso(),
                },
            },
            "last_cycle_id": "",
            "last_updated_utc": utc_now_iso(),
        }
        write_json_atomic(CANON_FILE, init)


def validate_reducer(obj: dict):
    if not isinstance(obj, dict):
        return False, "reducer_not_object", None

    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        return False, f"missing_keys:{','.join(missing)}", None

    if str(obj.get("schema_version")) != "1.0":
        return False, "bad_schema_version", None

    # Accept extras (additive). Do NOT drop parsed_tags; we need them.
    return True, "ok", obj


def next_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time())}_{os.urandom(3).hex()}"


def ingest_tags(canon: dict, reducer: dict):
    """
    parsed_tags optional field:
      "parsed_tags": {"self_observations":[], "beliefs":[], "knowledge":[], "rrr_queries":[]}
    """
    mem = canon.setdefault("memory", {})
    mem.setdefault("observations", [])
    mem.setdefault("beliefs", [])
    mem.setdefault("knowledge", [])
    mem.setdefault("entity_state", {})

    tags = reducer.get("parsed_tags") or {}
    cycle_id = str(reducer.get("cycle_id", ""))

    # Observations
    for text in tags.get("self_observations") or []:
        mem["observations"].append(
            {
                "id": next_id("obs"),
                "timestamp_utc": utc_now_iso(),
                "text": str(text),
                "source_cycle_id": cycle_id,
                "category": "self_observe",
            }
        )

    # Beliefs
    for text in tags.get("beliefs") or []:
        mem["beliefs"].append(
            {
                "id": next_id("bel"),
                "timestamp_utc": utc_now_iso(),
                "text": str(text),
                "source_cycle_id": cycle_id,
                "confidence": 0.5,
            }
        )

    # Knowledge
    for text in tags.get("knowledge") or []:
        # topic heuristic: first token before colon, else "general"
        topic = "general"
        s = str(text)
        if ":" in s:
            topic = s.split(":", 1)[0].strip()[:64] or "general"
        mem["knowledge"].append(
            {
                "id": next_id("kn"),
                "timestamp_utc": utc_now_iso(),
                "text": s,
                "source_cycle_id": cycle_id,
                "topic": topic,
            }
        )

    # entity_state maintenance
    es = mem["entity_state"]
    es.setdefault("name", "Praxis")
    es["last_active_utc"] = utc_now_iso()
    es["conversation_count"] = int(es.get("conversation_count", 0)) + (1 if reducer.get("conversation_id") else 0)
    es["total_messages"] = int(es.get("total_messages", 0)) + int(reducer.get("message_count", 2))


def update_memory_shards(canon: dict, reducer: dict, cap: int = DEFAULT_SHARDS_CAP):
    """
    Keep existing behavior: rolling memory_shards[] capped at 50.
    """
    shards = list(canon.get("memory_shards") or [])
    snippet = (reducer.get("summary") or "")[:600].strip()
    if snippet:
        shards.append(
            {
                "id": next_id("shard"),
                "timestamp_utc": utc_now_iso(),
                "text": snippet,
                "source_cycle_id": reducer.get("cycle_id", ""),
            }
        )
    canon["memory_shards"] = shards[-cap:]


def archive_and_trim_cycles(canon: dict, cycles_cap: int):
    cycles = list(canon.get("cycles") or [])
    if len(cycles) <= cycles_cap:
        return canon

    overflow = cycles[:-cycles_cap]
    keep = cycles[-cycles_cap:]

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    ts = utc_now_iso().replace(":", "").replace("-", "")
    arch = ARCHIVE / f"cycles_archive_{ts}.json"
    write_json_atomic(arch, {"archived_utc": utc_now_iso(), "count": len(overflow), "cycles": overflow})

    canon["cycles"] = keep
    canon["last_cycle_trimmed_utc"] = utc_now_iso()
    canon["last_cycle_archive_file"] = str(arch)
    return canon


def ingest(reducer: dict):
    canon = stable_read_json(CANON_FILE) or {}
    canon.setdefault("cycles", [])
    canon.setdefault("memory_shards", [])
    canon.setdefault("memory", {})
    canon.setdefault("last_cycle_id", "")
    canon.setdefault("last_updated_utc", utc_now_iso())

    # Append cycle (keep required keys + allow extras like conversation_id/mode/parsed_tags/reflection)
    canon["cycles"].append(reducer)

    canon["last_cycle_id"] = reducer.get("cycle_id", "")
    canon["last_updated_utc"] = utc_now_iso()

    canon["last_task"] = reducer.get("task", "")
    canon["last_summary"] = reducer.get("summary", "")
    canon["last_final_answer"] = reducer.get("final_answer", "")

    # Memory shards (legacy)
    update_memory_shards(canon, reducer, cap=DEFAULT_SHARDS_CAP)

    # Parsed tags ingestion (new)
    ingest_tags(canon, reducer)

    # Cap cycles (and archive overflow)
    canon = archive_and_trim_cycles(canon, cycles_cap=DEFAULT_CYCLES_CAP)

    write_json_atomic(CANON_FILE, canon)


def main():
    ensure_dirs()
    log("[START] PRAXIS ingestor online")

    last_seen_mtime = None

    while True:
        if STOP_FILE.exists():
            log("[STOP] STOP file present, exiting cleanly")
            break

        if REDUCER_FILE.exists():
            try:
                mtime = REDUCER_FILE.stat().st_mtime
                if last_seen_mtime is None or mtime != last_seen_mtime:
                    obj = stable_read_json(REDUCER_FILE)
                    if obj is None:
                        time.sleep(0.2)
                        continue

                    ok, reason, reducer = validate_reducer(obj)
                    if not ok:
                        log(f"[REJECT] reducer_output.json invalid: {reason}")
                    else:
                        ingest(reducer)
                        log(f"[INGEST] cycle_id={reducer.get('cycle_id')} mode={reducer.get('mode','single')}")
                        last_seen_mtime = mtime
            except Exception as e:
                log(f"[ERROR] ingest loop -> {e}")

        time.sleep(0.35)

    log("[EXIT] PRAXIS ingestor exited")


if __name__ == "__main__":
    main()

