import json
import os
import sqlite3
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import urllib.request

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "uri_config.json"

LOG_DIR = BASE_DIR / "logs" / "system"
LOG_FILE = LOG_DIR / "uri_server.log"

DATA_DIR = BASE_DIR / "data"
DB_FILE = DATA_DIR / "conversations.db"
MEM_FILE = DATA_DIR / "entity_memory.json"

DEFAULT_CONFIG = {
    "ollama_host": "http://localhost:11434",
    "default_model": "qwen2.5:32b-instruct",
    "temperature": 0.2,
    "top_p": 0.9,
    "max_new_tokens": 512,
    "context_window": 8192,
    "host": "127.0.0.1",
    "port": 8088,
    "defaults": {
        "mode": "single",
        "reflect": False,
        "system_prompt_path": r"C:\ai_control\ORCHESTRA\system_prompt.txt",
    },
    "external_bus": {
        "enabled": True,
        "orchestra_inbox_job": r"C:\ai_control\ORCHESTRA\inbox\job.json",
        "praxis_prompt": r"C:\ai_control\PRAXIS\inbox\user_prompt.txt",
        "orchestra_outbox": r"C:\ai_control\ORCHESTRA\outbox",
        "orchestra_archive": r"C:\ai_control\ORCHESTRA\archive",
        "orchestra_log": r"C:\ai_control\ORCHESTRA\logs\orchestra.log",
        "orchestra_conversations_db": r"C:\ai_control\ORCHESTRA\data\conversations.db",
        "praxis_log": r"C:\ai_control\PRAXIS\logs\praxis.log",
        "praxis_canonical": r"C:\ai_control\PRAXIS\outbox\canonical_state.json",
    },
}

CONFIG = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs" / "dialog").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs" / "reducer").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "rrr_queue").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "rrr_responses" / "processed").mkdir(parents=True, exist_ok=True)

    if not LOG_FILE.exists():
        LOG_FILE.write_text("", encoding="utf-8")

    if not MEM_FILE.exists():
        MEM_FILE.write_text(json.dumps({"entities": {}, "updated_utc": utc_now_iso()}, indent=2), encoding="utf-8")

    if not DB_FILE.exists():
        conn = sqlite3.connect(DB_FILE)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_utc TEXT NOT NULL,
                    title TEXT,
                    payload_json TEXT NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()


def log_line(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def atomic_write_text(path: Path, text: str, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(str(tmp), str(path))


def atomic_write_json(path: Path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2), encoding="utf-8")


def load_config():
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        return DEFAULT_CONFIG

    try:
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}

    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg or {})

    # Deep merge
    dflt_defaults = dict(DEFAULT_CONFIG.get("defaults") or {})
    dflt_defaults.update((cfg or {}).get("defaults") or {})
    merged["defaults"] = dflt_defaults

    eb = dict(DEFAULT_CONFIG.get("external_bus") or {})
    eb.update((cfg or {}).get("external_bus") or {})
    merged["external_bus"] = eb

    return merged


def resolve_bus_paths(cfg):
    eb = cfg.get("external_bus") or {}
    return {
        "job": Path(eb.get("orchestra_inbox_job", r"C:\ai_control\ORCHESTRA\inbox\job.json")),
        "prompt": Path(eb.get("praxis_prompt", r"C:\ai_control\PRAXIS\inbox\user_prompt.txt")),
        "orchestra_outbox": Path(eb.get("orchestra_outbox", r"C:\ai_control\ORCHESTRA\outbox")),
        "orchestra_archive": Path(eb.get("orchestra_archive", r"C:\ai_control\ORCHESTRA\archive")),
        "orchestra_log": Path(eb.get("orchestra_log", r"C:\ai_control\ORCHESTRA\logs\orchestra.log")),
        "orchestra_conversations_db": Path(eb.get("orchestra_conversations_db", r"C:\ai_control\ORCHESTRA\data\conversations.db")),
        "praxis_log": Path(eb.get("praxis_log", r"C:\ai_control\PRAXIS\logs\praxis.log")),
        "praxis_canonical": Path(eb.get("praxis_canonical", r"C:\ai_control\PRAXIS\outbox\canonical_state.json")),
    }


def safe_stat(path: Path):
    if not path.exists():
        return {"exists": False}
    try:
        st = path.stat()
        return {
            "exists": True,
            "size": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
    except Exception:
        return {"exists": True}


def read_tail_lines(file_path: Path, lines: int = 200):
    if not file_path.exists():
        return ""
    try:
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            block = 4096
            data = b""
            while end > 0 and data.count(b"\n") <= lines:
                start = max(0, end - block)
                f.seek(start)
                chunk = f.read(end - start)
                data = chunk + data
                end = start
            text = data.decode("utf-8", errors="replace")
            parts = text.splitlines()[-lines:]
            return "\n".join(parts)
    except Exception as e:
        return f"[tail_error] {e}"


def http_get_json(url: str, timeout=4):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data)


def list_orchestra_conversations(db_path: Path, limit: int = 200):
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT conversation_id, MAX(timestamp_utc) AS last_utc, COUNT(*) AS n
                FROM messages
                GROUP BY conversation_id
                ORDER BY last_utc DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            out = []
            for cid, last_utc, n in rows:
                out.append({"conversation_id": cid, "last_utc": last_utc, "message_count": int(n)})
            return out
        finally:
            conn.close()
    except Exception:
        return []


class URIHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        rel = path.split("?", 1)[0].split("#", 1)[0]
        rel = rel.lstrip("/")
        if rel == "":
            rel = "index.html"
        return str((BASE_DIR / rel).resolve())

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text, content_type="text/plain; charset=utf-8", status=200):
        data = text.encode("utf-8", errors="replace")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            return None

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self.handle_health()
                return
            if parsed.path == "/api/status":
                self.handle_status()
                return
            if parsed.path == "/api/tail":
                self.handle_tail(parsed)
                return
            if parsed.path == "/api/artifact":
                self.handle_artifact(parsed)
                return
            if parsed.path == "/api/archive/list":
                self.handle_archive_list()
                return
            if parsed.path == "/api/archive/get":
                self.handle_archive_get(parsed)
                return
            if parsed.path == "/api/memory":
                self.handle_memory()
                return
            if parsed.path == "/api/conversations":
                self.handle_conversations()
                return
        except Exception as e:
            log_line(f"[ERROR] GET {self.path} -> {e}")
            self._send_json({"error": str(e)}, status=500)
            return

        super().do_GET()

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/submit_job":
                self.handle_submit_job()
                return
            if parsed.path == "/api/archive/replay":
                self.handle_archive_replay()
                return
        except Exception as e:
            log_line(f"[ERROR] POST {self.path} -> {e}")
            self._send_json({"error": str(e)}, status=500)
            return
        self._send_json({"error": "not_found"}, status=404)

    def handle_health(self):
        cfg = CONFIG
        ollama_host = cfg.get("ollama_host")
        models = []
        ollama_ok = False
        try:
            tags = http_get_json(f"{ollama_host}/api/tags", timeout=4)
            models = [m.get("name") for m in (tags.get("models") or []) if m.get("name")]
            ollama_ok = True
        except Exception:
            ollama_ok = False

        self._send_json(
            {
                "status": "ok",
                "ollama": ollama_ok,
                "models": models,
                "current_model": cfg.get("default_model"),
                "time_utc": utc_now_iso(),
            }
        )

    def handle_status(self):
        cfg = CONFIG
        paths = resolve_bus_paths(cfg)
        job_path = paths["job"]
        outbox = paths["orchestra_outbox"]
        canonical = paths["praxis_canonical"]

        raw_path = outbox / "raw_model_output.txt"
        reducer_path = outbox / "reducer_output.json"
        md_path = outbox / "response_to_user.md"
        rrr_json = outbox / "rrr_dialogue.json"
        reflection = outbox / "reflection_output.txt"

        self._send_json(
            {
                "time_utc": utc_now_iso(),
                "job": safe_stat(job_path),
                "artifacts": {
                    "raw_model_output": safe_stat(raw_path),
                    "reducer_output": safe_stat(reducer_path),
                    "response_to_user": safe_stat(md_path),
                    "rrr_dialogue": safe_stat(rrr_json),
                    "reflection_output": safe_stat(reflection),
                    "canonical_state": safe_stat(canonical),
                },
            }
        )

    def handle_tail(self, parsed):
        qs = parse_qs(parsed.query or "")
        target = (qs.get("target") or ["uri"])[0]
        lines = int((qs.get("lines") or ["200"])[0])

        cfg = CONFIG
        paths = resolve_bus_paths(cfg)

        if target == "orchestra":
            fp = paths["orchestra_log"]
        elif target == "praxis":
            fp = paths["praxis_log"]
        else:
            fp = LOG_FILE

        text = read_tail_lines(fp, lines=lines)
        self._send_text(text, content_type="text/plain; charset=utf-8", status=200)

    def handle_artifact(self, parsed):
        qs = parse_qs(parsed.query or "")
        name = (qs.get("name") or [""])[0].strip()

        cfg = CONFIG
        paths = resolve_bus_paths(cfg)
        outbox = paths["orchestra_outbox"]

        mapping = {
            "raw_model_output": outbox / "raw_model_output.txt",
            "reducer_output": outbox / "reducer_output.json",
            "response_to_user": outbox / "response_to_user.md",
            "rrr_dialogue": outbox / "rrr_dialogue.json",
            "rrr_dialogue_md": outbox / "rrr_dialogue.md",
            "reflection_output": outbox / "reflection_output.txt",
            "canonical_state": paths["praxis_canonical"],
        }

        if name not in mapping:
            self._send_json({"error": "bad_name"}, status=400)
            return

        fp = mapping[name]
        if not fp.exists():
            self._send_json({"error": "not_found"}, status=404)
            return

        content = fp.read_text(encoding="utf-8", errors="replace")
        ctype = "application/json; charset=utf-8" if fp.suffix.lower() == ".json" else "text/plain; charset=utf-8"
        self._send_text(content, content_type=ctype, status=200)

    def handle_archive_list(self):
        cfg = CONFIG
        paths = resolve_bus_paths(cfg)
        arch = paths["orchestra_archive"]
        arch.mkdir(parents=True, exist_ok=True)
        items = []
        for p in sorted(arch.glob("job_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            items.append(
                {
                    "name": p.name,
                    "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "size": p.stat().st_size,
                }
            )
        self._send_json({"archive": items})

    def handle_archive_get(self, parsed):
        qs = parse_qs(parsed.query or "")
        name = (qs.get("name") or [""])[0].strip()
        if not name.startswith("job_") or not name.endswith(".json") or "/" in name or "\\" in name:
            self._send_json({"error": "bad_name"}, status=400)
            return

        cfg = CONFIG
        paths = resolve_bus_paths(cfg)
        fp = paths["orchestra_archive"] / name
        if not fp.exists():
            self._send_json({"error": "not_found"}, status=404)
            return
        self._send_text(fp.read_text(encoding="utf-8", errors="replace"), content_type="application/json; charset=utf-8", status=200)

    def handle_archive_replay(self):
        body = self._read_json_body()
        if body is None:
            self._send_json({"error": "bad_json"}, status=400)
            return
        name = (body.get("name") or "").strip()
        if not name.startswith("job_") or not name.endswith(".json") or "/" in name or "\\" in name:
            self._send_json({"error": "bad_name"}, status=400)
            return

        cfg = CONFIG
        paths = resolve_bus_paths(cfg)
        src = paths["orchestra_archive"] / name
        if not src.exists():
            self._send_json({"error": "not_found"}, status=404)
            return

        try:
            job = json.loads(src.read_text(encoding="utf-8"))
        except Exception:
            self._send_json({"error": "archive_bad_json"}, status=400)
            return

        # Replay as a new cycle (additive, no contract break)
        job = dict(job or {})
        old_cycle = job.get("cycle_id", "")
        job["cycle_id"] = f"{old_cycle}_replay_{int(datetime.now().timestamp())}"
        job["timestamp_utc"] = utc_now_iso()
        job["replay_of"] = name  # additive

        # Write prompt file if it references a path but doesn't exist
        prompt_path = Path(job.get("user_prompt_path") or paths["prompt"])
        if not prompt_path.exists():
            prompt_text = (body.get("prompt_fallback") or "").strip()
            if prompt_text:
                atomic_write_text(prompt_path, prompt_text + "\n", encoding="utf-8")
                job["user_prompt_path"] = str(prompt_path)

        atomic_write_json(paths["job"], job)
        log_line(f"[REPLAY] {name} -> cycle_id={job['cycle_id']}")
        self._send_json({"status": "replayed", "cycle_id": job["cycle_id"], "replay_of": name})

    def handle_memory(self):
        cfg = CONFIG
        paths = resolve_bus_paths(cfg)
        canon = paths["praxis_canonical"]
        if not canon.exists():
            self._send_json({"error": "canonical_state_not_found"}, status=404)
            return
        try:
            obj = json.loads(canon.read_text(encoding="utf-8"))
        except Exception:
            self._send_json({"error": "canonical_state_bad_json"}, status=500)
            return
        mem = obj.get("memory") or {}
        self._send_json({"memory": mem, "last_updated_utc": obj.get("last_updated_utc", "")})

    def handle_conversations(self):
        cfg = CONFIG
        paths = resolve_bus_paths(cfg)
        db_path = paths["orchestra_conversations_db"]
        items = list_orchestra_conversations(db_path, limit=200)
        self._send_json({"conversations": items})

    def handle_submit_job(self):
        body = self._read_json_body()
        if body is None:
            self._send_json({"error": "bad_json"}, status=400)
            return

        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            self._send_json({"error": "prompt_required"}, status=400)
            return

        cfg = CONFIG
        defaults = cfg.get("defaults") or {}
        paths = resolve_bus_paths(cfg)

        # Locked required pieces
        cycle_id = (body.get("cycle_id") or "").strip() or f"cycle_{os.urandom(4).hex()}"
        model = (body.get("model") or cfg.get("default_model") or "").strip() or "qwen2.5:32b-instruct"

        # Optional new fields (all additive)
        conversation_id = body.get("conversation_id", None)
        system_prompt_path = body.get("system_prompt_path", None) or defaults.get("system_prompt_path")
        context_paths = body.get("context_paths", None) or []

        mode = (body.get("mode") or defaults.get("mode") or "single").strip().lower()
        if mode not in ("single", "rrr"):
            mode = "single"

        reflect = bool(body.get("reflect", defaults.get("reflect", False)))

        rrr_config = body.get("rrr_config", None) or {}

        # Options with compat shims
        options_in = body.get("options") or {}
        temperature = float(options_in.get("temperature", cfg.get("temperature", 0.2)))
        top_p = float(options_in.get("top_p", cfg.get("top_p", 0.9)))
        max_new_tokens = int(options_in.get("max_new_tokens", cfg.get("max_new_tokens", 512)))
        context_window = int(options_in.get("context_window", cfg.get("context_window", 8192)))

        # Write prompt into PRAXIS inbox path (locked)
        prompt_path = paths["prompt"]
        atomic_write_text(prompt_path, prompt + "\n", encoding="utf-8")

        job = {
            "cycle_id": cycle_id,
            "timestamp_utc": utc_now_iso(),
            "user_prompt_path": str(prompt_path),
            "options": {
                "ollama_host": cfg.get("ollama_host"),
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "context_window": context_window,
            },
        }

        # Additive fields only
        if conversation_id:
            job["conversation_id"] = str(conversation_id)
        if system_prompt_path:
            job["system_prompt_path"] = str(system_prompt_path)
        if context_paths:
            job["context_paths"] = [str(p) for p in context_paths]

        job["mode"] = mode
        if mode == "rrr":
            job["rrr_config"] = rrr_config

        job["reflect"] = reflect

        atomic_write_json(paths["job"], job)

        log_line(f"[SUBMIT] cycle_id={cycle_id} mode={mode} model={model} conv={conversation_id or 'null'} reflect={reflect}")
        self._send_json({"status": "submitted", "cycle_id": cycle_id, "mode": mode, "model": model, "time_utc": utc_now_iso()})


def main():
    global CONFIG
    ensure_dirs()
    CONFIG = load_config()

    host = CONFIG.get("host", "127.0.0.1")
    port = int(CONFIG.get("port", 8088))

    log_line(f"[START] URI server host={host} port={port}")
    server = ThreadingHTTPServer((host, port), URIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log_line("[STOP] URI server exiting")
        server.server_close()


if __name__ == "__main__":
    main()

