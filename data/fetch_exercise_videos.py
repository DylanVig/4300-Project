"""
Backfill YouTube video IDs for every exercise in
``data/datasets/exercises_free_db.json`` and write them to a sidecar JSON
file (``data/datasets/exercise_videos.json``) keyed by exercise ``id``.

The sidecar exists separately so the upstream-managed exercises file can be
re-downloaded without losing the video links. The Flask backend reads this
sidecar at startup and joins it to search results by id.

Quota notes (--source api):
    YouTube Data API v3 ``search.list`` costs 100 units/call. Default daily
    quota is 10,000 units → ~100 lookups per day at the free tier. The
    script is fully resumable: it loads any existing sidecar, skips ids
    already present, and persists progress every PERSIST_EVERY lookups so
    a 403 ``quotaExceeded`` mid-run loses at most that many entries.

Usage:
    source 4300-IR/bin/activate
    python data/fetch_exercise_videos.py --dry-run --limit 3       # zero cost
    python data/fetch_exercise_videos.py --source api --limit 90   # API path
    python data/fetch_exercise_videos.py --source ytdlp --limit 100  # yt-dlp path
    python data/fetch_exercise_videos.py --source ytdlp            # all remaining

Reads YOUTUBE_API_KEY from .env only when --source api (default: ytdlp).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

HERE = Path(__file__).parent
EXERCISES_PATH = HERE / "datasets" / "exercises_free_db.json"
SIDECAR_PATH = HERE / "datasets" / "exercise_videos.json"
ENV_PATH = HERE.parent / ".env"

API_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"
PERSIST_EVERY = 20
SLEEP_API_S = 0.2
SLEEP_YTDLP_S = 0.5


def load_env_key(env_path: Path, key_name: str) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key_name:
            return v.strip().strip('"').strip("'")
    return None


def load_sidecar(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"WARN: existing sidecar at {path} is invalid JSON; starting fresh.")
        return {}


def save_sidecar(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def build_query(name: str) -> str:
    return f"{name} exercise proper form"


def fetch_video_id_api(api_key: str, query: str) -> str | None:
    params = {
        "key": api_key,
        "q": query,
        "type": "video",
        "part": "id",
        "maxResults": "1",
        "safeSearch": "strict",
        "relevanceLanguage": "en",
    }
    url = f"{API_ENDPOINT}?{urlencode(params)}"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    items = body.get("items") or []
    if not items:
        return None
    return items[0].get("id", {}).get("videoId")


def fetch_video_id_ytdlp(query: str) -> str | None:
    from yt_dlp import YoutubeDL
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "default_search": "ytsearch1",
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch1:{query}", download=False)
    entries = info.get("entries") or []
    return entries[0].get("id") if entries else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", choices=["api", "ytdlp"], default="ytdlp",
                        help="Data source: YouTube Data API v3 or yt-dlp scrape (default: ytdlp).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of calls in this run (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print queries without calling anything. Zero cost.")
    parser.add_argument("--retry-empty", action="store_true",
                        help="Re-query exercises whose sidecar entry has video_id=null.")
    args = parser.parse_args()

    if not EXERCISES_PATH.exists():
        print(f"ERROR: exercises file not found at {EXERCISES_PATH}", file=sys.stderr)
        return 1

    with EXERCISES_PATH.open("r", encoding="utf-8") as f:
        exercises = json.load(f)

    sidecar = load_sidecar(SIDECAR_PATH)

    pending = []
    for ex in exercises:
        ex_id = ex.get("id")
        if not ex_id:
            continue
        existing = sidecar.get(ex_id)
        if existing is None:
            pending.append(ex)
        elif args.retry_empty and existing.get("video_id") is None:
            pending.append(ex)

    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"{len(sidecar)} entries already in sidecar.")
    print(f"{len(pending)} exercises queued this run (source={args.source}).")
    if not pending:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        print("--- DRY RUN — no calls ---")
        for i, ex in enumerate(pending, 1):
            print(f"  [{i}/{len(pending)}] id={ex['id']} q={build_query(ex['name'])!r}")
        return 0

    api_key = None
    if args.source == "api":
        api_key = load_env_key(ENV_PATH, "YOUTUBE_API_KEY")
        if not api_key:
            print(f"ERROR: YOUTUBE_API_KEY not found in {ENV_PATH}", file=sys.stderr)
            return 1

    sleep_s = SLEEP_API_S if args.source == "api" else SLEEP_YTDLP_S
    completed = 0
    try:
        for i, ex in enumerate(pending, 1):
            ex_id = ex["id"]
            name = ex.get("name", "")
            query = build_query(name)
            print(f"[{i}/{len(pending)}] id={ex_id} q={query!r}", flush=True)
            try:
                if args.source == "api":
                    video_id = fetch_video_id_api(api_key, query)
                else:
                    video_id = fetch_video_id_ytdlp(query)
            except HTTPError as e:
                if args.source == "api" and e.code == 403:
                    body = e.read().decode("utf-8", errors="replace")
                    if "quotaExceeded" in body or "dailyLimitExceeded" in body:
                        print(f"  → quota exceeded ({e.code}). Saving and exiting.",
                              file=sys.stderr)
                        return 2
                print(f"  → HTTP {e.code}: {e.reason}. Skipping.", file=sys.stderr)
                continue
            except Exception as e:
                print(f"  → {type(e).__name__}: {e}. Skipping.", file=sys.stderr)
                continue

            sidecar[ex_id] = {"video_id": video_id}
            print(f"  → video_id={video_id}")
            completed += 1

            if completed % PERSIST_EVERY == 0:
                save_sidecar(SIDECAR_PATH, sidecar)
                print(f"  (checkpoint: {len(sidecar)} total entries saved)")

            time.sleep(sleep_s)
    finally:
        save_sidecar(SIDECAR_PATH, sidecar)
        print(f"Saved {len(sidecar)} total entries to {SIDECAR_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
