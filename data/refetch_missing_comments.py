"""
Patch missing top_comments in data/datasets/reddit_posts.json.

Reddit's old.reddit JSON endpoint silently soft-throttles under sustained
load: instead of HTTP 429 it serves an HTML challenge page. The original
fetcher's bare `except Exception: return []` swallowed those, so ~57% of
posts ended up with top_comments=[] even though their threads have public
comments. This script targets just those rows, retries on non-JSON
responses, and logs every failure loudly.

Run with:
    conda activate analytics && python data/refetch_missing_comments.py
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PATH = Path(__file__).parent / "datasets" / "reddit_posts.json"
UA = "atf-scraper/0.1 (cs4300 project)"
SLEEP = 2.0
RETRY_DELAYS = [15, 45]
COMMENTS_PER_POST = 5
SAVE_EVERY = 25


def _get_json(url):
    """Return parsed JSON. Raises RuntimeError when the body is non-JSON
    (Reddit's soft rate-limit returns HTML — surface it so the caller can
    retry instead of silently storing [])."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        preview = raw[:120].decode("utf-8", errors="replace")
        raise RuntimeError(f"non-JSON response (likely soft-429): {preview}") from e


def _fetch_top_comments(permalink):
    path = (permalink or "") \
        .replace("https://reddit.com", "") \
        .replace("https://www.reddit.com", "")
    # Reddit permalinks can contain non-ASCII slug chars (è, é, …). urllib's
    # HTTP layer is ASCII-only, so percent-encode everything except path
    # separators before sending.
    path = urllib.parse.quote(path, safe="/")
    url = f"https://old.reddit.com{path}.json?limit=50&sort=top&raw_json=1"
    data = _get_json(url)
    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError(f"unexpected payload shape: {type(data).__name__}")
    kids = data[1].get("data", {}).get("children", [])
    cs = []
    for k in kids:
        if k.get("kind") != "t1":
            continue
        d = k.get("data", {})
        body = d.get("body") or ""
        if body in ("", "[deleted]", "[removed]"):
            continue
        cs.append({"score": d.get("score", 0), "body": body[:2000]})
    cs.sort(key=lambda c: c["score"], reverse=True)
    return cs[:COMMENTS_PER_POST]


def fetch_with_retry(permalink, post_id, n, total):
    """Returns (comments, success_flag). False ⇒ exhausted retries."""
    attempts = [0] + RETRY_DELAYS
    last_err = None
    for attempt, delay in enumerate(attempts):
        if delay:
            print(f"  [{n}/{total}] {post_id}: retry {attempt} in {delay}s ({last_err})")
            time.sleep(delay)
        try:
            return _fetch_top_comments(permalink), True
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
            last_err = e
    print(f"  [{n}/{total}] {post_id}: PERMANENT FAIL: {last_err}")
    return [], False


def main():
    posts = json.loads(PATH.read_text(encoding="utf-8"))
    targets = [
        i for i, p in enumerate(posts)
        if not p.get("top_comments") and p.get("num_comments", 0) > 0
    ]
    print(f"need to patch {len(targets)} of {len(posts)} posts")
    ok = fail = got_zero = 0
    for n, idx in enumerate(targets, start=1):
        p = posts[idx]
        cs, success = fetch_with_retry(p.get("permalink", ""), p.get("id"), n, len(targets))
        p["top_comments"] = cs
        if success:
            ok += 1
            if not cs:
                got_zero += 1
        else:
            fail += 1
        if n % SAVE_EVERY == 0:
            PATH.write_text(json.dumps(posts, indent=2), encoding="utf-8")
            print(f"  checkpoint @ {n}/{len(targets)} (ok={ok} fail={fail} zero={got_zero})")
        time.sleep(SLEEP)
    PATH.write_text(json.dumps(posts, indent=2), encoding="utf-8")
    print(f"done: ok={ok} fail={fail} zero={got_zero}")


if __name__ == "__main__":
    main()
