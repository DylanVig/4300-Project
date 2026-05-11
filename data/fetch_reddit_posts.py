"""
Fetch top posts + comments from coaching subreddits via anonymous old.reddit JSON.
No auth required. Outputs data/datasets/reddit_posts.json.

Run with analytics conda env:
    conda activate analytics && python data/fetch_reddit_posts.py
"""
import json
import time
import urllib.request
from pathlib import Path

SUBS = [
    ("basketballcoach", "basketball"),
    #("SoccerCoachResources", "soccer"),
]
POSTS_PER_SUB = 500
COMMENTS_PER_POST = 5
FETCH_COMMENTS = True   # flip False if rate-limited (cuts ~95% of requests)
UA = "atf-scraper/0.1 (cs4300 project)"
SLEEP = 1.1
OUT_PATH = Path(__file__).parent / "datasets" / "reddit_posts.json"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _fetch_posts(sub):
    out, after = [], None
    while len(out) < POSTS_PER_SUB:
        qs = "?t=all&limit=100" + (f"&after={after}" if after else "")
        try:
            data = _get(f"https://old.reddit.com/r/{sub}/top.json{qs}")
        except Exception as e:
            print(f"  listing error: {e}")
            break
        kids = data.get("data", {}).get("children", [])
        if not kids:
            break
        for k in kids:
            d = k.get("data", {})
            if d.get("stickied") or not (d.get("title") or "").strip():
                continue
            out.append(d)
            if len(out) >= POSTS_PER_SUB:
                break
        after = data.get("data", {}).get("after")
        if not after:
            break
        time.sleep(SLEEP)
    return out


def _fetch_top_comments(permalink):
    try:
        data = _get(
            f"https://old.reddit.com{permalink}.json?limit=20&sort=top"
        )
    except Exception:
        return []
    if not isinstance(data, list) or len(data) < 2:
        return []
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


def _merge_rows(new_rows):
    """Load existing rows from OUT_PATH (if any), dedupe by post id, return
    the union with new rows winning. Lets the user run the script one
    subreddit at a time without clobbering data from earlier runs.
    """
    existing = []
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"existing file unreadable, treating as empty: {e}")
            existing = []
    by_id = {p.get("id"): p for p in existing if p.get("id")}
    for row in new_rows:
        rid = row.get("id")
        if rid:
            by_id[rid] = row
    return list(by_id.values())


def fetch():
    out = []
    for sub_name, sport in SUBS:
        print(f"[{sub_name}] fetching listing…")
        posts = _fetch_posts(sub_name)
        print(f"[{sub_name}] {len(posts)} posts found, fetching comments…")
        for i, d in enumerate(posts):
            row = {
                "id": d.get("id"),
                "subreddit": sub_name,
                "sport": sport,
                "title": d.get("title") or "",
                "selftext": (d.get("selftext") or "")[:8000],
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0),
                "permalink": f"https://reddit.com{d.get('permalink', '')}",
                "created_utc": d.get("created_utc"),
                "top_comments": [],
            }
            if FETCH_COMMENTS:
                row["top_comments"] = _fetch_top_comments(
                    d.get("permalink", "")
                )
                time.sleep(SLEEP)
            out.append(row)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(posts)}")
        print(f"[{sub_name}] done")
        time.sleep(SLEEP)

    merged = _merge_rows(out)
    OUT_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"wrote {len(merged)} rows ({len(out)} new/refreshed) → {OUT_PATH}")


if __name__ == "__main__":
    fetch()
