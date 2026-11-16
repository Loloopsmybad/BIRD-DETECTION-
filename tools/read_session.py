"""read_session.py - read any OpenCode session's history from the local DB.

Lets any OpenCode session (this or another one) catch up on what a
different session did, without the user pasting anything.

Usage:
    python tools/read_session.py                     # list recent sessions
    python tools/read_session.py <query>             # search sessions (id/slug/title)
    python tools/read_session.py <session-id-or-slug> --last 20
    python tools/read_session.py <session-id-or-slug> --full

Examples:
    python tools/read_session.py                     # list
    python tools/read_session.py curious-mountain --last 15
"""

import argparse
import datetime
import json
import os
import sqlite3
import sys

DB = os.path.expandvars(r"%USERPROFILE%\.local\share\opencode\opencode.db")


def connect():
    uri = "file:" + DB.replace("\\", "/") + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def fmt_time(ms):
    if not ms:
        return "-"
    return datetime.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def find_sessions(conn, query=None, limit=25):
    rows = conn.execute(
        "SELECT id, slug, title, directory, agent, time_created, time_updated, "
        "tokens_input, tokens_output FROM session_v2 ORDER BY time_updated DESC"
    ).fetchall()
    out = []
    for r in rows:
        text = f"{r[0]} {r[1]} {r[2] or ''} {r[3]}".lower()
        if query and query.lower() not in text:
            continue
        out.append(
            {
                "id": r[0],
                "slug": r[1],
                "title": r[2] or "",
                "directory": r[3],
                "agent": r[4],
                "created": fmt_time(r[5]),
                "updated": fmt_time(r[6]),
                "tokens_in": r[7],
                "tokens_out": r[8],
            }
        )
        if len(out) >= limit:
            break
    return out


def resolve_session(conn, ref):
    # exact id
    row = conn.execute(
        "SELECT id FROM session_v2 WHERE id=? OR slug=?", (ref, ref)
    ).fetchone()
    if row:
        return row[0]
    # substring match on slug/title
    rows = conn.execute("SELECT id, slug, title FROM session_v2").fetchall()
    ref_l = ref.lower()
    for r in rows:
        if ref_l in r[0].lower() or ref_l == r[1].lower() or ref_l in (r[2] or "").lower():
            return r[0]
    return None


def text_of_content(content):
    """Flatten a message's content list into readable text."""
    if isinstance(content, str):
        return content
    chunks = []
    if isinstance(content, list):
        for p in content:
            t = p.get("type")
            if t in ("text", "reasoning"):
                txt = p.get("text") or ""
                if t == "reasoning":
                    continue  # skip thinking by default
                if txt:
                    chunks.append(txt)
            elif t == "tool":
                tool = p.get("tool") or ""
                state = p.get("state", {})
                title = state.get("title") or ""
                chunks.append(f"[tool:{tool}] {title}")
    return "\n".join(chunks)


def dump_messages(conn, sid, last=0, full=False, show_tool=True):
    rows = conn.execute(
        "SELECT type, seq, data FROM session_message "
        "WHERE session_id=? AND type IN ('user','assistant') ORDER BY seq",
        (sid,),
    ).fetchall()
    if last:
        rows = rows[-last:]
    for typ, seq, data in rows:
        try:
            d = json.loads(data)
        except Exception:
            continue
        t = fmt_time(d.get("time", {}).get("created"))
        who = "USER" if typ == "user" else "AI"
        txt = text_of_content(d.get("content"))
        if not full:
            # compact: trim long text, drop tool spam unless --show-tool
            lines = []
            for ln in txt.splitlines():
                if ln.startswith("[tool:") and not show_tool:
                    continue
                lines.append(ln)
            txt = "\n".join(lines)
            if len(txt) > 1200:
                txt = txt[:1200] + f" ...[+{len(txt)-1200} chars]"
        print(f"--- [{seq}] {who} @ {t} ---")
        print(txt or "(empty)")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="search text, or session id/slug")
    ap.add_argument("--last", type=int, default=0, help="only last N messages")
    ap.add_argument("--full", action="store_true", help="no truncation")
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    conn = connect()

    if not args.query:
        sessions = find_sessions(conn, limit=args.limit)
        print(f"recent OpenCode sessions ({len(sessions)}):")
        for s in sessions:
            print(
                f"  {s['id']} | {s['slug']:16} | {s['agent']:7} | "
                f"{s['title'][:45]:45} | {s['directory'][:35]:35} | upd {s['updated']}"
            )
        return

    sid = resolve_session(conn, args.query)
    if not sid:
        print(f"no session matching {args.query!r}; run with no args to list.")
        return
    meta = conn.execute(
        "SELECT id, slug, title, directory, agent, time_created, time_updated FROM session_v2 WHERE id=?",
        (sid,),
    ).fetchone()
    print(f"session: {meta[0]} | {meta[1]} | {meta[2]} | {meta[3]} | {meta[4]}")
    print(f"created {fmt_time(meta[5])} -> updated {fmt_time(meta[6])}")
    print()
    dump_messages(conn, sid, last=args.last, full=args.full)


if __name__ == "__main__":
    main()
