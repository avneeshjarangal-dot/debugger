"""
fetch_logs.py  —  Fetch, filter, and group error/warning logs from MongoDB.

Defaults (no flags needed for normal use):
  - levels  : ERROR and WARNING
  - window  : last 24 hours
  - grouping: logs with the same (service, function, message) are clubbed

Usage (run from inside debugger/):
    python fetch_logs.py
    python fetch_logs.py --service order-service
    python fetch_logs.py --level ERROR
    python fetch_logs.py --level all
    python fetch_logs.py --hours 48
    python fetch_logs.py --no-group
    python fetch_logs.py --service order-service --hours 6 --level ERROR

Mongo connection can also be set via env vars:
    MONGO_URI   MONGO_DB   MONGO_COLL
"""

import os
import sys
import re
import argparse
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DEFAULT_MONGO_URI  = os.environ.get("MONGO_URI")
DEFAULT_DB_NAME    = os.environ.get("MONGO_DB", "log")
DEFAULT_COLLECTION = os.environ.get("MONGO_COLLECTION", "logs")
DEFAULT_LEVELS     = ["ERROR", "WARNING"]   # fetched by default
DEFAULT_HOURS      = 100                    # rolling window
DEFAULT_LIMIT      = 500                    # mongo fetch cap before grouping


# ---------------------------------------------------------------------------
# Fetch — last N hours, ERROR + WARNING (or custom levels)
# ---------------------------------------------------------------------------
def fetch_logs(
    uri: str,
    db_name: str,
    collection_name: str,
    levels: list[str] | None = None,
    service: str | None = None,
    hours: int = DEFAULT_HOURS,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    try:
        from pymongo import MongoClient, DESCENDING
    except ImportError:
        print("\n  ERROR: pymongo not installed.  Run: pip install pymongo\n")
        sys.exit(1)

    if not uri:
        raise RuntimeError("MongoDB connection is not configured. Set MONGO_URI on the server.")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")          # fail fast if unreachable

    collection = client[db_name][collection_name]

    since = datetime.now(tz=timezone.utc) - timedelta(hours=hours)

    query: dict = {"timestamp": {"$gte": since}}

    if levels:
        upper = [l.upper() for l in levels]
        query["level"] = {"$in": upper} if len(upper) > 1 else upper[0]

    if service:
        query["service"] = service

    cursor = (
        collection
        .find(query)
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )

    return list(cursor)


# ---------------------------------------------------------------------------
# Group — club similar logs by (service, function, message_signature)
#
# "message signature" strips variable parts like IDs, numbers, and quoted
# strings so that repeated errors with different payloads collapse together.
# ---------------------------------------------------------------------------
_VAR_PATTERN = re.compile(
    r"'[^']*'"           # single-quoted strings
    r'|"[^"]*"'          # double-quoted strings
    r"|[0-9a-f]{8,}"     # hex IDs / ObjectIds
    r"|\b\d+\b"          # plain numbers
)

def _message_sig(message: str) -> str:
    """Strip variable parts from a message to get a stable grouping key."""
    return _VAR_PATTERN.sub("?", message).strip()


def group_logs(logs: list[dict]) -> list[dict]:
    """
    Club logs that share (service, function, message_signature) into one
    group. Each group keeps the most recent log as the representative and
    records all unique (file, line) locations and occurrence timestamps.

    Returns a list of group dicts, sorted by occurrence count descending.
    """
    groups: dict[tuple, dict] = {}

    for log in logs:
        service  = log.get("service",  "unknown")
        function = log.get("function", "unknown")
        message  = log.get("message",  "")
        sig      = _message_sig(message)

        key = (service, function, sig)

        if key not in groups:
            groups[key] = {
                # representative — most recent (logs are already sorted desc)
                "representative": log,
                "level":          log.get("level", "ERROR"),
                "service":        service,
                "function":       function,
                "message_sig":    sig,
                # accumulate across occurrences
                "count":          0,
                "timestamps":     [],
                "locations":      set(),   # (file, line) pairs
                "extra_samples":  [],      # up to 3 unique extra payloads
            }

        g = groups[key]
        g["count"] += 1
        g["timestamps"].append(log.get("timestamp"))

        loc = (log.get("file", ""), str(log.get("line", "")))
        g["locations"].add(loc)

        extra = log.get("extra")
        if extra and len(g["extra_samples"]) < 3:
            if extra not in g["extra_samples"]:
                g["extra_samples"].append(extra)

    # convert sets to sorted lists for display
    result = []
    for g in groups.values():
        g["locations"] = sorted(g["locations"])
        result.append(g)

    result.sort(key=lambda g: g["count"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
SEP  = "─" * 72
SEP2 = "═" * 72

LEVEL_BADGE = {
    "ERROR":   "[ ERROR ]",
    "WARNING": "[ WARN  ]",
}


def fmt_ts(ts) -> str:
    if ts is None:
        return "—"
    if hasattr(ts, "isoformat"):
        return ts.replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if isinstance(ts, dict) and "$date" in ts:
        return ts["$date"]
    return str(ts)


def display_grouped(groups: list[dict], total_raw: int):
    if not groups:
        print("\n  No logs found in the given window.\n")
        return

    print(f"\n  {total_raw} log(s) fetched  →  {len(groups)} unique issue(s)\n")

    for i, g in enumerate(groups, 1):
        rep   = g["representative"]
        badge = LEVEL_BADGE.get(g["level"], f"[ {g['level']} ]")

        print(SEP2)
        print(f"  {badge}  #{i}  ×{g['count']} occurrence(s)")
        print(SEP2)

        def row(label, value):
            print(f"  {label:<18} {value}")

        row("service",   g["service"])
        row("function",  g["function"])
        row("message",   rep.get("message", "—"))

        # time range
        valid_ts = [t for t in g["timestamps"] if t is not None]
        if valid_ts:
            row("first seen",  fmt_ts(min(valid_ts, default=None)))
            row("last seen",   fmt_ts(max(valid_ts, default=None)))

        # all (file, line) locations seen in this group
        print()
        print("  [ location(s) ]")
        print(SEP)
        for file_path, line in g["locations"]:
            print(f"  {file_path}  line {line}")

        # traceback from the representative log
        tb = rep.get("traceback")
        if tb:
            print()
            print("  [ traceback ]")
            print(SEP)
            for line in tb.splitlines():
                print(f"  {line}")

        # sample extra payloads
        if g["extra_samples"]:
            print()
            print(f"  [ extra  (up to 3 samples) ]")
            print(SEP)
            for sample in g["extra_samples"]:
                parts = "  |  ".join(f"{k}: {v}" for k, v in sample.items())
                print(f"  {parts}")

        print()

    print(SEP2)
    print()


def display_flat(logs: list[dict]):
    """Plain display when --no-group is passed."""
    if not logs:
        print("\n  No logs found.\n")
        return

    print(f"\n  {len(logs)} log(s)\n")

    for i, log in enumerate(logs, 1):
        print(SEP2)
        print(f"  LOG {i}")
        print(SEP2)

        def row(label, value):
            print(f"  {label:<16} {value}")

        row("timestamp", fmt_ts(log.get("timestamp")))
        row("level",     log.get("level",    "—"))
        row("service",   log.get("service",  "—"))
        row("message",   log.get("message",  "—"))
        row("file",      log.get("file",     "—"))
        row("function",  log.get("function", "—"))
        row("line",      log.get("line",     "—"))

        extra = log.get("extra")
        if extra:
            print()
            print("  [ extra ]")
            print(SEP)
            for k, v in extra.items():
                row(k, v)

        tb = log.get("traceback")
        if tb:
            print()
            print("  [ traceback ]")
            print(SEP)
            for line in tb.splitlines():
                print(f"  {line}")

        print()

    print(SEP2)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch and group logs from MongoDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri",        default=DEFAULT_MONGO_URI,
                        help="MongoDB connection URI")
    parser.add_argument("--db",         default=DEFAULT_DB_NAME,
                        help="Database name")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help="Collection name")
    parser.add_argument("--level",      default=None,
                        help="ERROR | WARNING | INFO | all  (default: ERROR+WARNING)")
    parser.add_argument("--service",    default=None,
                        help="Filter by service name")
    parser.add_argument("--hours",      default=DEFAULT_HOURS, type=int,
                        help="Rolling window in hours (default: 24)")
    parser.add_argument("--no-group",   action="store_true",
                        help="Skip grouping, show raw logs")
    return parser.parse_args()


def main():
    args = parse_args()

    # resolve levels
    if args.level is None:
        levels = DEFAULT_LEVELS          # ERROR + WARNING
    elif args.level.lower() == "all":
        levels = None                    # no level filter
    else:
        levels = [args.level.upper()]

    print(f"\n  Connecting to MongoDB...")
    print(f"  URI        : {args.uri}")
    print(f"  Database   : {args.db}")
    print(f"  Collection : {args.collection}")
    print(f"  Levels     : {', '.join(levels) if levels else 'all'}")
    print(f"  Window     : last {args.hours}h")
    print(f"  Service    : {args.service or 'all'}")
    print(f"  Grouping   : {'off' if args.no_group else 'on'}")

    try:
        logs = fetch_logs(
            uri=args.uri,
            db_name=args.db,
            collection_name=args.collection,
            levels=levels,
            service=args.service,
            hours=args.hours,
        )
    except Exception as e:
        print(f"\n  ERROR: {e}\n")
        sys.exit(1)

    if args.no_group:
        display_flat(logs)
    else:
        groups = group_logs(logs)
        display_grouped(groups, total_raw=len(logs))


if __name__ == "__main__":
    main()