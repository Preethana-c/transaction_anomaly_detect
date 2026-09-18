"""
Transaction + anomaly-score logging.

Two implementations behind the same three methods (`insert`, `recent`,
`clear`) so the rest of the app doesn't care which one is active:

- `SupabaseStorage` -- the real thing, used when SUPABASE_URL/SUPABASE_KEY
  are set (see README.md "Supabase setup" for the table schema to create).
- `InMemoryStorage` -- a capped ring buffer, used otherwise. This lets the
  streaming endpoints be built and tested without a live Supabase project,
  and keeps local development from requiring cloud credentials.

`get_storage()` picks one at import time based on config and is what the
rest of the API imports.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

from api import config


class InMemoryStorage:
    def __init__(self, maxlen: int = 5000):
        self._rows: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._next_id = 1

    def insert(self, row: dict[str, Any]) -> None:
        row = {**row, "id": self._next_id, "ingested_at": datetime.now(timezone.utc).isoformat()}
        self._next_id += 1
        self._rows.appendleft(row)

    def recent(self, limit: int = 50, flagged_only: bool = False) -> list[dict[str, Any]]:
        rows = self._rows
        if flagged_only:
            rows = [r for r in rows if r["flagged"]]
        return list(rows)[:limit]

    def clear(self) -> None:
        self._rows.clear()


class SupabaseStorage:
    def __init__(self):
        from supabase import create_client

        self._client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        self._table = config.SUPABASE_TABLE

    def insert(self, row: dict[str, Any]) -> None:
        self._client.table(self._table).insert(row).execute()

    def recent(self, limit: int = 50, flagged_only: bool = False) -> list[Any]:
        query = self._client.table(self._table).select("*").order("ingested_at", desc=True).limit(limit)
        if flagged_only:
            query = query.eq("flagged", True)
        return query.execute().data

    def clear(self) -> None:
        # Demo convenience only -- deletes every row. Never exposed to
        # the dashboard directly, only used by the backend's own reset.
        self._client.table(self._table).delete().neq("id", -1).execute()


_storage = None


def get_storage():
    global _storage
    if _storage is None:
        _storage = SupabaseStorage() if config.SUPABASE_CONFIGURED else InMemoryStorage()
    return _storage
