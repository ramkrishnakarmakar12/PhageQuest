"""Cached proxy for public data (spec section 03.1).

    "PhagesDB serves JSON only, one gene per call, and ignores query filters --
    cache aggressively server-side. NCBI E-utilities allows 3 req/s without a
    key; thirty students will trip that in the first minute. Get an API key
    before the first pilot."

So this module is not a performance optimisation. Without it, the first lesson
of the first pilot fails in its opening minutes, in front of a class.

Two mechanisms:
  * a persistent cache keyed on (source, path, query), with per-source TTLs;
  * a token-bucket rate limiter per upstream, so thirty simultaneous students
    queue politely behind one shared budget instead of each hitting NCBI.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx
from sqlalchemy.orm import Session

from .models import CacheEntry

SOURCES = {
    "phagesdb": {
        "base": "https://phagesdb.org/api/",
        "ttl_hours": 24 * 7,     # the corpus changes slowly; a week is generous
        "rate_per_second": 2.0,
        "note": "JSON only, one gene per call, ignores query filters.",
    },
    "ncbi": {
        "base": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
        "ttl_hours": 24 * 30,    # a published record does not change
        # 3 req/s without a key, 10 with one. We stay under either.
        "rate_per_second": 3.0 if not os.getenv("NCBI_API_KEY") else 9.0,
        "note": "E-utilities. Set NCBI_API_KEY before a pilot.",
    },
}

_buckets: Dict[str, Dict[str, float]] = {
    k: {"tokens": v["rate_per_second"], "last": time.monotonic()}
    for k, v in SOURCES.items()
}
_locks: Dict[str, asyncio.Lock] = {}


def _key(source: str, path: str, params: Dict[str, Any]) -> str:
    raw = json.dumps({"s": source, "p": path, "q": sorted(params.items())}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:48]


async def _take_token(source: str) -> None:
    """Token bucket. Thirty students share one upstream budget."""
    cfg = SOURCES[source]
    lock = _locks.setdefault(source, asyncio.Lock())
    async with lock:
        b = _buckets[source]
        while True:
            now = time.monotonic()
            b["tokens"] = min(cfg["rate_per_second"],
                              b["tokens"] + (now - b["last"]) * cfg["rate_per_second"])
            b["last"] = now
            if b["tokens"] >= 1.0:
                b["tokens"] -= 1.0
                return
            await asyncio.sleep((1.0 - b["tokens"]) / cfg["rate_per_second"])


async def fetch_cached(db: Session, source: str, path: str,
                       params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = SOURCES.get(source)
    if cfg is None:
        return {"error": f"Unknown source {source!r}.", "available": sorted(SOURCES)}
    params = params or {}
    if source == "ncbi" and os.getenv("NCBI_API_KEY"):
        params = {**params, "api_key": os.environ["NCBI_API_KEY"]}

    key = _key(source, path, {k: v for k, v in params.items() if k != "api_key"})
    row = db.get(CacheEntry, key)
    now = datetime.now(timezone.utc)
    if row is not None:
        expires = row.expires_at.replace(tzinfo=timezone.utc) if row.expires_at else None
        if expires is None or expires > now:
            return {"source": source, "path": path, "cached": True,
                    "fetched_at": row.fetched_at, "data": row.payload}

    await _take_token(source)
    url = cfg["base"].rstrip("/") + "/" + path.lstrip("/")
    try:
        async with httpx.AsyncClient(timeout=20.0,
                                     headers={"User-Agent": "PhageQuest/0.1 (schools)"}) as c:
            resp = await c.get(url, params=params)
            resp.raise_for_status()
            try:
                payload = resp.json()
            except Exception:
                payload = {"text": resp.text}
    except httpx.HTTPError as exc:
        if row is not None:
            # Serve the stale copy rather than failing a lesson. Say it is stale.
            return {"source": source, "path": path, "cached": True, "stale": True,
                    "fetched_at": row.fetched_at, "data": row.payload,
                    "warning": (f"Could not reach {source} ({type(exc).__name__}); this is a "
                                f"cached copy from {row.fetched_at:%d %b %Y}.")}
        return {"error": "upstream_unavailable", "source": source,
                "message": (f"Could not reach {source} and nothing is cached for this request. "
                            f"The rest of the platform works offline -- this is the one part "
                            f"that needs the internet.")}

    expires = now + timedelta(hours=cfg["ttl_hours"])
    if row is None:
        db.add(CacheEntry(key=key, source=source, payload=payload,
                          fetched_at=now, expires_at=expires))
    else:
        row.payload, row.fetched_at, row.expires_at = payload, now, expires
    db.commit()
    return {"source": source, "path": path, "cached": False, "fetched_at": now,
            "data": payload}


def status(db: Session) -> Dict[str, Any]:
    from sqlalchemy import func, select
    counts = dict(db.execute(
        select(CacheEntry.source, func.count()).group_by(CacheEntry.source)).all())
    return {
        "sources": {k: {**{kk: vv for kk, vv in v.items() if kk != "base"},
                        "cached_entries": counts.get(k, 0)}
                    for k, v in SOURCES.items()},
        "ncbi_api_key_configured": bool(os.getenv("NCBI_API_KEY")),
        "warning": (None if os.getenv("NCBI_API_KEY") else
                    "No NCBI API key is configured. E-utilities allows 3 requests per second "
                    "without one; a class of thirty will trip that in the first minute of the "
                    "first lesson. Get a key before the first pilot."),
    }
