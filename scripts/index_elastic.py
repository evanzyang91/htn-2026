#!/usr/bin/env python3
"""Load a skillweaver data directory into Elasticsearch, so ``skillweaver search`` can
ask it questions.

    uv run python scripts/index_elastic.py                       # everything under data/
    uv run python scripts/index_elastic.py --only steps,runs     # just the recordings
    uv run python scripts/index_elastic.py --recreate            # drop and rebuild
    uv run python scripts/index_elastic.py --data-dir other/data --url http://es:9200

This script is the ONLY writer. The agent never indexes anything: a run leaves its
records on disk as before, and a person runs this afterwards to make them searchable.
That is what "read-only" means for the integration - the reader
(``skillweaver.dashboard.elastic.search``) has no write path, and if the address it is
given carries a read-only API key, nothing in ``src/`` can ever fail for lack of a
write permission because nothing there tries one.

Documents are keyed by content identity (``run_id:index``, ``domain:fingerprint``,
``domain:name``), so re-running over the same directory REPLACES rather than duplicates.
``--recreate`` is for a mapping change: it drops each index before writing it.

Prints one line per index with how many documents went in, and the wall time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skillweaver.dashboard.elastic import (  # noqa: E402
    INDICES,
    KINDS,
    MAPPINGS,
    ElasticClient,
    ElasticError,
    documents,
)

BATCH = 500
"""Documents per ``_bulk`` request. A step document with two screens of text is ~10 KB;
500 keeps a request well under Elasticsearch's default 100 MB limit."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SKILLWEAVER_DATA_DIR") or "data"),
        help="The skillweaver data directory (default: $SKILLWEAVER_DATA_DIR or data/).",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("SKILLWEAVER_ELASTIC_URL") or "http://127.0.0.1:9200",
        help="Elasticsearch base URL (default: $SKILLWEAVER_ELASTIC_URL or localhost:9200).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SKILLWEAVER_ELASTIC_API_KEY"),
        help="API key with write access to skillweaver-* ($SKILLWEAVER_ELASTIC_API_KEY).",
    )
    parser.add_argument(
        "--only",
        default=",".join(KINDS),
        help=f"Comma-separated kinds to index (default: all of {','.join(KINDS)}).",
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Drop each index before writing it."
    )
    args = parser.parse_args(argv)

    kinds = [k.strip() for k in args.only.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        parser.error(f"unknown kinds {unknown}; choose from {list(KINDS)}")
    if not args.data_dir.is_dir():
        parser.error(f"{args.data_dir} is not a directory")

    client = ElasticClient(args.url, args.api_key, timeout=60.0)
    started = time.perf_counter()
    try:
        info = client.info()
        version = info.get("version", {}).get("number", "?")
        print(f"elasticsearch {version} at {client.url}")
        for kind in kinds:
            made = client.put_index(INDICES[kind], MAPPINGS[kind], recreate=args.recreate)
            if made:
                print(f"  created {INDICES[kind]}")

        counts: dict[str, int] = defaultdict(int)
        batches: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for doc in documents(args.data_dir, kinds):
            batches[doc.kind].append((doc.id, doc.body))
            if len(batches[doc.kind]) >= BATCH:
                counts[doc.kind] += client.bulk(INDICES[doc.kind], batches.pop(doc.kind))
        for kind, pending in batches.items():
            counts[kind] += client.bulk(INDICES[kind], pending)
    except ElasticError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for kind in kinds:
        total = client.count(INDICES[kind])
        print(f"  {INDICES[kind]:<22} {counts[kind]:>6} indexed, {total:>6} total")
    print(f"done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
