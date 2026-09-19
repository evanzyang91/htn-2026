#!/usr/bin/env python3
"""Fetch the retrieval embedding model so skills can be ranked by meaning.

    uv run python scripts/fetch_embedder.py          # ~90 MB, once per machine
    uv run python scripts/fetch_embedder.py --check  # is it here, and is it intact?

This is the ONE network step the embedder needs, and it is deliberately a step rather
than a dependency or a committed file:

* ``pyproject.toml`` is shared surface, and the model is not a package anyway;
* ``data/`` is git-ignored except for the UI detector, and 90 MB in a repository that
  a demo clones back is a cost with no payoff;
* and without it everything still works. Retrieval falls back to token overlap and
  SAYS so in every candidate's ``why``, which is what keeps ``make test`` network-free
  (see :mod:`skillweaver.skills.embed`).

The files land in ``<data_dir>/models/text_embedder``. Point ``SKILLWEAVER_EMBEDDER_DIR``
at one directory to share a single download between worktrees.

Each file is verified against the SHA-256 published by the Hugging Face API for that
exact revision, so a truncated download fails here rather than becoming a ranking that
is quietly wrong. Nothing is moved into place until it has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillweaver.config import settings  # noqa: E402
from skillweaver.skills.embed import MODEL_FILE, VOCAB_FILE  # noqa: E402

REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "main"

FILES = {
    MODEL_FILE: (
        "onnx/model.onnx",
        "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    ),
    VOCAB_FILE: (
        "vocab.txt",
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
}
"""Local name -> (path in the model repository, SHA-256 of its contents).

The digests are the ones the Hugging Face API reports for these files (the LFS ``oid``
for the model, the blob's own hash for the vocabulary), recorded 2026-09-19. The
unquantized fp32 export is the one taken: the quantized variants are a quarter of the
size and change the vectors, and a stored skill ranked by one set of weights should not
be scored against another.
"""

_CHUNK = 1 << 20


def download(url: str, into: Path) -> str:
    """Stream ``url`` to ``into`` and return the SHA-256 of what arrived."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, into.open("wb") as out:  # noqa: S310
        while chunk := response.read(_CHUNK):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def check(target: Path) -> int:
    """Report whether the weights are present and intact. Exit code, not an exception:
    this is what a setup script or a person asks before a measurement run."""
    ok = True
    for name, (_, expected) in FILES.items():
        path = target / name
        if not path.is_file():
            print(f"missing: {path}")
            ok = False
            continue
        actual = sha256_of(path)
        state = "ok" if actual == expected else f"CORRUPT (sha256 {actual})"
        print(f"{path}: {state}")
        ok = ok and actual == expected
    print("retrieval will rank by meaning" if ok else "retrieval will rank on keywords")
    return 0 if ok else 1


def fetch(target: Path, force: bool) -> int:
    target.mkdir(parents=True, exist_ok=True)
    for name, (remote, expected) in FILES.items():
        destination = target / name
        if destination.is_file() and not force and sha256_of(destination) == expected:
            print(f"{destination}: already here")
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
        print(f"fetching {url}")
        with tempfile.TemporaryDirectory(dir=target) as scratch:
            staged = Path(scratch) / name
            actual = download(url, staged)
            if actual != expected:
                print(f"REFUSED: {remote} hashed {actual}, expected {expected}")
                return 1
            shutil.move(str(staged), destination)
        print(f"{destination}: {destination.stat().st_size:,} bytes, sha256 ok")
    print(f"done - retrieval will rank by meaning from {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify, download nothing")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument(
        "--dir", type=Path, default=None, help="where to put the weights (default: settings)"
    )
    args = parser.parse_args()
    target = args.dir if args.dir is not None else settings().embedder_dir
    return check(target) if args.check else fetch(target, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
