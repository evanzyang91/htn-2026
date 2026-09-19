"""OCR: reading the text on a screenshot with RapidOCR (PP-OCRv4, ONNX Runtime, local).

:class:`RapidOcrReader` is the project's :class:`skillweaver.contracts.TextReader`. It
runs the bundled ONNX models on the CPU, with no network and no API key, which is why
it can be the default: perception must work in a test, in an eval loop and on a plane.

Two decisions worth knowing about before using it:

**The model loads on the first :meth:`RapidOcrReader.read`, not on import.**
    Importing this module costs nothing, so a CLI that lists skills does not pay for
    ONNX Runtime. Construct the reader wherever you like; the 1-2 second load happens
    when text is first actually needed, and only once per reader.

**Failure is loud.** If the engine cannot be imported or built, or inference raises,
    :meth:`read` raises :class:`~skillweaver.errors.PerceptionError`. It never returns
    an empty list to mean "OCR is broken", because an agent cannot tell that apart from
    "this screen has no text" and would go on to click blindly. An empty list means the
    engine ran and found nothing.

OCR runs on the PHYSICAL-resolution image (sharper - it is what the model wants) and the
boxes are divided by ``Screenshot.scale`` on the way out, so everything this module
returns is in LOGICAL pixels like the rest of the system.

Not reading the same pixels twice
---------------------------------

Reading is the expensive half of perception by an enormous margin. Profiled against live
pages at 1280x800, a full-page read costs 0.44s to over 7s while capture costs 0.02-0.05s
and detection 0.06s - **84% to 97% of all perception time**, on every real page measured.
Two ways of making the read itself cheaper were measured and both lost: cropping to the
detector's boxes took 53.8s against 7.4s for one full-page read of the same frame (the
engine's per-call overhead dwarfs the pixel saving), and downscaling showed no reliable
win. So one saving is to not read at all.

:class:`CachingTextReader` is that saving, and :class:`PerceptionCounters` is how you know
it worked. Counts, unlike seconds, do not move when the machine is busy, so "this task
went from 14 reads to 3" is a claim that survives being measured on a loaded laptop.
It only ever fires on a frame nothing has touched, though, and a replay changes the
screen at every step, so the other saving had to be inside one read. That one is
:data:`DEFAULT_REC_BATCH`.

Recognition is the read
-----------------------

"OCR is expensive" is too coarse to optimize against, because RapidOCR is three models
and they are not close to equal. Timed by the engine's own per-stage clock on live
Wikipedia at 1280x800, medians of five, ``rec_batch_num=6`` as shipped::

    frame            lines     det      cls      rec    total
    Ada Lovelace        70    89 ms    28 ms   899 ms   1033 ms
    Photosynthesis      75    84 ms    32 ms   804 ms    938 ms
    dense table         58    87 ms    27 ms   923 ms   1055 ms

**Recognition is ~87% of a read; detection is ~8% and the angle classifier ~3%.**
Three consequences, each of which kills a plausible idea:

*Resolution cannot help.* Every crop is resized to height 48 before recognition, so
what sets rec's cost is each line's ASPECT RATIO, and halving the capture's pixel
density leaves every ratio exactly where it was. Downscaling can only touch det's 8% -
which is why the measurement above records no reliable win rather than a small one.
Browser captures are ``scale=1.0`` anyway (``BrowserController``'s default), so there
is no Retina factor to give back.

*Reading only where the detector looked is what already happens.* RapidOCR IS detect,
crop, recognize; rec never sees a pixel outside a detected line. Substituting the YOLO
detector's boxes would not read less, it would only pay the engine's per-call overhead
once per box - the 53.8s above.

*Skipping the classifier is not worth its 3%*, and it changes one or two lines' text
per page, so it buys the smallest win on offer at the price of a correctness argument.

What is worth it is how many lines go into one ONNX Runtime call. RapidOCR sorts the
crops by aspect ratio, batches ``rec_batch_num`` of them, and pads each batch to its
widest member. Six is its default. Measured on the frames above, speedup against that
default::

    rec_batch_num      1       2       4       6      12
    speedup         1.51x   0.84x   1.16x   1.00x    (slower)

**One line per call is ~1.5x on the whole read, and the ordering is not monotone**, so
the padding arithmetic is not the explanation: sorted crops waste only ~14% on a batch
of six, nowhere near 1.5x, and padding cannot make batch 2 the worst of the five. What
fits is memory: a batch of six lines of Wikipedia body text is a 3x48x~1900 input per
line and the intermediate feature maps are far larger again, so batch 1 stays in cache
where batch 6 streams. Whatever the mechanism, the number is measured on every page
tried and in both directions, and :data:`DEFAULT_REC_BATCH` is 1 because of it.

It costs no accuracy. Checked on eight live frames, including three scrolled ones:
every text element the batch-6 read found is still found by ``find_text`` at the same
place - 0 lost of 548 - and recall against the DOM's own labels goes from 338/385 to
340/385, because a batch padded less loses fewer of the spaces between words.

A cheaper read is not the same as a cached one, and the cache cannot be pushed down to
the line to make up the difference. Keyed on a line's exact pixels, a per-line cache
saves 3-4% of recognition across a NAVIGATION and 12-13% across a scroll (one pixel of
difference in a detected box is a different key, so it misses even where the content is
plainly unchanged); on a re-observation of one screen it would save everything, which is
the case :class:`CachingTextReader` already serves for free.

Ending a read that has gone wrong
---------------------------------

A read that never returns used to be unstoppable. A 12-task evaluation stopped
producing output after 49 runs and sat at 98.8% CPU for over 36 minutes with every
thread parked in ONNX Runtime's own ``WorkerLoop``, spinning - no log line, no error,
no timeout. Nothing caught it because the skill clock is enforced from Python: the
sandbox's trace hook fires on Python frames, and a native call executes none, so the
one place a bound is most needed is the one place it could not reach.

Two deliberate settings fix that here, where the native call is actually made.

**The read runs in a child process** (:class:`OcrWorker`), so abandoning it is a
``SIGKILL`` rather than a request the spinning threads are never going to read.
Nothing else can end such a call: a signal handler runs at a Python bytecode
boundary and there is no such boundary inside ONNX Runtime; a watchdog thread can
stop *waiting* but leaves the pool spinning at full CPU, which is the very symptom.
Killing the process is the only mechanism that both frees the caller and gives the
cores back, and ONNX Runtime itself offers no wall-clock cap on a ``Run``.
:data:`DEFAULT_READ_TIMEOUT_S` is the budget; exceeding it is a
:class:`PerceptionTimeout`, which is a :class:`~skillweaver.errors.PerceptionError`
- the eyes failed, and whatever asked for the read did not.

**The engine's thread pool is sized explicitly** (:data:`DEFAULT_OCR_THREADS`), not
inherited from the core count. ONNX Runtime sizes its intra-op pool from
``os.cpu_count()`` and then SPINS while waiting for work, which on a machine already
running several agents is all cost. Measured on this project's ``invoices@2x``
fixture, a 14-core machine, seconds per read and CPU-seconds burned per wall second::

    threads   idle wall   idle cpu/wall   6 readers at once   cpu-s per read
    default   0.245 s     6.11            1.80 s              3.6
    6         0.342 s     5.19            1.62 s              3.4
    4         0.293 s     3.81            1.38 s              2.3
    2         0.443 s     1.97            0.78 s              1.3
    1         0.811 s     1.01            0.97 s              1.0

The default pool is the FASTEST choice on an idle machine and the SLOWEST under the
contention that produced the hang - 14 spinning threads per reader, six readers, one
14-core machine. Four is the knee: 20% off the idle read, 1.3x faster than the
default under six-way load, and 1.6x less CPU burned to get there. Set
``SKILLWEAVER_OCR_THREADS=2`` for a fleet run, where it is 2.3x faster still.

Constraining the pool is a mitigation and not the bound: it makes the pathological
case far less likely and cannot make it impossible, which is why the kill exists.
"""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import json
import os
import queue
import struct
import subprocess
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillweaver.contracts import Box, Element, ElementKind, ElementSource, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.logging_ import get_logger
from skillweaver.perception.elements import reading_order, stable_id

__all__ = [
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_OCR_THREADS",
    "DEFAULT_READ_TIMEOUT_S",
    "DEFAULT_REC_BATCH",
    "CachingTextReader",
    "OcrWorker",
    "PerceptionCounters",
    "PerceptionCounts",
    "PerceptionTimeout",
    "RapidOcrReader",
    "content_key",
    "ocr_threads",
    "read_timeout_s",
    "rec_batch",
]

#: Recognitions below this score are dropped: at that level RapidOCR is reporting
#: shapes it could not read, and a wrong label is worse for an agent than no label.
DEFAULT_MIN_CONFIDENCE = 0.5

DEFAULT_READ_TIMEOUT_S = 60.0
"""Seconds one text read may take before it is abandoned and the engine killed.

Generous on purpose: this is a dead-man's switch, not a performance target. The
slowest honest read this project has measured is 7.4s on a dense live article, and
the same read under six-way contention is a few times that; 60s is comfortably above
anything real and four orders of magnitude below the 36 minutes the hang it exists
for actually ran. Override with ``SKILLWEAVER_OCR_TIMEOUT_S``; ``0`` disables the
bound, which only a test has any business doing.
"""

DEFAULT_OCR_THREADS = 4
"""Intra- and inter-op threads the ONNX Runtime sessions are given.

Explicit rather than inherited: left alone ONNX Runtime sizes its pool from
``os.cpu_count()`` and spins. See the module docstring for the measured table this
number comes from, and ``SKILLWEAVER_OCR_THREADS`` to change it.
"""

DEFAULT_REC_BATCH = 1
"""Text lines the recognizer is given per ONNX Runtime call.

One, which reads like a mistake and is the fastest setting measured, by a lot. See
"Recognition is the read" in the module docstring for the table and for why the
padding arithmetic does not explain it. ``SKILLWEAVER_OCR_REC_BATCH`` changes it,
which is how the number gets re-derived on a machine that is not this one rather
than nudged.
"""

log = get_logger(__name__)

_TIMEOUT_ENV = "SKILLWEAVER_OCR_TIMEOUT_S"
_THREADS_ENV = "SKILLWEAVER_OCR_THREADS"
_REC_BATCH_ENV = "SKILLWEAVER_OCR_REC_BATCH"


class PerceptionTimeout(PerceptionError):
    """A text read was abandoned because it did not return within its budget.

    A :class:`~skillweaver.errors.PerceptionError` and not a new top-level kind, so
    every caller that already treats a broken pair of eyes as "not the skill's fault"
    treats this the same way. What it must never be mistaken for is a skill failing
    its task: nothing has been learned about the skill, and recording one would
    demote a good one.
    """


def _positive_float(name: str, fallback: float) -> float:
    """``name`` from the environment as a non-negative float, else ``fallback``.

    A malformed value falls back rather than raising: perception refusing to start
    because a variable was misspelt would be a worse failure than the one this
    module is here to prevent.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        value = float(raw)
    except ValueError:
        return fallback
    return value if value >= 0 else fallback


def read_timeout_s() -> float:
    """The configured per-read budget: ``SKILLWEAVER_OCR_TIMEOUT_S``, else
    :data:`DEFAULT_READ_TIMEOUT_S`. Read per reader rather than at import, so a
    process that sets the variable gets the bound it asked for."""
    return _positive_float(_TIMEOUT_ENV, DEFAULT_READ_TIMEOUT_S)


def ocr_threads() -> int:
    """The configured ONNX Runtime pool size: ``SKILLWEAVER_OCR_THREADS``, else
    :data:`DEFAULT_OCR_THREADS`. Clamped to at least one thread."""
    return max(1, int(_positive_float(_THREADS_ENV, DEFAULT_OCR_THREADS)))


def rec_batch() -> int:
    """The configured recognizer batch: ``SKILLWEAVER_OCR_REC_BATCH``, else
    :data:`DEFAULT_REC_BATCH`. Clamped to at least one line."""
    return max(1, int(_positive_float(_REC_BATCH_ENV, DEFAULT_REC_BATCH)))


class RapidOcrReader:
    """A :class:`~skillweaver.contracts.TextReader` backed by RapidOCR.

    One element of kind :attr:`~skillweaver.contracts.ElementKind.text` per recognized
    line, with ``source=ElementSource.ocr``, a real confidence from the recognizer, a
    box in LOGICAL pixels and a :func:`~skillweaver.perception.elements.stable_id`
    already filled in.

    By default the engine runs in a CHILD PROCESS, one per reader, reused across
    reads: a read that does not come back within ``timeout_s`` is abandoned by killing
    that process, which is the only thing that ends a call executing no Python. See
    the module docstring for why nothing lighter works. The child is started on the
    first read and lives until :meth:`close`, the reader is garbage collected, or the
    parent exits.

    Instances are safe to share between threads (the engine is built once under a lock,
    and the worker serves one read at a time) and are cheap to create.

    Args:
        min_confidence: Recognitions scoring below this are dropped.
        engine: An already-built RapidOCR callable, for tests or for reusing one engine
            across readers. When given, nothing is imported, loaded or spawned, and the
            read happens IN THIS PROCESS - so it is not bounded, because there is no
            process of ours to kill.
        isolate: ``False`` builds the engine in this process instead of a child. The
            read is then unbounded; only use it to exercise the engine itself.
        timeout_s: Seconds one read may take. ``0`` disables the bound. Defaults to
            :func:`read_timeout_s`.
        threads: ONNX Runtime intra-/inter-op threads. Defaults to :func:`ocr_threads`.
        worker: A prepared :class:`OcrWorker` to read through, for tests that need to
            drive the bound itself.
    """

    __slots__ = (
        "_engine",
        "_isolate",
        "_lock",
        "_threads",
        "_worker",
        "_worker_lock",
        "min_confidence",
        "timeout_s",
    )

    def __init__(
        self,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        engine: Any | None = None,
        isolate: bool = True,
        timeout_s: float | None = None,
        threads: int | None = None,
        worker: OcrWorker | None = None,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.timeout_s = read_timeout_s() if timeout_s is None else float(timeout_s)
        self._threads = ocr_threads() if threads is None else max(1, int(threads))
        self._engine = engine
        self._isolate = isolate and engine is None
        self._worker = worker
        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()

    def __repr__(self) -> str:
        where = "worker" if self._isolate else "in-process"
        return f"RapidOcrReader({where}, timeout={self.timeout_s:g}s, threads={self._threads})"

    @property
    def loaded(self) -> bool:
        """Whether an engine has been built in THIS process yet.

        Always ``False`` for an isolated reader however many times it has read: the
        engine belongs to the child, and this process never imports ONNX Runtime.
        """
        return self._engine is not None

    @property
    def isolated(self) -> bool:
        """Whether reads run in a child process, and are therefore bounded."""
        return self._isolate

    def close(self) -> None:
        """Shut the worker process down. Reading again starts a new one."""
        with self._worker_lock:
            worker, self._worker = self._worker, None
        if worker is not None:
            worker.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.close()
        except Exception:
            pass

    def _ensure_engine(self) -> Any:
        """Build the engine in THIS process on first use.

        Raises:
            PerceptionError: if RapidOCR is not installed or its models cannot be
                loaded. The message names the cause, because "OCR silently found no
                text" is the single most expensive failure mode in this pipeline.
        """
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is None:
                self._engine = build_engine(self._threads)
            return self._engine

    def read(self, screenshot: Screenshot) -> list[Element]:
        """Read the text on ``screenshot``.

        Returns one element per recognized line, in reading order. An empty list means
        the engine ran and found no text.

        Raises:
            PerceptionTimeout: if the read did not return within ``timeout_s``. The
                engine process was killed; the next read starts a fresh one.
            PerceptionError: if the screenshot cannot be decoded, the engine cannot be
                loaded, or inference fails.
        """
        scale = screenshot.scale if screenshot.scale > 0 else 1.0
        raw = self._raw_rows(screenshot)

        elements: list[Element] = []
        for entry in raw or ():
            parsed = _parse_entry(entry)
            if parsed is None:
                continue
            polygon, text, confidence = parsed
            if not text.strip() or confidence < self.min_confidence:
                continue
            box = _polygon_to_box(polygon, scale, screenshot.width, screenshot.height)
            if box is None:
                continue
            element = Element(
                box=box,
                kind=ElementKind.text,
                text=text.strip(),
                confidence=confidence,
                stable_id=None,
                source=ElementSource.ocr,
            )
            elements.append(dataclasses.replace(element, stable_id=stable_id(element)))
        return reading_order(elements)

    def _raw_rows(self, screenshot: Screenshot) -> Any:
        """The recognizer's own ``[polygon, text, score]`` rows, however they are got."""
        if self._isolate:
            return self._worker_rows(screenshot)
        engine = self._ensure_engine()
        # Decoding is outside the try: an undecodable PNG is its own PerceptionError
        # and must not be reported as the engine having failed.
        image = screenshot.to_array(logical=False)
        try:
            raw, _elapsed = engine(image)
        except Exception as exc:
            raise PerceptionError(f"OCR inference failed: {exc}") from exc
        return raw

    def _worker_rows(self, screenshot: Screenshot) -> Any:
        """One read through the child process, at most one at a time per reader."""
        with self._worker_lock:
            if self._worker is None:
                self._worker = OcrWorker(threads=self._threads)
            worker = self._worker
            try:
                return worker.rows(screenshot, self.timeout_s)
            except PerceptionTimeout:
                # ``rows`` has already killed it. Drop it so the next read - which may
                # well be of a screen this engine can handle - starts somewhere clean.
                self._worker = None
                raise
            except PerceptionError:
                if not worker.alive:
                    self._worker = None
                raise


def _parse_entry(entry: Any) -> tuple[Any, str, float] | None:
    """Pull ``(polygon, text, confidence)`` out of one RapidOCR result row.

    RapidOCR returns ``[polygon, text, score]`` rows, but the exact container types have
    moved between releases, so this stays shape-driven rather than trusting one version.
    """
    try:
        polygon, text, score = entry[0], entry[1], entry[2]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        confidence = float(score)
    except (TypeError, ValueError):
        confidence = 0.0
    return polygon, str(text), max(0.0, min(1.0, confidence))


def _polygon_to_box(polygon: Any, scale: float, width: int, height: int) -> Box | None:
    """Convert a physical-pixel quadrilateral to a clamped LOGICAL-pixel ``Box``.

    Dividing by ``scale`` here is the one conversion that keeps OCR honest: skip it on a
    Retina capture and every click derived from text lands twice as far down the screen.
    """
    import numpy as np

    try:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if points.size == 0 or not np.all(np.isfinite(points)):
        return None
    x0 = float(points[:, 0].min()) / scale
    y0 = float(points[:, 1].min()) / scale
    x1 = float(points[:, 0].max()) / scale
    y1 = float(points[:, 1].max()) / scale

    left = max(0, min(int(np.floor(x0)), width))
    top = max(0, min(int(np.floor(y0)), height))
    right = max(left, min(int(np.ceil(x1)), width))
    bottom = max(top, min(int(np.ceil(y1)), height))
    box = Box(left, top, max(right - left, 1), max(bottom - top, 1))
    return box if box.area > 0 else None


# --------------------------------------------------------------------------------------
# The engine, and the process it is kept in
# --------------------------------------------------------------------------------------


def build_engine(threads: int, batch: int | None = None) -> Any:
    """Import RapidOCR and build it with an EXPLICIT thread pool and batch size.

    The one place the engine is constructed, so the parent process (``isolate=False``)
    and the worker child get identical settings and identical failure messages.

    ``intra_op_num_threads`` is the number that matters for the pool: ONNX Runtime
    spawns that many workers per session and spins them while they wait. RapidOCR
    forwards both values from its global config into the detection, classification
    and recognition sessions, so passing them here sizes all three.

    ``rec_batch_num`` is the number that matters for the clock. It defaults to
    :func:`rec_batch`, which is ONE line per call; RapidOCR ships six. The module
    docstring has the measurements.

    Raises:
        PerceptionError: if RapidOCR is not installed or its models cannot be loaded.
            The message names the cause, because "OCR silently found no text" is the
            single most expensive failure mode in this pipeline.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise PerceptionError(
            "OCR is unavailable: rapidocr-onnxruntime is not installed "
            f"({exc}). Install the project's dependencies with `make install`."
        ) from exc
    try:
        return RapidOCR(
            intra_op_num_threads=threads,
            inter_op_num_threads=threads,
            rec_batch_num=rec_batch() if batch is None else max(1, int(batch)),
        )
    except Exception as exc:  # model files missing, ONNX Runtime broken, ...
        raise PerceptionError(f"OCR engine could not be loaded: {exc}") from exc


_FRAME = struct.Struct("<I")
"""Every message on the worker pipe is a 4-byte little-endian length then that many
bytes. A length prefix rather than a delimiter because one of the messages is a PNG."""

_WORKER_BOOTSTRAP = "from skillweaver.perception.ocr import _worker_main; _worker_main()"


def _send_frame(stream: Any, payload: bytes) -> None:
    stream.write(_FRAME.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _recv_frame(stream: Any) -> bytes | None:
    """One frame, or ``None`` at end of stream or on a truncated one."""
    head = stream.read(_FRAME.size)
    if not head or len(head) < _FRAME.size:
        return None
    (size,) = _FRAME.unpack(head)
    body = stream.read(size)
    if body is None or len(body) < size:
        return None
    return body


def _worker_argv() -> list[str]:
    """The command that runs :func:`_worker_main` in a fresh interpreter."""
    return [sys.executable, "-c", _WORKER_BOOTSTRAP]


def _worker_env(threads: int) -> dict[str, str]:
    """The child's environment: this package importable, and every thread pool named.

    ``SKILLWEAVER_OCR_THREADS`` is what the child actually reads; the three
    ``*_NUM_THREADS`` variables are set because they are read by OpenMP and the BLAS
    libraries underneath ONNX Runtime AT IMPORT TIME, which is before the child can
    pass anything to a session. Setting them in the environment we hand to
    ``Popen`` is the only moment early enough.
    """
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[2])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{package_root}{os.pathsep}{existing}" if existing else package_root
    env[_THREADS_ENV] = str(threads)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    return env


_LIVE: set[subprocess.Popen[bytes]] = set()
"""Every engine process this interpreter has started and not yet let go of.

Kept so :func:`_kill_live_workers` can be certain. A reader that is simply dropped
gets its child cleaned up by ``__del__``; a process that exits while a reader is
still referenced somewhere would otherwise leave a child alive, and this child is
one that can be burning every core on the machine.
"""


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL and reap. Not ``terminate``: a process wedged in a native spin is not
    going to run a signal handler, and a polite request that is never read is exactly
    the failure this module exists to end."""
    _LIVE.discard(proc)
    try:
        proc.kill()
    except OSError:  # pragma: no cover - already gone
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL is not refusable
        log.warning("ocr.worker.unreaped", pid=proc.pid)


def _close_streams(proc: subprocess.Popen[bytes], *, stdin: bool, stdout: bool) -> None:
    """Close each pipe, but ONLY the ones no thread is still inside.

    Closing a pipe a thread is blocked reading is a fatal interpreter error
    (``_enter_buffered_busy``), which a real headless run produced on the first try.
    Leaking a file descriptor for a process that has already been killed is the
    cheaper mistake by a wide margin, so an unjoined thread means its pipe is left to
    the garbage collector.
    """
    for stream, joined in ((proc.stdin, stdin), (proc.stdout, stdout)):
        if stream is not None and joined:
            try:
                stream.close()
            except OSError:
                pass


@atexit.register
def _kill_live_workers() -> None:
    """Kill every engine process still running, before the interpreter tears down.

    Registered rather than left to ``__del__`` because ``atexit`` runs while threads
    and pipes still work, and because the cost of getting this wrong is a child at
    98.8% CPU that outlives the run that started it.
    """
    for proc in list(_LIVE):
        _kill(proc)


class OcrWorker:
    """A child process holding one OCR engine, whose reads can be abandoned.

    The point of the process is the kill. A read that has wedged inside ONNX Runtime
    is executing no Python, so nothing inside that interpreter will ever look at a
    flag, a signal or a trace hook again; ``SIGKILL`` is what ends it, and it is also
    what gives the spinning cores back. Everything else here is plumbing around that.

    One request is in flight at a time. The child is started lazily on the first
    :meth:`rows` and restarted automatically after it has been abandoned.

    Args:
        threads: ONNX Runtime pool size to build the engine with.
        argv: The command to run. Defaults to this module's own worker; a test may
            point it at a process that deliberately never answers.
        env: Environment for the child. Defaults to :func:`_worker_env`.
    """

    __slots__ = ("_argv", "_env", "_proc", "_pump", "_replies", "_sender", "_threads")

    def __init__(
        self,
        *,
        threads: int | None = None,
        argv: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._threads = ocr_threads() if threads is None else max(1, int(threads))
        self._argv = list(argv) if argv is not None else _worker_argv()
        self._env = dict(env) if env is not None else _worker_env(self._threads)
        self._proc: subprocess.Popen[bytes] | None = None
        self._replies: queue.Queue[bytes | None] = queue.Queue()
        self._pump: threading.Thread | None = None
        self._sender: threading.Thread | None = None

    def __repr__(self) -> str:
        return f"OcrWorker(pid={self.pid}, threads={self._threads})"

    @property
    def pid(self) -> int | None:
        """The child's pid, or ``None`` when it is not running."""
        return self._proc.pid if self._proc is not None else None

    @property
    def alive(self) -> bool:
        """Whether a child is running right now."""
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Spawn the child if it is not already running.

        Raises:
            PerceptionError: if the interpreter could not be started at all.
        """
        if self._proc is not None and self._proc.poll() is None:
            return
        self._reap()
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - argv is ours, not a shell
                self._argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                env=self._env,
                close_fds=True,
            )
        except OSError as exc:
            raise PerceptionError(f"the OCR engine process could not be started: {exc}") from exc
        _LIVE.add(self._proc)
        self._replies = queue.Queue()
        self._pump = threading.Thread(
            target=_pump_frames,
            args=(self._proc.stdout, self._replies),
            name="ocr-worker-reader",
            daemon=True,
        )
        self._pump.start()

    def rows(self, screenshot: Screenshot, timeout_s: float) -> Any:
        """Read ``screenshot`` in the child and return the recognizer's raw rows.

        Args:
            screenshot: The frame to read. Its PNG bytes and geometry go over the
                pipe; the child decodes them the same way this process would.
            timeout_s: Seconds to wait. ``0`` waits forever, which is what this class
                exists to avoid - only a test should pass it.

        Raises:
            PerceptionTimeout: the budget was spent. The child has been killed and the
                message says so, because a caller needs to know the cores came back.
            PerceptionError: the child died, could not be started, or reported that
                the engine failed. The message is the child's own.
        """
        self.start()
        proc = self._proc
        assert proc is not None and proc.stdin is not None  # noqa: S101 - start() guarantees it
        header = json.dumps(
            {
                "width": screenshot.width,
                "height": screenshot.height,
                "scale": screenshot.scale,
            }
        ).encode()
        # The request goes out on its own thread. A pipe holds 64 KiB and a screenshot
        # is bigger, so writing one to a child that has stopped reading BLOCKS - which
        # would be a second unbounded wait, in the caller, for exactly the reason the
        # first one exists. Only the reply is waited on here, with a deadline; a send
        # that never finishes ends when the kill closes the pipe under it.
        self._join_sender()
        self._sender = threading.Thread(
            target=_send_request,
            args=(proc.stdin, header, screenshot.png),
            name="ocr-worker-writer",
            daemon=True,
        )
        self._sender.start()

        try:
            reply = self._replies.get(timeout=timeout_s) if timeout_s > 0 else self._replies.get()
        except queue.Empty:
            pid = self.pid
            self.abandon()
            raise PerceptionTimeout(
                f"OCR did not return within {timeout_s:g}s and was abandoned; the engine "
                f"process (pid {pid}) was killed. The eyes failed - nothing has been "
                "learned about whatever asked for the read."
            ) from None
        if reply is None:
            code = proc.poll()
            self.abandon()
            raise PerceptionError(
                f"the OCR engine process exited (status {code}) without answering; "
                "its own output is on stderr"
            )
        return _decode_reply(reply)

    def abandon(self) -> None:
        """Kill the child now and forget it. Safe to call when none is running.

        ``kill`` and not ``terminate``: a process wedged in a native spin is not going
        to run a signal handler, and a polite request that is never read is exactly
        the failure this class exists to end.
        """
        proc, self._proc = self._proc, None
        if proc is None:
            return
        # Kill BEFORE touching anything. The child dying is what releases a writer
        # blocked on a full pipe and a reader blocked on an empty one; closing a
        # stream another thread is mid-call on is the way to turn this into a new bug.
        _kill(proc)
        if sys.is_finalizing():
            return
        sent = self._join_sender()
        pumped = self._join_pump()
        _close_streams(proc, stdin=sent, stdout=pumped)

    def close(self) -> None:
        """Ask the child to exit, and kill it if it will not.

        Closing its stdin is end-of-stream to the worker loop, which returns. A child
        that is mid-read cannot notice, so the wait is short and the kill is certain.
        """
        proc = self._proc
        if proc is None:
            return
        if sys.is_finalizing():
            # Interpreter teardown: the threads are about to be frozen wherever they
            # stand, so joining them cannot finish and closing a pipe one of them is
            # reading is a fatal error. Kill the child and let the OS do the rest.
            self._proc = None
            _kill(proc)
            return
        sent = self._join_sender()
        if sent and proc.stdin is not None:
            try:
                proc.stdin.close()  # end of stream: the worker loop returns
            except OSError:
                pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.abandon()
            return
        self._proc = None
        _LIVE.discard(proc)
        pumped = self._join_pump()
        _close_streams(proc, stdin=sent, stdout=pumped)

    def _join_pump(self) -> bool:
        """Wait for the reader thread. ``False`` means it is still in the pipe, and
        that the pipe must therefore be left alone."""
        pump, self._pump = self._pump, None
        if pump is None or pump is threading.current_thread():
            return True
        pump.join(timeout=2)
        return not pump.is_alive()

    def _join_sender(self) -> bool:
        """Wait for the writer thread, with the same meaning as :meth:`_join_pump`."""
        sender, self._sender = self._sender, None
        if sender is None or sender is threading.current_thread():
            return True
        sender.join(timeout=2)
        return not sender.is_alive()

    def _reap(self) -> None:
        """Clear away a child that has already exited."""
        if self._proc is not None:
            try:
                self._proc.wait(timeout=0)
            except subprocess.TimeoutExpired:  # pragma: no cover - poll() said it exited
                pass
            _LIVE.discard(self._proc)
            self._proc = None
        self._join_sender()
        self._join_pump()


def _send_request(stream: Any, header: bytes, png: bytes) -> None:
    """Write one request to the child, off the caller's thread.

    Failures are swallowed on purpose. A send can only fail because the child is gone
    or being killed, and both of those reach the caller as the reply that never
    arrives - reported once, in one place, rather than as two races for the same news.
    """
    try:
        _send_frame(stream, header)
        _send_frame(stream, png)
    except (OSError, ValueError):
        log.debug("ocr.worker.send_failed")


def _pump_frames(stream: Any, replies: queue.Queue[bytes | None]) -> None:
    """Move frames off the child's stdout so a reader can wait on them with a timeout.

    A queue and a thread rather than a plain blocking read, because a blocking read is
    precisely what cannot be given a deadline. The thread ends when the pipe closes,
    which killing the child does.
    """
    try:
        while True:
            frame = _recv_frame(stream)
            if frame is None:
                break
            replies.put(frame)
    except (OSError, ValueError):  # the pipe was closed under us, which is the kill
        pass
    finally:
        replies.put(None)


def _decode_reply(reply: bytes) -> Any:
    """The rows out of one worker reply, or the child's failure raised as ours."""
    try:
        message = json.loads(reply)
    except ValueError as exc:
        raise PerceptionError(f"the OCR engine process sent a reply we cannot read: {exc}") from exc
    if not isinstance(message, dict):
        raise PerceptionError("the OCR engine process sent a reply we cannot read")
    if message.get("ok"):
        return message.get("rows") or []
    # The child raises the same PerceptionError messages this module would, so its
    # wording is passed through verbatim rather than wrapped in ours.
    raise PerceptionError(str(message.get("error") or "the OCR engine process failed"))


def _jsonable_rows(raw: Any) -> list[list[Any]]:
    """RapidOCR's rows reduced to ``[[[x, y], ...], text, score]``, JSON-safe.

    Parsed in the child so numpy never has to cross the pipe, and so a row the
    recognizer returned in a shape we do not understand is dropped where it is made
    rather than travelling as far as the element list.
    """
    import numpy as np

    rows: list[list[Any]] = []
    for entry in raw or ():
        parsed = _parse_entry(entry)
        if parsed is None:
            continue
        polygon, text, confidence = parsed
        try:
            points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        if points.size == 0 or not np.all(np.isfinite(points)):
            continue
        rows.append([points.tolist(), text, confidence])
    return rows


def _worker_main() -> int:
    """The child: read frames, answer with rows, until stdin ends.

    Started by :class:`OcrWorker`, never by a person. Two things matter here.

    The reply channel is a DUPLICATE of the original stdout and file descriptor 1 is
    then pointed at stderr, so RapidOCR's logging, a stray ``print`` in a dependency
    and anything else that writes to stdout cannot corrupt the frames. The engine is
    built on the first request rather than at startup, so a failure to load is
    reported down the same channel as any other failure instead of killing a child
    the parent is still waiting on.
    """
    channel = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    stdin = sys.stdin.buffer
    threads = ocr_threads()
    engine: Any | None = None

    while True:
        header_frame = _recv_frame(stdin)
        if header_frame is None:
            return 0
        png = _recv_frame(stdin)
        if png is None:
            return 0
        try:
            header = json.loads(header_frame)
            shot = Screenshot(
                png=png,
                width=int(header["width"]),
                height=int(header["height"]),
                scale=float(header["scale"]),
                captured_at=_EPOCH,
            )
            image = shot.to_array(logical=False)
            if engine is None:
                engine = build_engine(threads)
            try:
                raw, _elapsed = engine(image)
            except Exception as exc:
                raise PerceptionError(f"OCR inference failed: {exc}") from exc
            reply = {"ok": True, "rows": _jsonable_rows(raw)}
        except PerceptionError as exc:
            reply = {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - the parent gets every failure, not a corpse
            reply = {"ok": False, "error": f"the OCR engine process failed: {exc!r}"}
        _send_frame(channel, json.dumps(reply).encode())


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
"""``Screenshot.captured_at`` for the frame the worker rebuilds. Nothing on the read
path looks at it, and inventing "now" in the child would make a reply depend on when
it was answered."""


# --------------------------------------------------------------------------------------
# Counting what perception did
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerceptionCounts:
    """How much work perception did over some window, as COUNTS rather than seconds.

    Seconds are the number everybody wants and the number nobody can trust: the same
    frame read on a quiet machine and on one running four other workers differs by more
    than any optimization in this module could ever buy. Counts do not move, so this is
    the type a run reports alongside its model calls and its dollars.

    Attributes:
        observations: Completed :meth:`~skillweaver.contracts.Perceiver.observe` calls.
        captures: Frames taken from the controller.
        detections: Detector (YOLO) invocations.
        ocr_reads: Text reads that actually ran the OCR engine. **This is the number
            an optimization here has to move.**
        ocr_hits: Text reads answered from cache without touching the engine.
        ocr_timeouts: Reads abandoned because the engine did not come back. A SUBSET
            of ``ocr_reads`` - the work was started and charged - and a number that
            should be zero. One of these is a killed engine process, and a run that
            reports any has had eyes that stopped working rather than skills that
            stopped working.
    """

    observations: int = 0
    captures: int = 0
    detections: int = 0
    ocr_reads: int = 0
    ocr_hits: int = 0
    ocr_timeouts: int = 0

    @property
    def text_reads(self) -> int:
        """Every text read asked for, served by the engine or by the cache."""
        return self.ocr_reads + self.ocr_hits

    @property
    def hit_rate(self) -> float:
        """Fraction of text reads the cache answered, in ``0.0..1.0``.

        ``0.0`` when no text read was asked for, rather than a division by zero.
        """
        asked = self.text_reads
        return self.ocr_hits / asked if asked else 0.0

    def __add__(self, other: PerceptionCounts) -> PerceptionCounts:
        return PerceptionCounts(
            observations=self.observations + other.observations,
            captures=self.captures + other.captures,
            detections=self.detections + other.detections,
            ocr_reads=self.ocr_reads + other.ocr_reads,
            ocr_hits=self.ocr_hits + other.ocr_hits,
            ocr_timeouts=self.ocr_timeouts + other.ocr_timeouts,
        )

    def __sub__(self, other: PerceptionCounts) -> PerceptionCounts:
        """The work done SINCE ``other``, which is how a per-attempt figure is taken.

        Counters only ever climb, so every field is clamped at zero rather than
        reporting a negative amount of work if the two came from different counters.
        """
        return PerceptionCounts(
            observations=max(self.observations - other.observations, 0),
            captures=max(self.captures - other.captures, 0),
            detections=max(self.detections - other.detections, 0),
            ocr_reads=max(self.ocr_reads - other.ocr_reads, 0),
            ocr_hits=max(self.ocr_hits - other.ocr_hits, 0),
            ocr_timeouts=max(self.ocr_timeouts - other.ocr_timeouts, 0),
        )

    def __bool__(self) -> bool:
        """Whether anything at all was counted, so a report can stay silent otherwise."""
        return bool(self.observations or self.captures or self.detections or self.text_reads)

    def __str__(self) -> str:
        # The abandoned clause appears only when there is one, so the ordinary line
        # stays the line every report and test already reads.
        abandoned = f", {self.ocr_timeouts} ABANDONED" if self.ocr_timeouts else ""
        return (
            f"{self.observations} observation(s), {self.ocr_reads} OCR read(s) "
            f"+ {self.ocr_hits} cached, {self.detections} detection(s){abandoned}"
        )


@dataclass(slots=True)
class PerceptionCounters:
    """MUTABLE running tally of perception work, shared by everything that does some.

    Modelled on :class:`~skillweaver.contracts.Spend`: one instance is threaded through
    a perceiver and its reader, each of them charges its own work to it, and a caller
    takes :meth:`snapshot` before and after a stretch to get that stretch's
    :class:`PerceptionCounts`. Not thread-safe, like ``Spend``; the counts are
    diagnostics, and a lost increment under concurrency is not worth a lock on the
    hot path.
    """

    observations: int = 0
    captures: int = 0
    detections: int = 0
    ocr_reads: int = 0
    ocr_hits: int = 0
    ocr_timeouts: int = 0

    def snapshot(self) -> PerceptionCounts:
        """An immutable copy of the tally as it stands."""
        return PerceptionCounts(
            observations=self.observations,
            captures=self.captures,
            detections=self.detections,
            ocr_reads=self.ocr_reads,
            ocr_hits=self.ocr_hits,
            ocr_timeouts=self.ocr_timeouts,
        )

    def since(self, mark: PerceptionCounts) -> PerceptionCounts:
        """The work done since ``mark`` was taken from this counter."""
        return self.snapshot() - mark

    def reset(self) -> None:
        """Zero every field, for a caller measuring one stretch in isolation."""
        self.observations = self.captures = self.detections = 0
        self.ocr_reads = self.ocr_hits = self.ocr_timeouts = 0

    def __str__(self) -> str:
        return str(self.snapshot())


# --------------------------------------------------------------------------------------
# Not reading the same pixels twice
# --------------------------------------------------------------------------------------

DEFAULT_CACHE_SIZE = 32
"""Frames a :class:`CachingTextReader` remembers.

Small on purpose. The hit this cache exists to catch is the one the agent hands it
immediately - a check, a critic and the next loop iteration all looking at the screen
the last action left - and a run that comes back to a frame it last saw thirty frames
ago has almost certainly re-rendered it in the meantime. Each entry holds a frame's
text elements, not its pixels, so the bound is on entries rather than bytes.
"""


def content_key(screenshot: Screenshot) -> str:
    """The identity of the pixels a text read would be performed on.

    Two screenshots share a key exactly when reading them is guaranteed to produce the
    same elements: the same PNG bytes *and* the same declared geometry. The geometry
    belongs in the key because :attr:`~skillweaver.contracts.Screenshot.scale` is what
    divides OCR's physical coordinates down to logical ones - the same bytes declared at
    ``scale=2.0`` yield boxes at half the position of the same bytes at ``scale=1.0``,
    and serving one for the other is the doubled-coordinate bug this project warns about
    everywhere else.

    Hashing the whole PNG on every read is affordable by four orders of magnitude: 0.26ms
    median for a 334 KiB live Wikipedia frame, against the 440ms to 7.4s read it may save.
    """
    digest = hashlib.blake2b(screenshot.png, digest_size=16).hexdigest()
    return f"{digest}:{screenshot.width}x{screenshot.height}@{screenshot.scale:g}"


class CachingTextReader:
    """A :class:`~skillweaver.contracts.TextReader` that never reads the same pixels twice.

    Wraps another reader with a bounded LRU keyed on :func:`content_key`, so an
    observation of a screen nothing has changed reuses the previous read instead of
    paying for it again. Measured on live Wikipedia pages, four captures of an untouched
    page are byte-identical, so this is the common case and not a corner one.

    Why the key is exact, and not "the same state"
    ----------------------------------------------

    The obvious improvement is to serve a cached read whenever the new frame is the SAME
    STATE as the cached one, by
    :meth:`~skillweaver.contracts.Fingerprint.similarity` against
    :data:`~skillweaver.perception.fingerprint.SAME_STATE_THRESHOLD`. That judgment is
    the right one for its own question and the wrong one for this one, and the
    fingerprinter's own measured table says why: ``dense_text_scrolled_slightly`` scores
    0.750 and ``same_screen_clock_tick`` scores 1.000. Both are correctly the same state;
    in both the text has MOVED or CHANGED.

    That is not a theoretical worry. Taking the 12 frames one live Wikipedia exploration
    actually captured, grouping them by perceptual hash and normalized URL - a far
    STRICTER key than the 0.62 same-state cut, since it demands every pixel row and the
    URL agree - and reading all 12 for real: three frames would have been served another
    frame's text. The worst of them differed by twelve strings that were still on screen
    but at a different box, because the page had been scrolled. Those boxes are precisely
    what a skill clicks, and a skill cannot tell a stale read from a true one - it just
    clicks. A slow read costs seconds; a stale one costs a wrong click on a real site, so
    this cache only ever answers for pixels it has literally seen.

    Nothing is given up for that. The exact key caught every duplicate those 12 frames
    contained: 12 frames, 9 distinct, 3 reads saved, which is the whole of what was
    safely available.

    Args:
        inner: The reader that does the actual work on a miss.
        capacity: How many frames to remember. ``0`` disables caching while leaving the
            counting intact.
        counters: The tally to charge reads and hits to. A fresh one is made when not
            given; share one to count a whole perceiver's work together.
    """

    __slots__ = ("_cache", "_capacity", "_counters", "_inner", "_lock")

    def __init__(
        self,
        inner: Any,
        *,
        capacity: int = DEFAULT_CACHE_SIZE,
        counters: PerceptionCounters | None = None,
    ) -> None:
        self._inner = inner
        self._capacity = max(int(capacity), 0)
        self._counters = counters if counters is not None else PerceptionCounters()
        self._cache: OrderedDict[str, tuple[Element, ...]] = OrderedDict()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"CachingTextReader({self._inner!r}, capacity={self._capacity}, "
            f"cached={len(self._cache)})"
        )

    @property
    def inner(self) -> Any:
        """The wrapped reader."""
        return self._inner

    @property
    def counters(self) -> PerceptionCounters:
        """The tally this reader charges its work to."""
        return self._counters

    @property
    def capacity(self) -> int:
        """How many frames are remembered at most."""
        return self._capacity

    def clear(self) -> None:
        """Forget every remembered frame. The counters are left alone."""
        with self._lock:
            self._cache.clear()

    def read(self, screenshot: Screenshot) -> list[Element]:
        """The text on ``screenshot``, read once and remembered.

        The returned list is a fresh one every time, so a caller that sorts or trims it
        cannot corrupt what the next caller is served.

        Raises:
            PerceptionTimeout: if ``inner`` abandoned the read. Counted in
                ``counters.ocr_timeouts`` on the way past, because this is the one
                layer every text read in the system goes through, so it is the one
                place a run's tally of abandoned reads can be complete.
            PerceptionError: whatever else ``inner`` raises. A failed read is NOT
                cached: an engine that could not load this time may load next time,
                and caching the failure would turn a transient fault into a permanent
                blind spot.
        """
        if self._capacity == 0:
            self._counters.ocr_reads += 1
            with self._charged():
                return self._inner.read(screenshot)

        key = content_key(screenshot)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
        if hit is not None:
            self._counters.ocr_hits += 1
            return list(hit)

        self._counters.ocr_reads += 1
        with self._charged():
            elements = tuple(self._inner.read(screenshot))
        with self._lock:
            self._cache[key] = elements
            self._cache.move_to_end(key)
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)
        return list(elements)

    @contextmanager
    def _charged(self) -> Iterator[None]:
        """Count an abandoned read on its way out, and let it keep going."""
        try:
            yield
        except PerceptionTimeout:
            self._counters.ocr_timeouts += 1
            raise
