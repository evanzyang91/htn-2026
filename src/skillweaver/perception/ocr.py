"""OCR through RapidOCR (PP-OCRv4, ONNX Runtime, local: no network, no API key).

Runs on the PHYSICAL-resolution image and divides boxes by ``Screenshot.scale`` on the
way out, so everything here is in logical pixels. Failure raises rather than returning
an empty list, which an agent cannot tell apart from "this screen has no text".
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

#: Below this RapidOCR reports shapes it could not read, and a wrong label is worse
#: for an agent than no label.
DEFAULT_MIN_CONFIDENCE = 0.5

DEFAULT_READ_TIMEOUT_S = 60.0
"""Seconds one read may take. A dead-man's switch, not a target: the slowest honest read
measured is 7.4s and the hang it exists for ran 36 minutes. ``SKILLWEAVER_OCR_TIMEOUT_S``
overrides; ``0`` disables the bound."""

DEFAULT_OCR_THREADS = 4
"""ONNX Runtime intra-/inter-op threads. Explicit because left alone it sizes the pool
from ``os.cpu_count()`` and SPINS: on 14 cores, four is 20% off the idle read, 1.3x
faster than the default under six concurrent readers and 1.6x less CPU burned.
``SKILLWEAVER_OCR_THREADS`` changes it (2 for a fleet run, 2.3x faster still)."""

DEFAULT_REC_BATCH = 1
"""Text lines per ONNX Runtime recognizer call. One beats RapidOCR's default of six in
36 of 36 matched adjacent pairs, median 1.51x, and costs no accuracy (0 of 548 text
elements lost); the 1/2/4/6/12 ordering is not monotone, so batch padding does not
explain it. Re-derive on another machine with ``SKILLWEAVER_OCR_REC_BATCH``."""

log = get_logger(__name__)

_TIMEOUT_ENV = "SKILLWEAVER_OCR_TIMEOUT_S"
_THREADS_ENV = "SKILLWEAVER_OCR_THREADS"
_REC_BATCH_ENV = "SKILLWEAVER_OCR_REC_BATCH"


class PerceptionTimeout(PerceptionError):
    """A text read was abandoned: the eyes failed, not whatever asked for the read."""


def _positive_float(name: str, fallback: float) -> float:
    """``name`` as a non-negative float; a malformed value falls back rather than raising."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        value = float(raw)
    except ValueError:
        return fallback
    return value if value >= 0 else fallback


def read_timeout_s() -> float:
    """The per-read budget, read per reader rather than at import."""
    return _positive_float(_TIMEOUT_ENV, DEFAULT_READ_TIMEOUT_S)


def ocr_threads() -> int:
    """The configured ONNX Runtime pool size, at least one thread."""
    return max(1, int(_positive_float(_THREADS_ENV, DEFAULT_OCR_THREADS)))


def rec_batch() -> int:
    """The configured recognizer batch, at least one line."""
    return max(1, int(_positive_float(_REC_BATCH_ENV, DEFAULT_REC_BATCH)))


class RapidOcrReader:
    """A ``TextReader`` backed by RapidOCR: one ``text`` element per recognized line.

    Boxes are in logical pixels with a ``stable_id`` filled in. By default the engine
    runs in a CHILD PROCESS, one per reader and reused across reads, because killing it
    is the only thing that ends a read executing no Python. Built on the first read;
    safe to share between threads.

    Args:
        min_confidence: Recognitions scoring below this are dropped.
        engine: An already-built RapidOCR callable. Reads then happen IN THIS PROCESS
            and are unbounded, because there is no process of ours to kill.
        isolate: ``False`` builds the engine here rather than in a child; unbounded.
        timeout_s: Seconds one read may take. ``0`` disables the bound.
        threads: ONNX Runtime intra-/inter-op threads.
        worker: A prepared :class:`OcrWorker` to read through, for driving the bound.
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
        """Whether an engine was built in THIS process; always ``False`` when isolated."""
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
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is None:
                self._engine = build_engine(self._threads)
            return self._engine

    def read(self, screenshot: Screenshot) -> list[Element]:
        """The text on ``screenshot``, in reading order. Empty means none was found.

        Raises:
            PerceptionTimeout: the read overran ``timeout_s``; the engine process was
                killed and the next read starts a fresh one.
            PerceptionError: the screenshot cannot be decoded or inference failed.
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
        # Outside the try: an undecodable PNG is its own error, not the engine failing.
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
                # ``rows`` already killed it; drop it so the next read starts clean.
                self._worker = None
                raise
            except PerceptionError:
                if not worker.alive:
                    self._worker = None
                raise


def _parse_entry(entry: Any) -> tuple[Any, str, float] | None:
    """``(polygon, text, confidence)`` from one RapidOCR row.

    Shape-driven rather than version-specific: the container types have moved between
    RapidOCR releases.
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
    """A physical-pixel quadrilateral as a clamped LOGICAL-pixel box.

    Skip the ``scale`` division on a Retina capture and every click derived from text
    lands twice as far down the screen.
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


def build_engine(threads: int, batch: int | None = None) -> Any:
    """Build the engine with an explicit thread pool and batch size.

    The one construction site, so ``isolate=False`` and the worker child get identical
    settings and identical failure messages. RapidOCR forwards both thread counts into
    its three sessions, so passing them here sizes all of them.

    Raises:
        PerceptionError: RapidOCR is not installed or its models cannot be loaded. The
            message names the cause; "OCR silently found no text" is the most expensive
            failure mode in this pipeline.
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
"""4-byte little-endian length prefix; a prefix rather than a delimiter because one of
the messages is a PNG."""

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
    return [sys.executable, "-c", _WORKER_BOOTSTRAP]


def _worker_env(threads: int) -> dict[str, str]:
    """The child's environment: this package importable, every thread pool named.

    The three ``*_NUM_THREADS`` variables are set because OpenMP and the BLAS libraries
    under ONNX Runtime read them AT IMPORT TIME, before the child can pass anything to a
    session; the environment handed to ``Popen`` is the only moment early enough.
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
"""Engine processes this interpreter started, so :func:`_kill_live_workers` can be
certain: an abandoned child here can be burning every core on the machine."""


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL and reap. Not ``terminate``: a process wedged in a native spin never runs
    a signal handler."""
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
    (``_enter_buffered_busy``); leaking an fd of an already-killed process is cheaper.
    """
    for stream, joined in ((proc.stdin, stdin), (proc.stdout, stdout)):
        if stream is not None and joined:
            try:
                stream.close()
            except OSError:
                pass


@atexit.register
def _kill_live_workers() -> None:
    """Kill every engine process before teardown.

    ``atexit`` rather than ``__del__`` because it runs while threads and pipes still work.
    """
    for proc in list(_LIVE):
        _kill(proc)


class OcrWorker:
    """A child process holding one OCR engine, whose reads can be abandoned.

    The point of the process is the kill: a read wedged inside ONNX Runtime executes no
    Python, so no flag, signal or trace hook will ever be looked at again, and SIGKILL is
    also what gives the spinning cores back. One request in flight at a time; the child
    starts lazily and is restarted after being abandoned.

    Args:
        threads: ONNX Runtime pool size to build the engine with.
        argv: The command to run. A test may point it at a process that never answers.
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
        """Spawn the child if it is not already running."""
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
            screenshot: The frame to read; its PNG and geometry go over the pipe.
            timeout_s: Seconds to wait. ``0`` waits forever - only a test should.

        Raises:
            PerceptionTimeout: the budget was spent and the child was killed.
            PerceptionError: the child died or reported that the engine failed.
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
        # On its own thread: a pipe holds 64 KiB and a screenshot is bigger, so writing
        # one to a child that has stopped reading BLOCKS - a second unbounded wait, in
        # the caller. Only the reply is waited on, and the kill closes the pipe.
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
        """Kill the child now and forget it. Safe to call when none is running."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        # Kill FIRST: the child dying is what releases a writer blocked on a full pipe
        # and a reader blocked on an empty one.
        _kill(proc)
        if sys.is_finalizing():
            return
        sent = self._join_sender()
        pumped = self._join_pump()
        _close_streams(proc, stdin=sent, stdout=pumped)

    def close(self) -> None:
        """Ask the child to exit - closing stdin ends its loop - and kill it if it will not."""
        proc = self._proc
        if proc is None:
            return
        if sys.is_finalizing():
            # Teardown: threads are about to freeze, so joining cannot finish and closing
            # a pipe one is reading is fatal. Kill and let the OS do the rest.
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
        """Wait for the reader thread. ``False`` means the pipe must be left alone."""
        pump, self._pump = self._pump, None
        if pump is None or pump is threading.current_thread():
            return True
        pump.join(timeout=2)
        return not pump.is_alive()

    def _join_sender(self) -> bool:
        """Wait for the writer thread; same meaning as :meth:`_join_pump`."""
        sender, self._sender = self._sender, None
        if sender is None or sender is threading.current_thread():
            return True
        sender.join(timeout=2)
        return not sender.is_alive()

    def _reap(self) -> None:
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

    Failures are swallowed: a send can only fail because the child is gone, which
    reaches the caller as the reply that never arrives.
    """
    try:
        _send_frame(stream, header)
        _send_frame(stream, png)
    except (OSError, ValueError):
        log.debug("ocr.worker.send_failed")


def _pump_frames(stream: Any, replies: queue.Queue[bytes | None]) -> None:
    """Move frames off the child's stdout so a reader can wait with a deadline.

    A blocking read is precisely what cannot be given one.
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
    """The rows of one worker reply, or the child's failure raised as ours."""
    try:
        message = json.loads(reply)
    except ValueError as exc:
        raise PerceptionError(f"the OCR engine process sent a reply we cannot read: {exc}") from exc
    if not isinstance(message, dict):
        raise PerceptionError("the OCR engine process sent a reply we cannot read")
    if message.get("ok"):
        return message.get("rows") or []
    # The child raises this module's own messages, so they pass through verbatim.
    raise PerceptionError(str(message.get("error") or "the OCR engine process failed"))


def _jsonable_rows(raw: Any) -> list[list[Any]]:
    """RapidOCR's rows as JSON-safe ``[[[x, y], ...], text, score]``.

    Parsed in the child so numpy never crosses the pipe.
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

    The reply channel is a DUPLICATE of stdout and fd 1 is repointed at stderr, so
    RapidOCR's logging and any stray ``print`` cannot corrupt the frames. The engine is
    built on the first request so a load failure is reported rather than killing a child
    the parent is waiting on.
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
"""``captured_at`` for the frame the worker rebuilds; nothing reads it, and inventing
"now" in the child would make a reply depend on when it was answered."""


@dataclass(frozen=True, slots=True)
class PerceptionCounts:
    """Perception work over some window, as COUNTS rather than seconds.

    Seconds move with machine load; counts do not, so this is what a run reports.

    Attributes:
        ocr_reads: Text reads that ran the engine. An optimization here has to move this.
        ocr_hits: Text reads answered from cache.
        ocr_timeouts: Reads abandoned. A SUBSET of ``ocr_reads``, and should be zero:
            one means eyes that stopped working, not skills that did.
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
        """Fraction of text reads the cache answered; ``0.0`` when none was asked for."""
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
        """The work done SINCE ``other``. Clamped at zero per field: counters only climb,
        so a negative would mean the two came from different counters."""
        return PerceptionCounts(
            observations=max(self.observations - other.observations, 0),
            captures=max(self.captures - other.captures, 0),
            detections=max(self.detections - other.detections, 0),
            ocr_reads=max(self.ocr_reads - other.ocr_reads, 0),
            ocr_hits=max(self.ocr_hits - other.ocr_hits, 0),
            ocr_timeouts=max(self.ocr_timeouts - other.ocr_timeouts, 0),
        )

    def __bool__(self) -> bool:
        """Whether anything was counted, so a report can stay silent otherwise."""
        return bool(self.observations or self.captures or self.detections or self.text_reads)

    def __str__(self) -> str:
        abandoned = f", {self.ocr_timeouts} ABANDONED" if self.ocr_timeouts else ""
        return (
            f"{self.observations} observation(s), {self.ocr_reads} OCR read(s) "
            f"+ {self.ocr_hits} cached, {self.detections} detection(s){abandoned}"
        )


@dataclass(slots=True)
class PerceptionCounters:
    """MUTABLE running tally of perception work, shared by everything that does some.

    Like ``Spend``: threaded through a perceiver and its reader, snapshotted before and
    after a stretch. Not thread-safe - a lost increment is not worth a hot-path lock.
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
        """Zero every field."""
        self.observations = self.captures = self.detections = 0
        self.ocr_reads = self.ocr_hits = self.ocr_timeouts = 0

    def __str__(self) -> str:
        return str(self.snapshot())


DEFAULT_CACHE_SIZE = 32
"""Frames a :class:`CachingTextReader` remembers. Small because the hit it exists for is
immediate - a check, a critic and the next loop iteration on the screen the last action
left. Entries hold text elements, not pixels, so the bound is on entries."""


def content_key(screenshot: Screenshot) -> str:
    """The identity of the pixels a text read would be performed on.

    The same PNG bytes *and* the same declared geometry: ``scale`` divides OCR's physical
    coordinates down, so the same bytes at ``scale=2.0`` yield boxes at half the position
    of those at ``1.0``. Hashing the whole PNG costs 0.26ms median against a 0.44-7.4s read.
    """
    digest = hashlib.blake2b(screenshot.png, digest_size=16).hexdigest()
    return f"{digest}:{screenshot.width}x{screenshot.height}@{screenshot.scale:g}"


class CachingTextReader:
    """A ``TextReader`` that never reads the same pixels twice.

    A bounded LRU keyed on :func:`content_key`. Four captures of an untouched live page
    are byte-identical, so this is the common case.

    The key is EXACT and not "the same state": a same-state judgment is right for its own
    question and wrong here, because ``dense_text_scrolled_slightly`` scores 0.750 and is
    correctly the same state while every box has moved. Replayed over the 12 frames one
    live Wikipedia exploration captured, a stricter-than-same-state key (perceptual hash
    plus normalized URL) would still have served three frames another frame's text, the
    worst differing by twelve strings at a different box. Boxes are what a skill clicks.
    The exact key lost nothing for that: 12 frames, 9 distinct, 3 reads saved.

    Args:
        inner: The reader that does the work on a miss.
        capacity: Frames to remember. ``0`` disables caching, leaving counting intact.
        counters: The tally to charge reads and hits to.
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

        The returned list is fresh every time, so a caller that sorts it cannot corrupt
        what the next caller is served.

        Raises:
            PerceptionTimeout: ``inner`` abandoned the read. Counted here because this is
                the one layer every text read goes through.
            PerceptionError: whatever else ``inner`` raises. A failed read is NOT cached:
                that would turn a transient fault into a permanent blind spot.
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
        """Count an abandoned read on its way out."""
        try:
            yield
        except PerceptionTimeout:
            self._counters.ocr_timeouts += 1
            raise
