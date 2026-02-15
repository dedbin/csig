from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from clang import cindex

from csig_core import Function, configure_libclang_from_env, iter_functions
from csig_db import (
    get_or_create_file,
    init_db,
    iter_file_states,
    mark_file_error,
    mark_file_parsed,
    open_db,
    replace_functions_for_file,
)


ProgressCallback = Callable[[Dict[str, Any]], None]

C_EXTENSIONS = {".c"}
CPP_EXTENSIONS = {".cc", ".cpp", ".cxx", ".c++"}
HEADER_EXTENSIONS = {".h", ".hh", ".hpp", ".hxx"}


def _clang_language_arg(language: str) -> str:
    lang = str(language).strip().lower()
    if lang in {"c++", "cpp", "cxx", "cc"}:
        return "-xc++"
    return "-xc"


def _language_candidates_for_path(path: str) -> List[str]:
    suffix = Path(path).suffix.lower()
    if suffix in C_EXTENSIONS:
        return ["c"]
    if suffix in CPP_EXTENSIONS:
        return ["c++"]
    if suffix in HEADER_EXTENSIONS:
        return ["c", "c++"]
    return ["c"]


class _ProgressTracker:
    def __init__(self, progress_cb: Optional[ProgressCallback]) -> None:
        self._lock = threading.Lock()
        self._progress: Dict[str, Any] = {
            "files_total": 0,
            "files_queued": 0,
            "files_done": 0,
            "files_skipped": 0,
            "files_indexed": 0,
            "files_failed": 0,
            "functions_total": 0,
            "running": False,
            "canceled": False,
            "start_time": None,
            "end_time": None,
        }
        self._progress_cb = progress_cb

    def set(self, **fields: Any) -> None:
        with self._lock:
            self._progress.update(fields)
            snapshot = dict(self._progress)
        self._emit(snapshot)

    def inc(self, **increments: int) -> None:
        with self._lock:
            for key, delta in increments.items():
                self._progress[key] = int(self._progress.get(key, 0)) + int(delta)
            snapshot = dict(self._progress)
        self._emit(snapshot)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._progress)

    def _emit(self, snapshot: Dict[str, Any]) -> None:
        if self._progress_cb is None:
            return
        try:
            self._progress_cb(snapshot)
        except Exception:
            # UI callback failure must not stop indexing.
            return


def parse_source_file(
    path: str,
    mtime: float,
    size: int,
    index: Optional[cindex.Index],
) -> Tuple[List[Function], Optional[str]]:
    del mtime, size
    if index is None:
        index = cindex.Index.create()

    parse_errors: List[str] = []
    for language in _language_candidates_for_path(path):
        try:
            tu = index.parse(
                path=path,
                args=[_clang_language_arg(language)],
                options=cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
            )
        except Exception as exc:
            parse_errors.append(f"{language}: libclang parse failed: {exc}")
            continue

        errors = [str(diag) for diag in tu.diagnostics if diag.severity >= diag.Error]
        if errors:
            parse_errors.append(f"{language}: " + "\n".join(errors))
            continue

        functions = iter_functions(tu, only_from_file=path)
        for function in functions:
            try:
                function.signature_norm = function.normalised_signature(index, language=language)
            except Exception:
                param_types = [param_type for (param_type, _) in function.parameters]
                if function.is_variadic:
                    param_types.append("...")
                function.signature_norm = f"{function.return_type} ( {', '.join(param_types)} )"
        return functions, None

    if parse_errors:
        return [], "\n".join(parse_errors)
    return [], "Unsupported file type"


_DEFAULT_PARSE_SOURCE_FILE = parse_source_file


def _discover_files(
    *,
    root: Path,
    workers: int,
    known_states: Dict[str, Tuple[float, int]],
    task_queue: "queue.Queue[Optional[Tuple[str, float, int]]]",
    cancel_event: threading.Event,
    tracker: _ProgressTracker,
) -> None:
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            if cancel_event.is_set():
                break
            for filename in filenames:
                if cancel_event.is_set():
                    break
                suffix = Path(filename).suffix.lower()
                if suffix not in (C_EXTENSIONS | CPP_EXTENSIONS | HEADER_EXTENSIONS):
                    continue
                file_path = str(Path(dirpath, filename).resolve())
                try:
                    stat = Path(file_path).stat()
                except OSError:
                    continue

                mtime = float(stat.st_mtime)
                size = int(stat.st_size)
                tracker.inc(files_total=1)

                old_state = known_states.get(file_path)
                if old_state is not None and old_state == (mtime, size):
                    tracker.inc(files_skipped=1, files_done=1)
                    continue

                task_queue.put((file_path, mtime, size))
                tracker.inc(files_queued=1)
    finally:
        for _ in range(workers):
            task_queue.put(None)


def _worker_loop(
    *,
    task_queue: "queue.Queue[Optional[Tuple[str, float, int]]]",
    result_queue: "queue.Queue[Optional[Dict[str, Any]]]",
    cancel_event: threading.Event,
) -> None:
    index: Optional[cindex.Index] = None
    index_error: Optional[str] = None

    if parse_source_file is _DEFAULT_PARSE_SOURCE_FILE:
        configure_libclang_from_env()
        try:
            index = cindex.Index.create()
        except Exception as exc:
            index_error = f"Failed to initialize libclang index: {exc}"

    while True:
        item = task_queue.get()
        if item is None:
            task_queue.task_done()
            break

        path, mtime, size = item
        try:
            if cancel_event.is_set():
                continue

            if index_error is not None:
                result_queue.put(
                    {
                        "path": path,
                        "mtime": mtime,
                        "size": size,
                        "functions": [],
                        "error": index_error,
                    }
                )
                continue

            try:
                functions, error = parse_source_file(path, mtime, size, index)
            except Exception:
                functions = []
                error = traceback.format_exc(limit=3)

            result_queue.put(
                {
                    "path": path,
                    "mtime": mtime,
                    "size": size,
                    "functions": functions,
                    "error": error,
                }
            )
        finally:
            task_queue.task_done()

    result_queue.put(None)


def _writer_loop(
    *,
    db_path: Path,
    workers: int,
    result_queue: "queue.Queue[Optional[Dict[str, Any]]]",
    tracker: _ProgressTracker,
) -> None:
    db = open_db(db_path)
    finished_workers = 0
    try:
        while finished_workers < workers:
            item = result_queue.get()
            if item is None:
                finished_workers += 1
                continue

            path = str(item["path"])
            mtime = float(item["mtime"])
            size = int(item["size"])
            functions = list(item["functions"])
            error = item["error"]

            file_id = get_or_create_file(db, path, mtime, size)

            if error:
                mark_file_error(db, file_id=file_id, mtime=mtime, size=size, error=str(error))
                db.commit()
                tracker.inc(files_failed=1, files_done=1)
                continue

            replace_functions_for_file(db, file_id, functions)
            mark_file_parsed(db, file_id=file_id, mtime=mtime, size=size)
            db.commit()
            tracker.inc(files_indexed=1, files_done=1, functions_total=len(functions))
    finally:
        db.close()


def run_index(
    root: str,
    db_path: str,
    workers: int,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_path}")

    workers = int(workers)
    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    db_file = Path(db_path).resolve()
    init_db(db_file)

    state_db = open_db(db_file)
    try:
        known_states = iter_file_states(state_db)
    finally:
        state_db.close()

    if cancel_event is None:
        cancel_event = threading.Event()

    tracker = _ProgressTracker(progress_cb)
    tracker.set(
        running=True,
        canceled=False,
        start_time=time.time(),
        end_time=None,
    )

    task_queue: "queue.Queue[Optional[Tuple[str, float, int]]]" = queue.Queue(maxsize=max(16, workers * 8))
    result_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()

    discovery_thread = threading.Thread(
        target=_discover_files,
        kwargs={
            "root": root_path,
            "workers": workers,
            "known_states": known_states,
            "task_queue": task_queue,
            "cancel_event": cancel_event,
            "tracker": tracker,
        },
        name="csig-discovery",
        daemon=True,
    )

    worker_threads = [
        threading.Thread(
            target=_worker_loop,
            kwargs={
                "task_queue": task_queue,
                "result_queue": result_queue,
                "cancel_event": cancel_event,
            },
            name=f"csig-worker-{idx + 1}",
            daemon=True,
        )
        for idx in range(workers)
    ]

    writer_thread = threading.Thread(
        target=_writer_loop,
        kwargs={
            "db_path": db_file,
            "workers": workers,
            "result_queue": result_queue,
            "tracker": tracker,
        },
        name="csig-writer",
        daemon=True,
    )

    writer_thread.start()
    for thread in worker_threads:
        thread.start()
    discovery_thread.start()

    discovery_thread.join()
    for thread in worker_threads:
        thread.join()
    writer_thread.join()

    tracker.set(
        running=False,
        canceled=bool(cancel_event.is_set()),
        end_time=time.time(),
    )

    summary = tracker.snapshot()
    summary["root"] = str(root_path)
    summary["db_path"] = str(db_file)
    summary["workers"] = workers
    if summary["start_time"] is not None and summary["end_time"] is not None:
        summary["duration_seconds"] = float(summary["end_time"]) - float(summary["start_time"])
    else:
        summary["duration_seconds"] = 0.0
    return summary
