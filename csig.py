#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List

from clang import cindex

import csig_core as _core
from csig_core import (
    Function,
    Location,
    Query,
    clang_c_include_path_args,
    configure_libclang_from_env,
    iter_functions,
    levenshtein_distance,
    normalise_signature,
    parse_query,
    score_function,
)
from csig_db import fetch_candidates, init_db, open_db
from csig_indexer import SKIPPED_DIRS, run_index

# Compatibility export for tests that patch subprocess used in core helpers.
subprocess = _core.subprocess


def default_db_path(root: str) -> str:
    return str(Path(root).resolve() / "csig.sqlite3")


def _is_path_under_skipped_dir(root: str, path: str) -> bool:
    try:
        relative = Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return any(part in SKIPPED_DIRS for part in relative.parts[:-1])


def _score_candidate(row: Dict[str, Any], query: Query) -> int:
    score = 0
    if query.name:
        score += levenshtein_distance(str(row["name"]), query.name)
    if query.normalised_signature:
        score += levenshtein_distance(str(row["signature_norm"]), query.normalised_signature)
    return score


def _score_candidate_worker(payload: tuple[Dict[str, Any], Query]) -> tuple[int, Dict[str, Any]]:
    row, query = payload
    return _score_candidate(row, query), row


def _sort_scored_candidates(scored: List[tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    scored.sort(
        key=lambda item: (
            item[0],
            str(item[1]["name"]).lower(),
            str(item[1]["path"]).lower(),
            int(item[1]["line"]),
            int(item[1]["column"]),
        )
    )
    return [row for _, row in scored]


def rank_candidates(
    candidates: List[Dict[str, Any]],
    query: Query,
    top: int,
    *,
    processes: int = 1,
) -> List[Dict[str, Any]]:
    limit = max(0, int(top))
    if limit == 0:
        return []

    process_count = max(1, int(processes))
    if process_count > 1 and len(candidates) > 1:
        payloads = [(row, query) for row in candidates]
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            scored = list(executor.map(_score_candidate_worker, payloads))
    else:
        scored = [(_score_candidate(row, query), row) for row in candidates]

    return _sort_scored_candidates(scored)[:limit]


class SignatureSearchEngine:
    def __init__(self, root: str, db_path: str | None = None, workers: int | None = None) -> None:
        self.root = str(Path(root).resolve())
        self.db_path = str(Path(db_path).resolve()) if db_path else default_db_path(self.root)
        self.workers = int(workers) if workers is not None else max(1, os.cpu_count() or 1)
        if self.workers <= 0:
            self.workers = max(1, os.cpu_count() or 1)

    def index(self, progress_cb: Callable[[Dict[str, Any]], None] | None = None) -> Dict[str, Any]:
        return run_index(self.root, self.db_path, workers=self.workers, progress_cb=progress_cb)

    def search(
        self,
        query_text: str,
        *,
        top: int = 20,
        rank_processes: int = 1,
        refresh_index: bool = True,
    ) -> List[Dict[str, Any]]:
        init_db(self.db_path)
        if refresh_index:
            self.index()

        configure_libclang_from_env()
        try:
            clang_index = cindex.Index.create()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize libclang: {exc}") from exc

        try:
            query = parse_query(query_text, clang_index)
        except Exception as exc:
            raise RuntimeError(f"Query parsing failed: {exc}") from exc

        db = open_db(self.db_path)
        try:
            candidates = fetch_candidates(db, query, limit=max(200, int(top) * 20))
        finally:
            db.close()

        candidates = [
            row
            for row in candidates
            if not _is_path_under_skipped_dir(self.root, str(row["path"]))
        ]
        return rank_candidates(candidates, query, top, processes=rank_processes)


def _format_params(params: List[List[str]]) -> str:
    chunks: List[str] = []
    for item in params:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        param_type = str(item[0])
        param_name = None
        if len(item) > 1:
            param_name = item[1]
        if param_name:
            chunks.append(f"{param_type} {param_name}")
        else:
            chunks.append(param_type)
    return ", ".join(chunks)


def _render_index_progress(snapshot: Dict[str, Any], *, verbose: bool = False) -> str:
    total = int(snapshot.get("files_total", 0) or 0)
    done = int(snapshot.get("files_done", 0) or 0)
    indexed = int(snapshot.get("files_indexed", 0) or 0)
    skipped = int(snapshot.get("files_skipped", 0) or 0)
    failed = int(snapshot.get("files_failed", 0) or 0)

    width = 30
    if total > 0:
        ratio = min(1.0, max(0.0, done / total))
    else:
        ratio = 0.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)

    base = f"[{bar}] {done}/{total} files (indexed={indexed}, skipped={skipped}, failed={failed})"
    if not verbose:
        return base

    last_file = snapshot.get("last_file")
    last_status = snapshot.get("last_status")
    if last_file and last_status:
        return f"{base} | {last_status}: {last_file}"
    return base


def _cmd_index(args: argparse.Namespace) -> int:
    db_path = args.db if args.db else default_db_path(args.root)
    engine = SignatureSearchEngine(args.root, db_path=db_path, workers=args.workers)

    def progress_cb(snapshot: Dict[str, Any]) -> None:
        if args.verbose:
            message = _render_index_progress(snapshot, verbose=True)
            print(message)
            return

        message = _render_index_progress(snapshot)
        print(f"\r{message}", end="", flush=True)

    summary = engine.index(progress_cb=progress_cb)

    if not args.verbose:
        print()
    print(f"Indexed root: {summary['root']}")
    print(f"DB path: {summary['db_path']}")
    print(f"Workers: {summary['workers']}")
    print(f"Files total: {summary['files_total']}")
    print(f"Files indexed: {summary['files_indexed']}")
    print(f"Files skipped: {summary['files_skipped']}")
    print(f"Files failed: {summary['files_failed']}")
    print(f"Functions indexed: {summary['functions_total']}")
    print(f"Duration: {summary['duration_seconds']:.3f}s")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    db_path = args.db if args.db else default_db_path(args.root)
    engine = SignatureSearchEngine(args.root, db_path=db_path, workers=args.workers)
    try:
        ranked = engine.search(args.query, top=args.top, rank_processes=args.rank_processes)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for row in ranked:
        params_text = _format_params(row["params"])
        print(
            f"{row['path']}:{row['line']}:{row['column']}: "
            f"{row['name']} :: {row['return_type']}({params_text})"
        )
    return 0


def _cmd_tui(args: argparse.Namespace) -> int:
    db_path = args.db if args.db else default_db_path(args.root)
    try:
        from csig_tui import run as run_tui
    except Exception as exc:
        print(
            f"TUI dependencies are not available ({exc}). Install requirements first.",
            file=sys.stderr,
        )
        return 1

    run_tui(args.root, db_path, args.workers)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csig", add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    default_workers = max(1, os.cpu_count() or 1)

    index_parser = subparsers.add_parser("index", help="Index C/C++ sources and headers in a project directory")
    index_parser.add_argument("root", help="Project root directory")
    index_parser.add_argument("--db", default=None, help="Path to sqlite database file")
    index_parser.add_argument("--workers", type=int, default=default_workers, help="Number of parser workers")
    index_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed indexing events for each processed file",
    )
    index_parser.set_defaults(handler=_cmd_index)

    search_parser = subparsers.add_parser("search", help="Search indexed function signatures")
    search_parser.add_argument("root", help="Project root directory")
    search_parser.add_argument("query", help='Query, e.g. "int (int, int)" or "foo :: int (int, int)"')
    search_parser.add_argument("--db", default=None, help="Path to sqlite database file")
    search_parser.add_argument("--top", type=int, default=20, help="How many results to print")
    search_parser.add_argument("--workers", type=int, default=default_workers, help="Workers used for refresh indexing")
    search_parser.add_argument(
        "--rank-processes",
        type=int,
        default=1,
        help="Processes used for candidate scoring; 1 disables multiprocessing",
    )
    search_parser.set_defaults(handler=_cmd_search)

    tui_parser = subparsers.add_parser("tui", help="Run interactive Textual UI")
    tui_parser.add_argument("root", help="Project root directory")
    tui_parser.add_argument("--db", default=None, help="Path to sqlite database file")
    tui_parser.add_argument("--workers", type=int, default=default_workers, help="Number of parser workers")
    tui_parser.set_defaults(handler=_cmd_tui)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
