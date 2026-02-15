#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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
from csig_indexer import run_index

# Compatibility export for tests that patch subprocess used in core helpers.
subprocess = _core.subprocess


def default_db_path(root: str) -> str:
    return str(Path(root).resolve() / "csig.sqlite3")


def rank_candidates(candidates: List[Dict[str, Any]], query: Query, top: int) -> List[Dict[str, Any]]:
    scored: List[tuple[int, Dict[str, Any]]] = []
    for row in candidates:
        score = 0
        if query.name:
            score += levenshtein_distance(str(row["name"]), query.name)
        if query.normalised_signature:
            score += levenshtein_distance(str(row["signature_norm"]), query.normalised_signature)
        scored.append((score, row))

    scored.sort(
        key=lambda item: (
            item[0],
            str(item[1]["name"]).lower(),
            str(item[1]["path"]).lower(),
            int(item[1]["line"]),
            int(item[1]["column"]),
        )
    )
    return [row for _, row in scored[: max(0, int(top))]]


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


def _cmd_index(args: argparse.Namespace) -> int:
    db_path = args.db if args.db else default_db_path(args.root)
    summary = run_index(args.root, db_path, workers=args.workers)

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
    init_db(db_path)

    # Keep DB fresh before querying.
    run_index(args.root, db_path, workers=args.workers)

    configure_libclang_from_env()
    try:
        clang_index = cindex.Index.create()
    except Exception as exc:
        print(f"Failed to initialize libclang: {exc}", file=sys.stderr)
        return 1

    try:
        query = parse_query(args.query, clang_index)
    except Exception as exc:
        print(f"Query parsing failed: {exc}", file=sys.stderr)
        return 1

    db = open_db(db_path)
    try:
        candidates = fetch_candidates(db, query, limit=max(200, args.top * 20))
    finally:
        db.close()

    ranked = rank_candidates(candidates, query, args.top)
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
    index_parser.set_defaults(handler=_cmd_index)

    search_parser = subparsers.add_parser("search", help="Search indexed function signatures")
    search_parser.add_argument("root", help="Project root directory")
    search_parser.add_argument("query", help='Query, e.g. "int (int, int)" or "foo :: int (int, int)"')
    search_parser.add_argument("--db", default=None, help="Path to sqlite database file")
    search_parser.add_argument("--top", type=int, default=20, help="How many results to print")
    search_parser.add_argument("--workers", type=int, default=default_workers, help="Workers used for refresh indexing")
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
