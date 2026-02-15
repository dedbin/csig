from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List

from clang import cindex
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Input, Static

from csig_core import configure_libclang_from_env, levenshtein_distance, parse_query
from csig_db import fetch_candidates, init_db, open_db
from csig_indexer import run_index


def _rank_candidates(candidates: List[Dict[str, Any]], query: Any, top: int) -> List[Dict[str, Any]]:
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


class CsigApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    #status {
        height: 1;
        content-align: left middle;
    }
    #controls {
        height: auto;
        layout: horizontal;
    }
    #query {
        width: 1fr;
    }
    #results {
        height: 1fr;
    }
    #progress {
        height: 1;
        content-align: left middle;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, root: str, db_path: str, workers: int) -> None:
        super().__init__()
        self.root = str(Path(root).resolve())
        self.db_path = str(Path(db_path).resolve())
        self.index_workers = int(workers)
        self.cancel_event = threading.Event()
        self._index_thread: threading.Thread | None = None
        self._search_task: asyncio.Task[None] | None = None
        self._progress_lock = threading.Lock()
        self._latest_progress: Dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(f"Root: {self.root}", id="status")
        with Vertical():
            with Horizontal(id="controls"):
                yield Input(placeholder='Search, e.g. "foo :: int (int, int)"', id="query")
                yield Button("Index", id="index", variant="primary")
                yield Button("Cancel", id="cancel", variant="warning")
            yield DataTable(id="results")
            yield Static("Progress: idle", id="progress")
        yield Footer()

    def on_mount(self) -> None:
        init_db(self.db_path)
        table = self.query_one("#results", DataTable)
        table.add_columns("Location", "Name", "Signature")
        table.cursor_type = "row"
        self.set_interval(0.3, self._refresh_progress)

    def _on_index_progress(self, snapshot: Dict[str, Any]) -> None:
        with self._progress_lock:
            self._latest_progress = dict(snapshot)

    def _refresh_progress(self) -> None:
        with self._progress_lock:
            snapshot = dict(self._latest_progress)

        if not snapshot:
            return

        status = self.query_one("#status", Static)
        progress = self.query_one("#progress", Static)

        running = bool(snapshot.get("running", False))
        state = "running" if running else "idle"
        status.update(f"Root: {self.root} | Index: {state}")
        progress.update(
            "Progress: "
            f"{snapshot.get('files_done', 0)}/{snapshot.get('files_total', 0)} files done, "
            f"indexed={snapshot.get('files_indexed', 0)}, "
            f"skipped={snapshot.get('files_skipped', 0)}, "
            f"errors={snapshot.get('files_failed', 0)}, "
            f"functions={snapshot.get('functions_total', 0)}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "index":
            self._start_index()
        elif event.button.id == "cancel":
            self.cancel_event.set()

    def _start_index(self) -> None:
        if self._index_thread is not None and self._index_thread.is_alive():
            return

        self.cancel_event.clear()

        def target() -> None:
            summary = run_index(
                self.root,
                self.db_path,
                workers=self.index_workers,
                progress_cb=self._on_index_progress,
                cancel_event=self.cancel_event,
            )
            self.call_from_thread(self._on_index_finished, summary)

        self._index_thread = threading.Thread(target=target, name="csig-tui-index", daemon=True)
        self._index_thread.start()

    def _on_index_finished(self, summary: Dict[str, Any]) -> None:
        status = self.query_one("#status", Static)
        status.update(
            f"Root: {self.root} | Indexed={summary.get('files_indexed', 0)} "
            f"Skipped={summary.get('files_skipped', 0)} Failed={summary.get('files_failed', 0)}"
        )

    async def on_input_changed(self, event: Input.Changed) -> None:
        if self._search_task and not self._search_task.done():
            self._search_task.cancel()
        self._search_task = asyncio.create_task(self._debounced_search(event.value))

    async def _debounced_search(self, query_text: str) -> None:
        try:
            await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            return
        rows = await asyncio.to_thread(self._search_sync, query_text)
        self._render_results(rows)

    def _search_sync(self, query_text: str) -> List[Dict[str, Any]]:
        value = query_text.strip()
        if not value:
            return []

        try:
            configure_libclang_from_env()
            clang_index = cindex.Index.create()
            query = parse_query(value, clang_index)
            db = open_db(self.db_path)
            try:
                candidates = fetch_candidates(db, query, limit=300)
            finally:
                db.close()
            return _rank_candidates(candidates, query, top=50)
        except Exception as exc:
            return [{"error": str(exc)}]

    def _render_results(self, rows: List[Dict[str, Any]]) -> None:
        table = self.query_one("#results", DataTable)
        table.clear()
        if not rows:
            return

        if "error" in rows[0]:
            status = self.query_one("#status", Static)
            status.update(f"Root: {self.root} | Search error: {rows[0]['error']}")
            return

        for row in rows:
            location = f"{row['path']}:{row['line']}:{row['column']}"
            table.add_row(location, str(row["name"]), str(row["signature_norm"]))


def run(root: str, db_path: str, workers: int) -> None:
    app = CsigApp(root=root, db_path=db_path, workers=workers)
    app.run()
