from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple

from csig_core import Function, Query


def open_db(db_path: str | Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    return db


def init_db(db_path: str | Path) -> None:
    db = open_db(db_path)
    try:
        db.execute("PRAGMA journal_mode=WAL;")
        db.execute("PRAGMA synchronous=NORMAL;")
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                parsed_at REAL,
                last_error TEXT
            );
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                return_type TEXT NOT NULL,
                params_json TEXT NOT NULL,
                signature_norm TEXT NOT NULL,
                line INTEGER NOT NULL,
                column INTEGER NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
            );
            """
        )
        db.execute("CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name);")
        db.execute("CREATE INDEX IF NOT EXISTS idx_functions_sig ON functions(signature_norm);")
        db.execute("CREATE INDEX IF NOT EXISTS idx_functions_file_id ON functions(file_id);")
        db.commit()
    finally:
        db.close()


def get_or_create_file(db: sqlite3.Connection, path: str, mtime: float, size: int) -> int:
    row = db.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
    if row is not None:
        db.execute("UPDATE files SET mtime = ?, size = ? WHERE id = ?", (mtime, size, row["id"]))
        return int(row["id"])

    cursor = db.execute(
        "INSERT INTO files(path, mtime, size, parsed_at, last_error) VALUES (?, ?, ?, NULL, NULL)",
        (path, mtime, size),
    )
    return int(cursor.lastrowid)


def replace_functions_for_file(db: sqlite3.Connection, file_id: int, functions: List[Function]) -> None:
    db.execute("DELETE FROM functions WHERE file_id = ?", (file_id,))
    if not functions:
        return

    payload = []
    for function in functions:
        params_json = json.dumps(function.parameters, ensure_ascii=True)
        signature_norm = function.signature_norm
        if not signature_norm:
            param_types = [str(param_type) for (param_type, _name) in function.parameters]
            if function.is_variadic:
                param_types.append("...")
            signature_norm = f"{function.return_type} ( {', '.join(param_types)} )"
        payload.append(
            (
                file_id,
                function.name,
                function.return_type,
                params_json,
                signature_norm,
                function.location.line,
                function.location.column,
            )
        )

    db.executemany(
        """
        INSERT INTO functions(
            file_id, name, return_type, params_json, signature_norm, line, column
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )


def mark_file_parsed(
    db: sqlite3.Connection,
    *,
    file_id: int,
    mtime: float,
    size: int,
) -> None:
    db.execute(
        "UPDATE files SET mtime = ?, size = ?, parsed_at = ?, last_error = NULL WHERE id = ?",
        (mtime, size, time.time(), file_id),
    )


def mark_file_error(
    db: sqlite3.Connection,
    *,
    file_id: int,
    mtime: float,
    size: int,
    error: str,
) -> None:
    db.execute(
        "UPDATE files SET mtime = ?, size = ?, parsed_at = ?, last_error = ? WHERE id = ?",
        (mtime, size, time.time(), error, file_id),
    )


def iter_file_states(db: sqlite3.Connection) -> Dict[str, Tuple[float, int]]:
    rows = db.execute("SELECT path, mtime, size FROM files").fetchall()
    return {str(row["path"]): (float(row["mtime"]), int(row["size"])) for row in rows}


def fetch_candidates(db: sqlite3.Connection, query: Query, limit: int = 500) -> List[dict]:
    where = ""
    args: List[object] = []

    if query.name and query.normalised_signature:
        where = "WHERE f.name LIKE ? OR f.signature_norm LIKE ?"
        args.extend([f"%{query.name}%", f"%{query.normalised_signature}%"])
    elif query.name:
        where = "WHERE f.name LIKE ?"
        args.append(f"%{query.name}%")
    elif query.normalised_signature:
        where = "WHERE f.signature_norm LIKE ?"
        args.append(f"%{query.normalised_signature}%")

    rows = db.execute(
        f"""
        SELECT
            f.id,
            fi.path AS path,
            f.name,
            f.return_type,
            f.params_json,
            f.signature_norm,
            f.line,
            f.column
        FROM functions AS f
        JOIN files AS fi ON fi.id = f.file_id
        {where}
        ORDER BY f.name COLLATE NOCASE, fi.path COLLATE NOCASE, f.line, f.column
        LIMIT ?
        """,
        (*args, limit),
    ).fetchall()

    if not rows and where:
        rows = db.execute(
            """
            SELECT
                f.id,
                fi.path AS path,
                f.name,
                f.return_type,
                f.params_json,
                f.signature_norm,
                f.line,
                f.column
            FROM functions AS f
            JOIN files AS fi ON fi.id = f.file_id
            ORDER BY f.name COLLATE NOCASE, fi.path COLLATE NOCASE, f.line, f.column
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result: List[dict] = []
    for row in rows:
        params_raw = row["params_json"]
        try:
            params = json.loads(params_raw) if params_raw else []
        except Exception:
            params = []
        result.append(
            {
                "id": int(row["id"]),
                "path": str(row["path"]),
                "name": str(row["name"]),
                "return_type": str(row["return_type"]),
                "params": params,
                "signature_norm": str(row["signature_norm"]),
                "line": int(row["line"]),
                "column": int(row["column"]),
            }
        )
    return result


def get_error_file_count(db: sqlite3.Connection) -> int:
    row = db.execute("SELECT COUNT(*) AS cnt FROM files WHERE last_error IS NOT NULL").fetchone()
    return int(row["cnt"]) if row is not None else 0
