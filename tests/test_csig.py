import threading
import time
import types
from pathlib import Path

import csig
import csig_core
import csig_db
import csig_indexer


def _function_for_test(path: str, name: str, line: int = 1) -> csig.Function:
    return csig.Function(
        name=name,
        location=csig.Location(file_name=path, line=line, column=1),
        return_type="int",
        parameters=[("int", "x")],
        signature_norm="int ( int )",
    )


def test_levenshtein_distance_case_insensitive_ascii():
    assert csig.levenshtein_distance("Foo", "foo") == 0
    assert csig.levenshtein_distance("", "abc") == 3
    assert csig.levenshtein_distance("kitten", "sitting") == 3


def test_parse_query_builds_fake_signature_and_strips_q_name(monkeypatch):
    captured = []

    def fake_normalise(index, query_string):
        captured.append(query_string)
        return "int __q__ ( int , int ) ;"

    monkeypatch.setattr(csig_core, "normalise_signature", fake_normalise)
    query = csig.parse_query("int (int, int)", index=object())

    assert captured == ["int __q__ (int, int);"]
    assert query.name is None
    assert query.normalised_signature == "int ( int , int )"


def test_parse_query_with_name_and_empty_signature(monkeypatch):
    called = False

    def fake_normalise(index, query_string):
        nonlocal called
        called = True
        return "ignored"

    monkeypatch.setattr(csig_core, "normalise_signature", fake_normalise)
    query = csig.parse_query("foo ::   ", index=object())

    assert query.name == "foo"
    assert query.normalised_signature is None
    assert called is False


def test_function_normalised_signature_uses_param_types(monkeypatch):
    def fake_normalise(index, query_string):
        assert query_string == "int __f__(int, const char *);"
        return "int __f__ ( int , const char * ) ;"

    monkeypatch.setattr(csig_core, "normalise_signature", fake_normalise)
    fn = csig.Function(
        name="f",
        location=csig.Location(file_name="x.c", line=1, column=1),
        return_type="int",
        parameters=[("int", "a"), ("const char *", "b")],
    )
    assert fn.normalised_signature(index=object()) == "int ( int , const char * )"


def test_score_function_combines_name_and_signature(monkeypatch):
    def fake_normalised_signature(self, index):
        return "int ( int )"

    monkeypatch.setattr(csig.Function, "normalised_signature", fake_normalised_signature)
    fn = csig.Function(
        name="add",
        location=csig.Location(file_name="x.c", line=1, column=1),
        return_type="int",
        parameters=[("int", "a")],
    )
    query = csig.Query(name="add", normalised_signature="int ( int )")
    assert csig.score_function(fn, query, index=object()) == 0


def test_clang_c_include_path_args_parsing(monkeypatch):
    stderr = "\n".join(
        [
            "random",
            '#include "..." search starts here:',
            " /usr/include",
            "#include <...> search starts here:",
            " /usr/local/include",
            "End of search list.",
        ]
    )
    fake_proc = types.SimpleNamespace(returncode=0, stderr=stderr)

    def fake_run(*_args, **_kwargs):
        return fake_proc

    monkeypatch.setattr(csig_core.subprocess, "run", fake_run)
    args = csig.clang_c_include_path_args()
    assert args == ["-isystem", "/usr/include", "-isystem", "/usr/local/include"]


def test_init_db_creates_schema(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    csig_db.init_db(db_path)
    db = csig_db.open_db(db_path)
    try:
        rows = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {str(row["name"]) for row in rows}
    finally:
        db.close()
    assert "files" in names
    assert "functions" in names


def test_replace_functions_for_file_overwrites_old_rows(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    csig_db.init_db(db_path)
    db = csig_db.open_db(db_path)
    try:
        file_id = csig_db.get_or_create_file(db, "a.c", mtime=1.0, size=10)
        csig_db.replace_functions_for_file(
            db,
            file_id,
            [_function_for_test("a.c", "foo"), _function_for_test("a.c", "bar", line=2)],
        )
        db.commit()
        count1 = db.execute("SELECT COUNT(*) AS cnt FROM functions WHERE file_id = ?", (file_id,)).fetchone()["cnt"]

        csig_db.replace_functions_for_file(db, file_id, [_function_for_test("a.c", "baz")])
        db.commit()
        rows = db.execute(
            "SELECT name FROM functions WHERE file_id = ? ORDER BY id ASC",
            (file_id,),
        ).fetchall()
    finally:
        db.close()

    assert int(count1) == 2
    assert [str(row["name"]) for row in rows] == ["baz"]


def test_rank_candidates_stable_top_n(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    csig_db.init_db(db_path)
    db = csig_db.open_db(db_path)
    try:
        file_id = csig_db.get_or_create_file(db, "a.c", mtime=1.0, size=10)
        f1 = _function_for_test("a.c", "add", line=10)
        f1.signature_norm = "int ( int , int )"
        f2 = _function_for_test("a.c", "subtract", line=20)
        f2.signature_norm = "int ( int , int )"
        csig_db.replace_functions_for_file(db, file_id, [f2, f1])
        db.commit()
        query = csig.Query(name="add", normalised_signature="int ( int , int )")
        candidates = csig_db.fetch_candidates(db, query, limit=50)
    finally:
        db.close()

    top1 = csig.rank_candidates(candidates, query, top=1)
    top1_again = csig.rank_candidates(candidates, query, top=1)

    assert len(top1) == 1
    assert top1[0]["name"] == "add"
    assert top1[0]["name"] == top1_again[0]["name"]


def test_rank_candidates_processes_match_sequential():
    query = csig.Query(name="add", normalised_signature="int ( int , int )")
    candidates = [
        {
            "id": 1,
            "path": "b.c",
            "name": "subtract",
            "return_type": "int",
            "params": [["int", "a"], ["int", "b"]],
            "signature_norm": "int ( int , int )",
            "line": 20,
            "column": 1,
        },
        {
            "id": 2,
            "path": "a.c",
            "name": "add",
            "return_type": "int",
            "params": [["int", "a"], ["int", "b"]],
            "signature_norm": "int ( int , int )",
            "line": 10,
            "column": 1,
        },
    ]

    sequential = csig.rank_candidates(candidates, query, top=2, processes=1)
    parallel = csig.rank_candidates(candidates, query, top=2, processes=2)

    assert [row["name"] for row in parallel] == [row["name"] for row in sequential]


def test_signature_search_engine_index_uses_configured_paths(monkeypatch, tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    db_path = tmp_path / "idx.sqlite3"
    calls = {}

    def fake_run_index(root_arg, db_arg, workers, progress_cb=None):
        calls["root"] = root_arg
        calls["db_path"] = db_arg
        calls["workers"] = workers
        calls["has_progress"] = progress_cb is not None
        return {
            "root": root_arg,
            "db_path": db_arg,
            "workers": workers,
            "files_total": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "functions_total": 0,
            "duration_seconds": 0.0,
        }

    monkeypatch.setattr(csig, "run_index", fake_run_index)
    engine = csig.SignatureSearchEngine(str(root), db_path=str(db_path), workers=3)
    summary = engine.index(progress_cb=lambda _snapshot: None)

    assert calls["root"] == str(root.resolve())
    assert calls["db_path"] == str(db_path.resolve())
    assert calls["workers"] == 3
    assert calls["has_progress"] is True
    assert summary["workers"] == 3


def test_search_filters_stale_candidates_from_skipped_dirs(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    active_path = root / "test.c"
    skipped_path = root / "projs" / "raylib" / "src" / "raylib.c"
    explicit_library_root = root / "projs" / "raylib"

    assert csig._is_path_under_skipped_dir(str(root), str(active_path)) is False
    assert csig._is_path_under_skipped_dir(str(root), str(skipped_path)) is True
    assert csig._is_path_under_skipped_dir(str(explicit_library_root), str(skipped_path)) is False


def test_indexer_skips_unchanged_files_by_mtime_and_size(tmp_path, monkeypatch):
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.c").write_text("int a(void){return 1;}\n", encoding="utf-8")
    (root / "b.h").write_text("int b(void);\n", encoding="utf-8")
    db_path = tmp_path / "idx.sqlite3"

    def fake_parse(path, mtime, size, index):
        del mtime, size, index
        name = Path(path).stem
        return [_function_for_test(path, name)], None

    monkeypatch.setattr(csig_indexer, "parse_source_file", fake_parse)

    first = csig_indexer.run_index(str(root), str(db_path), workers=2)
    second = csig_indexer.run_index(str(root), str(db_path), workers=2)

    assert first["files_total"] == 2
    assert first["files_indexed"] == 2
    assert second["files_total"] == 2
    assert second["files_skipped"] == 2
    assert second["files_indexed"] == 0


def test_indexer_skips_local_project_corpus_by_default(tmp_path, monkeypatch):
    root = tmp_path / "src"
    root.mkdir()
    (root / "test.c").write_text("int add(int a, int b){return a + b;}\n", encoding="utf-8")
    local_corpus = root / "projs"
    local_corpus.mkdir()
    (local_corpus / "external.c").write_text("int external(void){return 0;}\n", encoding="utf-8")
    db_path = tmp_path / "idx.sqlite3"

    def fake_parse(path, mtime, size, index):
        del mtime, size, index
        return [_function_for_test(path, Path(path).stem)], None

    monkeypatch.setattr(csig_indexer, "parse_source_file", fake_parse)
    summary = csig_indexer.run_index(str(root), str(db_path), workers=1)

    assert summary["files_total"] == 1
    assert summary["files_indexed"] == 1


def test_indexer_cancel_stops_processing_and_returns(tmp_path, monkeypatch):
    root = tmp_path / "src"
    root.mkdir()
    for idx in range(20):
        (root / f"f{idx}.c").write_text("int f(void){return 0;}\n", encoding="utf-8")
    db_path = tmp_path / "idx.sqlite3"
    cancel_event = threading.Event()

    def fake_parse(path, mtime, size, index):
        del mtime, size, index
        time.sleep(0.05)
        return [_function_for_test(path, Path(path).stem)], None

    def progress_cb(snapshot):
        if int(snapshot.get("files_done", 0)) >= 1:
            cancel_event.set()

    monkeypatch.setattr(csig_indexer, "parse_source_file", fake_parse)
    summary = csig_indexer.run_index(
        str(root),
        str(db_path),
        workers=2,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )

    assert summary["canceled"] is True
    assert int(summary["files_done"]) < int(summary["files_total"])


def test_cmd_index_verbose_prints_events(monkeypatch, capsys, tmp_path):
    root = tmp_path / "src"
    root.mkdir()

    def fake_run_index(root, db_path, workers, progress_cb=None, cancel_event=None):
        del root, db_path, workers, cancel_event
        assert progress_cb is not None
        progress_cb(
            {
                "files_total": 1,
                "files_done": 1,
                "files_indexed": 1,
                "files_skipped": 0,
                "files_failed": 0,
                "last_file": "x.c",
                "last_status": "indexed",
            }
        )
        return {
            "root": str(tmp_path),
            "db_path": str(tmp_path / "idx.sqlite3"),
            "workers": 1,
            "files_total": 1,
            "files_indexed": 1,
            "files_skipped": 0,
            "files_failed": 0,
            "functions_total": 1,
            "duration_seconds": 0.1,
        }

    monkeypatch.setattr(csig, "run_index", fake_run_index)
    args = csig.build_parser().parse_args(["index", str(root), "--verbose", "--workers", "1"])

    rc = args.handler(args)

    out = capsys.readouterr().out
    assert rc == 0
    assert "indexed: x.c" in out
    assert "Files indexed: 1" in out
