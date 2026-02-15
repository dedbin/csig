#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from clang import cindex


def _clang_language_arg(language: str) -> str:
    lang = str(language).strip().lower()
    if lang in {"c++", "cpp", "cxx", "cc"}:
        return "-xc++"
    return "-xc"


@dataclass(frozen=True)
class Location:
    file_name: str
    line: int
    column: int


@dataclass
class Function:
    name: str
    location: Location
    return_type: str
    parameters: List[Tuple[str, Optional[str]]]
    is_variadic: bool = False
    signature_norm: Optional[str] = None

    def normalised_signature(self, index: cindex.Index, language: str = "c") -> str:
        param_types = [param_type for (param_type, _) in self.parameters]
        if self.is_variadic:
            param_types.append("...")
        proto = f"{self.return_type} __f__({', '.join(param_types)});"
        if str(language).strip().lower() in {"c++", "cpp", "cxx", "cc"}:
            sig = normalise_signature_with_language(index, proto, language=language)
        else:
            sig = normalise_signature(index, proto)
        sig = re.sub(r"\b__f__\b", "", sig)
        sig = sig.replace(" ;", "").strip()
        sig = re.sub(r"\s+", " ", sig).strip()
        return sig


@dataclass(frozen=True)
class Query:
    name: Optional[str]
    normalised_signature: Optional[str]


def configure_libclang_from_env() -> None:
    libclang_path = os.environ.get("LIBCLANG_PATH")
    if not libclang_path:
        return
    try:
        cindex.Config.set_library_path(libclang_path)
    except Exception:
        # Non-fatal, parsing will produce a clear runtime error later.
        return


def clang_c_include_path_args() -> List[str]:
    """
    Read default include search paths from `cc -xc -E -v -`.
    """
    try:
        proc = subprocess.run(
            ["cc", "-xc", "-E", "-v", "-"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError("Failed to execute default C compiler (cc). Is it in PATH?") from exc

    if proc.returncode != 0:
        raise RuntimeError(f"Default C compiler (cc) failed.\n{proc.stderr}")

    lines = [line.strip() for line in proc.stderr.splitlines()]
    start_marker = '#include "..." search starts here:'
    mid_marker = "#include <...> search starts here:"
    end_marker = "End of search list."

    try:
        start_idx = lines.index(start_marker) + 1
    except ValueError:
        return []

    include_args: List[str] = []
    for line in lines[start_idx:]:
        if line == mid_marker:
            continue
        if line == end_marker:
            break
        if line:
            include_args.extend(["-isystem", line])
    return include_args


def normalise_signature(index: cindex.Index, query_string: str) -> str:
    return normalise_signature_with_language(index, query_string, language="c")


def normalise_signature_with_language(index: cindex.Index, query_string: str, language: str) -> str:
    """
    Normalize C declaration text using clang tokenizer.
    """
    unsaved_path = "<query>"
    unsaved = [(unsaved_path, query_string)]

    tu = index.parse(
        path=unsaved_path,
        args=[_clang_language_arg(language), "-E", "-w", "-ferror-limit=1"],
        unsaved_files=unsaved,
        options=cindex.TranslationUnit.PARSE_INCOMPLETE
        | cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
    )

    tokens = [tok.spelling for tok in tu.get_tokens(extent=tu.cursor.extent)]
    return " ".join(tokens).strip()


def parse_query(query_str: str, index: cindex.Index) -> Query:
    """
    Supported forms:
    - "<signature>"
    - "<name> :: <signature>"
    """
    if "::" in query_str:
        name_part, sig_part = query_str.split("::", 1)
        name = name_part.strip() or None
        sig_part = sig_part.strip()
    else:
        name = None
        sig_part = query_str.strip()

    if not sig_part:
        return Query(name=name, normalised_signature=None)

    fake = _build_fake_declaration(sig_part)
    try:
        norm = normalise_signature(index, fake)
    except Exception:
        norm = normalise_signature_with_language(index, fake, language="c++")
    norm = re.sub(r"\b__q__\b", "", norm)
    norm = norm.replace(" ;", "").strip()
    norm = re.sub(r"\s+", " ", norm).strip()
    return Query(name=name, normalised_signature=norm or None)


def _build_fake_declaration(sig_part: str) -> str:
    if "(" not in sig_part or ")" not in sig_part:
        return sig_part if sig_part.rstrip().endswith(";") else f"{sig_part};"

    if "__q__" in sig_part:
        return sig_part if sig_part.rstrip().endswith(";") else f"{sig_part};"

    # Heuristic for "ret (args)" format (no function name in source string).
    # Keep existing named prototypes untouched, for example "int foo(int)".
    if re.search(r"\w+\s*\(", sig_part) and " (" not in sig_part:
        return sig_part if sig_part.rstrip().endswith(";") else f"{sig_part};"

    open_paren = re.search(r"\(", sig_part)
    if open_paren is None:
        return sig_part if sig_part.rstrip().endswith(";") else f"{sig_part};"
    ret = sig_part[: open_paren.start()].strip()
    params = sig_part[open_paren.start() :].strip()
    return f"{ret} __q__ {params};"


def iter_functions(
    tu: cindex.TranslationUnit,
    *,
    only_from_file: Optional[str] = None,
) -> List[Function]:
    funcs: List[Function] = []
    only_path: Optional[str] = None

    if only_from_file is not None:
        only_path = str(Path(only_from_file).resolve())

    def collect(node: cindex.Cursor) -> None:
        if node.kind != cindex.CursorKind.FUNCTION_DECL or not node.location.file:
            return

        loc = node.location
        loc_file = str(loc.file)

        if only_path is not None:
            try:
                if str(Path(loc_file).resolve()) != only_path:
                    return
            except Exception:
                if loc_file != only_from_file:
                    return

        params: List[Tuple[str, Optional[str]]] = []
        for child in node.get_children():
            if child.kind == cindex.CursorKind.PARM_DECL:
                params.append((child.type.spelling, child.spelling or None))

        is_variadic = False
        try:
            is_variadic = bool(node.type.is_function_variadic())
        except Exception:
            is_variadic = False

        funcs.append(
            Function(
                name=node.spelling,
                location=Location(file_name=loc_file, line=loc.line, column=loc.column),
                return_type=node.result_type.spelling,
                parameters=params,
                is_variadic=is_variadic,
            )
        )

    def visit(node: cindex.Cursor) -> None:
        collect(node)
        for child in node.get_children():
            visit(child)

    visit(tu.cursor)
    return funcs


def levenshtein_distance(a: str, b: str) -> int:
    a_bytes = a.lower().encode("utf-8", errors="ignore")
    b_bytes = b.lower().encode("utf-8", errors="ignore")
    n, m = len(a_bytes), len(b_bytes)

    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    cur = [0] * (m + 1)

    for i in range(1, n + 1):
        cur[0] = i
        ai = a_bytes[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b_bytes[j - 1] else 1
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev
    return prev[m]


def score_function(func: Function, query: Query, index: cindex.Index) -> int:
    score = 0
    if query.name is not None:
        score += levenshtein_distance(func.name, query.name)
    if query.normalised_signature is not None:
        score += levenshtein_distance(func.normalised_signature(index), query.normalised_signature)
    return score
