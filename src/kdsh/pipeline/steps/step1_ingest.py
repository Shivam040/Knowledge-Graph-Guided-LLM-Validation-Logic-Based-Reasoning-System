from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from kdsh.common.utils import (
    IST,
    sha256_file,
    read_text,
    book_code_from_name,
    time_bucket,
    extract_entities,
)
from kdsh.common.config import ChunkingConfig


def split_chapters(text: str, chapter_regex: str) -> List[Tuple[str, str, int, int]]:
    pat = re.compile(chapter_regex)
    matches = list(pat.finditer(text))
    if not matches:
        return [("ch000", "NO_CHAPTER", 0, len(text))]
    chapters = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group(0).strip()
        chapter_id = f"ch{i+1:03d}"
        chapters.append((chapter_id, title, start, end))
    return chapters


def chunk_chapter(
    chapter_text: str,
    book_code: str,
    chapter_id: str,
    chapter_title: str,
    global_offset: int,
    full_len: int,
    cfg: ChunkingConfig,
) -> List[Dict[str, Any]]:
    words = re.findall(r"\S+", chapter_text)
    if not words:
        return []

    W = cfg.window_words
    O = cfg.overlap_words
    step = max(1, W - O)

    lens = [len(w) + 1 for w in words]
    cum = [0]
    for L in lens:
        cum.append(cum[-1] + L)

    chunks = []
    win_idx = 0
    for start_w in range(0, len(words), step):
        end_w = min(len(words), start_w + W)
        win_words = words[start_w:end_w]
        if len(win_words) < cfg.min_chunk_words and start_w != 0:
            break

        text_win = " ".join(win_words)
        ch_start_char = cum[start_w]
        ch_end_char = cum[end_w]
        start_char = global_offset + ch_start_char
        end_char = global_offset + ch_end_char
        pos = ((start_char + end_char) / 2) / max(1, full_len)

        chunk_id = f"{book_code}_{chapter_id}_w{win_idx:04d}"

        chunks.append(
            dict(
                book_name=None,
                chunk_id=chunk_id,
                chunk_text=text_win,
                chunk_pos=round(float(pos), 6),

                chapter_id=chapter_id,
                chapter_title=chapter_title,

                # offsets for provenance
                word_start=int(start_w),
                word_end=int(end_w),
                char_start=int(start_char),
                char_end=int(end_char),
                chunk_word_count=int(len(win_words)),

                time_bucket=time_bucket(pos),

                # keep old + add new alias for later steps
                entities=extract_entities(text_win, 12),
                entity_mentions=extract_entities(text_win, 12),

                # NOTE: this was “token_count” but actually word count; keep for compatibility
                token_count=int(len(win_words)),
            )
        )

        win_idx += 1
        if end_w == len(words):
            break
    return chunks

def step1_ingest_and_chunk(
    novels: List[Tuple[str, Path]],
    out_silver: Path,
    out_bronze_novels: Path,
    run_id: str,
    cfg: ChunkingConfig,
) -> Tuple[Path, Path]:
    registry_rows = []
    chunk_rows = []

    for book_name, novel_path in novels:
        txt, enc = read_text(novel_path)
        book_code = book_code_from_name(book_name)

        registry_rows.append(
            dict(
                book_name=book_name,
                file_name=novel_path.name,
                file_path=str((out_bronze_novels / novel_path.name)),
                sha256=sha256_file(novel_path),
                encoding=enc,
                byte_len=novel_path.stat().st_size,
                ingest_ts=datetime.now(IST).isoformat(),
                run_id=run_id,
            )
        )

        chapters = split_chapters(txt, cfg.chapter_regex)
        for (chapter_id, title, start, end) in chapters:
            ch_txt = txt[start:end]
            rows = chunk_chapter(ch_txt, book_code, chapter_id, title, start, len(txt), cfg)
            for r in rows:
                r["book_name"] = book_name
                r["run_id"] = run_id
                chunk_rows.append(r)

    registry_df = pd.DataFrame(registry_rows)
    chunks_df = pd.DataFrame(chunk_rows)

    registry_path = out_silver / "novel_registry.csv"
    chunks_path = out_silver / "chunks.csv"
    registry_df.to_csv(registry_path, index=False)
    chunks_df.to_csv(chunks_path, index=False)
    return registry_path, chunks_path
