"""Chunking strategies: naive fixed-window vs section-aware Markdown chunking."""

from __future__ import annotations

import re

from .schemas import PolicyChunk, PolicyDocument


def fixed_window_chunker(
    docs: list[PolicyDocument],
    window_chars: int = 240,
    overlap_chars: int = 35,
) -> list[PolicyChunk]:
    """Split policy text blindly by fixed character windows (baseline chunker).

    This intentionally mirrors naive RAG pipelines that cut across headings,
    sentences, and numeric compliance thresholds.
    """
    chunks: list[PolicyChunk] = []
    step = max(1, window_chars - overlap_chars)

    for doc in docs:
        clean = re.sub(r"\s+", " ", doc.content_markdown).strip()
        idx = 0
        chunk_num = 1
        while idx < len(clean):
            slice_text = clean[idx : idx + window_chars].strip()
            if slice_text:
                chunks.append(
                    PolicyChunk(
                        chunk_id=f"{doc.doc_id}-FIX-{chunk_num:02d}",
                        doc_id=doc.doc_id,
                        doc_title=doc.title,
                        section_title=f"Fixed Window #{chunk_num}",
                        text=slice_text,
                    )
                )
                chunk_num += 1
            idx += step

    return chunks


def section_aware_chunker(docs: list[PolicyDocument]) -> list[PolicyChunk]:
    """Split Markdown manuals cleanly on level-2 (`## `) policy section headings.

    Preserves full regulatory clauses, standard codes, and numeric SLAs intact
    inside each semantic unit.
    """
    chunks: list[PolicyChunk] = []

    for doc in docs:
        sections = re.split(r"(?m)^##\s+", doc.content_markdown)
        sec_idx = 1
        for raw_sec in sections[1:]:
            lines = [ln.strip() for ln in raw_sec.strip().splitlines() if ln.strip()]
            if not lines:
                continue
            section_title = lines[0]
            body = " ".join(lines[1:]).strip()
            full_text = f"{doc.title} — {section_title}: {body}"
            chunks.append(
                PolicyChunk(
                    chunk_id=f"{doc.doc_id}-SEC-{sec_idx:02d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section_title=section_title,
                    text=full_text,
                )
            )
            sec_idx += 1

    return chunks
