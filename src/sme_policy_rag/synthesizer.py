"""Grounded extractive answer synthesizer, obligation extractor, abstention gate, and Ollama adapter."""

from __future__ import annotations

import json
import re
import time
from typing import Protocol
import urllib.request

from .retrieval import tokenize
from .schemas import Citation, ExtractedObligation, GroundedAnswer, PolicyChunk

SLA_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:to|-)\s*\d+(?:\.\d+)?)?\s*"
    r"(?:hours?|minutes?|seconds?|days?|months?|meters?|cm|mL|m/s|characters?|copies)\b",
    re.IGNORECASE,
)
STD_PATTERN = re.compile(
    r"\b(?:ISO\s*27001(?:\s*control\s*A\.\d+\.\d+)?|ISO\s*A\.\d+\.\d+|"
    r"GDPR(?:\s*Article\s*\d+)?|EN\s*\d{3}|SOC\s*2(?:\s*Type\s*II)?|"
    r"EU\s*AI\s*Act|TPM\s*2\.0|AES-256|FIDO2|CVSS\s*>=\s*\d\.\d)\b",
    re.IGNORECASE,
)
ROLE_PATTERN = re.compile(
    r"(?:Data Protection Officer \(DPO\)|"
    r"\b(?:IT Security Lead|Incident Response Lead|"
    r"IT Operations|IT Helpdesk|IT Administrators|Operations Director|"
    r"Occupational Safety Technician|Industrial Hygiene Coordinator|"
    r"Laboratory Technicians|Procurement)\b)"
)


class RetrieverProtocol(Protocol):
    name: str

    def search(self, query: str, top_k: int = 3) -> list[tuple[PolicyChunk, float]]:
        ...


def extract_obligations(texts: list[str]) -> ExtractedObligation:
    """Extract verifiable SLAs, standards, and responsible roles strictly from cited chunks."""
    combined = " ".join(texts)
    slas = sorted({m.group(0).strip() for m in SLA_PATTERN.finditer(combined)})
    stds = sorted({m.group(0).strip() for m in STD_PATTERN.finditer(combined)})
    roles = sorted({m.group(0).strip() for m in ROLE_PATTERN.finditer(combined)})
    return ExtractedObligation(
        sla_or_threshold=slas[:8],
        referenced_standards=stds[:8],
        responsible_roles=roles[:6],
    )


def compute_grounding_score(answer: str, cited_texts: list[str]) -> float:
    """Measure fraction of informative tokens in the answer verbatim-grounded in citations."""
    # Exclude explicit bracketed citation tags like [POL-SEC-02-SEC-02]
    stripped = re.sub(r"\[[A-Z0-9-]+\]", " ", answer)
    ans_tokens = set(tokenize(stripped))
    if not ans_tokens:
        return 1.0
    ev_tokens = set(tokenize(" ".join(cited_texts)))
    overlap = len(ans_tokens & ev_tokens)
    return round(overlap / len(ans_tokens), 4)


def _call_ollama(prompt: str, model: str, base_url: str, timeout_s: float = 20.0) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("response", "")).strip()


def answer_query(
    query: str,
    retriever: RetrieverProtocol,
    top_k: int = 3,
    abstention_threshold: float = 0.24,
    ollama_model: str | None = None,
    ollama_url: str = "http://localhost:11434",
) -> GroundedAnswer:
    """Retrieve evidence, enforce abstention gate, and emit a cited Pydantic response."""
    t0 = time.perf_counter()
    hits = retriever.search(query, top_k=top_k)
    top_score = hits[0][1] if hits else 0.0

    citations = [
        Citation(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            section_title=chunk.section_title,
            score=score,
            excerpt=chunk.text[:260] + ("..." if len(chunk.text) > 260 else ""),
        )
        for chunk, score in hits
        if score > 0.0
    ]

    if not hits or top_score < abstention_threshold:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return GroundedAnswer(
            query=query,
            status="INSUFFICIENT_EVIDENCE",
            confidence=round(top_score, 4),
            answer=(
                "Insufficient evidence in the indexed SME compliance and safety manuals "
                "to answer this question reliably without speculation."
            ),
            citations=citations[:1],
            obligations=ExtractedObligation(),
            grounding_score=1.0,
            latency_ms=elapsed_ms,
            provider="abstention-gate",
        )

    # Select top cited chunk (plus secondary chunk if close in relevance)
    primary_chunk = hits[0][0]
    evidence_chunks = [primary_chunk]
    if len(hits) > 1 and hits[1][1] >= top_score * 0.78:
        evidence_chunks.append(hits[1][0])

    evidence_texts = [c.text for c in evidence_chunks]
    obligations = extract_obligations(evidence_texts)

    provider_used = "extractive-grounded"
    if ollama_model:
        context_block = "\n\n".join(
            f"[{c.chunk_id} | {c.section_title}]\n{c.text}" for c in evidence_chunks
        )
        prompt = (
            "Answer the compliance question using ONLY the provided policy excerpts. "
            "Cite the chunk_id in brackets and never invent rules not present in context.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
        )
        answer_text = _call_ollama(prompt, model=ollama_model, base_url=ollama_url)
        provider_used = f"ollama:{ollama_model}"
    else:
        # Deterministic extractive synthesis strictly grounded in top cited evidence
        body = primary_chunk.text
        answer_text = f"[{primary_chunk.chunk_id}] {body}"

    grounding = compute_grounding_score(answer_text, evidence_texts)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return GroundedAnswer(
        query=query,
        status="ANSWERED",
        confidence=round(top_score, 4),
        answer=answer_text,
        citations=citations,
        obligations=obligations,
        grounding_score=grounding,
        latency_ms=elapsed_ms,
        provider=provider_used,
    )
