"""Pydantic v2 contracts for policy chunks, citations, grounded answers, and evaluation."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class PolicyDocument(BaseModel):
    doc_id: str
    title: str
    domain: str
    content_markdown: str


class PolicyChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    text: str


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    section_title: str
    score: float
    excerpt: str


class ExtractedObligation(BaseModel):
    sla_or_threshold: list[str] = Field(default_factory=list)
    referenced_standards: list[str] = Field(default_factory=list)
    responsible_roles: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    query: str
    status: Literal["ANSWERED", "INSUFFICIENT_EVIDENCE"]
    confidence: float
    answer: str
    citations: list[Citation]
    obligations: ExtractedObligation
    grounding_score: float
    latency_ms: float
    provider: str = "extractive-grounded"


class EvalCase(BaseModel):
    case_id: str
    query: str
    category: Literal["exact_code", "paraphrased_policy", "procedural_threshold", "unanswerable"]
    expected_doc_id: str | None
    expected_section_keyword: str | None
    required_answer_terms: list[str] = Field(default_factory=list)
    answerable: bool = True
