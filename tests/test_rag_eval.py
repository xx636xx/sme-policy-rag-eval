"""Automated pytest suite for chunking, hybrid retrieval, abstention gating, and RAG eval."""

from __future__ import annotations

from sme_policy_rag.chunking import fixed_window_chunker, section_aware_chunker
from sme_policy_rag.corpus import load_policy_corpus
from sme_policy_rag.evaluator import run_full_evaluation
from sme_policy_rag.retrieval import HybridRRFRetriever
from sme_policy_rag.synthesizer import answer_query


def test_section_aware_chunker_preserves_headings_and_clauses() -> None:
    docs = load_policy_corpus()
    sec_chunks = section_aware_chunker(docs)
    fix_chunks = fixed_window_chunker(docs)

    assert len(docs) == 8
    assert len(sec_chunks) == 23
    assert len(fix_chunks) > len(sec_chunks)
    assert all(c.section_title and not c.section_title.startswith("Fixed") for c in sec_chunks)


def test_grounded_answer_extracts_obligations_and_abstains_on_out_of_scope() -> None:
    docs = load_policy_corpus()
    retriever = HybridRRFRetriever(section_aware_chunker(docs))

    # Answerable GDPR question
    ans = answer_query(
        "Under GDPR Article 33, who notifies the AEPD and within what deadline?",
        retriever=retriever,
    )
    assert ans.status == "ANSWERED"
    assert ans.grounding_score == 1.0
    assert ans.citations[0].doc_id == "POL-SEC-02"
    assert any("72 hours" in s.lower() for s in ans.obligations.sla_or_threshold)
    assert any("dpo" in r.lower() for r in ans.obligations.responsible_roles)

    # Unanswerable out-of-scope trap must abstain instead of hallucinating
    trap = answer_query(
        "What is the maximum reimbursement limit for international business-class flights?",
        retriever=retriever,
    )
    assert trap.status == "INSUFFICIENT_EVIDENCE"


def test_hybrid_rrf_outperforms_fixed_window_bm25_on_benchmark() -> None:
    report = run_full_evaluation(abstention_threshold=0.24)
    fixed = report["strategies"]["fixed_bm25"]
    hybrid = report["strategies"]["section_hybrid_rrf"]

    assert hybrid["recall_at_3"] > fixed["recall_at_3"]
    assert hybrid["mrr"] > fixed["mrr"]
    assert hybrid["fact_coverage"] > fixed["fact_coverage"]
    assert hybrid["recall_at_3"] >= 0.90
    assert hybrid["abstention_accuracy"] >= 0.83
    assert hybrid["citation_grounding"] == 1.0
