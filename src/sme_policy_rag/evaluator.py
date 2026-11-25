"""Automated RAG evaluation harness comparing chunking & retrieval strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from .chunking import fixed_window_chunker, section_aware_chunker
from .corpus import load_eval_benchmark, load_policy_corpus
from .retrieval import BM25Retriever, HybridRRFRetriever, TfidfNgramRetriever
from .schemas import EvalCase, PolicyChunk
from .synthesizer import RetrieverProtocol, answer_query


def _chunk_matches_case(chunk: PolicyChunk, case: EvalCase) -> bool:
    """Check whether a retrieved chunk contains the expected policy document and facts."""
    if case.expected_doc_id is None or chunk.doc_id != case.expected_doc_id:
        return False
    lower_text = chunk.text.lower()
    if case.expected_section_keyword and case.expected_section_keyword.lower() in chunk.section_title.lower():
        return True
    # For fixed-window chunks (which lose section headers), require all required answer terms
    if case.required_answer_terms:
        return all(term.lower() in lower_text for term in case.required_answer_terms)
    return True


def evaluate_strategy(
    strategy_name: str,
    retriever: RetrieverProtocol,
    cases: list[EvalCase],
    abstention_threshold: float = 0.24,
) -> dict[str, Any]:
    """Evaluate one chunker + retriever configuration across all 30 labeled cases."""
    answerable_cases = [c for c in cases if c.answerable]
    unanswerable_cases = [c for c in cases if not c.answerable]

    r1_hits = 0
    r3_hits = 0
    rr_sum = 0.0
    fact_coverage_scores: list[float] = []
    grounding_scores: list[float] = []
    latencies: list[float] = []
    false_abstentions = 0
    answerable_confidences: list[float] = []
    unanswerable_confidences: list[float] = []

    category_stats: dict[str, list[float]] = {
        "exact_code": [],
        "paraphrased_policy": [],
        "procedural_threshold": [],
    }
    sample_outputs: list[dict[str, Any]] = []

    for case in answerable_cases:
        hits = retriever.search(case.query, top_k=3)
        hit_rank: int | None = None
        for rank_idx, (chunk, _score) in enumerate(hits, start=1):
            if _chunk_matches_case(chunk, case):
                hit_rank = rank_idx
                break

        hit_at_1 = 1.0 if hit_rank == 1 else 0.0
        hit_at_3 = 1.0 if hit_rank is not None else 0.0
        r1_hits += int(hit_at_1)
        r3_hits += int(hit_at_3)
        rr_sum += (1.0 / hit_rank) if hit_rank is not None else 0.0
        category_stats[case.category].append(hit_at_3)

        ans = answer_query(
            case.query,
            retriever=retriever,
            top_k=3,
            abstention_threshold=abstention_threshold,
        )
        answerable_confidences.append(ans.confidence)
        latencies.append(ans.latency_ms)
        grounding_scores.append(ans.grounding_score)

        if ans.status == "INSUFFICIENT_EVIDENCE":
            false_abstentions += 1
            fact_coverage_scores.append(0.0)
        else:
            if case.required_answer_terms:
                ans_lower = ans.answer.lower()
                matched = sum(1 for t in case.required_answer_terms if t.lower() in ans_lower)
                fact_coverage_scores.append(matched / len(case.required_answer_terms))
            else:
                fact_coverage_scores.append(1.0)

        if case.case_id in ("Q07", "Q14", "Q22"):
            sample_outputs.append(ans.model_dump())

    correct_abstentions = 0
    for case in unanswerable_cases:
        ans = answer_query(
            case.query,
            retriever=retriever,
            top_k=3,
            abstention_threshold=abstention_threshold,
        )
        unanswerable_confidences.append(ans.confidence)
        latencies.append(ans.latency_ms)
        if ans.status == "INSUFFICIENT_EVIDENCE":
            correct_abstentions += 1
        if case.case_id == "U01":
            sample_outputs.append(ans.model_dump())

    n_ans = len(answerable_cases)
    n_unans = len(unanswerable_cases)

    return {
        "strategy": strategy_name,
        "abstention_threshold": abstention_threshold,
        "recall_at_1": round(r1_hits / n_ans, 4),
        "recall_at_3": round(r3_hits / n_ans, 4),
        "mrr": round(rr_sum / n_ans, 4),
        "fact_coverage": round(float(np.mean(fact_coverage_scores)), 4),
        "citation_grounding": round(float(np.mean(grounding_scores)), 4),
        "abstention_accuracy": round(correct_abstentions / n_unans, 4),
        "false_abstention_rate": round(false_abstentions / n_ans, 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "category_recall_at_3": {
            cat: round(float(np.mean(vals)), 4) for cat, vals in category_stats.items()
        },
        "answerable_confidences": answerable_confidences,
        "unanswerable_confidences": unanswerable_confidences,
        "sample_outputs": sample_outputs,
    }


def run_full_evaluation(abstention_threshold: float = 0.24) -> dict[str, Any]:
    """Run head-to-head chunking & retrieval ablation across all 4 RAG pipelines."""
    docs = load_policy_corpus()
    cases = load_eval_benchmark()

    fixed_chunks = fixed_window_chunker(docs, window_chars=240, overlap_chars=35)
    section_chunks = section_aware_chunker(docs)

    pipelines: list[tuple[str, RetrieverProtocol]] = [
        ("fixed_bm25", BM25Retriever(fixed_chunks)),
        ("section_bm25", BM25Retriever(section_chunks)),
        ("section_tfidf", TfidfNgramRetriever(section_chunks)),
        ("section_hybrid_rrf", HybridRRFRetriever(section_chunks)),
    ]

    results = {
        name: evaluate_strategy(name, retriever, cases, abstention_threshold=abstention_threshold)
        for name, retriever in pipelines
    }

    return {
        "corpus_summary": {
            "documents": len(docs),
            "fixed_window_chunks": len(fixed_chunks),
            "section_aware_chunks": len(section_chunks),
            "total_eval_cases": len(cases),
            "answerable_cases": sum(1 for c in cases if c.answerable),
            "unanswerable_cases": sum(1 for c in cases if not c.answerable),
        },
        "abstention_threshold": abstention_threshold,
        "strategies": results,
    }
