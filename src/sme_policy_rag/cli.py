"""CLI entrypoint for running the RAG evaluation harness or asking grounded compliance questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import section_aware_chunker
from .corpus import load_policy_corpus
from .evaluator import run_full_evaluation
from .plotting import render_rag_eval_dashboard
from .retrieval import HybridRRFRetriever
from .synthesizer import answer_query


def cmd_eval(output_dir: Path, chart_path: Path, threshold: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_full_evaluation(abstention_threshold=threshold)
    render_rag_eval_dashboard(report, output_path=chart_path)

    # Compact printable summary (without raw confidence arrays)
    compact_strategies = {}
    for name, metrics in report["strategies"].items():
        compact_strategies[name] = {
            k: v
            for k, v in metrics.items()
            if k not in ("answerable_confidences", "unanswerable_confidences", "sample_outputs")
        }

    out_summary = {
        "corpus_summary": report["corpus_summary"],
        "abstention_threshold": report["abstention_threshold"],
        "strategies": compact_strategies,
        "hybrid_rrf_samples": report["strategies"]["section_hybrid_rrf"]["sample_outputs"],
        "chart": str(chart_path),
    }
    (output_dir / "eval_report.json").write_text(
        json.dumps(out_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(out_summary, indent=2))


def cmd_ask(question: str, threshold: float, ollama_model: str | None, ollama_url: str) -> None:
    docs = load_policy_corpus()
    chunks = section_aware_chunker(docs)
    retriever = HybridRRFRetriever(chunks)
    ans = answer_query(
        query=question,
        retriever=retriever,
        top_k=3,
        abstention_threshold=threshold,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
    )
    print(json.dumps(ans.model_dump(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sme-policy-rag-eval",
        description="Grounded SME compliance Q&A and automated RAG evaluation harness",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run full 4-pipeline RAG evaluation benchmark and render chart")
    p_eval.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p_eval.add_argument("--chart-path", type=Path, default=Path("assets/rag_eval_overview.png"))
    p_eval.add_argument("--threshold", type=float, default=0.24)

    p_ask = sub.add_parser("ask", help="Query the SME compliance manuals with grounded citations")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--threshold", type=float, default=0.24)
    p_ask.add_argument("--ollama-model", type=str, default=None, help="Optional local Ollama model (e.g. qwen2.5:7b)")
    p_ask.add_argument("--ollama-url", type=str, default="http://localhost:11434")

    args = parser.parse_args()
    if args.command == "eval":
        cmd_eval(output_dir=args.output_dir, chart_path=args.chart_path, threshold=args.threshold)
    elif args.command == "ask":
        cmd_ask(
            question=args.question,
            threshold=args.threshold,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
        )


if __name__ == "__main__":
    main()
