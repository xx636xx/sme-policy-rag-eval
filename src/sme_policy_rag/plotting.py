"""Publication-grade 4-panel RAG evaluation dashboard renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def render_rag_eval_dashboard(report: dict[str, Any], output_path: Path) -> Path:
    """Render a 4-panel visual comparing chunking, retrieval, categories, and abstention gating."""
    strategies = report["strategies"]
    summary = report["corpus_summary"]
    thresh = report["abstention_threshold"]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel 1 (Top-Left): Strategy Benchmark — Recall@1, Recall@3, MRR
    ax1 = axes[0, 0]
    ax1.set_facecolor("#F8FAFC")
    order = [
        ("fixed_bm25", "Fixed-Window\nBM25"),
        ("section_bm25", "Section-Aware\nBM25"),
        ("section_tfidf", "Section-Aware\nTF-IDF N-Gram"),
        ("section_hybrid_rrf", "Section-Aware\nHybrid RRF"),
    ]
    x = np.arange(len(order))
    w = 0.25

    r1 = [strategies[k]["recall_at_1"] * 100 for k, _ in order]
    r3 = [strategies[k]["recall_at_3"] * 100 for k, _ in order]
    mrr = [strategies[k]["mrr"] * 100 for k, _ in order]

    b1 = ax1.bar(x - w, r1, w, label="Recall@1 (%)", color="#2563EB", alpha=0.90)
    b2 = ax1.bar(x, r3, w, label="Recall@3 (%)", color="#059669", alpha=0.90)
    b3 = ax1.bar(x + w, mrr, w, label="MRR (%)", color="#7C3AED", alpha=0.90)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
                color="#0F172A",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in order], fontsize=8.4, color="#1E293B")
    ax1.set_ylim(0, 126)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylabel("Score (%)", fontsize=9.4, color="#334155")
    ax1.set_title(
        "1. Retrieval Benchmark Across 24 Answerable Policy Queries",
        fontsize=10.6,
        fontweight="bold",
        color="#0F172A",
        loc="left",
        pad=8,
    )
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35, color="#94A3B8")
    ax1.legend(loc="upper center", frameon=True, facecolor="#FFFFFF", fontsize=8.0, ncol=3)

    # Panel 2 (Top-Right): Chunking Ablation (Fixed-Window vs Section-Aware on BM25 & Hybrid RRF)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#F8FAFC")
    ablation_metrics = [
        ("recall_at_1", "Recall@1"),
        ("recall_at_3", "Recall@3"),
        ("fact_coverage", "Answer Fact\nCoverage"),
        ("abstention_accuracy", "Unanswerable\nAbstention"),
    ]
    ax_x = np.arange(len(ablation_metrics))
    aw = 0.26

    fix_vals = [strategies["fixed_bm25"][m] * 100 for m, _ in ablation_metrics]
    sec_bm_vals = [strategies["section_bm25"][m] * 100 for m, _ in ablation_metrics]
    hyb_vals = [strategies["section_hybrid_rrf"][m] * 100 for m, _ in ablation_metrics]

    ab1 = ax2.bar(ax_x - aw, fix_vals, aw, label="Fixed-Window BM25", color="#DC2626", alpha=0.86)
    ab2 = ax2.bar(ax_x, sec_bm_vals, aw, label="Section-Aware BM25", color="#D97706", alpha=0.88)
    ab3 = ax2.bar(ax_x + aw, hyb_vals, aw, label="Section-Aware Hybrid RRF", color="#16A34A", alpha=0.90)

    for bars in (ab1, ab2, ab3):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
                color="#0F172A",
            )

    ax2.set_xticks(ax_x)
    ax2.set_xticklabels([lbl for _, lbl in ablation_metrics], fontsize=8.4, color="#1E293B")
    ax2.set_ylim(0, 126)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylabel("Score (%)", fontsize=9.4, color="#334155")
    ax2.set_title(
        "2. Chunking & Grounding Impact (Fixed vs Section-Aware)",
        fontsize=10.6,
        fontweight="bold",
        color="#0F172A",
        loc="left",
        pad=8,
    )
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35, color="#94A3B8")
    ax2.legend(loc="upper center", frameon=True, facecolor="#FFFFFF", fontsize=7.6, ncol=3)

    # Panel 3 (Bottom-Left): Query Category Recall@3 Breakdown
    ax3 = axes[1, 0]
    ax3.set_facecolor("#F8FAFC")
    cats = [
        ("exact_code", "Exact Standard\n/ Control Code"),
        ("paraphrased_policy", "Paraphrased\nPolicy Question"),
        ("procedural_threshold", "Procedural /\nSLA Threshold"),
    ]
    cx = np.arange(len(cats))
    cw = 0.25

    c_fix = [strategies["fixed_bm25"]["category_recall_at_3"][c] * 100 for c, _ in cats]
    c_tf = [strategies["section_tfidf"]["category_recall_at_3"][c] * 100 for c, _ in cats]
    c_hyb = [strategies["section_hybrid_rrf"]["category_recall_at_3"][c] * 100 for c, _ in cats]

    cb1 = ax3.bar(cx - cw, c_fix, cw, label="Fixed-Window BM25", color="#DC2626", alpha=0.86)
    cb2 = ax3.bar(cx, c_tf, cw, label="Section TF-IDF", color="#2563EB", alpha=0.88)
    cb3 = ax3.bar(cx + cw, c_hyb, cw, label="Section Hybrid RRF", color="#16A34A", alpha=0.90)

    for bars in (cb1, cb2, cb3):
        for bar in bars:
            h = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
                color="#0F172A",
            )

    ax3.set_xticks(cx)
    ax3.set_xticklabels([lbl for _, lbl in cats], fontsize=8.5, color="#1E293B")
    ax3.set_ylim(0, 126)
    ax3.set_yticks([0, 20, 40, 60, 80, 100])
    ax3.set_ylabel("Recall@3 (%)", fontsize=9.4, color="#334155")
    ax3.set_title(
        "3. Recall@3 Breakdown by Compliance Query Category",
        fontsize=10.6,
        fontweight="bold",
        color="#0F172A",
        loc="left",
        pad=8,
    )
    ax3.grid(True, axis="y", linestyle="--", alpha=0.35, color="#94A3B8")
    ax3.legend(loc="upper center", frameon=True, facecolor="#FFFFFF", fontsize=7.8, ncol=3)

    # Panel 4 (Bottom-Right): Hybrid RRF Confidence Separation (Answerable vs Unanswerable)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#F8FAFC")
    hyb = strategies["section_hybrid_rrf"]
    ans_c = hyb["answerable_confidences"]
    unans_c = hyb["unanswerable_confidences"]

    bins = np.linspace(0.0, 1.0, 21)
    ax4.hist(unans_c, bins=bins, color="#DC2626", alpha=0.75, edgecolor="#FFFFFF", label=f"Unanswerable Traps (n={len(unans_c)})")
    ax4.hist(ans_c, bins=bins, color="#059669", alpha=0.78, edgecolor="#FFFFFF", label=f"Answerable Policies (n={len(ans_c)})")
    ax4.axvline(thresh, color="#0F172A", linestyle="--", linewidth=1.8, label=f"Abstention Gate ({thresh:.2f})")

    blocked_n = round(hyb["abstention_accuracy"] * len(unans_c))
    ax4.annotate(
        f"Abstention Accuracy:\n{hyb['abstention_accuracy'] * 100:.1f}% ({blocked_n}/{len(unans_c)} traps blocked)",
        xy=(thresh, 3.2),
        xytext=(thresh + 0.08, 4.3),
        fontsize=8.3,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF3C7", edgecolor="#D97706", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#0F172A", lw=1.1),
    )

    ax4.set_xlabel("Hybrid RRF Calibrated Retrieval Confidence", fontsize=9.2, color="#334155")
    ax4.set_ylabel("Query Count", fontsize=9.2, color="#334155")
    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0, max(6, len(ans_c) // 3))
    ax4.set_title(
        "4. Hallucination Control — Abstention Gate Separation",
        fontsize=10.6,
        fontweight="bold",
        color="#0F172A",
        loc="left",
        pad=8,
    )
    ax4.grid(True, linestyle="--", alpha=0.35, color="#94A3B8")
    ax4.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", fontsize=7.8)

    fig.suptitle(
        "SME Policy RAG Eval — Hybrid Retrieval (BM25 + TF-IDF + RRF), Chunking Ablation & Abstention Gating",
        fontsize=13.2,
        fontweight="bold",
        color="#0F172A",
        y=0.985,
    )
    fig.text(
        0.5,
        0.946,
        f"Corpus: {summary['documents']} SME ISO 27001 / PRL / AI manuals ({summary['section_aware_chunks']} sections)  |  "
        f"Hybrid RRF Recall@3: {hyb['recall_at_3'] * 100:.1f}%  |  MRR: {hyb['mrr']:.3f}  |  "
        f"Citation Grounding: {hyb['citation_grounding'] * 100:.0f}%  |  Latency: {hyb['mean_latency_ms']:.2f} ms",
        ha="center",
        fontsize=9.1,
        color="#475569",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.93])
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    return output_path
