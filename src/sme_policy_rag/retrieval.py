"""Offline-reproducible retrieval engines: Okapi BM25, Word+Char TF-IDF, and Hybrid RRF."""

from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np

from .schemas import PolicyChunk

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "if", "in", "into", "is", "it", "must", "of",
    "on", "or", "our", "should", "that", "the", "their", "them", "they", "this",
    "to", "under", "what", "when", "where", "which", "who", "with", "within",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Tokenize preserving compliance codes like `a.8.5`, `en`, `166`, `2.0`, `3-2-1-1-0`."""
    raw = TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for tok in raw:
        if tok in STOPWORDS or len(tok) < 2:
            continue
        out.append(tok)
        # Also emit split subtokens for hyphenated/dotted identifiers (e.g., aes-256 -> aes, 256)
        if "-" in tok or "." in tok or "/" in tok:
            for sub in re.split(r"[-./]", tok):
                if len(sub) >= 2 and sub not in STOPWORDS:
                    out.append(sub)
    return out


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> list[str]:
    """Extract character n-grams to handle morphological variations and paraphrased roots."""
    cleaned = " ".join(tokenize(text))
    if not cleaned:
        return []
    grams: list[str] = []
    padded = f" {cleaned} "
    for n in range(n_min, n_max + 1):
        for i in range(len(padded) - n + 1):
            g = padded[i : i + n]
            if g.strip():
                grams.append(f"c{n}:{g}")
    return grams


class BM25Retriever:
    """Okapi BM25 lexical retriever."""

    name = "bm25"

    def __init__(self, chunks: list[PolicyChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        self.doc_lens = np.array([len(t) for t in self.doc_tokens], dtype=float)
        self.avg_dl = float(np.mean(self.doc_lens)) if len(self.doc_lens) else 1.0
        self.doc_freqs = [Counter(t) for t in self.doc_tokens]

        df: Counter[str] = Counter()
        for freq in self.doc_freqs:
            for term in freq:
                df[term] += 1
        n_docs = len(chunks)
        self.idf = {
            term: math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def score_all(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=float)
        if not q_tokens:
            return scores
        for i, freq in enumerate(self.doc_freqs):
            dl = self.doc_lens[i]
            s = 0.0
            for qt in q_tokens:
                tf = freq.get(qt, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(qt, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                s += idf * (tf * (self.k1 + 1.0)) / denom
            scores[i] = s
        return scores

    def search(self, query: str, top_k: int = 3) -> list[tuple[PolicyChunk, float]]:
        raw = self.score_all(query)
        # Normalize BM25 score relative to query length so abstention thresholding is scale-stable
        q_len = max(1, len(set(tokenize(query))))
        norm_scores = raw / (q_len * 2.5)
        order = np.argsort(-raw)[:top_k]
        return [(self.chunks[int(i)], round(float(norm_scores[int(i)]), 4)) for i in order]


class TfidfNgramRetriever:
    """Sublinear Word + Character N-Gram TF-IDF Cosine Retriever."""

    name = "tfidf_ngram"

    def __init__(self, chunks: list[PolicyChunk]) -> None:
        self.chunks = chunks
        doc_feats = [self._extract_features(c.text) for c in chunks]

        vocab: dict[str, int] = {}
        df: Counter[str] = Counter()
        for feats in doc_feats:
            for term in set(feats):
                if term not in vocab:
                    vocab[term] = len(vocab)
                df[term] += 1

        self.vocab = vocab
        n_docs = len(chunks)
        self.idf = np.zeros(len(vocab), dtype=float)
        for term, idx in vocab.items():
            self.idf[idx] = math.log((1.0 + n_docs) / (1.0 + df[term])) + 1.0

        mat = np.zeros((n_docs, len(vocab)), dtype=float)
        for row_i, feats in enumerate(doc_feats):
            counts = Counter(feats)
            for term, cnt in counts.items():
                col_i = vocab[term]
                mat[row_i, col_i] = (1.0 + math.log(cnt)) * self.idf[col_i]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = mat / norms

    @staticmethod
    def _extract_features(text: str) -> list[str]:
        words = [f"w:{w}" for w in tokenize(text)]
        # Double-weight whole tokens over char n-grams for clean precision + paraphrase recall
        return words + words + char_ngrams(text, 3, 5)

    def score_all(self, query: str) -> np.ndarray:
        feats = self._extract_features(query)
        if not feats or not self.vocab:
            return np.zeros(len(self.chunks), dtype=float)
        vec = np.zeros(len(self.vocab), dtype=float)
        for term, cnt in Counter(feats).items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = (1.0 + math.log(cnt)) * self.idf[idx]
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return np.zeros(len(self.chunks), dtype=float)
        vec /= norm
        return self.matrix @ vec

    def search(self, query: str, top_k: int = 3) -> list[tuple[PolicyChunk, float]]:
        scores = self.score_all(query)
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[int(i)], round(float(scores[int(i)]), 4)) for i in order]


class HybridRRFRetriever:
    """Hybrid Retriever fusing BM25 + Word/Char TF-IDF via Reciprocal Rank Fusion (RRF).

    Ranks documents via RRF (1 / (k + rank_bm25) + 1 / (k + rank_tfidf)) while
    computing a calibrated confidence score combining normalized BM25 and cosine
    similarity so out-of-scope queries cleanly separate below the abstention gate.
    """

    name = "hybrid_rrf"

    def __init__(self, chunks: list[PolicyChunk], rrf_k: int = 25) -> None:
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever(chunks)
        self.tfidf = TfidfNgramRetriever(chunks)

    def search(self, query: str, top_k: int = 3) -> list[tuple[PolicyChunk, float]]:
        bm_raw = self.bm25.score_all(query)
        tf_raw = self.tfidf.score_all(query)

        bm_order = np.argsort(-bm_raw)
        tf_order = np.argsort(-tf_raw)

        bm_ranks = np.empty(len(self.chunks), dtype=float)
        tf_ranks = np.empty(len(self.chunks), dtype=float)
        bm_ranks[bm_order] = np.arange(1, len(self.chunks) + 1)
        tf_ranks[tf_order] = np.arange(1, len(self.chunks) + 1)

        # Zero-signal terms shouldn't get free rank boost when raw score is 0
        bm_rrf = np.where(bm_raw > 0, 1.0 / (self.rrf_k + bm_ranks), 0.0)
        tf_rrf = np.where(tf_raw > 0, 1.0 / (self.rrf_k + tf_ranks), 0.0)
        rrf_scores = bm_rrf + tf_rrf

        # Calibrated semantic+lexical confidence for abstention gating (0..1 scale)
        q_len = max(1, len(set(tokenize(query))))
        bm_norm = np.clip(bm_raw / (q_len * 2.5), 0.0, 1.0)
        confidence = 0.50 * bm_norm + 0.50 * np.clip(tf_raw * 2.2, 0.0, 1.0)

        # Rank by RRF first, tie-break by calibrated confidence
        order = np.lexsort((-confidence, -rrf_scores))[:top_k]
        return [(self.chunks[int(i)], round(float(confidence[int(i)]), 4)) for i in order]
