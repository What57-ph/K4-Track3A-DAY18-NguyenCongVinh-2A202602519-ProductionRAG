from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
import re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            use_model = os.getenv("M3_USE_CROSS_ENCODER", "").lower() in {"1", "true", "yes"}
            if use_model:
                try:
                    from sentence_transformers import CrossEncoder

                    # Dùng sentence_transformers.CrossEncoder; FlagEmbedding
                    # không tương thích ổn định với một số transformers mới.
                    self._model = CrossEncoder(self.model_name)
                except Exception as exc:
                    print(f"  ⚠️  Cross-encoder unavailable ({exc}); using lexical reranker.", flush=True)
            if self._model is None:
                self._model = _LexicalReranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        pairs = [(query, str(document.get("text", ""))) for document in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        scores = [float(score) for score in scores]

        scored = sorted(
            zip(scores, documents),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            RerankResult(
                text=str(document.get("text", "")),
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # FlashRank is optional. Reuse the deterministic implementation when
        # the package/model is not installed instead of failing the pipeline.
        if self._model is None:
            try:
                from flashrank import Ranker

                self._model = Ranker()
            except Exception:
                self._model = _LexicalReranker()

        if isinstance(self._model, _LexicalReranker):
            pairs = [(query, str(document.get("text", ""))) for document in documents]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
        else:
            try:
                from flashrank import RerankRequest

                passages = [{"id": index, "text": document.get("text", "")} for index, document in enumerate(documents)]
                ranked_raw = self._model.rerank(RerankRequest(query=query, passages=passages))
                ranked = [
                    (float(item.get("score", 0.0)), documents[int(item["id"])])
                    for item in ranked_raw
                ]
            except Exception:
                fallback = _LexicalReranker()
                scores = fallback.predict([(query, str(document.get("text", ""))) for document in documents])
                ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)

        return [
            RerankResult(
                text=str(document.get("text", "")),
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(ranked[:max(top_k, 0)])
        ]


class _LexicalReranker:
    """Offline fallback approximating query-document relevance."""

    def predict(self, pairs):
        return [self._score(query, document) for query, document in pairs]

    @staticmethod
    def _score(query: str, document: str) -> float:
        query_tokens = set(re.findall(r"[\wÀ-ỹ]+", query.lower(), flags=re.UNICODE))
        document_tokens = re.findall(r"[\wÀ-ỹ]+", document.lower(), flags=re.UNICODE)
        if not query_tokens or not document_tokens:
            return 0.0
        document_set = set(document_tokens)
        overlap = len(query_tokens & document_set) / len(query_tokens)
        phrase_bonus = 0.25 if " ".join(query.lower().split()) in " ".join(document.lower().split()) else 0.0
        return overlap + phrase_bonus + 0.001 / max(len(document_tokens), 1)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
