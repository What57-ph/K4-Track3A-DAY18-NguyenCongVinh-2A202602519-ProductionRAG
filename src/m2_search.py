from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import math
import os, sys
import re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text or not text.strip():
        return ""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
    except Exception:
        # The search module remains usable when underthesea is not installed
        # or cannot handle an unusual input document.
        segmented = text

    # underthesea joins multi-syllable Vietnamese words with underscores.
    # BM25 must see the individual query terms as well, so restore spaces.
    return re.sub(r"\s+", " ", segmented.replace("_", " ")).strip()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks or [])
        self.corpus_tokens = [
            segment_vietnamese(str(chunk.get("text", ""))).split()
            for chunk in self.documents
        ]
        if not self.corpus_tokens:
            self.bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents or top_k <= 0:
            return []
        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (float(scores[index]), -index),
            reverse=True,
        )
        results = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            document = self.documents[index]
            results.append(SearchResult(
                text=str(document.get("text", "")),
                score=score,
                metadata=dict(document.get("metadata", {})),
                method="bm25",
            ))
            if len(results) >= top_k:
                break
        return results


class _SimpleBM25:
    """Small dependency-free BM25 fallback for constrained environments."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.avgdl = sum(map(len, corpus_tokens)) / max(len(corpus_tokens), 1)
        document_frequency = {}
        for document in corpus_tokens:
            for token in set(document):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log(1 + (len(corpus_tokens) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for document in self.corpus_tokens:
            frequencies = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                if token not in frequencies:
                    continue
                tf = frequencies[token]
                denominator = tf + self.k1 * (1 - self.b + self.b * len(document) / max(self.avgdl, 1e-9))
                score += self.idf.get(token, 0.0) * (tf * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        self._documents: list[dict] = []
        self._vectors: list[list[float]] = []
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            try:
                self.client = QdrantClient(":memory:")
            except Exception:
                # qdrant-client is optional for BM25-only or unit-test usage.
                self.client = None

    def _get_encoder(self):
        if self._encoder is None:
            # Loading bge-m3 is expensive and should be explicit in local
            # development. The hashing encoder keeps a deterministic dense
            # fallback when no model/Qdrant service is available.
            use_model = os.getenv("M2_USE_EMBEDDINGS", "").lower() in {"1", "true", "yes"}
            if use_model:
                try:
                    from sentence_transformers import SentenceTransformer

                    self._encoder = SentenceTransformer(EMBEDDING_MODEL)
                except Exception as exc:
                    print(f"  ⚠️  Dense model unavailable ({exc}); using local encoder.", flush=True)
            if self._encoder is None:
                self._encoder = _HashingEncoder(EMBEDDING_DIM)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._documents = list(chunks or [])
        self._vectors = []
        if not self._documents:
            return

        texts = [str(chunk.get("text", "")) for chunk in self._documents]
        self._vectors = self._encode(texts)

        if self.client is None:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=index,
                    vector=vector,
                    payload={**dict(chunk.get("metadata", {})), "text": text},
                )
                for index, (chunk, text, vector) in enumerate(
                    zip(self._documents, texts, self._vectors)
                )
            ]
            self.client.upsert(collection_name=collection, points=points, wait=True)
        except Exception as exc:
            # Preserve local vectors so DenseSearch still works if Qdrant is
            # down or has a client/server API mismatch.
            print(f"  ⚠️  Qdrant indexing unavailable ({exc}); using local index.", flush=True)
            self.client = None

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not self._documents or top_k <= 0:
            return []

        query_vector = self._encode([query])[0]
        if self.client is not None:
            try:
                response = self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
                results = []
                for point in response.points:
                    payload = dict(point.payload or {})
                    text = str(payload.get("text", ""))
                    results.append(SearchResult(
                        text=text,
                        score=float(point.score),
                        metadata=payload,
                        method="dense",
                    ))
                return results
            except Exception as exc:
                print(f"  ⚠️  Qdrant search unavailable ({exc}); using local index.", flush=True)
                self.client = None

        scored = [
            (self._cosine(query_vector, vector), index)
            for index, vector in enumerate(self._vectors)
        ]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        results = []
        for score, index in scored[:top_k]:
            document = self._documents[index]
            results.append(SearchResult(
                text=str(document.get("text", "")),
                score=float(score),
                metadata=dict(document.get("metadata", {})),
                method="dense",
            ))
        return results

    def _encode(self, texts: list[str]) -> list[list[float]]:
        encoded = self._get_encoder().encode(texts, show_progress_bar=False)
        if isinstance(encoded, (list, tuple)):
            vectors = encoded
        else:
            vectors = encoded.tolist()
        return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors]

    @staticmethod
    def _cosine(first: list[float], second: list[float]) -> float:
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if not first_norm or not second_norm:
            return 0.0
        return sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []

    rrf_scores: dict[str, dict] = {}
    for result_list in results_list or []:
        seen_in_list = set()
        for rank, result in enumerate(result_list or []):
            if result.text in seen_in_list:
                continue
            seen_in_list.add(result.text)
            entry = rrf_scores.setdefault(
                result.text,
                {"score": 0.0, "result": result},
            )
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(
        rrf_scores.values(),
        key=lambda entry: entry["score"],
        reverse=True,
    )[:top_k]
    return [
        SearchResult(
            text=entry["result"].text,
            score=float(entry["score"]),
            metadata=dict(entry["result"].metadata),
            method="hybrid",
        )
        for entry in ranked
    ]


class _HashingEncoder:
    """Deterministic local vectorizer used only when bge-m3 is unavailable."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, texts, show_progress_bar: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()
        self._documents: list[dict] = []
        self._parent_contexts: dict[tuple[str, str], str] = {}
        self._parent_metadata: dict[tuple[str, str], dict] = {}

    def index(self, chunks: list[dict]) -> None:
        self._documents = list(chunks or [])
        self.bm25.index(chunks)
        self.dense.index(chunks)
        self._build_parent_contexts()

    @staticmethod
    def _parent_key(metadata: dict, fallback: str = "") -> tuple[str, str]:
        """Return a document-scoped parent key.

        M1 creates local parent IDs per document. Combining the source with the
        ID prevents ``parent_0`` from one document overwriting ``parent_0``
        from another document.
        """
        metadata = dict(metadata or {})
        source = str(metadata.get("source", ""))
        parent_id = str(metadata.get("parent_id", fallback))
        return source, parent_id

    def _build_parent_contexts(self) -> None:
        """Build full parent contexts from indexed hierarchical children."""
        grouped: dict[tuple[str, str], list[tuple[int, str, dict]]] = {}
        for index, document in enumerate(self._documents):
            metadata = dict(document.get("metadata", {}) or {})
            parent_id = metadata.get("parent_id")
            if parent_id is None:
                continue
            key = self._parent_key(metadata, fallback=f"chunk_{index}")
            chunk_index = int(metadata.get("chunk_index", index))
            grouped.setdefault(key, []).append(
                (chunk_index, str(document.get("text", "")), metadata)
            )

        self._parent_contexts = {}
        self._parent_metadata = {}
        for key, chunks in grouped.items():
            chunks.sort(key=lambda item: item[0])
            self._parent_contexts[key] = "\n\n".join(
                text for _, text, _ in chunks if text.strip()
            )
            self._parent_metadata[key] = dict(chunks[0][2])

    def _expand_parent_contexts(self, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Expand child hits to unique, complete parent contexts."""
        expanded = []
        seen_parents = set()
        for result in results:
            metadata = dict(result.metadata or {})
            if "parent_id" not in metadata:
                expanded.append(result)
                if len(expanded) >= top_k:
                    break
                continue

            key = self._parent_key(metadata)
            if key in seen_parents:
                continue
            seen_parents.add(key)
            context = self._parent_contexts.get(key, result.text)
            expanded.append(SearchResult(
                text=context,
                score=result.score,
                metadata=self._parent_metadata.get(key, metadata),
                method="hybrid",
            ))
            if len(expanded) >= top_k:
                break
        return expanded

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        # Keep the full BM25/Dense candidate pool before collapsing children
        # into parents; otherwise a relevant sibling can be truncated before
        # its parent context is reconstructed.
        candidate_k = max(top_k, BM25_TOP_K + DENSE_TOP_K)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], top_k=candidate_k)
        return self._expand_parent_contexts(fused, top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
