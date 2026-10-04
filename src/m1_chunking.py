from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import math
import os, sys, glob, re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    if not text or not text.strip():
        return []
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    metadata = dict(metadata or {})
    sentences = _split_sentences(text)
    if not sentences:
        return []

    embeddings = _encode_sentences(sentences)
    groups: list[list[str]] = [[sentences[0]]]
    for index in range(1, len(sentences)):
        similarity = _cosine_similarity(embeddings[index - 1], embeddings[index])
        if similarity < threshold:
            groups.append([])
        groups[-1].append(sentences[index])

    chunks = []
    for index, group in enumerate(groups):
        chunk_metadata = {
            **metadata,
            "strategy": "semantic",
            "chunk_index": index,
        }
        chunks.append(Chunk(text=" ".join(group).strip(), metadata=chunk_metadata))
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split prose into non-empty units while retaining markdown lines."""
    units = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [re.sub(r"\s+", " ", unit).strip() for unit in units if unit.strip()]


@lru_cache(maxsize=1)
def _load_semantic_model():
    """Load the optional embedding model once per process.

    Importing/loading is deliberately lazy: chunking remains usable in a
    minimal or offline environment through the lexical fallback below.

    Set ``CHUNKING_USE_EMBEDDINGS=1`` after pre-downloading the model to use
    the SentenceTransformer implementation. The default keeps library calls
    deterministic and avoids a surprising network/model startup cost.
    """
    if os.getenv("CHUNKING_USE_EMBEDDINGS", "").lower() not in {"1", "true", "yes"}:
        raise RuntimeError("sentence-transformer embeddings are disabled")

    from sentence_transformers import SentenceTransformer

    # Do not initiate a network download from a library function. The model
    # can be pre-downloaded as documented in README.md; otherwise the caller
    # transparently falls back to lexical cosine similarity.
    return SentenceTransformer(
        "all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True},
    )


def _encode_sentences(sentences: list[str]):
    try:
        model = _load_semantic_model()
        return model.encode(sentences, normalize_embeddings=True)
    except Exception:
        # A deterministic lexical vector keeps unit tests and local/offline
        # development functional when the optional model is unavailable.
        return [_lexical_vector(sentence) for sentence in sentences]


def _lexical_vector(text: str) -> dict[str, float]:
    tokens = re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
    counts = Counter(tokens)
    magnitude = math.sqrt(sum(value * value for value in counts.values()))
    if magnitude == 0:
        return {}
    return {token: value / magnitude for token, value in counts.items()}


def _cosine_similarity(first, second) -> float:
    if isinstance(first, dict) and isinstance(second, dict):
        return sum(first.get(key, 0.0) * second.get(key, 0.0) for key in first)

    # Avoid requiring numpy for the fallback implementation. Numpy arrays
    # support iteration and regular Python arithmetic sufficiently here.
    first_values = list(first)
    second_values = list(second)
    first_norm = math.sqrt(sum(value * value for value in first_values))
    second_norm = math.sqrt(sum(value * value for value in second_values))
    if not first_norm or not second_norm:
        return 0.0
    return sum(a * b for a, b in zip(first_values, second_values)) / (first_norm * second_norm)


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    if not text or not text.strip():
        return ([], [])

    metadata = dict(metadata or {})
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text.strip())
                  if paragraph.strip()]
    parent_texts = _pack_paragraphs(paragraphs, parent_size)

    parents: list[Chunk] = []
    children: list[Chunk] = []
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "strategy": "hierarchical",
                "chunk_type": "parent",
                "chunk_index": parent_index,
                "parent_id": parent_id,
            },
            parent_id=parent_id,
        ))

        for child_text in _split_text_by_size(parent_text, child_size):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "strategy": "hierarchical",
                    "chunk_type": "child",
                    "chunk_index": len(children),
                    "parent_id": parent_id,
                },
                parent_id=parent_id,
            ))
    return (parents, children)


def _split_text_by_size(text: str, max_size: int) -> list[str]:
    """Split text at whitespace where possible, with a hard-size guarantee."""
    remaining = text.strip()
    parts = []
    while remaining:
        if len(remaining) <= max_size:
            parts.append(remaining)
            break
        cut = remaining.rfind(" ", 0, max_size + 1)
        if cut <= 0:
            cut = max_size
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [part for part in parts if part]


def _pack_paragraphs(paragraphs: list[str], max_size: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = _split_text_by_size(paragraph, max_size)
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if current and len(candidate) > max_size:
                packed.append(current)
                current = piece
            else:
                current = candidate
    if current:
        packed.append(current)
    return packed


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    if not text or not text.strip():
        return []

    metadata = dict(metadata or {})
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_header = ""
    current_level = 0
    in_fenced_code = False

    def flush() -> None:
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if not content:
            current_lines = []
            return
        chunks.append(Chunk(
            text=content,
            metadata={
                **metadata,
                "strategy": "structure",
                "section": current_header or "preamble",
                "section_level": current_level,
                "chunk_index": len(chunks),
            },
        ))
        current_lines = []

    for line in text.splitlines():
        fence = re.match(r"^\s*(```|~~~)", line)
        if fence:
            in_fenced_code = not in_fenced_code

        header_match = None if in_fenced_code and not fence else re.match(
            r"^(#{1,3})[ \t]+(.+?)\s*$", line
        )
        if header_match:
            flush()
            current_header = line.strip()
            current_level = len(header_match.group(1))
        current_lines.append(line)

    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
