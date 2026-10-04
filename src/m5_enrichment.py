from __future__ import annotations

"""Module 5: enrichment before embedding.

The module supports four independent techniques and a combined single-call
mode. Every technique has a deterministic local fallback so indexing still
works without an OpenAI key.
"""

import json
import os
import re
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """A raw chunk plus its enriched representation and metadata."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _use_openai() -> bool:
    """Avoid treating a short placeholder value as a real API key."""
    return (
        os.getenv("M5_USE_OPENAI", "").lower() in {"1", "true", "yes"}
        or len(OPENAI_API_KEY.strip()) >= 20
    )


def summarize_chunk(text: str) -> str:
    """Create a short summary using the LLM or extractive fallback."""
    clean_text = str(text or "").strip()
    if not clean_text:
        return ""
    if _use_openai():
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": clean_text},
                ],
                max_tokens=150,
            )
            summary = (response.choices[0].message.content or "").strip()
            if summary:
                return summary
        except Exception as exc:
            print(f"  ⚠️  OpenAI summarize failed ({exc}); using local fallback.", flush=True)
    return _fallback_summary(clean_text)


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions that the chunk can answer."""
    if n_questions <= 0 or not str(text or "").strip():
        return []
    clean_text = str(text).strip()
    if _use_openai():
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Mỗi câu hỏi trên một dòng.",
                    },
                    {"role": "user", "content": clean_text},
                ],
                max_tokens=200,
            )
            questions = [
                re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                for line in (response.choices[0].message.content or "").splitlines()
                if line.strip()
            ]
            if questions:
                return questions[:n_questions]
        except Exception as exc:
            print(f"  ⚠️  OpenAI HyQA failed ({exc}); using local fallback.", flush=True)
    return _fallback_questions(clean_text, n_questions)


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend a short description of the chunk's document context."""
    clean_text = str(text or "").strip()
    if not clean_text:
        return ""
    title = str(document_title or "").strip()
    if _use_openai():
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Viết một câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về một câu.",
                    },
                    {"role": "user", "content": f"Tài liệu: {title}\n\nĐoạn văn:\n{clean_text}"},
                ],
                max_tokens=80,
            )
            context = (response.choices[0].message.content or "").strip()
            if context:
                return f"{context}\n\n{clean_text}"
        except Exception as exc:
            print(f"  ⚠️  OpenAI contextual failed ({exc}); using local fallback.", flush=True)
    return _fallback_context(clean_text, title)


def extract_metadata(text: str) -> dict:
    """Extract topic, entities, date range, category and language."""
    clean_text = str(text or "").strip()
    if not clean_text:
        return {"topic": "general", "entities": [], "date_range": "", "category": "general", "language": "vi"}
    if _use_openai():
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": 'Trích xuất metadata và chỉ trả về JSON: {"topic":"...","entities":[],"date_range":"","category":"policy|hr|it|finance|general","language":"vi|en"}',
                    },
                    {"role": "user", "content": clean_text},
                ],
                max_tokens=180,
            )
            parsed = json.loads(_strip_json_fence(response.choices[0].message.content or ""))
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            print(f"  ⚠️  OpenAI metadata failed ({exc}); using local fallback.", flush=True)
    return _fallback_metadata(clean_text)


def _enrich_single_call(text: str, source: str) -> dict:
    """Return all enrichment fields using one LLM call when available."""
    clean_text = str(text or "").strip()
    if _use_openai():
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Phân tích đoạn văn và chỉ trả về JSON hợp lệ:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "một câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": [], "category": "policy|hr|it|finance|general", "language": "vi|en"}
}""",
                    },
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{clean_text}"},
                ],
                max_tokens=400,
            )
            parsed = json.loads(_strip_json_fence(response.choices[0].message.content or ""))
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            print(f"  ⚠️  Combined enrichment failed ({exc}); using local fallback.", flush=True)

    return {
        "summary": _fallback_summary(clean_text),
        "questions": _fallback_questions(clean_text, 3),
        "context": _fallback_context(clean_text, source).split("\n\n", 1)[0],
        "metadata": _fallback_metadata(clean_text),
    }


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    """Enrich chunks in combined mode or with selected individual techniques."""
    methods = ["combined"] if methods is None else list(methods)
    if not methods:
        methods = ["contextual"]
    valid_methods = {"summary", "hyqa", "contextual", "metadata", "combined"}
    unknown = set(methods) - valid_methods
    if unknown:
        raise ValueError(f"Unknown enrichment methods: {sorted(unknown)}")

    enriched = []
    for index, chunk in enumerate(chunks or []):
        text = str(chunk.get("text", ""))
        source = str(chunk.get("metadata", {}).get("source", ""))
        if "combined" in methods:
            result = _enrich_single_call(text, source)
            summary = str(result.get("summary", "") or "")
            questions = list(result.get("questions", []) or [])
            context_line = str(result.get("context", "") or "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_metadata = dict(result.get("metadata", {}) or {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = text
            if "contextual" in methods:
                enriched_text = contextual_prepend(text, source)
            elif "summary" in methods and summary:
                enriched_text = summary
            if "hyqa" in methods and questions:
                enriched_text = f"{enriched_text}\n\nQuestions:\n" + "\n".join(questions)
            auto_metadata = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**dict(chunk.get("metadata", {}) or {}), **auto_metadata},
            method="+".join(methods),
        ))
        if (index + 1) % 10 == 0 or index + 1 == len(chunks or []):
            print(f"  Enriched {index + 1}/{len(chunks or [])} chunks...", flush=True)
    return enriched


def _fallback_summary(text: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if not sentences:
        return text
    return " ".join(sentences[:2])


def _fallback_questions(text: str, n_questions: int) -> list[str]:
    lower = text.lower()
    if "nghỉ phép" in lower or ("nghỉ" in lower and "ngày" in lower):
        candidates = [
            "Nhân viên được nghỉ phép bao nhiêu ngày?",
            "Quy định nghỉ phép áp dụng cho đối tượng nào?",
            "Số ngày nghỉ phép được tính như thế nào?",
        ]
    elif "mật khẩu" in lower:
        candidates = [
            "Mật khẩu phải thay đổi bao lâu một lần?",
            "Quy định mật khẩu yêu cầu điều gì?",
        ]
    else:
        candidates = [
            "Đoạn văn này nói về vấn đề gì?",
            "Thông tin chính trong đoạn văn là gì?",
            "Quy định này áp dụng cho ai?",
        ]
    return candidates[:n_questions]


def _fallback_context(text: str, title: str) -> str:
    if title:
        prefix = f"Trích từ tài liệu '{title}', đoạn này chứa thông tin chính sách liên quan."
    else:
        prefix = "Đoạn trích này chứa thông tin chính sách liên quan."
    return f"{prefix}\n\n{text}"


def _fallback_metadata(text: str) -> dict:
    lower = text.lower()
    category = "general"
    category_keywords = {
        "it": ("vpn", "mật khẩu", "mfa", "phần mềm", "hệ thống", "bảo mật"),
        "finance": ("lương", "thưởng", "chi phí", "tài chính", "phụ cấp"),
        "hr": ("nghỉ", "thử việc", "nhân viên", "đào tạo", "đánh giá"),
        "policy": ("quy định", "chính sách", "phê duyệt", "an toàn"),
    }
    for candidate, keywords in category_keywords.items():
        if any(keyword in lower for keyword in keywords):
            category = candidate
            break
    topic_keywords = {
        "leave": ("nghỉ phép", "nghỉ ốm", "nghỉ không lương"),
        "password": ("mật khẩu", "mfa"),
        "remote_work": ("làm việc từ xa", "vpn"),
        "salary": ("lương", "thưởng", "phụ cấp"),
    }
    topic = next(
        (label for label, keywords in topic_keywords.items() if any(keyword in lower for keyword in keywords)),
        "general",
    )
    known_entities = [
        entity for entity in ("VPN", "MFA", "WireGuard", "AES-256", "Giám đốc")
        if entity.lower() in lower
    ]
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    language = "vi" if re.search(r"[À-ỹ]|\b(?:nhân viên|quy định|ngày)\b", lower) else "en"
    return {
        "topic": topic,
        "entities": known_entities,
        "date_range": ", ".join(dict.fromkeys(years)),
        "category": category,
        "language": language,
    }


def _strip_json_fence(content: str) -> str:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
    print(summarize_chunk(sample))
    print(generate_hypothesis_questions(sample))
    print(contextual_prepend(sample, "Sổ tay nhân viên"))
    print(extract_metadata(sample))
