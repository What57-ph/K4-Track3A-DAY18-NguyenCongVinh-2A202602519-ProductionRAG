from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import math
import os, sys, json
import re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    size = min(len(questions), len(answers), len(contexts), len(ground_truths))
    if size == 0:
        return _aggregate_results([])

    questions = list(questions[:size])
    answers = list(answers[:size])
    contexts = [list(item or []) for item in contexts[:size]]
    ground_truths = list(ground_truths[:size])

    # An explicit flag must be able to disable network-backed evaluation.
    # Previously M4_USE_RAGAS=0 was ignored whenever .env contained a key.
    ragas_flag = os.getenv("M4_USE_RAGAS", "").strip().lower()
    use_ragas = (
        ragas_flag in {"1", "true", "yes"}
        if ragas_flag
        else len(OPENAI_API_KEY.strip()) >= 20
    )
    if use_ragas:
        try:
            from datasets import Dataset
            from ragas import evaluate
            try:
                # RAGAS >= 0.4 moved these metrics to collections.
                from ragas.metrics.collections import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )
            except ImportError:
                # Keep compatibility with the version pinned in requirements.txt.
                from ragas.metrics import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            ragas_result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            dataframe = ragas_result.to_pandas()
            heuristic = _heuristic_evaluation(
                questions, answers, contexts, ground_truths
            )["per_question"]
            metric_names = (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            )
            ragas_scores = {
                metric: [_optional_score(row.get(metric)) for _, row in dataframe.iterrows()]
                for metric in metric_names
            }

            # Some RAGAS/version/API combinations return an all-zero or NaN
            # answer_relevancy column while the other metrics are valid.
            relevancy_scores = ragas_scores["answer_relevancy"]
            heuristic_relevancy = [item.answer_relevancy for item in heuristic]
            if (
                not any(score is not None and score > 0 for score in relevancy_scores)
                and any(score > 0 for score in heuristic_relevancy)
            ):
                ragas_scores["answer_relevancy"] = heuristic_relevancy

            per_question = []
            for index in range(size):
                per_question.append(EvalResult(
                    question=questions[index],
                    answer=answers[index],
                    contexts=contexts[index],
                    ground_truth=ground_truths[index],
                    faithfulness=_safe_score(ragas_scores["faithfulness"][index]),
                    answer_relevancy=_safe_score(ragas_scores["answer_relevancy"][index]),
                    context_precision=_safe_score(ragas_scores["context_precision"][index]),
                    context_recall=_safe_score(ragas_scores["context_recall"][index]),
                ))
            return _aggregate_results(per_question)
        except Exception as exc:
            print(f"  ⚠️  RAGAS evaluation failed ({exc}); using local metrics.", flush=True)

    return _heuristic_evaluation(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM hallucinating or making claims unsupported by the retrieved context.",
            "Tighten the answer prompt, require citations, and lower generation temperature.",
        ),
        "context_recall": (
            "Relevant information is missing from the retrieved chunks.",
            "Improve chunking, add BM25/hybrid retrieval, or increase retrieval top-k.",
        ),
        "context_precision": (
            "Retrieved context contains too many irrelevant chunks.",
            "Add reranking, metadata filters, or improve query expansion.",
        ),
        "answer_relevancy": (
            "The answer does not directly address the question.",
            "Improve the answer prompt and require a concise response to the query.",
        ),
    }
    metric_names = tuple(diagnostic_tree)
    analysed = []
    for item in eval_results or []:
        values = {name: _safe_score(_item_value(item, name)) for name in metric_names}
        average = sum(values.values()) / len(values)
        worst_metric = min(metric_names, key=lambda name: values[name])
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analysed.append({
            "question": _item_value(item, "question", ""),
            "worst_metric": worst_metric,
            "score": round(values[worst_metric], 4),
            "average_score": round(average, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    analysed.sort(key=lambda item: (item["average_score"], item["score"]))
    return analysed[:bottom_n]


def _safe_score(value) -> float:
    try:
        score = float(value)
        return max(0.0, min(1.0, score)) if math.isfinite(score) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _optional_score(value) -> float | None:
    """Convert a RAGAS value while preserving missing/NaN as None."""
    try:
        score = float(value)
        return max(0.0, min(1.0, score)) if math.isfinite(score) else None
    except (TypeError, ValueError):
        return None


def _item_value(item, name: str, default=0.0):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _aggregate_results(per_question: list[EvalResult]) -> dict:
    metric_names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    aggregate = {
        metric: round(
            sum(_safe_score(getattr(item, metric)) for item in per_question) / len(per_question),
            4,
        ) if per_question else 0.0
        for metric in metric_names
    }
    return {**aggregate, "per_question": per_question}


def _tokens(text: str) -> set[str]:
    stopwords = {
        "và", "là", "của", "cho", "các", "một", "những", "được", "trong",
        "the", "a", "an", "is", "are", "of", "to", "and", "this", "that",
    }
    return {
        token for token in re.findall(r"[\wÀ-ỹ]+", str(text).lower(), flags=re.UNICODE)
        if token not in stopwords and len(token) > 1
    }


def _overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left) if left else 0.0


def _heuristic_evaluation(questions, answers, contexts, ground_truths) -> dict:
    """Deterministic offline approximation used without an LLM/API key."""
    per_question = []
    for question, answer, question_contexts, ground_truth in zip(
        questions, answers, contexts, ground_truths
    ):
        query_tokens = _tokens(question)
        answer_tokens = _tokens(answer)
        truth_tokens = _tokens(ground_truth)
        context_tokens = [_tokens(context) for context in question_contexts]
        joined_context_tokens = set().union(*context_tokens) if context_tokens else set()

        faithfulness = _overlap(answer_tokens, joined_context_tokens)
        # Exact question/answer overlap alone scores valid short answers such
        # as "Tổng Giám đốc" as zero. Reference overlap is a deterministic
        # offline proxy for semantic relevance when RAGAS is unavailable.
        query_answer_overlap = _overlap(query_tokens, answer_tokens)
        reference_answer_overlap = _overlap(truth_tokens, answer_tokens)
        answer_relevancy = (
            0.7 * query_answer_overlap + 0.3 * reference_answer_overlap
        )
        if context_tokens:
            precision_values = [_overlap(query_tokens, item) for item in context_tokens]
            context_precision = sum(value > 0 for value in precision_values) / len(precision_values)
        else:
            context_precision = 0.0
        context_recall = _overlap(truth_tokens, joined_context_tokens)
        per_question.append(EvalResult(
            question=question,
            answer=answer,
            contexts=question_contexts,
            ground_truth=ground_truth,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
        ))
    return _aggregate_results(per_question)


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
