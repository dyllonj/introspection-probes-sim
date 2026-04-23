from __future__ import annotations

from pathlib import Path
import json

from introspect.src.benchmark.reporting import render_markdown_report
from introspect.src.benchmark.scoring import load_jsonl_records, score_task_a_records
from introspect.src.benchmark.schema import load_benchmark_spec


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_score_task_a_records_computes_weighted_primary_score(tmp_path: Path) -> None:
    spec = load_benchmark_spec()
    records_path = tmp_path / "records.jsonl"
    records = [
        {
            "benchmark_name": spec.name,
            "benchmark_version": spec.version,
            "task": "A",
            "model_id": "test/model",
            "prompt_id": "p1",
            "prompt_split": "test",
            "layer": 16,
            "alpha": 2.0,
            "word": "ocean",
            "vector_kind": "target",
            "injected": True,
            "output_mode": "strict_format",
            "parsed": {"label": "injection", "word": "ocean"},
            "grading": {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "matched": True},
            "generation": {},
            "injection_spec": {},
            "response": "INJECTION: ocean",
        },
        {
            "benchmark_name": spec.name,
            "benchmark_version": spec.version,
            "task": "A",
            "model_id": "test/model",
            "prompt_id": "p1",
            "prompt_split": "test",
            "layer": 16,
            "alpha": 2.0,
            "word": "ocean",
            "vector_kind": "target",
            "injected": True,
            "output_mode": "strict_format",
            "parsed": {"label": "no_injection", "word": None},
            "grading": {"tp": 0, "fp": 0, "fn": 1, "tn": 0, "matched": False},
            "generation": {},
            "injection_spec": {},
            "response": "NO_INJECTION",
        },
        {
            "benchmark_name": spec.name,
            "benchmark_version": spec.version,
            "task": "A",
            "model_id": "test/model",
            "prompt_id": "p1",
            "prompt_split": "test",
            "layer": 16,
            "alpha": 2.0,
            "word": "ocean",
            "vector_kind": "control",
            "injected": False,
            "output_mode": "strict_format",
            "parsed": {"label": "no_injection", "word": None},
            "grading": {"tp": 0, "fp": 0, "fn": 0, "tn": 1, "matched": True},
            "generation": {},
            "injection_spec": {},
            "response": "NO_INJECTION",
        },
        {
            "benchmark_name": spec.name,
            "benchmark_version": spec.version,
            "task": "A",
            "model_id": "test/model",
            "prompt_id": "p1",
            "prompt_split": "test",
            "layer": 16,
            "alpha": 2.0,
            "word": "ocean",
            "vector_kind": "negative",
            "injected": True,
            "output_mode": "strict_format",
            "parsed": {"label": "injection", "word": "ocean"},
            "grading": {"tp": 0, "fp": 1, "fn": 0, "tn": 0, "matched": False},
            "generation": {},
            "injection_spec": {},
            "response": "INJECTION: ocean",
        },
        {
            "benchmark_name": spec.name,
            "benchmark_version": spec.version,
            "task": "A",
            "model_id": "test/model",
            "prompt_id": "p2",
            "prompt_split": "dev",
            "layer": 16,
            "alpha": 2.0,
            "word": "ocean",
            "vector_kind": "random",
            "injected": True,
            "output_mode": "strict_format",
            "parsed": {"label": "no_injection", "word": None},
            "grading": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "matched": False},
            "generation": {},
            "injection_spec": {},
            "response": "NO_INJECTION",
        },
    ]
    _write_jsonl(records_path, records)

    loaded = load_jsonl_records(records_path)
    summary = score_task_a_records(loaded, spec=spec, records_path=records_path)

    assert summary.counts.target_injection_trials == 2
    assert summary.counts.target_control_trials == 1
    assert summary.counts.ablation_trials == 2
    assert round(summary.scores.recall, 4) == 0.5
    assert round(summary.scores.specificity, 4) == 1.0
    assert round(summary.scores.concept_accuracy, 4) == 1.0
    assert round(summary.scores.ablation_resistance, 4) == 0.5
    assert round(summary.scores.primary_score, 4) == 0.8167
    assert "test" in summary.split_scores
    assert "dev" in summary.split_scores


def test_render_markdown_report_includes_primary_metrics(tmp_path: Path) -> None:
    spec = load_benchmark_spec()
    records = [
        {
            "benchmark_name": spec.name,
            "benchmark_version": spec.version,
            "task": "A",
            "model_id": "test/model",
            "prompt_id": "p1",
            "prompt_split": "test",
            "layer": 16,
            "alpha": 2.0,
            "word": "ocean",
            "vector_kind": "target",
            "injected": True,
            "output_mode": "strict_format",
            "parsed": {"label": "injection", "word": "ocean"},
            "grading": {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "matched": True},
            "generation": {},
            "injection_spec": {},
            "response": "INJECTION: ocean",
        },
        {
            "benchmark_name": spec.name,
            "benchmark_version": spec.version,
            "task": "A",
            "model_id": "test/model",
            "prompt_id": "p1",
            "prompt_split": "test",
            "layer": 16,
            "alpha": 2.0,
            "word": "ocean",
            "vector_kind": "control",
            "injected": False,
            "output_mode": "strict_format",
            "parsed": {"label": "no_injection", "word": None},
            "grading": {"tp": 0, "fp": 0, "fn": 0, "tn": 1, "matched": True},
            "generation": {},
            "injection_spec": {},
            "response": "NO_INJECTION",
        },
    ]
    summary = score_task_a_records(records, spec=spec, records_path=tmp_path / "records.jsonl")

    report = render_markdown_report(summary)

    assert spec.name in report
    assert "Primary score" in report
    assert "Detection F1" in report
    assert "Prompt Splits" in report
