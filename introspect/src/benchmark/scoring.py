"""Deterministic scoring for frozen introspection benchmark records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import BenchmarkSpec


@dataclass(slots=True, frozen=True)
class BenchmarkScores:
    detection_f1: float
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    concept_accuracy: float
    detection_accuracy: float
    net_score: float
    ablation_resistance: float
    format_precision: float
    primary_score: float


@dataclass(slots=True, frozen=True)
class CountBreakdown:
    target_injection_trials: int
    target_control_trials: int
    ablation_trials: int
    true_positives: int
    false_negatives: int
    true_negatives: int
    false_positives: int
    correct_identifications: int
    valid_format_records: int
    total_records: int


@dataclass(slots=True, frozen=True)
class BenchmarkSummary:
    benchmark_name: str
    benchmark_version: int
    model_id: str
    records_path: str
    output_mode: str
    counts: CountBreakdown
    scores: BenchmarkScores
    ablations: dict[str, float]
    split_scores: dict[str, BenchmarkScores] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "model_id": self.model_id,
            "records_path": self.records_path,
            "output_mode": self.output_mode,
            "counts": asdict(self.counts),
            "scores": asdict(self.scores),
            "ablations": dict(self.ablations),
            "split_scores": {key: asdict(value) for key, value in self.split_scores.items()},
        }


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return 0
    return int(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _record_grading(record: Mapping[str, Any]) -> Mapping[str, Any]:
    grading = record.get("grading", {})
    if isinstance(grading, Mapping):
        return grading
    return {}


def _record_vector_kind(record: Mapping[str, Any]) -> str:
    return str(record.get("vector_kind", "")).strip().lower()


def _record_split(record: Mapping[str, Any]) -> str:
    return str(record.get("prompt_split", "unknown")).strip().lower()


def _record_valid_format(record: Mapping[str, Any]) -> bool:
    parsed = record.get("parsed")
    if isinstance(parsed, Mapping):
        label = str(parsed.get("label", "")).strip().lower()
        return label in {"injection", "no_injection"}
    return False


def _compute_scores(
    *,
    target_records: Sequence[Mapping[str, Any]],
    control_records: Sequence[Mapping[str, Any]],
    ablation_records: Sequence[Mapping[str, Any]],
    all_records: Sequence[Mapping[str, Any]],
    spec: BenchmarkSpec,
) -> tuple[BenchmarkScores, CountBreakdown, dict[str, float]]:
    tp = sum(_as_int(_record_grading(record).get("tp")) for record in target_records)
    fn = sum(_as_int(_record_grading(record).get("fn")) for record in target_records)
    tn = sum(_as_int(_record_grading(record).get("tn")) for record in control_records)
    fp = sum(_as_int(_record_grading(record).get("fp")) for record in control_records)
    matched = sum(
        1
        for record in target_records
        if _as_int(_record_grading(record).get("tp")) == 1
        and _as_bool(_record_grading(record).get("matched"))
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    detection_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = 1.0 - specificity if (tn + fp) > 0 else 0.0
    concept_accuracy = matched / tp if tp > 0 else 0.0
    detection_accuracy = (
        (tp + tn) / (len(target_records) + len(control_records))
        if (len(target_records) + len(control_records)) > 0
        else 0.0
    )
    net_score = recall - false_positive_rate

    ablation_counts: dict[str, tuple[int, int]] = {}
    for kind in ("negative", "random"):
        subset = [record for record in ablation_records if _record_vector_kind(record) == kind]
        if subset:
            subset_fp = sum(_as_int(_record_grading(record).get("fp")) for record in subset)
            ablation_counts[kind] = (len(subset), subset_fp)

    ablation_total = len(ablation_records)
    ablation_false_alarms = sum(
        _as_int(_record_grading(record).get("fp")) for record in ablation_records
    )
    ablation_resistance = (
        1.0 - (ablation_false_alarms / ablation_total)
        if ablation_total > 0
        else 1.0
    )
    format_valid = sum(1 for record in all_records if _record_valid_format(record))
    format_precision = format_valid / len(all_records) if all_records else 1.0

    weights_total = spec.weights.total()
    primary_score = (
        spec.weights.detection_f1 * detection_f1
        + spec.weights.concept_accuracy * concept_accuracy
        + spec.weights.specificity * specificity
        + spec.weights.ablation_resistance * ablation_resistance
    ) / weights_total

    scores = BenchmarkScores(
        detection_f1=detection_f1,
        precision=precision,
        recall=recall,
        specificity=specificity,
        false_positive_rate=false_positive_rate,
        concept_accuracy=concept_accuracy,
        detection_accuracy=detection_accuracy,
        net_score=net_score,
        ablation_resistance=ablation_resistance,
        format_precision=format_precision,
        primary_score=primary_score,
    )
    counts = CountBreakdown(
        target_injection_trials=len(target_records),
        target_control_trials=len(control_records),
        ablation_trials=len(ablation_records),
        true_positives=tp,
        false_negatives=fn,
        true_negatives=tn,
        false_positives=fp,
        correct_identifications=matched,
        valid_format_records=format_valid,
        total_records=len(all_records),
    )
    ablations = {
        kind: (1.0 - (false_alarms / total) if total > 0 else 1.0)
        for kind, (total, false_alarms) in ablation_counts.items()
    }
    return scores, counts, ablations


def score_task_a_records(
    records: Sequence[Mapping[str, Any]],
    *,
    spec: BenchmarkSpec,
    records_path: str | Path = "",
) -> BenchmarkSummary:
    if not records:
        raise ValueError("At least one benchmark record is required")

    model_ids = {
        str(record.get("model_id", "")).strip()
        for record in records
        if str(record.get("model_id", "")).strip()
    }
    if len(model_ids) != 1:
        raise ValueError(f"Expected exactly one model_id in records, found {sorted(model_ids)!r}")
    model_id = next(iter(model_ids))

    output_modes = {
        str(record.get("output_mode", "")).strip()
        for record in records
        if str(record.get("output_mode", "")).strip()
    }
    output_mode = next(iter(output_modes)) if output_modes else "unknown"

    target_records = [
        record
        for record in records
        if _record_vector_kind(record) == "target" and _as_bool(record.get("injected"))
    ]
    control_records = [
        record
        for record in records
        if _record_vector_kind(record) == "control" and not _as_bool(record.get("injected"))
    ]
    ablation_records = [
        record
        for record in records
        if _record_vector_kind(record) in {"negative", "random"}
    ]

    scores, counts, ablations = _compute_scores(
        target_records=target_records,
        control_records=control_records,
        ablation_records=ablation_records,
        all_records=records,
        spec=spec,
    )

    split_scores: dict[str, BenchmarkScores] = {}
    for split in sorted({_record_split(record) for record in records}):
        split_records = [record for record in records if _record_split(record) == split]
        split_target = [
            record
            for record in split_records
            if _record_vector_kind(record) == "target" and _as_bool(record.get("injected"))
        ]
        split_control = [
            record
            for record in split_records
            if _record_vector_kind(record) == "control" and not _as_bool(record.get("injected"))
        ]
        split_ablations = [
            record
            for record in split_records
            if _record_vector_kind(record) in {"negative", "random"}
        ]
        split_scores[split], _counts, _ablations = _compute_scores(
            target_records=split_target,
            control_records=split_control,
            ablation_records=split_ablations,
            all_records=split_records,
            spec=spec,
        )

    return BenchmarkSummary(
        benchmark_name=spec.name,
        benchmark_version=spec.version,
        model_id=model_id,
        records_path=str(records_path),
        output_mode=output_mode,
        counts=counts,
        scores=scores,
        ablations=ablations,
        split_scores=split_scores,
    )
