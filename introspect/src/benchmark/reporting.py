"""Template-based reporting for deterministic introspection benchmark payloads."""

from __future__ import annotations

from .scoring import BenchmarkSummary


def render_markdown_report(summary: BenchmarkSummary) -> str:
    scores = summary.scores
    counts = summary.counts

    lines = [
        f"# {summary.benchmark_name}",
        "",
        f"- Model: `{summary.model_id}`",
        f"- Benchmark version: `{summary.benchmark_version}`",
        f"- Output mode: `{summary.output_mode}`",
        f"- Records: `{summary.records_path}`",
        "",
        "## Summary",
        "",
        f"- Primary score: `{scores.primary_score:.4f}`",
        f"- Detection F1: `{scores.detection_f1:.4f}`",
        f"- Concept accuracy: `{scores.concept_accuracy:.4f}`",
        f"- Specificity: `{scores.specificity:.4f}`",
        f"- Ablation resistance: `{scores.ablation_resistance:.4f}`",
        f"- Net score: `{scores.net_score:.4f}`",
        f"- Format precision: `{scores.format_precision:.4f}`",
        "",
        "## Counts",
        "",
        f"- Target injection trials: `{counts.target_injection_trials}`",
        f"- Target control trials: `{counts.target_control_trials}`",
        f"- Ablation trials: `{counts.ablation_trials}`",
        f"- True positives: `{counts.true_positives}`",
        f"- False negatives: `{counts.false_negatives}`",
        f"- True negatives: `{counts.true_negatives}`",
        f"- False positives: `{counts.false_positives}`",
        f"- Correct identifications: `{counts.correct_identifications}`",
        "",
        "## Prompt Splits",
        "",
    ]

    if summary.split_scores:
        for split, split_scores in summary.split_scores.items():
            lines.append(
                f"- `{split}`: primary=`{split_scores.primary_score:.4f}`, "
                f"detection_f1=`{split_scores.detection_f1:.4f}`, "
                f"concept_accuracy=`{split_scores.concept_accuracy:.4f}`, "
                f"specificity=`{split_scores.specificity:.4f}`"
            )
    else:
        lines.append("- No prompt split metrics available.")

    lines.extend(
        [
            "",
            "## Ablations",
            "",
        ]
    )
    if summary.ablations:
        for name, value in sorted(summary.ablations.items()):
            lines.append(f"- `{name}` resistance: `{value:.4f}`")
    else:
        lines.append("- No ablation metrics available.")

    return "\n".join(lines) + "\n"
