"""Schemas for frozen introspection benchmark specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = PROJECT_ROOT / "benchmark" / "task_a_v0.yaml"


@dataclass(slots=True, frozen=True)
class BenchmarkPromptVariant:
    id: str
    split: str
    trial_question: str


@dataclass(slots=True, frozen=True)
class BenchmarkGenerationConfig:
    max_new_tokens: int
    stop_sequences: tuple[str, ...]
    allowed_formats: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class BenchmarkWeights:
    detection_f1: float
    concept_accuracy: float
    specificity: float
    ablation_resistance: float

    def total(self) -> float:
        return (
            self.detection_f1
            + self.concept_accuracy
            + self.specificity
            + self.ablation_resistance
        )


@dataclass(slots=True, frozen=True)
class BenchmarkSpec:
    name: str
    version: int
    task: str
    description: str
    seed: int
    n_concepts: int
    layers: tuple[int, ...]
    alphas: tuple[float, ...]
    words_file: Path
    cache_dir: Path
    vector_prompt_template: str
    baseline_sample_size: int
    response_instruction: str
    generation: BenchmarkGenerationConfig
    weights: BenchmarkWeights
    prompt_variants: tuple[BenchmarkPromptVariant, ...]
    source_path: Path

    @property
    def benchmark_slug(self) -> str:
        suffix = f"-v{self.version}"
        if self.name.endswith(suffix):
            return self.name
        return f"{self.name}{suffix}"

    @property
    def prompt_splits(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(variant.split for variant in self.prompt_variants))


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_benchmark_spec(path: str | Path = DEFAULT_SPEC_PATH) -> BenchmarkSpec:
    source_path = Path(path).resolve()
    with source_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark spec at {source_path} must be a mapping")

    base_dir = source_path.parent.parent

    generation_raw = raw.get("generation") or {}
    weights_raw = raw.get("weights") or {}
    prompt_variants_raw = raw.get("prompt_variants") or []

    prompt_variants = tuple(
        BenchmarkPromptVariant(
            id=str(item["id"]),
            split=str(item["split"]),
            trial_question=str(item["trial_question"]),
        )
        for item in prompt_variants_raw
    )
    if not prompt_variants:
        raise ValueError("Benchmark spec must define at least one prompt variant")

    spec = BenchmarkSpec(
        name=str(raw["name"]),
        version=int(raw["version"]),
        task=str(raw["task"]),
        description=str(raw.get("description", "")),
        seed=int(raw.get("seed", 42)),
        n_concepts=int(raw["n_concepts"]),
        layers=tuple(int(value) for value in raw.get("layers", [])),
        alphas=tuple(float(value) for value in raw.get("alphas", [])),
        words_file=_resolve_path(raw["words_file"], base_dir=base_dir),
        cache_dir=_resolve_path(raw["cache_dir"], base_dir=base_dir),
        vector_prompt_template=str(raw["vector_prompt_template"]),
        baseline_sample_size=int(raw["baseline_sample_size"]),
        response_instruction=str(raw["response_instruction"]),
        generation=BenchmarkGenerationConfig(
            max_new_tokens=int(generation_raw.get("max_new_tokens", 12)),
            stop_sequences=tuple(str(value) for value in generation_raw.get("stop_sequences", [])),
            allowed_formats=tuple(str(value) for value in generation_raw.get("allowed_formats", [])),
        ),
        weights=BenchmarkWeights(
            detection_f1=float(weights_raw.get("detection_f1", 0.4)),
            concept_accuracy=float(weights_raw.get("concept_accuracy", 0.3)),
            specificity=float(weights_raw.get("specificity", 0.2)),
            ablation_resistance=float(weights_raw.get("ablation_resistance", 0.1)),
        ),
        prompt_variants=prompt_variants,
        source_path=source_path,
    )

    if spec.weights.total() <= 0:
        raise ValueError("Benchmark weights must sum to a positive value")
    if not spec.layers:
        raise ValueError("Benchmark spec must define at least one layer")
    if not spec.alphas:
        raise ValueError("Benchmark spec must define at least one alpha")

    return spec
