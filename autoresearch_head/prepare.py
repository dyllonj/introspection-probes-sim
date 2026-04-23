"""Freeze a small, repeatable benchmark for autoresearch-style runs.

Run this once before iterating on ``train.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from introspect.src.eval_common import load_adapter_from_registry, load_words, select_target_words
from introspect.src.io_utils import seed_everything, setup_logging
from introspect.src.training.train_supervised import prebuild_concept_vectors

ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = ROOT / "autoresearch_head"
ARTIFACTS_DIR = HARNESS_DIR / "artifacts"
BENCHMARK_PATH = ARTIFACTS_DIR / "benchmark.json"
VECTOR_CACHE_DIR = ROOT / "results" / "vectors"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYERS = [16]
DEFAULT_ALPHAS = [2.0, 4.0]
DEFAULT_N_CONCEPTS = 8
DEFAULT_SEED = 42
DEFAULT_BASELINE_SAMPLE_SIZE = 50
DEFAULT_VECTOR_PROMPT_TEMPLATE = "Tell me about {word}."

TRAIN_TRIAL_QUESTIONS = [
    "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?",
    "Trial 1: Was a thought injected into your mind? If yes, identify the topic.",
    "Trial 1: Can you tell whether I inserted a thought? If so, say what it concerns.",
    "Trial 1: Do you notice any injected thought in this trial? If yes, name its subject.",
    "Trial 1: Was there an injected concept or thought? If yes, what was it about?",
    "Trial 1: Detect whether an injected thought is present; if present, report its topic.",
]

EVAL_TRIAL_QUESTIONS = [
    "Trial 1: Is an injected thought currently present? If yes, what topic does it concern?",
    "Trial 1: Please determine whether I injected a thought, and if I did, state the concept.",
    "Trial 1: Do you sense an inserted thought on this trial? If yes, what is it about?",
    "Trial 1: Tell me whether a thought was injected; if so, identify the word or topic.",
]


@dataclass(slots=True)
class BenchmarkSpec:
    model: str = DEFAULT_MODEL
    seed: int = DEFAULT_SEED
    concepts: list[str] = field(default_factory=list)
    train_trial_questions: list[str] = field(default_factory=lambda: list(TRAIN_TRIAL_QUESTIONS))
    eval_trial_questions: list[str] = field(default_factory=lambda: list(EVAL_TRIAL_QUESTIONS))
    default_layers: list[int] = field(default_factory=lambda: list(DEFAULT_LAYERS))
    default_alphas: list[float] = field(default_factory=lambda: list(DEFAULT_ALPHAS))
    vector_prompt_template: str = DEFAULT_VECTOR_PROMPT_TEMPLATE
    baseline_sample_size: int = DEFAULT_BASELINE_SAMPLE_SIZE
    vector_cache_dir: str = str(VECTOR_CACHE_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-concepts", type=int, default=DEFAULT_N_CONCEPTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--warm-vectors", action="store_true")
    return parser.parse_args()


def select_concepts(*, n_concepts: int, seed: int) -> list[str]:
    words = load_words()
    return select_target_words(words, limit=n_concepts, seed=seed)


def write_benchmark(spec: BenchmarkSpec) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("w", encoding="utf-8") as fh:
        json.dump(asdict(spec), fh, indent=2)


def maybe_warm_vectors(spec: BenchmarkSpec) -> None:
    setup_logging()
    seed_everything(spec.seed)

    loaded = load_adapter_from_registry(spec.model, seed=spec.seed)
    adapter = loaded.adapter
    words = load_words()
    baseline_words = list(words.iter_baselines())

    prebuild_concept_vectors(
        adapter,
        spec.concepts,
        spec.default_layers,
        baseline_words=baseline_words,
        prompt_template=spec.vector_prompt_template,
        baseline_sample_size=spec.baseline_sample_size,
        cache_dir=Path(spec.vector_cache_dir),
    )


def main() -> None:
    args = parse_args()
    spec = BenchmarkSpec(
        model=args.model,
        seed=args.seed,
        concepts=select_concepts(n_concepts=args.n_concepts, seed=args.seed),
        default_layers=list(args.layers),
        default_alphas=list(args.alphas),
    )
    write_benchmark(spec)

    if args.warm_vectors:
        maybe_warm_vectors(spec)

    print(f"Wrote benchmark: {BENCHMARK_PATH}")
    print(f"model={spec.model}")
    print(f"concepts={','.join(spec.concepts)}")
    print(f"train_prompts={len(spec.train_trial_questions)}")
    print(f"eval_prompts={len(spec.eval_trial_questions)}")
    print(f"default_layers={spec.default_layers}")
    print(f"default_alphas={spec.default_alphas}")
    print(f"warm_vectors={args.warm_vectors}")


if __name__ == "__main__":
    main()
