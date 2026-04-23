"""Run and score a frozen Task A introspection benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import random
from pathlib import Path
from typing import Any, Sequence

import torch

from ..eval_common import ensure_vector, load_adapter_from_registry, load_words, select_target_words
from ..generation import build_chat_prompt
from ..grading import grade_injection_detection, parse_injection_report
from ..inject import DEFAULT_GENERATION_KWARGS, InjectionSpec, inject_once, resolve_injection_positions
from ..io_utils import JsonlWriter, gather_runtime_metadata, seed_everything, setup_logging, truncate_text
from ..plotting import model_id_to_slug
from ..prompts import task_a_paper_messages
from .reporting import render_markdown_report
from .scoring import load_jsonl_records, score_task_a_records
from .schema import DEFAULT_SPEC_PATH, BenchmarkPromptVariant, BenchmarkSpec, load_benchmark_spec

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model identifier to benchmark")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH, help="Benchmark YAML spec")
    parser.add_argument("--adapter", help="Override adapter class name")
    parser.add_argument("--dtype", help="Preferred torch dtype (bf16/fp16/fp32)")
    parser.add_argument("--device-map", help="Device map forwarded to transformers (auto/cpu)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/benchmark"),
        help="Directory where benchmark artifacts are written",
    )
    parser.add_argument("--seed", type=int, help="Override benchmark seed")
    return parser


def _vector_variants(
    *,
    base_spec: InjectionSpec,
    concept_vector: torch.Tensor,
) -> list[tuple[str, InjectionSpec, bool]]:
    negative_spec = InjectionSpec(
        layer_idx=base_spec.layer_idx,
        alpha=base_spec.alpha,
        vector=-concept_vector,
        token_positions=base_spec.token_positions,
        apply_on_input=base_spec.apply_on_input,
        apply_to_generated=base_spec.apply_to_generated,
    )
    random_vec = torch.randn_like(concept_vector, dtype=torch.float32)
    random_vec = random_vec / torch.linalg.vector_norm(random_vec)
    random_spec = InjectionSpec(
        layer_idx=base_spec.layer_idx,
        alpha=base_spec.alpha,
        vector=random_vec,
        token_positions=base_spec.token_positions,
        apply_on_input=base_spec.apply_on_input,
        apply_to_generated=base_spec.apply_to_generated,
    )
    return [
        ("target", base_spec, True),
        ("negative", negative_spec, True),
        ("random", random_spec, True),
        ("control", base_spec, False),
    ]


def _build_messages(prompt_variant: BenchmarkPromptVariant, *, spec: BenchmarkSpec) -> list[dict[str, str]]:
    messages = task_a_paper_messages(trial_question=prompt_variant.trial_question)
    if len(messages) < 2:
        raise ValueError("Task A prompt must contain a final user turn and assistant turn")
    messages = [dict(message) for message in messages]
    messages[-2]["content"] = (
        f"{prompt_variant.trial_question}\n\n{spec.response_instruction}"
    )
    return messages


def _build_prompt(
    adapter,
    *,
    prompt_variant: BenchmarkPromptVariant,
    spec: BenchmarkSpec,
) -> tuple[str, list[int | str], str]:
    messages = _build_messages(prompt_variant, spec=spec)
    prompt, _helper_stops = build_chat_prompt(adapter.tokenizer, messages)
    token_positions, _suffix_start = resolve_injection_positions(
        adapter,
        prompt,
        mode="prefix",
    )
    positions = [*token_positions, "suffix"]
    return prompt, positions, messages[-2]["content"]


def _generation_kwargs(spec: BenchmarkSpec) -> dict[str, Any]:
    kwargs = dict(DEFAULT_GENERATION_KWARGS)
    kwargs["max_new_tokens"] = spec.generation.max_new_tokens
    kwargs["stop_sequences"] = spec.generation.stop_sequences
    kwargs["allowed_formats"] = spec.generation.allowed_formats
    return kwargs


def _record_schema() -> dict[str, type | tuple[type, ...]]:
    return {
        "benchmark_name": str,
        "benchmark_version": int,
        "task": str,
        "model_id": str,
        "prompt_id": str,
        "prompt_split": str,
        "layer": int,
        "alpha": float,
        "word": str,
        "vector_kind": str,
        "injected": bool,
        "response": str,
        "parsed": dict,
        "grading": dict,
        "generation": dict,
        "injection_spec": dict,
    }


def _artifact_paths(output_root: Path, *, model_id: str, spec: BenchmarkSpec) -> dict[str, Path]:
    model_slug = model_id_to_slug(model_id)
    root = output_root / model_slug / spec.benchmark_slug
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "records": root / "task_A_benchmark.jsonl",
        "payload": root / "payload.json",
        "report": root / "report.md",
    }


def run_benchmark(
    *,
    model_id: str,
    spec: BenchmarkSpec,
    adapter_name: str | None,
    dtype: str | None,
    device_map: str | None,
    output_root: Path,
    seed: int,
) -> dict[str, Path]:
    setup_logging()
    seed_everything(seed)

    adapter_loaded = load_adapter_from_registry(
        model_id,
        adapter_name=adapter_name,
        dtype=dtype,
        device_map=device_map,
        seed=seed,
    )
    adapter = adapter_loaded.adapter

    paths = _artifact_paths(output_root, model_id=model_id, spec=spec)
    metadata = gather_runtime_metadata(
        extra={
            "benchmark_name": spec.name,
            "benchmark_version": spec.version,
            "task": spec.task,
            "model_id": model_id,
            "seed": seed,
        }
    )

    word_set = load_words(spec.words_file)
    targets = select_target_words(word_set, limit=spec.n_concepts, seed=seed)
    baseline_words = list(word_set.iter_baselines())
    rng = random.Random(seed)
    generation_kwargs = _generation_kwargs(spec)

    with JsonlWriter(
        paths["records"],
        append=False,
        metadata=metadata,
        schema=_record_schema(),
    ) as writer:
        for prompt_variant in spec.prompt_variants:
            prompt, token_positions, full_trial_question = _build_prompt(
                adapter,
                prompt_variant=prompt_variant,
                spec=spec,
            )

            for layer_idx in spec.layers:
                for concept in targets:
                    vector = ensure_vector(
                        adapter=adapter,
                        model_id=model_id,
                        layer_idx=layer_idx,
                        word=concept,
                        cache_dir=spec.cache_dir,
                        baseline_words=baseline_words,
                        prompt_template=spec.vector_prompt_template,
                        baseline_sample_size=spec.baseline_sample_size,
                        rng=rng,
                    )
                    for alpha in spec.alphas:
                        base_spec = InjectionSpec(
                            layer_idx=layer_idx,
                            alpha=alpha,
                            vector=vector,
                            token_positions=token_positions,
                            apply_to_generated=True,
                        )
                        for vector_kind, variant_spec, injected in _vector_variants(
                            base_spec=base_spec,
                            concept_vector=vector,
                        ):
                            result = inject_once(
                                adapter,
                                prompt,
                                variant_spec,
                                gen_kwargs=generation_kwargs,
                                enable_injection=injected,
                            )
                            parsed = parse_injection_report(result.text)
                            expected_word = concept if vector_kind == "target" else None
                            grading = grade_injection_detection(
                                expected_word=expected_word,
                                report=parsed,
                            )
                            record = {
                                "benchmark_name": spec.name,
                                "benchmark_version": spec.version,
                                "task": spec.task,
                                "model_id": model_id,
                                "adapter": adapter_loaded.adapter_name,
                                "prompt_id": prompt_variant.id,
                                "prompt_split": prompt_variant.split,
                                "trial_question": prompt_variant.trial_question,
                                "full_trial_question": full_trial_question,
                                "layer": int(layer_idx),
                                "alpha": float(alpha),
                                "word": concept,
                                "vector_kind": vector_kind,
                                "injected": bool(injected),
                                "prompt": truncate_text(prompt),
                                "response": result.text,
                                "parsed": asdict(parsed),
                                "grading": grading,
                                "output_mode": "strict_format",
                                "generation": dict(result.generation),
                                "injection_spec": dict(result.injection_spec),
                            }
                            writer.write(record)

    records = load_jsonl_records(paths["records"])
    summary = score_task_a_records(records, spec=spec, records_path=paths["records"])
    with paths["payload"].open("w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, indent=2, default=str)
    paths["report"].write_text(render_markdown_report(summary), encoding="utf-8")

    LOGGER.info("Benchmark records written to %s", paths["records"])
    LOGGER.info("Benchmark payload written to %s", paths["payload"])
    LOGGER.info("Benchmark report written to %s", paths["report"])
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    spec = load_benchmark_spec(args.spec)
    seed = args.seed if args.seed is not None else spec.seed
    run_benchmark(
        model_id=args.model,
        spec=spec,
        adapter_name=args.adapter,
        dtype=args.dtype,
        device_map=args.device_map,
        output_root=args.output_root,
        seed=seed,
    )


if __name__ == "__main__":
    main()
