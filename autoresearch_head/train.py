"""Single-file experiment loop for autoresearch on the introspection head."""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from introspect.src.eval_common import load_adapter_from_registry, load_words
from introspect.src.generation import build_chat_prompt
from introspect.src.inject import InjectionSpec, attach_injection, resolve_injection_positions
from introspect.src.io_utils import seed_everything, setup_logging
from introspect.src.prompts import task_a_paper_messages
from introspect.src.training.introspection_head import IntrospectionHead, IntrospectionHeadConfig
from introspect.src.training.train_supervised import prebuild_concept_vectors

ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = ROOT / "autoresearch_head"
ARTIFACTS_DIR = HARNESS_DIR / "artifacts"
BENCHMARK_PATH = ARTIFACTS_DIR / "benchmark.json"
RESULTS_TSV_PATH = HARNESS_DIR / "results.tsv"
RUNS_DIR = HARNESS_DIR / "runs"
LAST_RUN_PATH = ARTIFACTS_DIR / "last_run.json"


@dataclass(slots=True)
class BenchmarkSpec:
    model: str
    seed: int
    concepts: list[str]
    train_trial_questions: list[str]
    eval_trial_questions: list[str]
    default_layers: list[int]
    default_alphas: list[float]
    vector_prompt_template: str
    baseline_sample_size: int
    vector_cache_dir: str


# Agent-editable knobs.
@dataclass(slots=True)
class ExperimentConfig:
    model: str | None = None
    layers: list[int] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)
    epochs: int = 2
    batch_size: int = 4
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    detection_weight: float = 0.3
    concept_weight: float = 0.7
    head_dropout: float = 0.1
    head_intermediate_size: int | None = None
    capture_offset: int = 4
    injection_mode: str = "prefix"
    assistant_marker: str | None = None
    max_length: int = 512
    seed: int | None = None


EXPERIMENT = ExperimentConfig()


@dataclass(slots=True)
class BenchmarkSample:
    prompt: str
    token_positions: list[int | str]
    concept_word: str
    concept_id: int
    layer_idx: int
    alpha: float
    is_injection: bool
    target_response: str


class BenchmarkDataset(Dataset):
    def __init__(self, samples: list[BenchmarkSample], tokenizer, max_length: int) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        prompt_tokens = self.tokenizer(
            sample.prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )
        full_text = sample.prompt + sample.target_response
        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )
        return {
            "input_ids": prompt_tokens["input_ids"].squeeze(0),
            "attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "full_input_ids": full_tokens["input_ids"].squeeze(0),
            "full_attention_mask": full_tokens["attention_mask"].squeeze(0),
            "prompt_length": prompt_tokens["input_ids"].shape[1],
            "concept_id": sample.concept_id,
            "is_injection": 1 if sample.is_injection else 0,
            "layer_idx": sample.layer_idx,
            "alpha": sample.alpha,
            "concept_word": sample.concept_word,
            "token_positions": sample.token_positions,
        }


def collate_samples(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    max_full_len = max(item["full_input_ids"].shape[0] for item in batch)

    def pad_tensor(tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        pad_len = target_len - tensor.shape[0]
        if pad_len <= 0:
            return tensor
        return torch.nn.functional.pad(tensor, (0, pad_len), value=pad_value)

    return {
        "input_ids": torch.stack([pad_tensor(item["input_ids"], max_len, pad_token_id) for item in batch]),
        "attention_mask": torch.stack([pad_tensor(item["attention_mask"], max_len, 0) for item in batch]),
        "full_input_ids": torch.stack(
            [pad_tensor(item["full_input_ids"], max_full_len, pad_token_id) for item in batch]
        ),
        "full_attention_mask": torch.stack(
            [pad_tensor(item["full_attention_mask"], max_full_len, 0) for item in batch]
        ),
        "prompt_length": torch.tensor([item["prompt_length"] for item in batch]),
        "concept_id": torch.tensor([item["concept_id"] for item in batch]),
        "is_injection": torch.tensor([item["is_injection"] for item in batch]),
        "layer_idx": torch.tensor([item["layer_idx"] for item in batch]),
        "alpha": torch.tensor([item["alpha"] for item in batch]),
        "concept_word": [item["concept_word"] for item in batch],
        "token_positions": [item["token_positions"] for item in batch],
    }


class BenchmarkTrainer:
    def __init__(
        self,
        adapter,
        head: IntrospectionHead,
        config: ExperimentConfig,
        concept_vectors: dict[tuple[str, int], torch.Tensor],
        *,
        device: str,
    ) -> None:
        self.adapter = adapter
        self.head = head
        self.config = config
        self.concept_vectors = concept_vectors
        self.device = device
        self.head.to(self.device)
        self.adapter.model.eval()
        for param in self.adapter.model.parameters():
            param.requires_grad = False
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=config.head_lr,
            weight_decay=config.weight_decay,
        )

    def _vector_for(self, concept: str, layer_idx: int) -> torch.Tensor:
        return self.concept_vectors[(concept, layer_idx)].to(self.device)

    def _forward_single(
        self,
        *,
        full_input_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        prompt_length: int,
        concept_word: str,
        layer_idx: int,
        alpha: float,
        is_injection: bool,
        token_positions: list[int | str],
    ) -> torch.Tensor:
        capture_layer = layer_idx + self.config.capture_offset
        n_layers = getattr(self.adapter.model.config, "num_hidden_layers", None)
        if n_layers is not None:
            capture_layer = max(0, min(capture_layer, n_layers - 1))

        captured: dict[str, torch.Tensor] = {}
        injection_handle = None
        if is_injection and alpha > 0:
            spec = InjectionSpec(
                layer_idx=layer_idx,
                alpha=alpha,
                vector=self._vector_for(concept_word, layer_idx),
                token_positions=token_positions,
                apply_to_generated=self.config.injection_mode == "suffix",
            )
            injection_handle = attach_injection(self.adapter, spec)

        def capture_fn(_module, _inputs, output):
            residual = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = residual[0, prompt_length - 1, :]
            return output

        capture_handle = self.adapter.register_residual_hook(capture_layer, capture_fn)
        try:
            with torch.no_grad():
                self.adapter.model(
                    input_ids=full_input_ids.unsqueeze(0).to(self.device),
                    attention_mask=full_attention_mask.unsqueeze(0).to(self.device),
                )
            if "hidden" not in captured:
                raise RuntimeError("Failed to capture hidden state")
            return captured["hidden"]
        finally:
            capture_handle.remove()
            if injection_handle is not None:
                injection_handle.remove()

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.head.train()
        metrics = {
            "loss": 0.0,
            "detection_loss": 0.0,
            "concept_loss": 0.0,
            "detection_acc": 0.0,
            "concept_acc": 0.0,
        }
        n_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            hidden_states: list[torch.Tensor] = []
            for i in range(batch["input_ids"].shape[0]):
                hidden = self._forward_single(
                    full_input_ids=batch["full_input_ids"][i],
                    full_attention_mask=batch["full_attention_mask"][i],
                    prompt_length=int(batch["prompt_length"][i].item()),
                    concept_word=batch["concept_word"][i],
                    layer_idx=int(batch["layer_idx"][i].item()),
                    alpha=float(batch["alpha"][i].item()),
                    is_injection=bool(batch["is_injection"][i].item()),
                    token_positions=batch["token_positions"][i],
                )
                hidden_states.append(hidden)

            hidden_stack = torch.stack(hidden_states, dim=0)
            concept_ids = batch["concept_id"].to(self.device)
            is_injection = batch["is_injection"].to(self.device).float()
            loss_dict = self.head.compute_loss(
                hidden_states=hidden_stack,
                concept_labels=concept_ids,
                is_injection=is_injection,
                detection_weight=self.config.detection_weight,
                concept_weight=self.config.concept_weight,
            )
            loss_dict["loss"].backward()
            self.optimizer.step()

            for key, value in loss_dict.items():
                metrics[key] = metrics.get(key, 0.0) + value.item()
            n_batches += 1

        if n_batches == 0:
            return metrics
        return {key: value / n_batches for key, value in metrics.items()}

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.head.eval()
        detection_preds: list[int] = []
        detection_labels: list[int] = []
        concept_preds: list[int] = []
        concept_labels: list[int] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                hidden_states: list[torch.Tensor] = []
                for i in range(batch["input_ids"].shape[0]):
                    hidden = self._forward_single(
                        full_input_ids=batch["full_input_ids"][i],
                        full_attention_mask=batch["full_attention_mask"][i],
                        prompt_length=int(batch["prompt_length"][i].item()),
                        concept_word=batch["concept_word"][i],
                        layer_idx=int(batch["layer_idx"][i].item()),
                        alpha=float(batch["alpha"][i].item()),
                        is_injection=bool(batch["is_injection"][i].item()),
                        token_positions=batch["token_positions"][i],
                    )
                    hidden_states.append(hidden)

                preds = self.head.predict(torch.stack(hidden_states, dim=0))
                detection_preds.extend(preds["detected"].cpu().int().tolist())
                detection_labels.extend(batch["is_injection"].cpu().int().tolist())
                concept_preds.extend(preds["concept_id"].cpu().int().tolist())
                concept_labels.extend(batch["concept_id"].cpu().int().tolist())

        detection_preds_t = torch.tensor(detection_preds)
        detection_labels_t = torch.tensor(detection_labels)
        concept_preds_t = torch.tensor(concept_preds)
        concept_labels_t = torch.tensor(concept_labels)

        tp = ((detection_preds_t == 1) & (detection_labels_t == 1)).sum().item()
        fp = ((detection_preds_t == 1) & (detection_labels_t == 0)).sum().item()
        tn = ((detection_preds_t == 0) & (detection_labels_t == 0)).sum().item()
        fn = ((detection_preds_t == 0) & (detection_labels_t == 1)).sum().item()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        correct_detections = (detection_preds_t == 1) & (detection_labels_t == 1)
        if correct_detections.any():
            concept_acc = (
                (concept_preds_t[correct_detections] == concept_labels_t[correct_detections])
                .float()
                .mean()
                .item()
            )
        else:
            concept_acc = 0.0

        return {
            "detection_accuracy": (tp + tn) / len(detection_labels_t) if detection_labels_t.numel() else 0.0,
            "tpr": tpr,
            "fpr": fpr,
            "net_score": tpr - fpr,
            "concept_accuracy": concept_acc,
            "n_correct_detections": int(correct_detections.sum().item()),
        }


def load_benchmark() -> BenchmarkSpec:
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Missing benchmark spec at {BENCHMARK_PATH}. Run `python3 autoresearch_head/prepare.py` first."
        )
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    return BenchmarkSpec(**data)


def build_prompt(adapter, *, trial_question: str, config: ExperimentConfig) -> tuple[str, list[int | str]]:
    messages = task_a_paper_messages(trial_question=trial_question)
    prompt, _ = build_chat_prompt(adapter.tokenizer, messages)
    token_positions, _suffix_start = resolve_injection_positions(
        adapter,
        prompt,
        mode=config.injection_mode,
        assistant_marker=config.assistant_marker,
    )
    if config.injection_mode == "suffix":
        return prompt, ["suffix"]
    return prompt, token_positions


def build_samples(
    adapter,
    *,
    trial_questions: list[str],
    concepts: list[str],
    layers: list[int],
    alphas: list[float],
    concept_to_id: dict[str, int],
    config: ExperimentConfig,
) -> list[BenchmarkSample]:
    prompts = [
        build_prompt(adapter, trial_question=trial_question, config=config)
        for trial_question in trial_questions
    ]
    samples: list[BenchmarkSample] = []
    for prompt, token_positions in prompts:
        for concept in concepts:
            concept_id = concept_to_id[concept]
            for layer_idx in layers:
                for alpha in alphas:
                    samples.append(
                        BenchmarkSample(
                            prompt=prompt,
                            token_positions=token_positions,
                            concept_word=concept,
                            concept_id=concept_id,
                            layer_idx=layer_idx,
                            alpha=alpha,
                            is_injection=True,
                            target_response=f"I detect an injected thought. It seems to be about {concept}.",
                        )
                    )
                    samples.append(
                        BenchmarkSample(
                            prompt=prompt,
                            token_positions=token_positions,
                            concept_word=concept,
                            concept_id=concept_id,
                            layer_idx=layer_idx,
                            alpha=0.0,
                            is_injection=False,
                            target_response="I do not detect any injected thought.",
                        )
                    )
    return samples


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def timestamp_slug() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"


def save_checkpoint(path: Path, head: IntrospectionHead, payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    head.save(path / "introspection_head")
    with (path / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def append_results_tsv(summary: dict[str, Any]) -> None:
    RESULTS_TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "run_dir",
        "primary_score",
        "net_score",
        "concept_accuracy",
        "detection_accuracy",
        "epochs",
        "batch_size",
        "layers",
        "alphas",
        "n_concepts",
        "n_train_prompts",
        "n_eval_prompts",
        "wall_seconds",
        "peak_vram_mb",
        "seed",
        "model",
        "git_commit",
    ]
    write_header = not RESULTS_TSV_PATH.exists()
    with RESULTS_TSV_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": summary["timestamp"],
                "run_dir": summary["run_dir"],
                "primary_score": f"{summary['primary_score']:.6f}",
                "net_score": f"{summary['net_score']:.6f}",
                "concept_accuracy": f"{summary['concept_accuracy']:.6f}",
                "detection_accuracy": f"{summary['detection_accuracy']:.6f}",
                "epochs": summary["config"]["epochs"],
                "batch_size": summary["config"]["batch_size"],
                "layers": json.dumps(summary["config"]["layers"]),
                "alphas": json.dumps(summary["config"]["alphas"]),
                "n_concepts": len(summary["benchmark"]["concepts"]),
                "n_train_prompts": len(summary["benchmark"]["train_trial_questions"]),
                "n_eval_prompts": len(summary["benchmark"]["eval_trial_questions"]),
                "wall_seconds": f"{summary['wall_seconds']:.2f}",
                "peak_vram_mb": f"{summary['peak_vram_mb']:.1f}",
                "seed": summary["config"]["seed"],
                "model": summary["config"]["model"],
                "git_commit": summary["git_commit"],
            }
        )


def main() -> None:
    setup_logging()
    benchmark = load_benchmark()
    config = deepcopy(EXPERIMENT)
    if config.model is None:
        config.model = benchmark.model
    if not config.layers:
        config.layers = list(benchmark.default_layers)
    if not config.alphas:
        config.alphas = list(benchmark.default_alphas)
    if config.seed is None:
        config.seed = benchmark.seed

    seed_everything(config.seed)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    loaded = load_adapter_from_registry(config.model, seed=config.seed)
    adapter = loaded.adapter
    device = str(next(adapter.model.parameters()).device)

    if adapter.tokenizer.pad_token_id is None:
        adapter.tokenizer.pad_token = adapter.tokenizer.eos_token

    concepts = list(benchmark.concepts)
    concept_to_id = {concept: idx for idx, concept in enumerate(concepts)}
    baseline_words = list(load_words().iter_baselines())
    concept_vectors = prebuild_concept_vectors(
        adapter,
        concepts,
        config.layers,
        baseline_words=baseline_words,
        prompt_template=benchmark.vector_prompt_template,
        baseline_sample_size=benchmark.baseline_sample_size,
        cache_dir=Path(benchmark.vector_cache_dir),
    )

    train_samples = build_samples(
        adapter,
        trial_questions=list(benchmark.train_trial_questions),
        concepts=concepts,
        layers=config.layers,
        alphas=config.alphas,
        concept_to_id=concept_to_id,
        config=config,
    )
    eval_samples = build_samples(
        adapter,
        trial_questions=list(benchmark.eval_trial_questions),
        concepts=concepts,
        layers=config.layers,
        alphas=config.alphas,
        concept_to_id=concept_to_id,
        config=config,
    )

    train_dataset = BenchmarkDataset(train_samples, adapter.tokenizer, max_length=config.max_length)
    eval_dataset = BenchmarkDataset(eval_samples, adapter.tokenizer, max_length=config.max_length)
    collate = lambda batch: collate_samples(batch, adapter.tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    head = IntrospectionHead(
        IntrospectionHeadConfig(
            hidden_size=adapter.model.config.hidden_size,
            n_concepts=len(concepts),
            intermediate_size=config.head_intermediate_size,
            dropout=config.head_dropout,
        )
    )
    trainer = BenchmarkTrainer(
        adapter,
        head,
        config,
        concept_vectors,
        device=device,
    )

    run_dir = RUNS_DIR / timestamp_slug()
    best_metrics: dict[str, float] | None = None
    best_epoch = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for epoch in range(config.epochs):
        logging.info("Epoch %d/%d", epoch + 1, config.epochs)
        train_metrics = trainer.train_epoch(train_loader)
        eval_metrics = trainer.evaluate(eval_loader)
        logging.info("Train metrics: %s", train_metrics)
        logging.info("Eval metrics: %s", eval_metrics)

        is_best = best_metrics is None or eval_metrics["net_score"] > best_metrics["net_score"]
        if is_best:
            best_epoch = epoch + 1
            best_metrics = dict(eval_metrics)
            save_checkpoint(
                run_dir / "best",
                trainer.head,
                {
                    "epoch": best_epoch,
                    "train_metrics": train_metrics,
                    "eval_metrics": eval_metrics,
                },
            )

    wall_seconds = time.perf_counter() - start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    final_eval = trainer.evaluate(eval_loader)
    save_checkpoint(
        run_dir / "final",
        trainer.head,
        {
            "epoch": config.epochs,
            "eval_metrics": final_eval,
        },
    )

    if best_metrics is None:
        raise RuntimeError("Training did not produce any evaluation metrics")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_dir": str(run_dir),
        "primary_score": best_metrics["net_score"],
        "net_score": best_metrics["net_score"],
        "concept_accuracy": best_metrics["concept_accuracy"],
        "detection_accuracy": best_metrics["detection_accuracy"],
        "best_epoch": best_epoch,
        "wall_seconds": wall_seconds,
        "peak_vram_mb": peak_vram_mb,
        "git_commit": current_git_commit(),
        "config": asdict(config),
        "benchmark": asdict(benchmark),
        "final_eval": final_eval,
    }
    with LAST_RUN_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    append_results_tsv(summary)

    print(f"PRIMARY_SCORE\t{summary['primary_score']:.6f}")
    print(f"NET_SCORE\t{summary['net_score']:.6f}")
    print(f"CONCEPT_ACCURACY\t{summary['concept_accuracy']:.6f}")
    print(f"DETECTION_ACCURACY\t{summary['detection_accuracy']:.6f}")
    print(f"WALL_SECONDS\t{summary['wall_seconds']:.2f}")
    print(f"PEAK_VRAM_MB\t{summary['peak_vram_mb']:.1f}")
    print(f"RUN_DIR\t{summary['run_dir']}")
    print(f"RUN_SUMMARY\t{json.dumps(summary, sort_keys=True, default=str)}")


if __name__ == "__main__":
    main()
