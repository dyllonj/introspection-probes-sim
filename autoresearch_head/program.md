# Autoresearch Program For `introspection-research`

Goal: maximize the `PRIMARY_SCORE` printed by:

```bash
python3 autoresearch_head/train.py
```

`PRIMARY_SCORE` is the fixed benchmark's `net_score = tpr - fpr` for a head-only introspection model on held-out prompt paraphrases. `CONCEPT_ACCURACY` is a guardrail and should not regress materially while improving `PRIMARY_SCORE`.

## Files

- Edit only `autoresearch_head/train.py`.
- Do not edit `autoresearch_head/prepare.py`.
- Do not edit `autoresearch_head/artifacts/benchmark.json` after it has been created.
- Treat `autoresearch_head/results.tsv` and `autoresearch_head/runs/` as disposable run artifacts.

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[train]
python3 autoresearch_head/prepare.py
```

Optional cache warm-up:

```bash
python3 autoresearch_head/prepare.py --warm-vectors
```

## Benchmark

- Fixed concept set: created by `prepare.py`.
- Fixed train prompts: stored in `benchmark.json`.
- Fixed eval prompts: stored in `benchmark.json`.
- Default model: `Qwen/Qwen2.5-7B-Instruct`.
- Default search surface: head-only training only.

## Search Policy

- Keep changes local to `train.py`.
- Prefer small, testable changes.
- Use `PRIMARY_SCORE` as the main objective.
- Track `CONCEPT_ACCURACY`, `DETECTION_ACCURACY`, `WALL_SECONDS`, and `PEAK_VRAM_MB`.
- Record every run by inspecting `autoresearch_head/results.tsv`.
- If a change increases `PRIMARY_SCORE` by overfitting runtime or collapsing `CONCEPT_ACCURACY`, reject it.

## Good First Moves

- Tune `layers` and `alphas`.
- Tune `epochs`, `batch_size`, `head_lr`, and loss weights.
- Adjust the head width or dropout.
- Improve the prompt mix or sampling logic inside `train.py` without changing the frozen eval prompts.

## Avoid

- Editing the core package unless explicitly asked.
- Expanding to LoRA or joint training in this loop.
- Optimizing against ad hoc prompts outside the fixed benchmark.
