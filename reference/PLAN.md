# Laguna-XS-2.1 Fine-Tune + Prune → Local Mac Deployment

## Context

The user wants to fine-tune poolside's **Laguna-XS-2.1** for **agentic coding / SWE work** using high-quality reasoning-trace datasets, apply **pruning**, and run the result **locally on a 24GB Mac via MLX**. Budget: **300 Google Colab compute units** (expandable).

Critical facts established during research:
- The linked repo (`mlx-community/Laguna-XS-2.1-4bit`) is an **Apple-Silicon inference conversion** — MLX can't run/train on Colab's NVIDIA GPUs. All Colab work uses the original **`poolside/Laguna-XS-2.1`** (Transformers v5.7+, also supported by vLLM/SGLang/llama.cpp).
- The model is a **33B-total / 3B-active MoE** (256 routed experts + 1 shared per layer, 10 global + 30 sliding-window attention layers, 262K ctx, native thinking + tool calling, OpenMDW-1.1 license — fine-tuning & redistribution allowed).
- The existing 4-bit build is **18GB** — it barely fits a 24GB Mac (macOS caps GPU-wired memory ~16–17GB). **Pruning is mandatory**, not optional: target **≤ ~12GB final artifact** (≈ 40–50% expert reduction → ~18–20B total params) so there's headroom for KV cache.
- Pruning order: **prune first, then QLoRA fine-tune** — the SFT run doubles as "healing" for pruning damage, and training the smaller model is cheaper.
- mlx-lm laguna support is still in open PRs ([mlx-lm#1378](https://github.com/ml-explore/mlx-lm/issues/1378)); workarounds: mlx-vlm (≥0.6.3) or community PR branches. **GGUF/llama.cpp is the reliable fallback** (official day-one support).

## Pipeline overview

```
poolside/Laguna-XS-2.1 (BF16, 66GB)
   │  Phase 2: router-stats calibration → expert pruning (REAP + baselines, 25%/50%)
   ▼
Pruned MoE (~18–25B, BF16 on HF Hub)
   │  Phase 4: QLoRA SFT on agentic/SWE + code-reasoning data (heals + specializes)
   ▼
Merged fine-tuned model
   │  Phase 6: quantize + convert
   ▼
MLX 4-bit (primary ask) + GGUF Q4_K_M (reliable fallback) → 24GB Mac
```

## Phase 0 — Setup + feasibility smoke test (~10 units)

**Do this before spending real budget — it de-risks everything.**

1. Local repo scaffold in `LagunaFineTune/`: `notebooks/` (numbered Colab notebooks), `src/` (shared Python utils synced to Colab via git or Drive), `README.md`.
2. Accounts/plumbing: HF token in Colab Secrets; **private HF Hub repos as the artifact store** (checkpoints are 20–40GB — Google Drive free tier can't hold them; Hub storage is free). Optional: W&B for loss curves.
3. **Smoke test on A100 40GB** (`notebooks/01_smoke_test.ipynb`):
   - Load `poolside/Laguna-XS-2.1` with `bitsandbytes` NF4 4-bit + Transformers ≥5.7; generate with the chat template (verify thinking tags render).
   - Attach a tiny LoRA (attention projections only) via PEFT/TRL and run ~10 training steps on dummy data.
   - **Check whether Unsloth/Axolotl have added laguna support since Jan 2026** (knowledge cutoff — it's now July 2026, likely improved). If Unsloth supports it, use it everywhere (2–5× faster, ~70% less VRAM → big budget savings).
   - Fallback if bnb QLoRA fails on this MoE: LoRA in BF16 on the *pruned* model (~18B BF16 = 36GB fits A100-40 with gradient checkpointing at seq ≤4K), or Colab's A100 80GB tier if available.

## Phase 1 — Baseline evaluation (~20 units)

`notebooks/02_baseline_eval.ipynb`, A100 + vLLM using poolside's own `Laguna-XS-2.1-INT4`:
- **HumanEval+ and MBPP+** via EvalPlus (cheap, standardized).
- **10–25 SWE-bench-Verified instances** via `mini-swe-agent` (agentic signal; full SWE-bench is out of budget).
- Record tokens/sec and thinking-length stats. These numbers are the yardstick for every later stage.

## Phase 2 — Pruning study (~60 units: ~4h A100 + CPU-runtime surgery + quick evals)

`notebooks/03_calibration_pruning.ipynb`. This is the "study the methods" deliverable:

1. **Calibration**: run the 4-bit model over ~0.5–1M tokens of agentic-coding calibration data with forward hooks collecting per-layer router statistics (expert activation frequency × routing weight × expert output norm).
2. **Methods compared** (same calibration data, same eval):
   - **REAP** (router-weighted expert activation pruning, Cerebras — check github.com/cerebras/reap for reusable code; it was designed for exactly this class of coding MoEs) — primary method.
   - **Frequency-based expert dropping** — simple baseline.
   - *(Stretch, only if budget allows)* expert merging, or layer pruning of redundant SWA layers via angular distance.
3. **Weight surgery on CPU high-RAM runtime** (cheap — no GPU needed): load BF16 shards, delete pruned experts, rewrite router weights/config (`num_routed_experts` only — keeping the config standard is what keeps vLLM/llama.cpp/MLX converters working). Produce **25% and 50% pruned variants**, push to Hub.
4. **Quick eval** (HumanEval+ subset, 4-bit vLLM) of each variant → pick the one meeting ≤12GB @ 4-bit with least degradation. Expected winner: REAP @ 40–50%.

## Phase 3 — Data curation (CPU runtime, ~0 GPU units)

`notebooks/04_data_prep.ipynb`. Goal: **small, genuinely high-quality, agentic + reasoning-trace mix** (~30–50K samples, seq cap 16K), formatted into Laguna's chat template **with thinking blocks preserved**:

| Slice | Datasets (user-specified in **bold**; verify others/newer options on HF at execution — knowledge cutoff Jan 2026) | Share |
|---|---|---|
| Agentic tool-loop SFT | **`Nexlab/fable5-agentic-coding-sft`** (~160K multi-turn tool-call trajectories with `<think>` preserved, MIT, 590MB) — take a filtered ~10–20K subset: prefer Unix-shell trajectories (card says tool calls are PowerShell-flavored — filter or translate for Mac use), dedupe, complete loops only | ~30% |
| Agentic SWE trajectories | `SWE-bench/SWE-smith-trajectories` (repo-fix trajectories used to train SWE-agent-LM-32B); R2E-Gym / SWE-Gym OpenHands SFT trajectories — **keep only verified-successful trajectories** | ~25% |
| Verified debugging traces | **`greghavens/kimi-k3-coding-and-debugging-traces`** (~3.9K Kimi-K3 traces, 14 languages, has verification fields — keep verified rows, CC-BY-4.0) | ~8% |
| Code reasoning traces | `nvidia/OpenCodeReasoning` / OpenCodeReasoning-2 (R1 traces, competitive programming); `open-r1/codeforces-cots` | ~25% |
| General-reasoning replay (anti-forgetting) | OpenThoughts3 subset | ~12% |

Curation rules: dedupe, drop failed/truncated traces, re-wrap all external formats (OpenAI-style tool calls in the Nexlab set, R1/Kimi think styles) into **Laguna's native chat template, think format, and tool-call syntax**, length-bucket for packing. Push processed dataset to Hub.

Notes on the two user-specified sets: the Nexlab card states refusals were filtered out ("reduced safety guardrails") and its provenance is a third-party redistribution of frontier-model traces — the MIT label is the redistributor's claim, so treat it as fine for a personal/educational project but not something to build a redistributed product on without checking. Its tool-call format retention role replaces the earlier generic "tool-calling slice".

## Phase 4 — QLoRA fine-tune of the pruned model (~150 units, the main spend)

`notebooks/05_sft.ipynb`, A100 40GB:
- TRL `SFTTrainer` + PEFT on the **pruned** checkpoint: NF4 4-bit base (~10–13GB), LoRA r=16–32 on attention projections (+ shared-expert MLP), **router frozen**, routed experts frozen (LoRA over 256 experts/layer is a VRAM trap).
- Single mixed run (healing + specialization together — cheaper than two stages), 1–2 epochs, seq len 8–16K with packing, ~15–18 A100-hours.
- **Colab-disconnect resilience is mandatory**: push adapter checkpoints to Hub every ~30 min, resume-from-checkpoint logic in the notebook, small eval-loss slice for monitoring.

## Phase 5 — Final evaluation + ablation table (~30 units)

`notebooks/06_eval.ipynb`: re-run Phase-1 suite on **base vs pruned vs pruned+SFT** → one comparison table (quality, size, tok/s). This is the report deliverable for the pruning study.

## Phase 6 — Package for the Mac (~15 units + local work)

`notebooks/07_convert.ipynb` + local steps:
1. Merge LoRA → BF16 pruned model (~36–40GB) on CPU high-RAM runtime; push to Hub.
2. **GGUF first** (official llama.cpp laguna support): `convert_hf_to_gguf.py` → Q4_K_M ≈ 10–11GB → runs via llama.cpp/Ollama/LM Studio on the Mac. This guarantees a working local deliverable.
3. **MLX 4-bit** (the original ask): `mlx_lm.convert` if the laguna PR has landed by execution time; else mlx-vlm or the community PR branch. group-size-64 4-bit ≈ 10–11GB.
4. **On-Mac acceptance test**: loads under ~14GB wired memory, ≥8K context usable, tok/s measured, one live agentic coding session end-to-end.

## Budget summary (Colab compute units; A100 ≈ 8.5 units/hr, verify current rates)

| Phase | Est. units |
|---|---|
| 0 Smoke test | 10 |
| 1 Baseline eval | 20 |
| 2 Pruning study | 60 |
| 3 Data prep | ~0 (CPU) |
| 4 QLoRA SFT | 150 |
| 5 Final eval | 30 |
| 6 Conversion | 15 |
| **Total** | **~285 of 300** |

If budget runs short: shrink SFT dataset (quality over quantity), drop the 25%-pruned variant, use L4 (~2.4 units/hr) for evals of quantized variants.

## Key risks & mitigations

1. **bnb-4bit QLoRA may not support the laguna MoE** → Phase 0 smoke test catches this before real spend; fallbacks: Unsloth (check July-2026 support), BF16 LoRA on pruned model, A100-80GB tier.
2. **Colab disconnects mid-training** → Hub checkpointing + resume built into the SFT notebook from day one.
3. **mlx-lm laguna support hasn't landed** → GGUF is the guaranteed deliverable; MLX via mlx-vlm/PR branch as best-effort.
4. **Pruned-config converter incompatibility** → only change `num_routed_experts`-style config fields (REAP approach), keep architecture name/layout standard.
5. **Dataset IDs stale** (my knowledge ends Jan 2026) → Phase 3 starts with a fresh HF Hub scout for newer agentic-SWE trajectory datasets.
6. **"300 compute hours" vs units** → Colab sells *units*; user should confirm their actual balance in Colab's usage panel before Phase 4 (the big spend).

## Verification

- Phase 0: 10 LoRA steps complete without OOM/NaN on A100.
- Phase 2: pruned variants load in vLLM and score within ~5–10% of base on HumanEval+ subset before any healing.
- Phase 4: eval-loss decreasing; spot-check generations keep thinking format + tool-call syntax.
- Phase 5: pruned+SFT ≥ pruned on all evals; ideally ≈ base on agentic subset.
- Phase 6 (final acceptance): model runs on the 24GB Mac under ~14GB wired memory with ≥8K context and completes a live agentic coding task.
