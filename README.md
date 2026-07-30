# LagunaFineTune

Prune + fine-tune [poolside/Laguna-XS-2.1](https://huggingface.co/poolside/Laguna-XS-2.1)
(33B-total / 3B-active MoE coding model) for **agentic coding / SWE work**, small enough to run
on a **24GB Mac** (target: ≤ ~12GB 4-bit artifact → MLX + GGUF).

Full plan: [reference/PLAN.md](reference/PLAN.md)

## Pipeline

```
poolside/Laguna-XS-2.1 (BF16, ~66GB)
  → expert pruning (REAP + frequency baseline, 25%/50%)   [Colab A100 + CPU high-RAM]
  → QLoRA SFT on agentic/reasoning data (heals + specializes)  [Colab A100]
  → merge → GGUF Q4_K_M + MLX 4-bit (~10–11GB)            [Colab CPU + local Mac]
```

## Phases / notebooks

| # | Notebook | Runtime | Est. units | What it does |
|---|----------|---------|-----------:|--------------|
| 0 | `notebooks/01_smoke_test.ipynb` | A100 | ~10 | Verify 4-bit load + tiny LoRA step **before spending real budget** |
| 1 | `notebooks/02_baseline_eval.ipynb` | A100 | ~20 | HumanEval+/MBPP+ (EvalPlus) + small SWE-bench slice on the base model |
| 2 | `notebooks/03_calibration_pruning.ipynb` | A100 → CPU high-RAM | ~60 | Router-stats calibration → REAP & frequency expert pruning @25%/50% → quick evals |
| 3 | `notebooks/04_data_prep.ipynb` | CPU | ~0 | Curate + re-template the SFT mix into Laguna's chat format |
| 4 | `notebooks/05_sft.ipynb` | A100 | ~150 | QLoRA SFT of the pruned model, Hub checkpointing + resume |
| 5 | `notebooks/06_eval.ipynb` | A100 | ~30 | base vs pruned vs pruned+SFT comparison table |
| 6 | `notebooks/07_convert.ipynb` | CPU high-RAM | ~15 | Merge LoRA → GGUF Q4_K_M → MLX 4-bit → Mac acceptance test |

Budget: ~285 of 300 Colab compute units (A100 ≈ 8.5 units/hr — **verify current rates and your
actual unit balance in Colab's usage panel before Phase 4**, the big spend).

## Setup

1. Edit `src/project_config.py`: set `HF_USER` to your HuggingFace username.
2. Push this repo to GitHub (private is fine) and set `REPO_URL` in `src/project_config.py` —
   each notebook clones it to get `src/`.
3. In Colab: add your HF token as a Secret named `HF_TOKEN` (write access — artifacts are stored
   in private HF Hub repos, since checkpoints are 20–40GB and Drive can't hold them).
4. Run notebooks in order. **Do not skip 01** — it validates that bitsandbytes QLoRA works on
   this MoE architecture before any real spend.

## Key model facts (from `reference/config.laguna-xs-2.1.json`)

- `model_type: laguna`, 40 layers: layer 0 dense MLP, layers 1–39 sparse MoE.
- 256 routed experts + 1 shared per sparse layer, top-8 routing, `moe_intermediate_size=512`.
- Routed experts ≈ **95% of all parameters** → expert pruning is extremely effective here
  (50% experts → ~17.5B total → ~10GB @ 4-bit).
- **Per-head router gating** (`gating: per_head`) — non-standard; router-stat hooks introspect
  shapes at runtime instead of assuming `[tokens, experts]`.
- Mixed attention: every 4th layer full (10), rest sliding-window 512 (30). 262K context (YaRN).
- Chat format (`reference/chat_template.jinja`): `<user>…</user>`, `<assistant><think>…</think>…</assistant>`,
  reasoning goes in the `reasoning_content` message field, tool calls are OpenAI-style dicts
  (the template renders them to `<tool_call>name<arg_key>…`). The template has `{% generation %}`
  markers → TRL `assistant_only_loss=True` works.

## SFT data mix (Phase 3)

| Slice | Dataset | Share |
|---|---|---|
| Agentic tool-loop SFT | `Nexlab/fable5-agentic-coding-sft` (filtered subset) | 18% |
| Verified debugging traces ("moonshiner" series, 4 source models) | `greghavens/{kimi-k3, fable-5, gpt-5.6-sol, glm-5.2}-coding-and-debugging-traces` | 29% |
| Fable-5 family extras | `Crownelius/Complete-FABLE.5-traces-2M` (filtered take), `Glint-Research/Fable-5-traces` (AGPL-3.0), `AlinCiocan/fable-5-claude-code-traces` | 8% |
| Agentic SWE trajectories | `SWE-bench/SWE-smith-trajectories` | 20% |
| Code reasoning | `nvidia/OpenCodeReasoning` | 17% |
| General-reasoning replay | OpenThoughts3 subset | 8% |

## Repo layout

```
reference/       fetched model config, chat template, generation config, approved plan
src/             shared utilities imported by the notebooks
notebooks_src/   notebook sources (jupytext percent format — edit these)
notebooks/       generated .ipynb for Colab (regenerate: python tools/py2ipynb.py)
tools/           py→ipynb converter (stdlib only)
```

## Known risks (updated 2026-07-30 after the research-agent audit)

1. **bnb-4bit skips fused MoE experts** (native transformers stores them as 3D nn.Parameters;
   bnb only converts nn.Linear) → all bnb loads use `trust_remote_code=True` (poolside's
   per-expert nn.Linear code) with a footprint assert. Fallbacks: Axolotl v0.18
   `quantize_moe_experts: true`, or `experts4bit-qlora`. Unsloth does NOT support laguna.
2. Colab disconnects → notebook 05 pushes checkpoints to Hub every save and auto-resumes.
   Unit rates are unpublished and vary — read them off Colab's live resource panel.
3. **Mac runtimes** (see notebook 07): stock llama.cpp Metal currently NaNs on Laguna XS 2.1
   (PR #25442 unmerged) → use Ollama ≥ 0.32.3 or a patched build. mlx-lm laguna merged on
   `main` only (no release, no sanitize()) → use mlx-vlm ≥ 0.6.8. DFlash speculative decoding
   is unusable on Mac and invalidated by pruning (pointer stripped from pruned configs).
4. **Pruning aggressiveness**: 25% is the published-safe zone; 50% is model-dependent and
   agentic/multi-turn quality degrades fastest. Default ship target: keep-160 (37.5%, ~11GB
   @ 4-bit). Layer/depth pruning was researched and REJECTED (breaks long-horizon generation).
5. Dataset schemas drift → adapters in `src/data_prep.py` inspect columns at runtime and fail loudly.
