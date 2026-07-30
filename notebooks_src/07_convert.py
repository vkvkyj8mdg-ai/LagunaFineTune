# %% [markdown]
# # 07 — Merge + convert for the 24GB Mac
#
# **Runtime: CPU** (the merge is shard-streaming now — peak RAM is a few GB;
# high-RAM only helps the GGUF conversion step's headroom).
#
# 1. Merge the LoRA adapter into the pruned BF16 model → push
# 2. Convert to **GGUF Q4_K_M** (llama.cpp has official laguna support) → push
# 3. MLX 4-bit — run ON YOUR MAC (instructions at the bottom)

# %%
# ── Bootstrap ────────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/vkvkyj8mdg-ai/LagunaFineTune.git"   # ← your fork
import os, sys
if not os.path.isdir("/content/LagunaFineTune"):
    !git clone {REPO_URL} /content/LagunaFineTune
sys.path.insert(0, "/content/LagunaFineTune")
os.environ["HF_HOME"] = "/content/hf_cache"
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from src import project_config as cfg
from src.project_config import ART

# %%
!pip install -q "transformers==5.12.0" gguf sentencepiece

# %%
PRUNED_REPO = ART.pruned_reap50

# %% [markdown]
# ## 1. Merge adapter → BF16 (CPU shard-streaming, low RAM — no PEFT involved)
# Folds the trainable-params checkpoint (LoRALinear + per-expert ExpertsLoRA
# deltas, scaling alpha/r) into the per-expert bf16 checkpoint one shard at a
# time. Code-reviewed but not yet run-verified: after the GGUF step, spot-check
# a generation against notebook 06's adapter-based outputs before trusting it.

# %%
!pip install -q experts4bit-qlora
from huggingface_hub import snapshot_download, hf_hub_download
from src.laguna_e4b import merge_adapter_into_checkpoint

src = snapshot_download(PRUNED_REPO)
adapter = hf_hub_download(ART.sft_adapter, "adapter-final.pt")
merge_adapter_into_checkpoint(src, adapter, "/content/merged")

from src.hub_utils import upload_dir
upload_dir("/content/merged", ART.sft_merged)
print("merged model →", ART.sft_merged)

# %% [markdown]
# ## 2. GGUF Q4_K_M via llama.cpp (~11GB output)

# %%
!git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
!pip install -q -r /content/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
!python /content/llama.cpp/convert_hf_to_gguf.py /content/merged \
    --outtype bf16 --outfile /content/laguna-agentic-bf16.gguf

# %%
!cmake -S /content/llama.cpp -B /content/llama.cpp/build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
!cmake --build /content/llama.cpp/build --target llama-quantize -j
!/content/llama.cpp/build/bin/llama-quantize /content/laguna-agentic-bf16.gguf \
    /content/laguna-agentic-Q4_K_M.gguf Q4_K_M
!ls -lh /content/*.gguf

# %%
import shutil
os.makedirs("/content/gguf_up", exist_ok=True)
shutil.move("/content/laguna-agentic-Q4_K_M.gguf", "/content/gguf_up/")
upload_dir("/content/gguf_up", ART.gguf)
print("GGUF →", ART.gguf)

# %% [markdown]
# ## 3. MLX 4-bit — run these ON YOUR MAC (not in Colab)
#
# Status (verified 2026-07-30): mlx-lm merged laguna on `main` (2026-07-26) but it is
# NOT in any PyPI release, cannot ingest per-expert HF checkpoints (no sanitize()),
# and lacks a laguna tool parser. Use **mlx-vlm 0.6.8+** — it converts, serves, and
# stacks the per-expert tensors correctly. (Never 0.6.4: known weight-corruption bug.)
# Needs ~45GB free disk for the BF16 download; output ≈10GB.
#
# ```bash
# pip install "mlx-vlm>=0.6.8" huggingface_hub
# hf download <you>/laguna-xs-2.1-agentic-pruned --local-dir ./pruned-bf16
# python -m mlx_vlm.convert --hf-path ./pruned-bf16 \
#     --mlx-path ./laguna-agentic-mlx-4bit \
#     --quantize --q-bits 4 --q-group-size 64 --q-mode affine
# # never --q-group-size 128: expert down_proj rows are only 512 wide
# python -m mlx_vlm.generate --model ./laguna-agentic-mlx-4bit \
#     --prompt "Fix this failing pytest..." --max-tokens 512
# python -m mlx_vlm.server --model ./laguna-agentic-mlx-4bit --port 8080  # OpenAI-compatible
# ```
# If 4-bit quality disappoints, next rungs: `mixed_3_6` quant predicate, or
# `mlx_lm.dynamic_quant --target-bpw 4.0` from a git-main mlx-lm (needs a
# pre-stacking step — see mlx-lm PR #1223's sanitize()).
#
# Environment gotcha (measured on this Mac): a stale mlx/mlx-metal version skew
# makes mlx-vlm fail on import (`no attribute 'new_thread_local_stream'`).
# Clean-reinstall the trio together: `pip install -U mlx mlx-metal "mlx-vlm>=0.6.8"`.
#
# MLX memory safety — MLX's DEFAULT memory limit (22.8GB) is ABOVE the Metal
# working-set ceiling (~17.8GB on a 24GB Mac): it will swap-thrash before erroring.
# Set limits before loading:
# ```python
# import mlx.core as mx; GB = 1024**3
# mx.set_memory_limit(14 * GB)   # hard stop instead of thrash
# mx.set_cache_limit(1 * GB)
# mx.set_wired_limit(12 * GB)    # keeps weights resident (must be < ~17.7GB)
# ```
# Don't raise iogpu.wired_limit_mb — the default (~74% = 17.8GB) already exceeds our
# 14GB budget. Biggest practical RAM win: quit Electron apps (~4GB measured).
#
# ### GGUF path — llama.cpp launch flags that matter
# ⚠ VERIFIED BLOCKER (2026-07-30): stock llama.cpp on Apple Silicon produces
# NaN/empty output on Laguna XS 2.1 — Metal MUL_MAT_ID casts MoE intermediates to
# f16 and Laguna's SwiGLU activations overflow it (PR #25442, OPEN/unmerged).
# Until it merges: use **Ollama ≥ 0.32.3** (ships an equivalent Metal fix) as the
# GGUF runtime on the Mac, or build llama.cpp with PR #25442 cherry-picked.
# First smoke test: if output is empty or garbage, this is why — not your model.
#
# llama.cpp merged laguna 2026-07-22 (release b10087). Two more verified gotchas:
# 1. **Thinking is silently OFF by default**: the chat template defaults
#    `enable_thinking=false` and llama.cpp does not read generation_config.json.
#    Set it explicitly AND pair with --reasoning-budget (either alone can fail).
# 2. After conversion, verify YaRN + EOG metadata: pass the rope flags explicitly
#    and check token 24 is registered as end-of-generation, else reasoning runs away.
#
# Build (brew's build is often behind; laguna needs ≥ b10087, plus the overflow patch):
# ```bash
# git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
# curl -sL https://github.com/ggml-org/llama.cpp/pull/25442.diff | git apply  # Metal MoE f16 fix
# cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# ```
# Quantize: plain Q4_K_M (~10.7GB) — NO imatrix needed (K-quants don't require one, and
# published Laguna imatrices won't fit the pruned expert dims anyway). IQ4_XS (~9.5GB)
# if you need the extra GB. Do NOT use MXFP4_MOE (ignores calibration, ~30% slower here).
#
# ```bash
# llama-server -m laguna-agentic-Q4_K_M.gguf --jinja -ngl 99 -fa on \
#   -c 32768 --swa-full -ctk q8_0 -ctv q8_0 -ctxcp 8 \
#   --rope-scaling yarn --rope-scale 32 --yarn-orig-ctx 8192 --yarn-beta-fast 64 \
#   --chat-template-kwargs '{"enable_thinking":true}' --reasoning-budget 4096 \
#   --spec-type ngram-map-k4v,ngram-cache --spec-draft-n-max 8
# llama-bench -m laguna-agentic-Q4_K_M.gguf -ngl 99 -fa 1 -p 512,2048 -n 128
# ```
# Flag rationale (measured on this Mac): --swa-full at 32K + q8_0 KV ≈ 14.6GB total and
# buys a measured ~522x prefill-reuse win on repeated agentic prompts (without it, SWA
# capping disables prompt-cache reuse and every turn re-prefills). For contexts ≥64K
# drop --swa-full and accept re-prefills. KV quant types must be SYMMETRIC on Metal
# (q8_0/q8_0 — mixed K/V types fail).
#
# Correctness checks before trusting ANY quant (the Metal overflow hits batched prefill
# only, so short interactive turns can look fine while long prompts return empty):
# ```bash
# ./build/bin/llama-cli -m laguna-agentic-Q4_K_M.gguf -ngl 99 -fa on --temp 0 -n 64 \
#   -f some_400_token_prompt.txt --single-turn        # must be non-empty
# ./build/bin/llama-perplexity -m laguna-agentic-Q4_K_M.gguf -ngl 99 -fa on \
#   -f wiki.test.raw --chunks 20                       # must not print nan
# ```
#
# Post-conversion GGUF metadata asserts (a pruned/hand-edited config can silently
# mangle these): rope.dimension_count == 64 (partial_rotary_factor 0.5!),
# rope.freq_base == 500000 with rope.freq_base_swa == 10000, attention.sliding_window
# == 512, expert_count == 128, head_count = 40-element array of 48/64 values.
# The `--spec-type ngram-*` line is model-free speculation — a free 1.2–2x on
# repetition-heavy agentic loops, immune to pruning distribution shift. For fast
# tool-loop steps, run a profile with `{"enable_thinking":false}` + `--reasoning-budget 0`.
# Do NOT use --cpu-moe or draft-model speculation on Apple Silicon (measured net loss);
# the DFlash drafter is unusable here (target-coupled EAGLE-style head, invalidated
# by pruning — its stale pointer is already stripped from our generation_config).
#
# ### Pass criteria (plan)
# Loads under ~14GB wired memory, 8K+ usable context, one live agentic session
# end-to-end. KV is cheap on this model: only the 10 full-attention layers grow
# (~40KB/token); the 30 sliding-window layers are pinned at 512 tokens (~63MB).
# The default macOS wired limit (~16–18GB on 24GB) should already suffice — only
# touch `sysctl iogpu.wired_limit_mb` if you measure otherwise.
