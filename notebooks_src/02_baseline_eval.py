# %% [markdown]
# # 02 — Baseline evaluation of the unmodified model
#
# **Runtime: A100 40GB.** Est. ~2h ≈ 20 units.
#
# Produces the yardstick every later stage is compared against:
# - HumanEval+ (full 164) and MBPP+ (first 100) pass@1, greedy, thinking ON
# - generation throughput
#
# Uses poolside's own INT4 build under vLLM (laguna is officially supported there).

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

# %%
# vllm 0.26.0 (0.23 lacks group-size-128 Marlin MoE support that the INT4
# checkpoint needs). Its wheels link CUDA-13 libs, which ship via its nvidia-*
# pip deps but OUTSIDE the dynamic-loader path on Colab's cu12.8 image —
# registering the cu13 lib dir with ldconfig fixes every one of them at once
# (verified on A100, 2026-07-30).
!pip install -q "vllm==0.26.0" evalplus
!echo /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib > /etc/ld.so.conf.d/nvidia-cu13.conf && ldconfig
# align flashinfer-cubin with whatever flashinfer version vllm resolved
!python -c "import importlib.metadata as m; print(m.version('flashinfer'))" | xargs -I{} pip install -q "flashinfer-cubin=={}"
!python -c "from vllm import LLM; print('vllm import OK')"

# %% [markdown]
# ## Run the eval as a PLAIN PROCESS (vLLM cannot start inside a Jupyter kernel:
# fork inherits kernel CUDA and dies silently, spawn re-imports Jupyter's
# __main__, in-process needs a real stdout fd — all verified 2026-07-30).
# The script generates for HumanEval+ (164) + MBPP+ (100), scores with EvalPlus,
# and pushes base_int4.json to the Hub (notebook 06 reads it for its table).

# %%
!cd /content/LagunaFineTune && python tools/eval_baseline.py

# %% [markdown]
# ## (Stretch) 10–25 SWE-bench-Verified instances
# Full SWE-bench is far outside budget; a small slice still gives an agentic signal.
# Rough recipe (budget ~2–4h extra — decide consciously):
# ```
# pip install mini-swe-agent
# # terminal 1: vllm serve poolside/Laguna-XS-2.1-INT4 --max-model-len 65536
# # terminal 2: point mini-swe-agent at the local OpenAI-compatible endpoint,
# #             run a fixed 10–25 instance subset, record % resolved
# ```
# Keep the SAME subset for notebook 06 so numbers are comparable.
