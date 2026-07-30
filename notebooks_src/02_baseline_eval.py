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
# vllm PINNED: 0.24.0+ wheels are built against CUDA 13 (libcudart.so.13) and fail
# to import on Colab's CUDA 12.8 / torch 2.11 image (verified 2026-07-30).
# 0.23.0 is the newest release that matches the image.
!pip install -q "vllm==0.23.0" evalplus

# %%
from transformers import AutoTokenizer
from vllm import LLM
tok = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
llm = LLM(model=cfg.BASE_MODEL_INT4, max_model_len=16384, gpu_memory_utilization=0.92,
          trust_remote_code=True)

# %% [markdown]
# ## HumanEval+ (full) and MBPP+ (subset)

# %%
from src.eval_utils import get_problems, vllm_generate_solutions, score
results = {}
for dataset, limit in (("humaneval", None), ("mbpp", 100)):
    problems = get_problems(dataset, limit=limit)
    samples, stats = vllm_generate_solutions(llm, tok, problems)
    print(f"{dataset}: {stats}")
    results[dataset] = {"stats": stats, "eval": score(samples, dataset, tag="base_int4")}
results

# %% [markdown]
# ## Persist results to the Hub (notebook 06 pulls these for the comparison table)

# %%
import json
from src.hub_utils import ensure_repo, upload_dir
os.makedirs("/content/results", exist_ok=True)
with open("/content/results/base_int4.json", "w") as f:
    json.dump(results, f, indent=1)
ensure_repo(cfg.hub("eval-results"), repo_type="dataset")
upload_dir("/content/results", cfg.hub("eval-results"), repo_type="dataset")

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
