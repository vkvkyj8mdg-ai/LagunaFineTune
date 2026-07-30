# %% [markdown]
# # 06 — Final evaluation: base vs pruned vs pruned+SFT
#
# **Runtime: A100 40GB.** Est. ~3h ≈ 30 units.
#
# Base numbers come from notebook 02's saved results (no need to rerun the 33B).
# Here we run HumanEval+ (full) + MBPP+ (100) on:
# - the pruned checkpoint (quantifies pruning damage)
# - pruned + SFT adapter (quantifies healing + specialization)
#
# Output: the comparison table for the pruning-study write-up.

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
!pip install -q -U "transformers>=5.7" accelerate bitsandbytes peft evalplus
!pip install -q experts4bit-qlora

# %%
PRUNED_REPO = ART.pruned_reap50            # same choice as notebook 05

# %%
import gc, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.eval_utils import get_problems, generate_solutions, score

tok = AutoTokenizer.from_pretrained(PRUNED_REPO)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

def evaluate(tag, with_adapter):
    # fused 3D experts aren't quantized by plain bnb — use the loader from notebook 01
    try:
        import experts4bit_qlora
        model = experts4bit_qlora.load_moe_4bit(PRUNED_REPO, device_map={"": 0})
    except Exception as e:
        print(f"expert-quantized loader unavailable ({e!r}); plain bnb load")
        model = AutoModelForCausalLM.from_pretrained(PRUNED_REPO, quantization_config=bnb,
                                                     device_map="auto", dtype=torch.bfloat16)
    assert model.get_memory_footprint() / 2**30 < 30, "experts not quantized — see notebook 01"
    if with_adapter:
        model = PeftModel.from_pretrained(model, ART.sft_adapter, subfolder="final")
    model.eval()
    out = {}
    for dataset, limit in (("humaneval", None), ("mbpp", 100)):
        problems = get_problems(dataset, limit=limit)
        samples, stats = generate_solutions(model, tok, problems)
        out[dataset] = {"stats": stats, "eval": score(samples, dataset, tag=tag)}
    del model; gc.collect(); torch.cuda.empty_cache()
    return out

results = {"pruned": evaluate("pruned", with_adapter=False),
           "pruned_sft": evaluate("pruned_sft", with_adapter=True)}

# %%
# Pull the base numbers saved by notebook 02 and print the study table
from huggingface_hub import hf_hub_download
base = json.load(open(hf_hub_download(cfg.hub("eval-results"), "base_int4.json",
                                      repo_type="dataset")))
from src.eval_utils import pass1
def p1(r, ds):
    return pass1(r[ds]["eval"])  # EvalPlus nests scores under pass_at_k.plus/base
rows = [("base 33B (INT4)", base), ("pruned", results["pruned"]),
        ("pruned + SFT", results["pruned_sft"])]
print(f"{'model':<18} {'HumanEval+':<12} {'MBPP+':<12} {'tok/s':<8}")
for name, r in rows:
    print(f"{name:<18} {str(p1(r, 'humaneval')):<12} {str(p1(r, 'mbpp')):<12} "
          f"{r['humaneval']['stats']['tok_per_s']:<8}")

# %%
os.makedirs("/content/results", exist_ok=True)
with open("/content/results/final_comparison.json", "w") as f:
    json.dump(results, f, indent=1)
from src.hub_utils import upload_dir
upload_dir("/content/results", cfg.hub("eval-results"), repo_type="dataset")

# %% [markdown]
# ### (Stretch) rerun the notebook-02 SWE-bench slice against pruned+SFT
# Same fixed instance subset, served via vLLM after notebook 07's merge.
# Success criterion from the plan: **pruned+SFT ≥ pruned everywhere, ≈ base on agentic tasks.**
