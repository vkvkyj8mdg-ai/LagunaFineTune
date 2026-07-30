# %% [markdown]
# # 03 — Calibration + expert pruning (the pruning study)
#
# Three parts on DIFFERENT runtimes — run top to bottom, switching runtime between parts:
#
# | Part | Runtime | What | Est. |
# |------|---------|------|------|
# | A | **A100 40GB** | collect router/expert stats over ~1M calibration tokens | ~2h |
# | B | **CPU high-RAM** | weight surgery → reap25 / reap50 / freq50 checkpoints | ~2–3h, ~0 GPU units |
# | C | **A100 40GB** | quick HumanEval+ (30 problems) per variant | ~2h |
#
# Methods compared: **REAP** (freq × gate-weight × output-norm saliency) vs
# **frequency-only** dropping — at keep-192 (25% pruned) and keep-128 (50% pruned).

# %%
# ── Bootstrap (run on every runtime) ─────────────────────────────────────────
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

# %% [markdown]
# ## Part A — calibration statistics (A100)
# Calibration data = the same domains we fine-tune on (agentic + code reasoning),
# so pruning keeps the experts THAT DOMAIN actually uses.

# %%
!pip install -q -U "transformers>=5.7" accelerate bitsandbytes datasets

# %%
# Build ~600 calibration texts from the agentic/code sources (small streamed takes)
from datasets import load_dataset
from src.data_prep import adapt_row, to_laguna_text, is_complete_trajectory
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)

def take(repo, n, split="train", config=None):
    ds = load_dataset(repo, config, split=split, streaming=True)
    rows = []
    for row in ds:
        try:
            s = adapt_row(row, repo)
            if is_complete_trajectory(s):
                rows.append(s)
        except ValueError as e:
            print(e); break
        if len(rows) >= n:
            break
    return rows

calib = (take("Nexlab/fable5-agentic-coding-sft", 250)
         + take("SWE-bench/SWE-smith-trajectories", 150, split="tool")  # no "train" split
         + take("nvidia/OpenCodeReasoning", 200, config="split_0", split="split_0"))
texts = [to_laguna_text(tok, s) for s in calib]
print(f"{len(texts)} calibration docs, ~{sum(map(len, texts)) / 4 / 1e6:.1f}M tokens (char/4 est)")

# %%
# Loader path validated by notebook 01: NF4 experts via the streaming loader
# (plain bnb skips fused 3D experts). The experts module becomes a fused
# container, so per-expert output-norm hooks can't attach — REAP saliency uses
# its documented gate×freq fallback (src/router_stats.py). Router hooks on
# mlp.gate capture the exact selection rule regardless.
!pip install -q experts4bit-qlora
import torch
from src.laguna_e4b import load_laguna_4bit
model, _ = load_laguna_4bit(cfg.BASE_MODEL)   # LoRA rank irrelevant: forward-only
model.eval()
alloc = torch.cuda.memory_allocated() / 2**30
print(f"cuda allocated: {alloc:.1f} GB (expect ~19-22)")
assert alloc < 30

# %%
from src.router_stats import RouterStatsCollector, ExpertNormCollector, save_stats
router_c = RouterStatsCollector(model).attach(model)
norm_c = ExpertNormCollector(model).attach()

# batch of ONE, no padding: the MoE block routes every position regardless of the
# attention mask, so pad tokens would pollute the very statistics that decide
# which experts survive. ~2x slower than batching; correctness wins.
SEQ = 2048
enc = tok(texts, truncation=True, max_length=SEQ, padding=False,
          add_special_tokens=False).input_ids  # template already emitted BOS
enc = [e for e in enc if len(e) > 256]
with torch.inference_mode():
    for i, e in enumerate(enc):
        model(input_ids=torch.tensor([e], device=model.device))
        if i % 50 == 0:
            print(f"{i}/{len(enc)}  tokens so far: {router_c.tokens.sum() / cfg.NUM_LAYERS:.0f}")
router_c.detach(); norm_c.detach()

rs, ns = router_c.stats(), norm_c.stats()
save_stats("/content/router_stats.json", rs, ns, meta={"docs": len(enc), "seq": SEQ})

# %%
from src.hub_utils import ensure_repo, upload_dir
os.makedirs("/content/stats_up", exist_ok=True)
!cp /content/router_stats.json /content/stats_up/
ensure_repo(ART.router_stats, repo_type="dataset")
upload_dir("/content/stats_up", ART.router_stats, repo_type="dataset")

# %% [markdown]
# ## Part B — weight surgery (switch to **CPU high-RAM** runtime, rerun Bootstrap)
# Downloads the 66GB BF16 model once, then produces each variant **sequentially**
# (prune → upload → delete) to stay inside Colab's disk.

# %%
!pip install -q -U safetensors huggingface_hub torch --index-url https://download.pytorch.org/whl/cpu
from huggingface_hub import snapshot_download, hf_hub_download
stats_path = hf_hub_download(ART.router_stats, "router_stats.json", repo_type="dataset")
src_dir = snapshot_download(cfg.BASE_MODEL)   # ~66GB, ~30–60 min
print("snapshot at", src_dir)

# %%
# !!! Verify the per-head router row layout before trusting any output !!!
# Prints gating code from modeling_laguna.py — decide heads_major vs experts_major.
import pathlib, re as _re
code = pathlib.Path(src_dir, "modeling_laguna.py").read_text() \
    if pathlib.Path(src_dir, "modeling_laguna.py").exists() else ""
for m in _re.finditer(r"(?s)class \w*(Gate|Router|Moe|MoE)\w*.*?(?=\nclass |\Z)", code):
    print(m.group(0)[:1500], "\n" + "=" * 80)
if not code:
    print("modeling_laguna.py not in snapshot — model runs on native transformers code; "
          "inspect transformers/models/laguna/modeling_laguna.py instead:")
    import transformers, inspect
    from transformers.models import laguna  # noqa — fails loudly if missing
    print(inspect.getsource(laguna.modeling_laguna)[:4000])

# %%
ROUTER_LAYOUT = "verify_me"   # ← set to "heads_major" or "experts_major" after reading above.
                              #   If the router weight is plain [256, hidden] this is ignored.

# %%
import shutil, torch
from src.router_stats import load_stats, reap_scores, freq_scores, keep_lists
from src.prune_experts import prune_checkpoint
from src.hub_utils import upload_dir
rs, ns = load_stats(stats_path)

# Published evidence (REAP paper + shipped checkpoints): 25% pruning is the
# well-supported zone (~0-3pt EvalPlus loss one-shot); 50% is model-dependent
# (2-11pt) and agentic/multi-turn degrades FASTEST (Kimi-K2 BFCL -54% at 50%).
# keep-160 (37.5%) is our fit/quality sweet spot for the 24GB Mac (~11GB @4-bit).
# Order matters for DISK: base snapshot is 62GB and Colab CPU runtimes have ~100GB.
# Smallest outputs first (33GB), reap25 (48GB) last — and check free space first.
!df -h /content
variants = {
    ART.pruned_reap50: keep_lists(reap_scores(rs, ns), keep_fraction=0.50),    # keep 128, ~33GB
    ART.pruned_freq50: keep_lists(freq_scores(rs), keep_fraction=0.50),        # baseline, ~33GB
    ART.pruned_reap375: keep_lists(reap_scores(rs, ns), keep_fraction=0.625),  # keep 160, ~40GB
    ART.pruned_reap25: keep_lists(reap_scores(rs, ns), keep_fraction=0.75),    # keep 192, ~48GB
}
for repo_id, keep in variants.items():
    out = "/content/pruned_out"
    prune_checkpoint(src_dir, out, keep, router_layout=ROUTER_LAYOUT)
    upload_dir(out, repo_id)
    shutil.rmtree(out)
    print("uploaded", repo_id)

# %% [markdown]
# ## Part C — quick quality check (switch back to **A100**, rerun Bootstrap)
# 30 HumanEval+ problems per variant, 4-bit transformers generate (uniform + reliable;
# ~30–45 min per variant). Compare against notebook 02's base score.

# %%
!pip install -q -U "transformers>=5.7" accelerate bitsandbytes evalplus
!pip install -q experts4bit-qlora
import gc, json, torch
from transformers import AutoTokenizer
from src.eval_utils import get_problems, generate_solutions, score
from src.hub_utils import upload_dir, ensure_repo

tok = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
problems = get_problems("humaneval", limit=30)
from src.laguna_e4b import load_laguna_4bit
quick = {}
for repo_id in (ART.pruned_reap25, ART.pruned_reap375, ART.pruned_reap50, ART.pruned_freq50):
    model, _ = load_laguna_4bit(repo_id)
    samples, stats = generate_solutions(model, tok, problems, max_new_tokens=2048)
    quick[repo_id] = {"stats": stats, "eval": score(samples, "humaneval",
                                                    tag=repo_id.split("/")[-1])}
    # persist per-variant so a disconnect doesn't lose earlier evals
    os.makedirs("/content/results", exist_ok=True)
    with open("/content/results/pruned_quickeval.json", "w") as f:
        json.dump(quick, f, indent=1)
    del model; gc.collect(); torch.cuda.empty_cache()
quick

# %%
os.makedirs("/content/results", exist_ok=True)
with open("/content/results/pruned_quickeval.json", "w") as f:
    json.dump(quick, f, indent=1)
ensure_repo(cfg.hub("eval-results"), repo_type="dataset")
upload_dir("/content/results", cfg.hub("eval-results"), repo_type="dataset")

# %% [markdown]
# ### Decision
# Pick the SFT base for notebook 05. Default recommendation: **reap375 (keep 160,
# ~11GB @4-bit)** — the published evidence says 50% pruning is where agentic/tool-use
# quality becomes unpredictable, while 25% barely fits the 24GB Mac (~13GB @4-bit,
# no KV headroom). Use reap50 only if its quick score is within ~10% of base;
# fall back to reap25 + a 3-3.5 bit final quant if even reap375 shows damage.
# For the write-up: reap50 vs freq50 isolates the value of REAP saliency vs frequency.
