# %% [markdown]
# # 04 — SFT data curation
#
# **Runtime: CPU high-RAM (REQUIRED — the tokenized rows alone peak >10GB as Python
# objects; a standard 12.7GB runtime WILL OOM). ~0 GPU units.**
#
# Builds the training mix, re-templated into Laguna's chat format and
# **pre-tokenized with assistant-only loss masks** (so notebook 05 needs zero
# template logic). Output: private HF dataset `…/laguna-xs-2.1-sft-mix`.
#
# | Slice | Source | Target share |
# |---|---|---|
# | agentic tool-loop | Nexlab/fable5-agentic-coding-sft (filtered) | 18% |
# | verified debugging ("moonshiner" series) | greghavens/{kimi-k3, fable-5, gpt-5.6-sol, glm-5.2}-coding-and-debugging-traces | 29% |
# | Fable-5 family extras | Crownelius/Complete-FABLE.5-traces-2M (filtered take), Glint-Research/Fable-5-traces, AlinCiocan/fable-5-claude-code-traces | 8% |
# | agentic SWE | SWE-bench/SWE-smith-trajectories | 20% |
# | code reasoning | nvidia/OpenCodeReasoning | 17% |
# | general replay | open-thoughts/OpenThoughts3-1.2M | 8% |

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
!pip install -q "transformers==5.12.0"  # datasets/hf_hub: use the image versions

# %%
TOTAL = 40_000          # mix size; shrink if notebook 05's budget projection is over
MAX_LEN = 8192          # tokens per sample (truncated); raise to 16384 only if budget allows
MAX_CHARS = 200_000     # pre-filter: drop pathological mega-trajectories
RATIOS = {"nexlab": 0.18, "swe_smith": 0.20,
          "kimi": 0.07, "gh_fable5": 0.10, "gh_gpt56": 0.10, "gh_glm52": 0.02,
          "crownelius": 0.05, "glint": 0.02, "alin_cc": 0.01,
          "ocr": 0.17, "openthoughts": 0.08}
assert abs(sum(RATIOS.values()) - 1.0) < 1e-9

# %%
from datasets import load_dataset
from src.data_prep import (adapt_row, is_complete_trajectory, looks_windows_centric,
                           dedupe, char_len, build_mix)

def collect(repo, source, n_raw, split="train", config=None, extra_filter=None):
    """Stream-take raw rows, adapt, filter. Fails loudly on schema drift."""
    ds = load_dataset(repo, config, split=split, streaming=True)
    out, seen_raw = [], 0
    for row in ds:
        seen_raw += 1
        try:
            s = adapt_row(row, source)
        except ValueError as e:
            print(f"!! {e}")
            print("   first row keys:", sorted(row.keys()))
            raise
        if (is_complete_trajectory(s) and char_len(s) < MAX_CHARS
                and (extra_filter is None or extra_filter(row, s))):
            out.append(s)
        if seen_raw >= n_raw:
            break
    print(f"{source}: kept {len(out)}/{seen_raw}")
    return dedupe(out)

# %% [markdown]
# ### Nexlab agentic set — drop Windows/PowerShell-centric trajectories (target is a Mac)

# %%
pools = {}
pools["nexlab"] = collect("Nexlab/fable5-agentic-coding-sft", "nexlab", n_raw=40_000,
                          extra_filter=lambda row, s: not looks_windows_centric(s))

# %%
# Splits are tool/xml/ticks (there is NO "train"); `tool` is the OpenAI tool-call
# format we want, and `resolved` marks execution-verified successes (plan rule:
# verified-only). The messages column is a JSON string — adapt_row handles that.
pools["swe_smith"] = collect("SWE-bench/SWE-smith-trajectories", "swe_smith", n_raw=15_000,
                             split="tool",
                             extra_filter=lambda row, s: bool(row.get("resolved", False)))

# %% [markdown]
# ### greghavens "moonshiner" series — verified traces from FOUR frontier models
# kimi-k3 + fable-5 + gpt-5.6-sol + glm-5.2: same curator, harness, and schema,
# so one loop handles all of them. Source-model diversity reduces overfitting to
# any single model's quirks. Schema is nonstandard (trajectory/verification
# columns) — column mapping is inspected per repo and fails loudly.

# %%
import json as _json
MOONSHINER = {
    "kimi":      "greghavens/kimi-k3-coding-and-debugging-traces",
    "gh_fable5": "greghavens/fable-5-coding-and-debugging-traces",
    "gh_gpt56":  "greghavens/gpt-5.6-sol-coding-and-debugging-traces",
    "gh_glm52":  "greghavens/glm-5.2-coding-and-debugging-traces",   # tiny (~1.8K rows) — take all
}

def collect_moonshiner(repo, source):
    raw = load_dataset(repo, split="train")
    traj_col = next((c for c in raw.column_names
                     if c.lower() in ("messages", "trajectory", "conversations", "turns")), None)
    # the boolean success flag is `model_attested`; the `verifier` column is just
    # the name of the verification tool (a string — always truthy)
    verified_col = "model_attested" if "model_attested" in raw.column_names else None
    assert traj_col, f"{repo}: no trajectory column in {raw.column_names} — map it manually"
    # rows are step-expanded (one row per assistant turn, sharing long prefixes):
    # keep only the final step per trajectory or the mix fills with near-duplicates
    step_cols = {"assistant_step", "assistant_steps"} <= set(raw.column_names)
    out = []
    for row in raw:
        if verified_col and row[verified_col] is not True:
            continue
        if step_cols and row["assistant_step"] != row["assistant_steps"]:
            continue
        traj = row[traj_col]
        if isinstance(traj, str):
            traj = _json.loads(traj)
        s = adapt_row({"messages": traj} if isinstance(traj, list) else traj, source)
        if is_complete_trajectory(s) and char_len(s) < MAX_CHARS:
            out.append(s)
    out = dedupe(out)
    print(f"{source}: kept {len(out)}/{len(raw)} (traj: {traj_col}, verified: {verified_col})")
    return out

for source, repo in MOONSHINER.items():
    pools[source] = collect_moonshiner(repo, source)

# %% [markdown]
# ### Fable-5 family extras
# - **Crownelius/Complete-FABLE.5-traces-2M** — raw parent of the Nexlab set; we stream a
#   modest take through the same filters for extra volume. (Copy claims MIT; the upstream
#   Glint release is AGPL-3.0 — fine for this personal project, a caveat for redistribution.)
# - **Glint-Research/Fable-5-traces** — original upstream, "Pi session" format, AGPL-3.0.
# - **AlinCiocan/fable-5-claude-code-traces** — <1K real scrubbed Claude Code sessions.

# %%
# Crownelius is a MIRROR of 28 upstream sets (~229K rows, not 2M). It re-hosts the
# greghavens fable-5 set AND Glint-Research (both already ingested separately!).
# Filter by its `first_source_dataset` column to skip rows we already have.
# (AGPL-3.0 upstreams like 1EYE4ALL/lordx64/Swarm-AI-Research are kept: this model
# is personal-use only and will never be published.)
CROWNELIUS_EXCLUDE = (
    "greghavens/",                    # already ingested directly (moonshiner series)
    "Glint-Research/",                # already ingested directly (its own 2% slice)
    "armand0e/claude-fable-5-claude-code",  # verbatim copy of Glint-Research
)

def crownelius_ok(row, s):
    src = row.get("first_source_dataset") or ""
    return (not any(src.startswith(p) for p in CROWNELIUS_EXCLUDE)
            and not looks_windows_centric(s))

pools["crownelius"] = collect("Crownelius/Complete-FABLE.5-traces-2M", "crownelius",
                              n_raw=25_000, extra_filter=crownelius_ok)

# %%
for source, repo in (("glint", "Glint-Research/Fable-5-traces"),
                     ("alin_cc", "AlinCiocan/fable-5-claude-code-traces")):
    raw = load_dataset(repo, split="train")
    print(source, raw.column_names)
    rows = []
    for row in raw:
        try:
            s = adapt_row(row, source)
        except ValueError:
            # Pi-session exports usually nest turns under some list column — retry with it.
            list_col = next((c for c in raw.column_names if isinstance(row[c], list)), None)
            if list_col is None:
                raise
            s = adapt_row({"messages": row[list_col]}, source)
        if is_complete_trajectory(s) and char_len(s) < MAX_CHARS:
            rows.append(s)
    pools[source] = dedupe(rows)
    print(f"{source}: kept {len(pools[source])}/{len(raw)}")

# %%
pools["ocr"] = collect("nvidia/OpenCodeReasoning", "ocr", n_raw=30_000,
                       split="split_0", config="split_0")
pools["openthoughts"] = collect("open-thoughts/OpenThoughts3-1.2M", "openthoughts", n_raw=15_000)

# %% [markdown]
# ## Cross-pool dedupe, then mix, tokenize with assistant-only masks, push
# The Fable-5 family pools overlap heavily (mirrors of mirrors), so per-pool dedupe
# is not enough: drop any sample whose content hash already appeared in an
# earlier-priority pool before shares are computed.

# %%
from src.data_prep import content_hash
PRIORITY = ["nexlab", "gh_fable5", "gh_gpt56", "gh_glm52", "kimi", "glint", "alin_cc",
            "crownelius", "swe_smith", "ocr", "openthoughts"]
seen = set()
for name in PRIORITY:
    unique = []
    for s in pools[name]:
        h = content_hash(s)
        if h not in seen:
            seen.add(h)
            unique.append(s)
    if len(unique) < len(pools[name]):
        print(f"{name}: removed {len(pools[name]) - len(unique)} cross-pool duplicates")
    pools[name] = unique

# %%
mix = build_mix(pools, RATIOS, total=TOTAL)
from collections import Counter
print(Counter(s["source"] for s in mix))
# tiny pools (alin_cc has ~18 rows, glm-5.2 ~1.8K) underfill their share — that's
# expected; a large shortfall means a loader upstream silently failed
assert len(mix) >= 0.85 * TOTAL, f"mix underfilled: {len(mix)}/{TOTAL} — check pool sizes above"

# %%
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)

rows, dropped = [], 0
for i, s in enumerate(mix):
    enc = tok.apply_chat_template(s["messages"], enable_thinking=True,
                                  return_assistant_tokens_mask=True, return_dict=True,
                                  truncation=True, max_length=MAX_LEN)
    labels = [t if m else -100 for t, m in zip(enc["input_ids"], enc["assistant_masks"])]
    if sum(m != -100 for m in labels) < 16:   # truncation ate the assistant turns
        dropped += 1
        continue
    rows.append({"input_ids": enc["input_ids"], "labels": labels, "source": s["source"]})
    if i % 2000 == 0:
        print(f"{i}/{len(mix)} tokenized, {dropped} dropped")
print(f"final: {len(rows)} samples, {sum(len(r['input_ids']) for r in rows) / 1e6:.1f}M tokens")

# %%
from datasets import Dataset
from src.hub_utils import ensure_repo
del pools, mix  # free several GB before Arrow materializes alongside `rows`
ds = Dataset.from_list(rows)
ensure_repo(ART.sft_dataset, repo_type="dataset")
ds.push_to_hub(ART.sft_dataset, private=True)
ds

# %% [markdown]
# Sanity: decode one sample and eyeball that ONLY assistant spans carry labels.

# %%
r = ds[0]
print(tok.decode(r["input_ids"])[:2000])
supervised = tok.decode([t for t, l in zip(r["input_ids"], r["labels"]) if l != -100])
print("\n--- supervised tokens only ---\n", supervised[:1000])
