"""Single place for every name/knob the notebooks share.

Edit HF_USER and REPO_URL before running anything on Colab.
"""

from dataclasses import dataclass, field

HF_USER = "bobamelo"  # HuggingFace username — artifact repos are created under this namespace
REPO_URL = "https://github.com/vkvkyj8mdg-ai/LagunaFineTune.git"  # this repo, for cloning in Colab

BASE_MODEL = "poolside/Laguna-XS-2.1"
BASE_MODEL_INT4 = "poolside/Laguna-XS-2.1-INT4"  # poolside's own INT4, for cheap vLLM eval

# Architecture constants (mirrors reference/config.laguna-xs-2.1.json — do not edit)
NUM_LAYERS = 40
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE = 512
DENSE_MLP_LAYERS = (0,)  # mlp_only_layers: layer 0 has a dense MLP, no experts
SPARSE_LAYERS = tuple(i for i in range(NUM_LAYERS) if i not in DENSE_MLP_LAYERS)


def hub(name: str) -> str:
    """Namespaced private-repo id for artifacts, e.g. hub('reap50') -> '<you>/laguna-xs-2.1-reap50'."""
    return f"{HF_USER}/laguna-xs-2.1-{name}"


@dataclass
class Artifacts:
    router_stats: str = field(default_factory=lambda: hub("router-stats"))       # dataset repo
    sft_dataset: str = field(default_factory=lambda: hub("sft-mix"))             # dataset repo
    pruned_reap25: str = field(default_factory=lambda: hub("reap25"))    # keep 192 — safest quality
    pruned_reap375: str = field(default_factory=lambda: hub("reap375"))  # keep 160 — fit/quality sweet spot
    pruned_reap50: str = field(default_factory=lambda: hub("reap50"))    # keep 128 — most aggressive
    pruned_freq50: str = field(default_factory=lambda: hub("freq50"))    # baseline method (study)
    sft_adapter: str = field(default_factory=lambda: hub("sft-lora"))
    sft_merged: str = field(default_factory=lambda: hub("agentic-pruned"))
    gguf: str = field(default_factory=lambda: hub("agentic-pruned-gguf"))
    mlx: str = field(default_factory=lambda: hub("agentic-pruned-mlx-4bit"))


ART = Artifacts()

# SFT LoRA geometry — must be IDENTICAL wherever the adapter is built or reloaded
# (notebooks 05, 06, 07): attention r=32/alpha=32 ("LoRA Without Regret" SFT recipe),
# per-expert r=4 (= total 32 / top-8, its MoE recipe; attached by the streaming loader).
SFT_LORA_R = 32
SFT_LORA_ALPHA = 32
SFT_EXPERT_LORA_R = 4
SFT_SAVE_EVERY = 40   # steps between adapter checkpoints (~30-40 min)

# Budget guardrails (Colab compute units). MEASURED 2026-07-30 on this account:
# 15 units for ~50min A100 + ~10min T4 => A100 ≈ 17.5 units/hr. Balance then: 255.
TOTAL_UNITS = 255
A100_UNITS_PER_HOUR = 17.5
SFT_MAX_HOURS = 8          # ≈140 units at the measured rate; notebook 05's
                           # projection check aborts above this — trim the dataset
                           # (04 TOTAL) or MAX_LEN instead of blowing the budget
