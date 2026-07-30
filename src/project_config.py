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

# Budget guardrails (Colab compute units). Google does NOT publish unit rates and
# they vary over time — READ THE LIVE RATE off Colab's resource panel and update
# this number before the big Phase-4 spend. 8.5 is a conservative planning value.
TOTAL_UNITS = 300
A100_UNITS_PER_HOUR = 8.5
SFT_MAX_HOURS = 18         # notebook 05 aborts its projection check above this
