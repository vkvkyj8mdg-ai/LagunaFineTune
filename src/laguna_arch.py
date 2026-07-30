"""Runtime introspection of the laguna architecture.

The `laguna` model type is new and its module naming may differ between the
trust_remote_code implementation and the native Transformers v5.7+ one, so
nothing here hardcodes module paths — we discover them by pattern and shape.
"""

import re
from collections import defaultdict

from . import project_config as cfg

# Candidate substrings for the three MoE roles. Extend if discovery fails.
ROUTER_PATTERNS = (r"\.gate$", r"\.router$", r"\.gate_proj_router$")
EXPERT_PATTERNS = (r"\.experts$", r"\.experts\.\d+$")
SHARED_PATTERNS = (r"shared_expert", r"shared_experts")


def find_moe_modules(model):
    """Return {layer_idx: {"router": (name, module), "experts": [(name, module)...],
    "shared": (name, module) | None}} for every sparse layer.

    Handles both per-expert submodules (…experts.0, …experts.1, …) and fused
    expert containers (a single …experts module holding 3D weights).
    """
    layer_re = re.compile(r"\.layers\.(\d+)\.")
    out = defaultdict(lambda: {"router": None, "experts": [], "shared": None})

    for name, module in model.named_modules():
        m = layer_re.search(name)
        if not m:
            continue
        layer = int(m.group(1))
        if layer in cfg.DENSE_MLP_LAYERS:
            continue
        if any(re.search(p, name) for p in ROUTER_PATTERNS):
            out[layer]["router"] = (name, module)
        elif any(re.search(p, name) for p in SHARED_PATTERNS):
            # keep the outermost shared-expert module only
            if out[layer]["shared"] is None or len(name) < len(out[layer]["shared"][0]):
                out[layer]["shared"] = (name, module)
        elif any(re.search(p, name) for p in EXPERT_PATTERNS):
            out[layer]["experts"].append((name, module))

    out = dict(out)
    missing = [l for l in cfg.SPARSE_LAYERS if l not in out or out[l]["router"] is None]
    if missing:
        sample = [n for n, _ in list(model.named_modules())[:80]]
        raise RuntimeError(
            f"MoE discovery failed for layers {missing[:5]}… — module naming differs from "
            f"expected patterns. First module names for inspection:\n" + "\n".join(sample)
        )
    return out


def attention_lora_targets(model, include_shared_expert=True):
    """Leaf names of Linears to target with LoRA (PEFT accepts suffixes).

    Includes g_proj — laguna's per-head attention output gate, a real nn.Linear
    in every attention block (this is what config "gating: per-head" refers to;
    the MoE router is a separate plain [experts, hidden] matrix).

    include_shared_expert: also target mlp.shared_expert.{gate,up,down}_proj —
    the always-active MLP path, the highest-leverage healing surface after
    expert pruning that is reachable without touching fused 3D expert params.
    """
    leaves = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ not in ("Linear", "Linear4bit", "Linear8bitLt"):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if "self_attn" in name and re.fullmatch(r"[qkvog]_proj", leaf):
            leaves.add(leaf)
        elif include_shared_expert and "shared_expert" in name and leaf.endswith("_proj"):
            # derive the parent segment from the real module path: native transformers
            # names it `shared_experts` (plural) while the checkpoint / remote code use
            # `shared_expert` — hardcoding either silently matches nothing on the other
            parent = name.rsplit(".", 2)[-2]
            leaves.add(f"{parent}.{leaf}")
    if not leaves:
        raise RuntimeError("No attention projections found — inspect model.named_modules().")
    return sorted(leaves)


def estimate_params(keep_experts: int) -> dict:
    """Rough parameter counts after keeping `keep_experts` of 256 per sparse layer."""
    per_expert = 3 * cfg.HIDDEN_SIZE * cfg.MOE_INTERMEDIATE          # gate/up/down
    routed = len(cfg.SPARSE_LAYERS) * keep_experts * per_expert
    # everything else (attention, shared experts, dense layer 0, embeddings, norms): ~1.8B
    other = 1.8e9
    total = routed + other
    return {
        "total_B": round(total / 1e9, 2),
        "bf16_GB": round(total * 2 / 2**30, 1),
        "q4_GB": round(total * 0.56 / 2**30, 1),  # ~4.5 bpw effective incl. overhead
    }
