"""Calibration statistics for expert pruning (REAP + frequency baseline).

Verified Laguna routing semantics (transformers modeling_laguna.py):

    routing_scores    = sigmoid(router_logits)                  # NOT softmax
    scores_for_select = routing_scores + e_score_correction_bias  # selection ONLY
    selected          = topk(scores_for_select, top_k=8)
    routing_weights   = routing_scores.gather(selected)
    routing_weights  /= routing_weights.sum(-1, keepdim=True)   # renormalized
    output           *= moe_routed_scaling_factor (2.5)         # rank-invariant

REAP saliency (arXiv 2510.13999) is the CONDITIONAL mean over tokens where
expert j is active — deliberately decoupled from activation frequency:

    S_j = mean_{x : j selected}  g_j(x) * ||f_j(x)||_2

Two module layouts exist at runtime:
- per-expert nn.Linear submodules (poolside trust_remote_code code) → we hook
  each expert's forward and get true output norms;
- fused 3D nn.Parameter experts (native transformers LagunaExperts) → per-expert
  hooks are impossible; S_j falls back to the conditional mean gate alone
  (documented approximation; the gate term dominates ranking in practice).

Router hooks capture logits + the module's e_score_correction_bias when present.
"""

import json

import torch

from . import project_config as cfg
from .laguna_arch import find_moe_modules


class RouterStatsCollector:
    """Hooks each sparse layer's router; reproduces Laguna's selection exactly."""

    def __init__(self, model):
        self.moe = find_moe_modules(model)
        E, L = cfg.NUM_EXPERTS, cfg.NUM_LAYERS
        self.gate_sums = torch.zeros(L, E, dtype=torch.float64)   # sum of applied weights when active
        self.active_counts = torch.zeros(L, E, dtype=torch.float64)
        self.tokens = torch.zeros(L, dtype=torch.float64)
        self.bias = {}      # layer -> e_score_correction_bias tensor (if found)
        self.handles = []

    def _find_bias(self, layer, router_module, model):
        for owner in (router_module, getattr(router_module, "experts", None)):
            b = getattr(owner, "e_score_correction_bias", None) if owner is not None else None
            if torch.is_tensor(b) and b.numel() == cfg.NUM_EXPERTS:
                return b.detach().float().cpu()
        # fused layout keeps it on the experts module, sibling of the router
        for name, module in model.named_modules():
            if f".layers.{layer}." in name and hasattr(module, "e_score_correction_bias"):
                b = module.e_score_correction_bias
                if torch.is_tensor(b) and b.numel() == cfg.NUM_EXPERTS:
                    return b.detach().float().cpu()
        return None

    def _hook(self, layer):
        def hook(module, args, output):
            t = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(t) or t.shape[-1] != cfg.NUM_EXPERTS:
                return
            logits = t.reshape(-1, cfg.NUM_EXPERTS).float()
            scores = torch.sigmoid(logits)
            select_scores = scores + self.bias[layer].to(scores.device) \
                if self.bias.get(layer) is not None else scores
            _, idx = select_scores.topk(cfg.TOP_K, dim=-1)
            gates = scores.gather(-1, idx)
            gates = gates / gates.sum(-1, keepdim=True)           # norm_topk_prob=True
            flat_g = torch.zeros_like(scores)
            flat_g.scatter_(1, idx, gates)
            active = torch.zeros_like(scores)
            active.scatter_(1, idx, 1.0)
            self.gate_sums[layer] += flat_g.sum(dim=0).double().cpu()
            self.active_counts[layer] += active.sum(dim=0).double().cpu()
            self.tokens[layer] += logits.shape[0]
        return hook

    def attach(self, model):
        for layer, mods in self.moe.items():
            name, mod = mods["router"]
            self.bias[layer] = self._find_bias(layer, mod, model)
            self.handles.append(mod.register_forward_hook(self._hook(layer)))
        missing_bias = [l for l, b in self.bias.items() if b is None]
        if missing_bias:
            print(f"note: e_score_correction_bias not found for layers {missing_bias[:5]}… — "
                  f"selection approximated by plain sigmoid scores")
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stats(self):
        return {
            "freq": self.active_counts / self.tokens.unsqueeze(1).clamp(min=1),
            "mean_gate_active": self.gate_sums / self.active_counts.clamp(min=1),
        }


class ExpertNormCollector:
    """Per-expert output L2 norms — only possible with per-expert submodules
    (trust_remote_code layout). On the fused native layout attach() reports 0
    hooks and REAP saliency proceeds without the norm term."""

    def __init__(self, model):
        self.moe = find_moe_modules(model)
        E, L = cfg.NUM_EXPERTS, cfg.NUM_LAYERS
        self.norm_sums = torch.zeros(L, E, dtype=torch.float64)
        self.counts = torch.zeros(L, E, dtype=torch.float64)
        self.handles = []

    def _hook(self, layer, expert_idx):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            n = out.shape[0] if out.dim() > 1 else 1
            self.counts[layer, expert_idx] += n
            self.norm_sums[layer, expert_idx] += out.float().norm(dim=-1).sum().item()
        return hook

    def attach(self):
        import re
        for layer, mods in self.moe.items():
            for name, mod in mods["experts"]:
                if not re.search(r"\.experts\.\d+$", name):
                    continue  # fused container / ModuleList parent — not a single expert
                idx = int(name.rsplit(".", 1)[-1])
                self.handles.append(mod.register_forward_hook(self._hook(layer, idx)))
        print(f"ExpertNormCollector: {len(self.handles)} hooks "
              f"({'per-expert layout' if self.handles else 'fused layout — norm term unavailable'})")
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stats(self):
        return {"mean_norm": self.norm_sums / self.counts.clamp(min=1)}


def reap_scores(router_stats, norm_stats=None, min_active_frac=1e-4):
    """REAP saliency. With the norm term: conditional mean gate × conditional mean
    output norm (true REAP — NOT frequency-weighted). Without it (fused runtime
    layout, no per-expert hooks possible), the conditional mean gate alone is
    near-flat (~1/top_k for everyone), so fall back to gate × freq — the REAP
    repo's 'weighted frequency' criterion. Experts activated on fewer than
    min_active_frac of tokens rank last regardless (a single lucky activation
    must not outrank a workhorse)."""
    gate, freq = router_stats["mean_gate_active"], router_stats["freq"]
    if norm_stats is not None and norm_stats["mean_norm"].sum() > 0:
        s = gate * norm_stats["mean_norm"]
    else:
        s = gate * freq
    return torch.where(freq >= min_active_frac, s, torch.full_like(s, -1.0))


def freq_scores(router_stats, norm_stats=None):
    return router_stats["freq"]


def keep_lists(scores, keep_fraction: float) -> dict:
    """{layer_idx: sorted expert indices to KEEP}; uniform count per layer
    (required by converters and mlx-lm, which have no per-layer expert counts)."""
    k = int(round(cfg.NUM_EXPERTS * keep_fraction))
    out = {}
    for layer in cfg.SPARSE_LAYERS:
        row = scores[layer]
        # all-equal scores mean topk returns experts 0..k-1 — a meaningless prune
        # that would still "work" and upload; refuse instead
        assert row.max() > row.min(), f"degenerate saliency at layer {layer} — no calibration signal"
        top = torch.topk(row, k).indices.sort().values
        out[int(layer)] = [int(i) for i in top]
    return out


def save_stats(path, router_stats, norm_stats=None, meta=None):
    payload = {"meta": meta or {},
               "freq": router_stats["freq"].tolist(),
               "mean_gate_active": router_stats["mean_gate_active"].tolist(),
               "mean_norm": norm_stats["mean_norm"].tolist() if norm_stats else None}
    with open(path, "w") as f:
        json.dump(payload, f)


def load_stats(path):
    with open(path) as f:
        p = json.load(f)
    router = {"freq": torch.tensor(p["freq"], dtype=torch.float64),
              "mean_gate_active": torch.tensor(p["mean_gate_active"], dtype=torch.float64)}
    norm = ({"mean_norm": torch.tensor(p["mean_norm"], dtype=torch.float64)}
            if p.get("mean_norm") else None)
    return router, norm
