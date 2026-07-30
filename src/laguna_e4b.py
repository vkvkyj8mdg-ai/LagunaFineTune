"""Laguna adapter for the experts4bit-qlora streaming loader.

The package's `load_moe_4bit_streaming` handles per-expert SwiGLU checkpoints
(OLMoE/Qwen3-MoE layout — exactly Laguna's on-disk format) but gates on a
hardcoded `SUPPORTED_ARCHITECTURES` registry that lacks "laguna". Verified
compatible (2026-07-30, transformers main):

- LagunaSparseMoeBlock calls `self.experts(hidden_states, selected_experts,
  routing_weights)` — the exact `ExpertsLoRA.forward` contract.
- Routed ×2.5 scaling, sigmoid router, and the shared expert all live OUTSIDE
  the experts module, so the swap is transparent.
- One rename needed: on disk the router bias is `mlp.experts.
  e_score_correction_bias`; the built module owns it at `mlp.gate.
  e_score_correction_bias` (native transformers applies this same rename in
  from_pretrained; the streaming loader reads raw shard keys, so we inject it
  via its LEGACY_KEY_RENAMES hook).
- Dense layer 0 has no expert tensors and is skipped by the loader naturally.

The loader returns the model with trainable per-expert LoRA already attached
(ExpertsLoRA, rank `r` per expert — the 'LoRA Without Regret' MoE recipe);
add attention LoRA with `experts4bit_qlora.add_attention_lora`.
"""

import torch
from experts4bit_qlora import loader, verify_moe_4bit


def register_laguna():
    # plain assignment, NOT setdefault: kernels persist across runs, and a stale
    # registration from an earlier (buggy) shim version must be overwritten
    loader.SUPPORTED_ARCHITECTURES["laguna"] = "mlp.experts"
    loader.SUPPORTED_MODEL_TYPES.add("laguna")
    loader.LEGACY_KEY_RENAMES["laguna"] = (
        ("mlp.experts.e_score_correction_bias", "mlp.gate.e_score_correction_bias"),
        # disk stores the shared expert SINGULAR; the built module is PLURAL
        # (native from_pretrained applies this same rename via WeightRenaming)
        ("mlp.shared_expert.", "mlp.shared_experts."),
    )


def load_laguna_4bit(model_id, r=8, alpha=16, device="cuda", dtype=torch.bfloat16, **kw):
    """Stream-load a Laguna checkpoint with NF4 experts + per-expert LoRA(r).

    Returns (model, hf_config). Peak memory stays ~1 expert stack above the
    quantized footprint — the full bf16 model never materializes anywhere.
    """
    register_laguna()
    model, config = loader.load_moe_4bit_streaming(
        model_id, device=device, dtype=dtype, r=r, alpha=alpha, **kw)
    report = verify_moe_4bit(model)
    assert report["n_unquantized"] == 0, f"unquantized expert stacks: {report['unquantized']}"
    print(f"quantized expert stacks: {report['n_quantized']} "
          f"({report['quantized'][0]['quant_type']})")
    return model, config


def add_extra_lora(model, r=16, alpha=32, dtype=torch.bfloat16):
    """LoRA-wrap the Linears `add_attention_lora` misses: the per-head attention
    output gate (g_proj) and the always-active shared expert — the highest-
    leverage healing surface after expert pruning. Idempotent (a wrapped module
    is a LoRALinear, not nn.Linear)."""
    import torch.nn as nn
    from experts4bit_qlora import LoRALinear

    n = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        wanted = (leaf == "g_proj" and "self_attn" in name) or \
                 ("shared_expert" in name and leaf.endswith("_proj"))
        if wanted:
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, leaf, LoRALinear(module, r=r, alpha=alpha, dtype=dtype))
            n += 1
    return n


# ---- checkpointing: TRAINABLE PARAMS ONLY -----------------------------------
# save_pretrained / Trainer checkpoints cannot serialize Experts4bit's quant-
# state dicts, and the frozen NF4 base is reproducible from the Hub anyway.
# A checkpoint is therefore just {name: tensor} for requires_grad params
# (~1-2GB) plus the step counter.

def save_trainable(model, path, step=None, alpha=None):
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"step": step, "alpha": alpha, "state": state}, path)
    return len(state)


def load_trainable(model, path):
    """Load an adapter checkpoint; returns the saved step (or None)."""
    payload = torch.load(path, map_location="cpu")
    result = model.load_state_dict(payload["state"], strict=False)
    assert not result.unexpected_keys, f"unexpected keys: {result.unexpected_keys[:5]}"
    loaded = set(payload["state"])
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert loaded == trainable, (f"adapter/model mismatch: {len(loaded - trainable)} extra, "
                                 f"{len(trainable - loaded)} missing — same r/alpha/targets?")
    return payload.get("step")


def merge_adapter_into_checkpoint(src_dir, adapter_path, out_dir, alpha=None):
    """Fold a trainable-params adapter into a bf16 per-expert laguna checkpoint.

    CPU shard-streaming (same pattern as prune_experts): never materializes the
    full model. Handles both adapter kinds this pipeline trains:
    - LoRALinear (attention q/k/v/o/g_proj, shared expert): key `<path>.lora_A/B`
      merges into on-disk `<path>.weight` as W += (alpha/r) * B @ A.
    - ExpertsLoRA (per-expert, fused [E, r, ...] tensors under `...mlp.experts.`)
      merges into per-expert `experts.{e}.{gate,up,down}_proj.weight`; gate_up
      rows 0:inter are gate, inter:2*inter are up (the loader's concat order).
    scaling alpha comes from the adapter payload unless overridden.
    """
    import json
    import re
    import shutil
    from pathlib import Path

    from safetensors import safe_open

    from .prune_experts import _ShardWriter

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(adapter_path, map_location="cpu")
    alpha = alpha if alpha is not None else payload.get("alpha")
    assert alpha is not None, "adapter has no stored alpha — pass alpha= explicitly"
    state = payload["state"]

    linear_deltas, expert_adapters = {}, {}
    for key in list(state):
        if key.endswith(".lora_A"):
            path = key[: -len(".lora_A")]
            A, B = state[key].float(), state[path + ".lora_B"].float()
            linear_deltas[path + ".weight"] = ((alpha / A.shape[0]) * (B @ A))
        elif key.endswith(".gate_up_lora_A"):
            prefix = key[: -len(".gate_up_lora_A")]           # ...mlp.experts
            expert_adapters[prefix] = {
                "guA": state[prefix + ".gate_up_lora_A"].float(),
                "guB": state[prefix + ".gate_up_lora_B"].float(),
                "dA": state[prefix + ".down_lora_A"].float(),
                "dB": state[prefix + ".down_lora_B"].float(),
            }

    exp_re = re.compile(r"^(.*\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$")
    with open(src_dir / "model.safetensors.index.json") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    writer = _ShardWriter(out_dir)
    merged_linear = merged_expert = 0
    for shard in shards:
        with safe_open(str(src_dir / shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name in linear_deltas:
                    t = (t.float() + linear_deltas[name]).to(t.dtype)
                    merged_linear += 1
                elif (m := exp_re.match(name)) and m.group(1) in expert_adapters:
                    ad = expert_adapters[m.group(1)]
                    e, proj = int(m.group(2)), m.group(3)
                    if proj == "down":
                        r = ad["dA"].shape[1]
                        delta = (alpha / r) * (ad["dB"][e] @ ad["dA"][e])
                    else:
                        r = ad["guA"].shape[1]
                        gu = (alpha / r) * (ad["guB"][e] @ ad["guA"][e])
                        inter = gu.shape[0] // 2
                        delta = gu[:inter] if proj == "gate" else gu[inter:]
                    t = (t.float() + delta).to(t.dtype)
                    merged_expert += 1
                writer.add(name, t)
    writer.finish()
    for aux in src_dir.glob("*"):
        if aux.is_file() and not aux.name.endswith(".safetensors") \
                and aux.name != "model.safetensors.index.json":
            shutil.copy(aux, out_dir / aux.name)
    assert merged_linear == len(linear_deltas), \
        f"only {merged_linear}/{len(linear_deltas)} linear deltas found on disk"
    print(f"merged {merged_linear} linear + {merged_expert} per-expert projections -> {out_dir}")
