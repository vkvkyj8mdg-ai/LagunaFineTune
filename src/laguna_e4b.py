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
