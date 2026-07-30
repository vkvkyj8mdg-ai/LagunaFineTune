"""CPU weight surgery: build a smaller Laguna checkpoint keeping only selected experts.

Streams the BF16 safetensors shards one tensor at a time, so peak RAM stays at
one-tensor size — runs on a Colab high-RAM CPU runtime (no GPU needed).

Config compatibility rule (keeps vLLM / llama.cpp / MLX converters working):
we ONLY change `num_experts` in config.json; architecture name, layout and all
other fields stay untouched. All sparse layers keep the same expert COUNT
(different expert IDs per layer is fine — routers are re-sliced per layer).

Verified against the actual checkpoint index (30,513 tensors):
- experts are stored per-expert 2D (`…mlp.experts.<i>.{gate,up,down}_proj.weight`)
- router is `…mlp.gate.weight`, shape [256, hidden] (config's "per-head gating"
  refers to ATTENTION output gating `self_attn.g_proj`, not the MoE router)
- `…mlp.experts.e_score_correction_bias`, shape [256], exists in all 39 sparse
  layers and MUST be sliced too (handled by the generic dim0==num_experts rule).
The heads-folded router branch is kept only as a safety net for other layouts.
"""

import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import project_config as cfg

EXPERT_SUBMODULE_RE = re.compile(r"^(.*\.layers\.(\d+)\..*\.experts\.)(\d+)(\..+)$")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
ROUTER_HINT = re.compile(r"\.(gate|router)\.(weight|bias|e_score_correction_bias)$")

SHARD_BYTES = 5 * 2**30


def _layer_of(name):
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _slice_router_tensor(t, keep, router_layout):
    """Slice per-expert rows out of a router weight/bias. keep: LongTensor of expert ids."""
    E = cfg.NUM_EXPERTS
    if t.shape[0] == E:
        return t[keep]
    if t.shape[0] % E == 0:
        heads = t.shape[0] // E
        if router_layout == "heads_major":       # rows ordered [head0_e0..head0_e255, head1_e0..]
            v = t.reshape(heads, E, *t.shape[1:])[:, keep]
            return v.reshape(heads * len(keep), *t.shape[1:])
        elif router_layout == "experts_major":   # rows ordered [e0_head0..e0_headH, e1_head0..]
            v = t.reshape(E, heads, *t.shape[1:])[keep]
            return v.reshape(len(keep) * heads, *t.shape[1:])
        raise ValueError("router dim0 is heads*experts — set router_layout after checking modeling_laguna.py")
    raise ValueError(f"router tensor dim0 {t.shape[0]} not divisible by num_experts {E}")


class _ShardWriter:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.buf, self.buf_bytes, self.shard_idx = {}, 0, 0
        self.weight_map = {}

    def add(self, name, tensor):
        self.buf[name] = tensor.contiguous()
        self.buf_bytes += tensor.numel() * tensor.element_size()
        if self.buf_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        self.shard_idx += 1
        fname = f"model-{self.shard_idx:05d}.safetensors"  # renamed to -of-N in finish()
        save_file(self.buf, str(self.out_dir / fname), metadata={"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = fname
        self.buf, self.buf_bytes = {}, 0

    def finish(self):
        self.flush()
        total = self.shard_idx
        renames = {}
        for i in range(1, total + 1):
            old = f"model-{i:05d}.safetensors"
            new = f"model-{i:05d}-of-{total:05d}.safetensors"
            (self.out_dir / old).rename(self.out_dir / new)
            renames[old] = new
        self.weight_map = {k: renames[v] for k, v in self.weight_map.items()}
        size = sum((self.out_dir / v).stat().st_size for v in set(self.weight_map.values()))
        index = {"metadata": {"total_size": size}, "weight_map": self.weight_map}
        with open(self.out_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=1)


def prune_checkpoint(src_dir, out_dir, keep: dict, router_layout="verify_me"):
    """src_dir: local snapshot of the BF16 model. keep: {layer_idx: [expert ids]}.

    Every sparse layer must keep the same number of experts.
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {int(l): torch.tensor(sorted(ids), dtype=torch.long) for l, ids in keep.items()}
    counts = {len(v) for v in keep.values()}
    assert len(counts) == 1, f"all layers must keep the same expert count, got {counts}"
    new_E = counts.pop()
    renumber = {l: {int(old): new for new, old in enumerate(ids.tolist())} for l, ids in keep.items()}

    index_path = src_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = [p.name for p in sorted(src_dir.glob("*.safetensors"))]

    writer = _ShardWriter(out_dir)
    dropped = kept = fused = 0
    seen = set()
    for shard in shards:
        with safe_open(str(src_dir / shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name in seen:  # tensor duplicated across input shards — write once
                    continue
                seen.add(name)
                layer = _layer_of(name)
                sub = EXPERT_SUBMODULE_RE.match(name)
                if sub and layer in keep:
                    old_idx = int(sub.group(3))
                    if old_idx not in renumber[layer]:
                        dropped += 1
                        continue
                    new_name = f"{sub.group(1)}{renumber[layer][old_idx]}{sub.group(4)}"
                    writer.add(new_name, f.get_tensor(name))
                    kept += 1
                    continue
                t = f.get_tensor(name)
                if layer in keep and ".mlp." in name and t.shape and t.shape[0] == cfg.NUM_EXPERTS:
                    # any per-expert-indexed tensor in MoE scope: router gate.weight
                    # [E, hidden], e_score_correction_bias [E], fused 3D [E, out, in]
                    t = t[keep[layer]]
                    if t.dim() == 3:
                        fused += 1
                elif layer in keep and ROUTER_HINT.search(name) and t.shape[0] % cfg.NUM_EXPERTS == 0:
                    t = _slice_router_tensor(t, keep[layer], router_layout)
                writer.add(name, t)
    writer.finish()

    # a naming drift that slices routers but misses experts would silently destroy
    # the router/expert pairing — refuse to produce such a checkpoint
    expected = len(keep) * new_E * 3
    if kept != expected and fused != len(keep) * 2:
        raise RuntimeError(
            f"expert slicing mismatch: kept {kept} per-expert tensors (expected {expected}) "
            f"and sliced {fused} fused 3D tensors (expected {len(keep) * 2}) — naming drift?")

    # config: change ONLY the expert count
    with open(src_dir / "config.json") as f:
        config = json.load(f)
    config["num_experts"] = new_E
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=1)

    for aux in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "chat_template.jinja", "vocab.json", "merges.txt",
                "configuration_laguna.py", "modeling_laguna.py"):
        if (src_dir / aux).exists():
            shutil.copy(src_dir / aux, out_dir / aux)

    # generation_config: drop the DFlash speculative_config — that EAGLE-style drafter
    # is keyed to the ORIGINAL model's hidden states and is invalid after pruning/SFT;
    # a stale pointer makes engines try to download/load it.
    gen_cfg_path = src_dir / "generation_config.json"
    if gen_cfg_path.exists():
        with open(gen_cfg_path) as f:
            gen_cfg = json.load(f)
        gen_cfg.pop("speculative_config", None)
        with open(out_dir / "generation_config.json", "w") as f:
            json.dump(gen_cfg, f, indent=1)

    with open(out_dir / "pruning_manifest.json", "w") as f:
        json.dump({"kept_per_layer": {str(l): v.tolist() for l, v in keep.items()},
                   "new_num_experts": new_E, "router_layout": router_layout}, f)
    print(f"pruned checkpoint written to {out_dir}: kept {kept} expert tensors, dropped {dropped}, "
          f"num_experts {cfg.NUM_EXPERTS} -> {new_E}")
