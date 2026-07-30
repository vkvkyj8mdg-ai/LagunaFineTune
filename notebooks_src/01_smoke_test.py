# %% [markdown]
# # 01 — Smoke test (run FIRST, before spending real budget)
#
# **Runtime: A100 40GB** (Runtime → Change runtime type → A100).
# Est. cost: ~1h ≈ 10 units, mostly the 66GB weight download.
#
# Verifies, in order:
# 1. `poolside/Laguna-XS-2.1` loads in 4-bit (bitsandbytes NF4) under 40GB
# 2. the chat template renders thinking + tool calls the way our data prep assumes
# 3. MoE module discovery works (routers/experts found; per-head gating shape visible)
# 4. a tiny LoRA training step runs without OOM/NaN  ← the go/no-go signal
#
# If step 4 fails, STOP and read the fallback cell at the bottom.

# %%
# ── Bootstrap: repo, deps, auth ──────────────────────────────────────────────
REPO_URL = "https://github.com/vkvkyj8mdg-ai/LagunaFineTune.git"   # ← your fork
import os, sys
if not os.path.isdir("/content/LagunaFineTune"):
    !git clone {REPO_URL} /content/LagunaFineTune
sys.path.insert(0, "/content/LagunaFineTune")
os.environ["HF_HOME"] = "/content/hf_cache"
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from src import project_config as cfg
assert cfg.HF_USER != "CHANGE_ME", "Edit src/project_config.py (HF_USER + REPO_URL), push, re-clone."

# %%
!pip install -q -U "transformers>=5.7" accelerate bitsandbytes peft trl datasets
!nvidia-smi
import torch, transformers
# HARD gate: transformers < 5.7 has no laguna WeightRenaming rules and silently
# RE-INITIALIZES ~13B expert params instead of erroring (looks like "pruning broke it").
from packaging.version import Version
assert Version(transformers.__version__) >= Version("5.7.0"), transformers.__version__
gb = torch.cuda.get_device_properties(0).total_memory / 2**30
assert gb > 35, f"Need an A100 40GB, got {gb:.0f}GB — change runtime type."

# %% [markdown]
# ## 1. Tokenizer + chat template sanity
# The template stores reasoning in `reasoning_content` and renders OpenAI-style
# `tool_calls` into `<tool_call>name<arg_key>…` — confirm both visually.

# %%
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
demo = [
    {"role": "user", "content": "List the files here."},
    {"role": "assistant", "content": "", "reasoning_content": "I should run ls.",
     "tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "ls -la"}}}]},
    {"role": "tool", "content": "README.md src/"},
    {"role": "assistant", "content": "Two entries: README.md and src/.",
     "reasoning_content": "Simple summary."},
]
print(tok.apply_chat_template(demo, tokenize=False, enable_thinking=True))
ids = tok.apply_chat_template(demo, enable_thinking=True, return_assistant_tokens_mask=True,
                              return_dict=True)
n_assist = sum(ids["assistant_masks"])
print(f"\nassistant tokens: {n_assist}/{len(ids['input_ids'])}  (must be >0 → loss masking works)")
assert n_assist > 0

# %% [markdown]
# ## 2. Load the model with 4-bit EXPERTS (~66GB download, 20–40 min, cached per session)
# Verified blocker: BOTH the native transformers laguna class AND poolside's remote
# code store routed experts as fused 3D nn.Parameters. bitsandbytes only converts
# nn.Linear, so a plain `load_in_4bit` silently keeps 95% of params in bf16
# (~60GB → CPU-offloaded/dead on A100-40; bnb issue #1849, fix PR #1965 unmerged).
# Primary loader: `experts4bit-qlora` (standalone NF4 for fused experts, works with
# stock bnb). The footprint assert below is THE go/no-go for the whole pipeline.

# %%
!pip install -q experts4bit-qlora
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
try:
    # NOTE: verify the exact entrypoint against the package README —
    # https://github.com/pjordanandrsn/experts4bit-qlora
    import experts4bit_qlora
    model = experts4bit_qlora.load_moe_4bit(cfg.BASE_MODEL, device_map={"": 0})
except Exception as e:
    print(f"experts4bit-qlora path failed ({e!r}) — plain bnb load; expect the assert to fire")
    model = AutoModelForCausalLM.from_pretrained(cfg.BASE_MODEL, quantization_config=bnb,
                                                 device_map="auto", dtype=torch.bfloat16)
n4 = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
fp = model.get_memory_footprint() / 2**30
print(f"Linear4bit modules: {n4} | footprint: {fp:.1f} GB | "
      f"device_map: {getattr(model, 'hf_device_map', 'single device')}")
assert fp < 30, (f"{fp:.0f}GB → experts NOT quantized. GO/NO-GO fallbacks (in order): "
                 f"(1) Axolotl ≥0.18 `quantize_moe_experts: true`; "
                 f"(2) bf16 LoRA on the PRUNED reap50 model (~33GB, fits A100-40 at seq ≤4096); "
                 f"(3) A100 80GB tier if offered. Unsloth does NOT support laguna.")

# %%
# apply_chat_template returns a BatchEncoding dict in transformers v5
enc = tok.apply_chat_template([{"role": "user", "content": "Write a Python one-liner to reverse a string."}],
                              add_generation_prompt=True, enable_thinking=True,
                              return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=300, do_sample=False, pad_token_id=tok.pad_token_id)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False))

# %% [markdown]
# ## 3. MoE module discovery (needed by the pruning notebook)

# %%
from src.laguna_arch import find_moe_modules, attention_lora_targets, estimate_params
moe = find_moe_modules(model)
some_layer = sorted(moe)[0]
print(f"sparse layers discovered: {len(moe)} (expect {len(cfg.SPARSE_LAYERS)})")
print(f"layer {some_layer} router: {moe[some_layer]['router'][0]}")
print(f"layer {some_layer} expert modules: {len(moe[some_layer]['experts'])} "
      f"(256 = per-expert submodules; 1 = fused container)")
print(f"layer {some_layer} shared: {moe[some_layer]['shared']}")
targets = attention_lora_targets(model)
print("LoRA targets:", targets)
print("size projections:", {f"keep {k}": estimate_params(k) for k in (192, 128)})

# %% [markdown]
# ## 4. Tiny LoRA run — the go/no-go check

# %%
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from src.data_prep import to_laguna_text

model.config.use_cache = False
peft_model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM", target_modules=targets))
peft_model.print_trainable_parameters()

dummy = [{"messages": [
    {"role": "user", "content": f"Return the number {i} from a Python function."},
    {"role": "assistant", "reasoning_content": "Trivial.",
     "content": f"```python\ndef f():\n    return {i}\n```"}]} for i in range(32)]
def tok_fn(sample):
    enc = tok.apply_chat_template(sample["messages"], enable_thinking=True,
                                  return_assistant_tokens_mask=True, return_dict=True)
    labels = [t if m else -100 for t, m in zip(enc["input_ids"], enc["assistant_masks"])]
    return {"input_ids": enc["input_ids"], "labels": labels}
ds = Dataset.from_list([tok_fn(s) for s in dummy])

def collate(batch):
    import torch as T
    L = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    return {"input_ids": T.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch]),
            "labels": T.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in batch]),
            "attention_mask": T.tensor([[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"])) for b in batch])}

trainer = Trainer(model=peft_model, data_collator=collate, train_dataset=ds,
                  args=TrainingArguments(output_dir="/content/smoke", max_steps=10,
                                         per_device_train_batch_size=2, gradient_accumulation_steps=1,
                                         learning_rate=1e-4, bf16=True, logging_steps=1,
                                         gradient_checkpointing=True, report_to=[]))
result = trainer.train()
print(f"\npeak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
print("VERDICT: QLoRA works on laguna — proceed to notebook 02." )

# %% [markdown]
# ## (Optional) Check Unsloth support — big speed/VRAM win if it exists now
# My info is from Jan 2026; check https://docs.unsloth.ai for `laguna` support.
# If supported, adapt notebook 05 to `FastLanguageModel.from_pretrained(cfg.BASE_MODEL, load_in_4bit=True)`.
#
# ## If the expert-quantized load or step 4 FAILED (fallbacks, in order)
# 1. **Axolotl ≥ 0.18** with `quantize_moe_experts: true` — patches transformers
#    loading to NF4-quantize any 3D tensor named *expert* (arch-generic; GLM-4.7-Flash
#    QLoRA went 127GB → 23GB). Rework notebook 05 as an Axolotl config over the same
#    pre-tokenized dataset.
# 2. **bf16 LoRA on the PRUNED model only**: reap50 ≈ 33GB bf16 fits A100-40 with
#    gradient checkpointing at seq ≤4096 + packing. (keep-160/192 = 43-48GB: does NOT
#    fit bf16 — those variants require a working 4-bit expert loader.)
# 3. Colab A100 80GB tier, if offered — read the live unit rate first.
# Unsloth does NOT support laguna (verified July 2026), and its docs disclaim
# QLoRA-on-MoE for exactly this bitsandbytes reason.
