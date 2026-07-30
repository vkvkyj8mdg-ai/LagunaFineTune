# %% [markdown]
# # 05 — QLoRA fine-tune of the pruned model (the main spend, ~150 units)
#
# **Runtime: A100 40GB.**
#
# - Base: the pruned checkpoint chosen in notebook 03 (default: reap50)
# - LoRA on attention projections, router + routed experts frozen
# - Loss only on assistant tokens (masks precomputed in notebook 04)
# - Checkpoints pushed to the Hub every save → survives Colab disconnects;
#   rerunning this notebook auto-resumes from the newest Hub checkpoint.
#
# **Run the PROFILE pass first** (20 steps) — it projects total hours/units and
# tells you whether to shrink the dataset BEFORE you burn the budget.

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
!pip install -q -U "transformers>=5.7" accelerate bitsandbytes peft datasets kernels
!pip install -q experts4bit-qlora

# %%
PRUNED_REPO = ART.pruned_reap50   # ← decision from notebook 03 Part C
PROFILE = True                    # True: 20-step timing pass. Flip to False for the real run.
LORA_R = 32
LR = 1e-4
EPOCHS = 1

# %%
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model
from src.laguna_arch import attention_lora_targets

tok = AutoTokenizer.from_pretrained(PRUNED_REPO)
data = load_dataset(ART.sft_dataset, split="train")
print(data)

# Plain bitsandbytes does NOT quantize the fused 3D expert tensors (both native and
# remote laguna code fuse them) — use the expert-quantizing loader validated in
# notebook 01. device_map={"":0}, never "auto", for training — "auto" silently
# CPU-offloads overflow and turns 18h into 500h with no error.
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
try:
    import experts4bit_qlora
    model = experts4bit_qlora.load_moe_4bit(
        PRUNED_REPO, device_map={"": 0},
        attn_implementation="kernels-community/flash-attn2")
except Exception as e:
    print(f"expert-quantized loader failed ({e!r}) — plain bnb; the assert below decides")
    model = AutoModelForCausalLM.from_pretrained(
        PRUNED_REPO, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16,
        attn_implementation="kernels-community/flash-attn2")
model.config.use_cache = False

n4 = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
fp = model.get_memory_footprint() / 2**30
print(f"Linear4bit modules: {n4} | footprint: {fp:.1f} GB | device_map: {getattr(model, 'hf_device_map', 'n/a')}")
assert fp < 20, (f"{fp:.0f}GB — experts not quantized. Fallbacks: Axolotl "
                 f"`quantize_moe_experts: true`, or bf16 LoRA (reap50 only, seq ≤4096).")

# alpha fixed at 32 with alpha/r scaling keeps optimal LR ~rank-independent; dropout 0
# per every serious LoRA-SFT reproduction. Targets = attention (incl. g_proj) + shared
# expert — NOT "all-linear", which in the per-expert layout would LoRA all ~15K routed
# expert Linears.
targets = attention_lora_targets(model)
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=32, lora_dropout=0.0,
                                         bias="none", task_type="CAUSAL_LM",
                                         target_modules=targets))
model.enable_input_require_grads()   # required with grad checkpointing + frozen quantized base
model.print_trainable_parameters()

# %%
# Packed flattening collator: concatenates each batch into one sequence with
# cu_seqlens boundaries (no cross-contamination; FA2 varlen handles the 512-token
# sliding-window layers correctly). Our -100 label masking passes through untouched.
# This amortizes the per-expert kernel-launch overhead — expect ~2.5-3.5x wall-clock
# vs one-sample batches on this architecture.
from transformers import DataCollatorWithFlattening
collate = DataCollatorWithFlattening(return_flash_attn_kwargs=True)

# %%
# Resume from the newest Hub checkpoint if one exists (Colab died mid-run)
from src.hub_utils import latest_hub_checkpoint, download_checkpoint, HubCheckpointCallback
resume_dir = None
found = latest_hub_checkpoint(ART.sft_adapter)
if found and not PROFILE:
    name, step = found
    print(f"resuming from Hub checkpoint {name}")
    resume_dir = download_checkpoint(ART.sft_adapter, name, "/content/resume")

args = TrainingArguments(
    output_dir="/content/ckpt",
    max_steps=20 if PROFILE else -1,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=LR, lr_scheduler_type="cosine",
    warmup_steps=0.03,  # v5: warmup_ratio was removed; a float in [0,1) here acts as a ratio
    bf16=True, gradient_checkpointing=True,
    # if you ever extend this dict, KEEP use_reentrant: False — passing any dict
    # replaces the default wholesale, and reentrant + frozen 4-bit base silently
    # produces no gradients
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    logging_steps=5, save_steps=40, save_total_limit=2, report_to=[],
    # save_steps=40 ≈ every ~640 samples ≈ 30-40 min — the plan's disconnect budget
    # (200 was 1.5-3h of lost work per disconnect)
)
trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=collate,
                  callbacks=[] if PROFILE else [HubCheckpointCallback(ART.sft_adapter)])

# %%
import time
t0 = time.time()
trainer.train(resume_from_checkpoint=resume_dir)
elapsed = time.time() - t0

# %%
if PROFILE:
    steps_total = len(data) * EPOCHS // (args.per_device_train_batch_size
                                         * args.gradient_accumulation_steps)
    s_per_step = elapsed / 20
    hours = steps_total * s_per_step / 3600
    units = hours * cfg.A100_UNITS_PER_HOUR
    print(f"projected: {steps_total} steps × {s_per_step:.1f}s = {hours:.1f}h ≈ {units:.0f} units")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    if hours > cfg.SFT_MAX_HOURS:
        print(f"⚠ over the {cfg.SFT_MAX_HOURS}h cap — rebuild the mix in notebook 04 with "
              f"TOTAL ≈ {int(len(data) * cfg.SFT_MAX_HOURS / hours)} samples, or lower MAX_LEN.")
    else:
        print("within budget — set PROFILE = False and rerun from the top.")
else:
    model.save_pretrained("/content/final_adapter")
    tok.save_pretrained("/content/final_adapter")
    from src.hub_utils import upload_dir
    upload_dir("/content/final_adapter", ART.sft_adapter, path_in_repo="final")
    print("adapter pushed →", ART.sft_adapter, "/final")

# %% [markdown]
# ### Quick vibe check before spending eval budget

# %%
model.eval(); model.config.use_cache = True
msgs = [{"role": "user", "content":
         "There's a failing test in my repo: `tests/test_auth.py::test_expired_token`. "
         "Walk me through how you'd debug it, then show the likely fix."}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True,
                              return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=700, do_sample=False, pad_token_id=tok.pad_token_id)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False))
