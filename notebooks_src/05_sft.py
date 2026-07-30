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
LR = 1e-4
EPOCHS = 1

# %%
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer

tok = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
data = load_dataset(ART.sft_dataset, split="train")
print(data)

# Loader path validated end-to-end by notebook 01: NF4 experts via the streaming
# loader (plain bnb skips fused 3D experts), with per-expert LoRA(r=4) attached
# during loading. PEFT is deliberately NOT used — get_peft_model would freeze the
# per-expert adapters; the package's structural wrappers + plain Trainer train
# every requires_grad param.
from src.laguna_e4b import load_laguna_4bit, add_extra_lora
from experts4bit_qlora import add_attention_lora

model, _ = load_laguna_4bit(PRUNED_REPO, r=cfg.SFT_EXPERT_LORA_R, alpha=cfg.SFT_LORA_ALPHA)
model.config.use_cache = False
n_attn = add_attention_lora(model, r=cfg.SFT_LORA_R, alpha=cfg.SFT_LORA_ALPHA,
                            dtype=torch.bfloat16)
n_extra = add_extra_lora(model, r=cfg.SFT_LORA_R, alpha=cfg.SFT_LORA_ALPHA)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"attention wraps: {n_attn} | g_proj+shared-expert wraps: {n_extra} | "
      f"trainable: {trainable / 1e6:.0f}M | cuda: {torch.cuda.memory_allocated() / 2**30:.1f} GB")

# %%
# Packed flattening collator: concatenates each batch into one sequence with
# cu_seqlens boundaries (no cross-contamination; FA2 varlen handles the 512-token
# sliding-window layers correctly). Our -100 label masking passes through untouched.
# This amortizes the per-expert kernel-launch overhead — expect ~2.5-3.5x wall-clock
# vs one-sample batches on this architecture.
from transformers import DataCollatorWithFlattening
collate = DataCollatorWithFlattening(return_flash_attn_kwargs=True)

# %%
# Adapter-only checkpointing: Trainer's own save path crashes on the quantized
# expert modules (unserializable quant-state dicts), so save_strategy is OFF and
# a callback pushes {trainable params, step} (~1-2GB) to the Hub instead.
# Resume restores adapter WEIGHTS with a fresh optimizer/schedule — acceptable
# for a 1-epoch SFT; worst case a disconnect costs some LR-schedule fidelity.
import re as _re
from huggingface_hub import hf_hub_download
from transformers import TrainerCallback
from src.hub_utils import api, ensure_repo
from src.laguna_e4b import load_trainable, save_trainable

ensure_repo(ART.sft_adapter)
resume_step = None
if not PROFILE:
    ckpts = sorted(int(m.group(1)) for f in api().list_repo_files(ART.sft_adapter)
                   if (m := _re.match(r"adapter-step-(\d+)\.pt$", f)))
    if ckpts:
        path = hf_hub_download(ART.sft_adapter, f"adapter-step-{ckpts[-1]}.pt")
        resume_step = load_trainable(model, path)
        print(f"resumed adapter weights from step {resume_step}")


class AdapterHubCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step and state.global_step % cfg.SFT_SAVE_EVERY == 0:
            path = f"/content/adapter-step-{state.global_step}.pt"
            n = save_trainable(model, path, step=state.global_step, alpha=cfg.SFT_LORA_ALPHA)
            api().upload_file(path_or_fileobj=path, repo_id=ART.sft_adapter,
                              path_in_repo=f"adapter-step-{state.global_step}.pt")
            print(f"pushed adapter checkpoint @ step {state.global_step} ({n} tensors)")


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
    logging_steps=5, save_strategy="no", report_to=[],
)
trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=collate,
                  callbacks=[] if PROFILE else [AdapterHubCallback()])

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
    save_trainable(model, "/content/adapter-final.pt",
                   step=trainer.state.global_step, alpha=cfg.SFT_LORA_ALPHA)
    api().upload_file(path_or_fileobj="/content/adapter-final.pt",
                      repo_id=ART.sft_adapter, path_in_repo="adapter-final.pt")
    print("final adapter pushed →", ART.sft_adapter, "adapter-final.pt")

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
