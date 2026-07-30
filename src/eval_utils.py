"""Shared evaluation helpers: EvalPlus (HumanEval+/MBPP+) + throughput.

Design choice: generation runs through *our* chat-template loop (greedy,
thinking enabled, code extracted from the final answer) so base / pruned /
pruned+LoRA are compared apples-to-apples regardless of backend.
"""

import json
import re
import subprocess
import time
from pathlib import Path

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)


def get_problems(dataset="humaneval", limit=None):
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
    elif dataset == "mbpp":
        from evalplus.data import get_mbpp_plus
        problems = get_mbpp_plus()
    else:
        raise ValueError(dataset)
    items = list(problems.items())
    return dict(items[:limit] if limit else items)


def extract_code(text: str) -> str:
    """Final answer code: last fenced python block; else text after </think>."""
    text = text.split("</think>")[-1]
    blocks = CODE_BLOCK_RE.findall(text)
    return blocks[-1].strip() if blocks else text.strip()


PROMPT = ("Complete the following Python function. Reply with the complete, "
          "runnable implementation in a single ```python code block.\n\n```python\n{prompt}\n```")


def generate_solutions(model, tokenizer, problems, max_new_tokens=3072, enable_thinking=True):
    """Greedy decode, one problem at a time. Returns (samples, stats)."""
    import torch

    samples, gen_tokens, t0 = [], 0, time.time()
    for i, (task_id, prob) in enumerate(problems.items()):
        messages = [{"role": "user", "content": PROMPT.format(prompt=prob["prompt"])}]
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=enable_thinking,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        gen_tokens += new.shape[0]
        text = tokenizer.decode(new, skip_special_tokens=True)
        samples.append({"task_id": task_id, "solution": extract_code(text)})
        if i % 10 == 0:
            print(f"[{i + 1}/{len(problems)}] {task_id} "
                  f"({gen_tokens / max(time.time() - t0, 1):.1f} tok/s avg)")
    stats = {"gen_tokens": gen_tokens, "seconds": round(time.time() - t0, 1),
             "tok_per_s": round(gen_tokens / max(time.time() - t0, 1), 2)}
    return samples, stats


def vllm_generate_solutions(llm, tokenizer, problems, max_new_tokens=3072, enable_thinking=True):
    """Same contract as generate_solutions, but batched through a vllm.LLM instance."""
    from vllm import SamplingParams

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(prompt=p["prompt"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        for p in problems.values()
    ]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new_tokens))
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    samples = [{"task_id": tid, "solution": extract_code(o.outputs[0].text)}
               for tid, o in zip(problems.keys(), outs)]
    stats = {"gen_tokens": gen_tokens, "seconds": round(time.time() - t0, 1),
             "tok_per_s": round(gen_tokens / max(time.time() - t0, 1), 2)}
    return samples, stats


def score(samples, dataset="humaneval", workdir="eval_out", tag="run"):
    """Write samples and invoke the EvalPlus sandboxed scorer. Returns parsed results."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    samples_path = workdir / f"{tag}.{dataset}.jsonl"
    with open(samples_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    subprocess.run(["evalplus.evaluate", "--dataset", dataset,
                    "--samples", str(samples_path)], check=True)
    for suffix in ("_eval_results.json", ".eval_results.json"):  # evalplus has used both
        results_path = samples_path.with_name(samples_path.stem + suffix)
        if results_path.exists():
            with open(results_path) as f:
                full = json.load(f)
            # return only the scores — the full record carries per-task solutions
            # (megabytes) and would bloat the Hub results files and comparison table
            return {"pass_at_k": full.get("pass_at_k"), "n": len(samples)}
    return {"note": "see stdout above for pass@1 (results file layout changed?)"}


def pass1(result, variant="plus"):
    """Extract pass@1 (percent) from a score() result; None if unavailable."""
    try:
        return round(100 * result["pass_at_k"][variant]["pass@1"], 1)
    except (KeyError, TypeError):
        return None
