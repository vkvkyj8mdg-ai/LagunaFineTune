#!/usr/bin/env python3
"""Baseline eval of poolside's INT4 build via vLLM — run as a PLAIN PROCESS.

vLLM cannot start inside a Jupyter kernel (verified on Colab, 2026-07-30):
fork inherits the kernel's initialized CUDA and dies silently, spawn re-imports
Jupyter's __main__, and the in-process engine needs sys.stdout.fileno(). A real
python subprocess has none of those problems — notebook 02 invokes this with
`!python`. HF_TOKEN must be in the environment (the CLI prelude sets it).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import project_config as cfg  # noqa: E402
from src.eval_utils import get_problems, score, vllm_generate_solutions  # noqa: E402
from src.hub_utils import ensure_repo, upload_dir  # noqa: E402


def main():
    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
    llm = LLM(model=cfg.BASE_MODEL_INT4, max_model_len=16384,
              gpu_memory_utilization=0.92, trust_remote_code=True)
    results = {}
    # full sets only: evalplus's scorer asserts every problem has a sample,
    # and at vLLM speeds (~3400 tok/s batched) the full run is ~3 minutes
    for dataset, limit in (("humaneval", None), ("mbpp", None)):
        problems = get_problems(dataset, limit=limit)
        samples, stats = vllm_generate_solutions(llm, tok, problems)
        print(dataset, stats, flush=True)
        results[dataset] = {"stats": stats,
                            "eval": score(samples, dataset, tag="base_int4")}

    os.makedirs("/content/results", exist_ok=True)
    with open("/content/results/base_int4.json", "w") as f:
        json.dump(results, f, indent=1)
    ensure_repo(cfg.hub("eval-results"), repo_type="dataset")
    upload_dir("/content/results", cfg.hub("eval-results"), repo_type="dataset")
    print("BASELINE_RESULTS:", json.dumps(
        {k: {"pass_at_k": v["eval"].get("pass_at_k"),
             "tok_per_s": v["stats"]["tok_per_s"]} for k, v in results.items()}))


if __name__ == "__main__":
    main()
