"""HF Hub as the artifact + checkpoint store (Colab disconnect insurance).

Colab sessions die without warning; Drive can't hold 20–40GB checkpoints.
Every phase pushes its outputs to private Hub repos, and notebook 05 resumes
training from the newest checkpoint found on the Hub.
"""

import os
import re
import time

from huggingface_hub import HfApi, create_repo, snapshot_download


def api():
    return HfApi(token=os.environ.get("HF_TOKEN"))


def ensure_repo(repo_id, repo_type="model"):
    create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True,
                token=os.environ.get("HF_TOKEN"))
    return repo_id


def upload_dir(local_dir, repo_id, repo_type="model", path_in_repo=".", retries=3):
    ensure_repo(repo_id, repo_type)
    for attempt in range(retries):
        try:
            api().upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                                repo_type=repo_type, path_in_repo=path_in_repo)
            return
        except Exception as e:  # noqa: BLE001 — flaky Colab networking, retry then re-raise
            if attempt == retries - 1:
                raise
            print(f"upload failed ({e}); retrying in 30s…")
            time.sleep(30)


def latest_hub_checkpoint(repo_id):
    """Return (checkpoint_name, step) for the highest checkpoint-N dir in the repo, or None."""
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    try:
        files = api().list_repo_files(repo_id)
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None  # genuinely no checkpoints yet; auth/network errors re-raise
        # (a swallowed auth error here would silently restart training from step 0)
    steps = {int(m.group(1)) for f in files if (m := re.match(r"checkpoint-(\d+)/", f))}
    if not steps:
        return None
    step = max(steps)
    return f"checkpoint-{step}", step


def download_checkpoint(repo_id, checkpoint_name, local_dir):
    snapshot_download(repo_id, allow_patterns=[f"{checkpoint_name}/*"],
                      local_dir=local_dir, token=os.environ.get("HF_TOKEN"))
    return os.path.join(local_dir, checkpoint_name)


class HubCheckpointCallback:
    """TrainerCallback: after each local save, push that checkpoint dir to the Hub.

    Defined lazily (transformers import inside) so this module stays importable
    on runtimes without transformers.
    """

    def __new__(cls, repo_id):
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self, repo_id):
                self.repo_id = ensure_repo(repo_id)

            def on_save(self, args, state, control, **kwargs):
                ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                if os.path.isdir(ckpt):
                    print(f"pushing {ckpt} → {self.repo_id}")
                    upload_dir(ckpt, self.repo_id,
                               path_in_repo=f"checkpoint-{state.global_step}")

        return _Callback(repo_id)
