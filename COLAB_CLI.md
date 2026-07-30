# Running this pipeline headlessly with the Google Colab CLI

Verified 2026-07-30 against `google-colab-cli` **0.6.0** (latest on PyPI, released 2026-06-16).

**Verdict: usable, with two mandatory workarounds.** You can create an A100 runtime, run
the existing `.ipynb` files unchanged, and stream logs from a Mac terminal. Two things do
*not* work out of the box and are handled below: the released package installs broken
dependencies, and Colab Secrets (`google.colab.userdata`) do not exist in CLI sessions.

The one thing the CLI genuinely cannot do is **re-attach to the log stream of a run in
progress**. That matters less here than it would elsewhere: notebook 05 already pushes a
checkpoint to the Hub every ~40 steps and auto-resumes from the newest one, so a dropped
terminal costs you visibility, not work. See [Disconnects](#disconnects-and-resume).

Read [Known limitations](#known-limitations) before spending compute units. The maintainer
describes the project as "still very much experimental" and there has been no release in
the six weeks since 0.6.0.

---

## 1. Install

Requires Python >= 3.12, macOS or Linux (Windows unsupported). Verified here on 3.14.6;
`uv` will fetch an interpreter if the requested version isn't installed.

`google-colab-cli` 0.6.0 declares `jupyter-kernel-client` **with no version bound**. Pip
resolves 1.0.0 (2026-07-26), which removed the `KernelClient` class the CLI calls at
`colab_cli/runtime.py:106`, so every `exec`/`repl`/`new` dies with
`AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'`
([issue #94](https://github.com/googlecolab/google-colab-cli/issues/94)). You must pin:

```bash
uv venv --python 3.12 ~/.venvs/colab-cli
uv pip install --python ~/.venvs/colab-cli/bin/python \
    google-colab-cli 'jupyter-kernel-client==0.9.0'
export PATH="$HOME/.venvs/colab-cli/bin:$PATH"   # add to ~/.zshrc
```

Verify the pin took effect — if this prints `False`, nothing else will work:

```bash
colab version   # -> Version: 0.6.0
python -c "import jupyter_kernel_client as j; print(hasattr(j,'KernelClient'))"
```

Install into a **fresh** venv. Installing over an environment that already has `pyzmq<26`
produces an unrelated `zmq.backend.cython` import error
([#81](https://github.com/googlecolab/google-colab-cli/issues/81)).

## 2. Authenticate

Do this once, interactively. It needs your Google account, so it is not scriptable.

The bundled README and skill file both claim `--auth` defaults to `adc`. **They are wrong** —
`colab_cli/cli.py:78` is `AuthProvider.OAUTH2`. Pass `--auth` explicitly.

**Option A — OAuth copy/paste (recommended; no gcloud needed).** Prints a URL, you approve
in any browser, Google shows a code, you paste it back. Works headless; there is no
localhost redirect. Token cached at `~/.config/colab-cli/token.json`.

```bash
colab --auth=oauth2 sessions      # triggers the flow, then lists sessions (read-only)
```

**Option B — Application Default Credentials.** Needs all four scopes or the keep-alive
RPC 403s:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
colab --auth=adc sessions
```

Whichever you pick, **use it consistently**: `colab new` propagates `--auth` (and `--config`)
to the detached keep-alive daemon it spawns.

`colab auth` is unrelated — it injects GCP credentials *into the VM* for BigQuery/GCS. It
will not fix a CLI 401/403.

## 3. Create the A100 runtime

```bash
colab --auth=oauth2 new -s laguna05 --gpu A100
colab status -s laguna05      # hardware, IDLE/BUSY, last execution
colab sessions                # all live assignments
```

- Always pass `-s <name>`; an omitted name becomes a random 6-hex string.
- Supported: `T4 L4 G4 H100 A100`. An **unrecognised `--gpu` value silently falls back to
  A100**, so typos fail confusingly later.
- A100 availability is tier-gated and often needs several retries; users report `H100`
  never allocating and failures surfacing as raw `Service Unavailable` or a mislabelled
  `TooManyAssignmentsError` on HTTP 412
  ([#73](https://github.com/googlecolab/google-colab-cli/issues/73)).
- **High-RAM cannot be requested from the CLI.** `Shape.HIGH_RAM` exists in `client.py` but
  no flag exposes it ([#47](https://github.com/googlecolab/google-colab-cli/issues/47)). If a
  phase needs the high-memory A100 shape, use the browser for that phase.
- `colab new` allocates a **billable** VM immediately. Nothing reclaims it automatically
  except the 24 h keep-alive cap. Always `colab stop` when done.

**Start a dedicated session for notebook 05.** The keep-alive daemon's 24 h clock starts at
`colab new`, so don't burn hours of it on notebooks 01–04 first.

## 4. HF_TOKEN handling

`google.colab.userdata.get()` **raises `TimeoutException` for every key in CLI sessions** —
maintainer-confirmed in [#48](https://github.com/googlecolab/google-colab-cli/issues/48)
("User secrets are a separate API that needs some extra development"). Colab Secrets are
brokered by the browser frontend, which does not exist here. `--env KEY=VALUE` was written
in PR #65 but is **unmerged and absent from 0.6.0**.

Do **not** inline the token into piped code: `colab exec` records every executed code string
verbatim in `~/.config/colab-cli/history/<session>.jsonl`, and `colab log` re-exports it.
`colab upload` logs only paths, so upload the token as a file.

```bash
mkdir -p ~/.config/laguna && chmod 700 ~/.config/laguna
read -rs -p 'HF token: ' T && printf '%s' "$T" > ~/.config/laguna/hf_token && unset T
chmod 600 ~/.config/laguna/hf_token
colab upload -s laguna05 ~/.config/laguna/hf_token /content/.hf_token
```

`read -rs` keeps the token out of shell history and out of `ps`. The path is outside the
repo, so there is nothing to gitignore.

## 5. Bootstrap the session (once per `colab new`)

`tools/colab_cli_prelude.py` emits a prelude that clones your fork, loads the token from
`/content/.hf_token` into `os.environ`, and monkeypatches `google.colab.userdata.get` to
fall back to the environment. Because **kernel state persists across separate `colab exec`
calls**, the patch is still active when you run a notebook afterwards — so the notebooks
need no edits.

```bash
python3 tools/colab_cli_prelude.py --repo "$(git remote get-url origin)" \
  | colab exec -s laguna05 --timeout 900
```

(`git remote get-url origin` currently resolves to your
`github.com/vkvkyj8mdg-ai/LagunaFineTune` fork. The clone must be **pushed and public**, or
the VM cannot fetch it — the prelude does an unauthenticated `git clone`.)

Expect `patched google.colab.userdata.get -> os.environ fallback`, a `gpu: NVIDIA A100 …`
line, and `prelude OK`. It prints the token's byte length, never its value. Re-running is
safe (it fast-forwards the checkout).

Because the prelude clones to `/content/LagunaFineTune`, each notebook's own bootstrap cell
(`if not os.path.isdir("/content/LagunaFineTune"): !git clone {REPO_URL} …`) short-circuits —
so the `REPO_URL = ".../CHANGE_ME/..."` placeholder in the notebooks never gets used and
needs no editing for CLI runs.

`--help` covers `--branch`, `--checkout`, `--token-file`, `--hf-home`.

## 6. Run a phase

```bash
colab exec -s laguna05 -f notebooks/01_smoke_test.ipynb --timeout 3600
```

`colab exec -f` reads the **local** file and sends each code cell to the remote kernel, so
local edits take effect with no push and no upload. `!pip install` and other magics work —
it is a real IPython kernel. Outputs stream live to your terminal and a copy is written to
`notebooks/01_smoke_test_output.ipynb` next to the input (consider adding
`*_output.ipynb` to `.gitignore`).

`src/` is imported from the **git clone**, not your local tree, so changes under `src/` must
be pushed to your fork and picked up by re-running the prelude.

Two traps:

- **`--timeout` defaults to 30 seconds** and is a *total wall-clock deadline for one cell*,
  set once and never reset. Always pass a value larger than the whole phase. On the pinned
  `jupyter-kernel-client==0.9.0` an exceeded deadline does not even raise — the local
  process spins one CPU core at 100% indefinitely while the remote run continues
  ([#82](https://github.com/googlecolab/google-colab-cli/issues/82)).
- **A failing cell does not stop the run and does not affect the exit code.** `exec_command`
  prints the traceback to stderr and continues to the next cell; `colab exec` exits 0
  regardless. Grep your logs for `Traceback` rather than trusting `$?`.

Use `colab run` only for plain `.py` scripts — it propagates exit codes but feeds `.ipynb`
files to Python as raw JSON ([#74](https://github.com/googlecolab/google-colab-cli/issues/74)).

## 7. Notebook 05 — the 15–18 h SFT run

First do the profile pass. `notebooks/05_sft.ipynb` ships with `PROFILE = True` (20 steps,
prints a projected hours/units estimate):

```bash
colab exec -s laguna05 -f notebooks/05_sft.ipynb --timeout 5400
```

If the projection is within budget, set `PROFILE = False` — edit `notebooks_src/05_sft.py`
and run `python tools/py2ipynb.py`, or edit the one line in the `.ipynb`. No push needed.

Then the real run. `caffeinate` is **not optional**: the keep-alive daemon is a local
process on your Mac, and if the machine sleeps it stops pinging and Colab reclaims the VM
after roughly 90 minutes of apparent idleness.

```bash
mkdir -p logs
caffeinate -dims colab exec -s laguna05 -f notebooks/05_sft.ipynb \
    --timeout 86400 2>&1 | tee -a logs/05_sft.$(date +%FT%H%M).log
```

`--timeout 86400` exceeds any possible run length, which is what you want given the 0.9.0
spin bug. Closing the laptop lid is what breaks this, not closing the terminal.

**Session ceilings.** Free/Pro runtimes die at 12 h; **Colab Pro+ supports up to 24 h of
continuous execution** provided compute units last. The CLI's keep-alive daemon also
hard-stops at exactly 24 h. So:

- On **Pro+**, an 18 h run fits in one session.
- On **Pro**, it does not. Expect the runtime to die around 12 h — then just re-run the same
  command against a fresh session; the notebook resumes from the newest Hub checkpoint.

### Monitoring while it runs

While the kernel is BUSY on an 18 h cell, **every other `colab exec`/`repl` call fails after
~30 s** (they each begin with an `os.chdir('/content')` cell that queues behind your run;
the CLI passes no timeout for it, so `jupyter-kernel-client`'s 30 s default applies). That
is a fail-fast, not a hang, but it means you cannot poll the VM. Monitor instead via:

```bash
colab status -s laguna05     # REST call, works while BUSY: shows BUSY + last execution
```

plus the commit history of your `ART.sft_adapter` repo on the Hub — checkpoints land every
~30–40 min, which is the real progress signal.

Do **not** use `colab url` to peek: it currently opens a scratchpad that is *not* connected
to your CLI session, and clicking Connect silently allocates a **second billable runtime**
([#24](https://github.com/googlecolab/google-colab-cli/issues/24)).

### Disconnects and resume

Three independent things, and only two survive losing your terminal:

| | Survives terminal close? | Notes |
|---|---|---|
| The VM / runtime | **Yes** | Independent of any client. |
| The keep-alive daemon | **Yes** | Spawned detached via `start_new_session=True`, so no SIGHUP. Dies on Mac sleep/shutdown, or after 24 h. |
| Your `colab exec` client + its output | **No** | No reattach, follow, or tail command exists. `colab log` reads a *local* file written by the client, so output produced after the client dies is gone for good. |

Critically, **the remote execution is never interrupted**. The CLI sets
`_own_kernel = False`, `runtime.stop()` closes only the websocket, and there is no
`interrupt()` call anywhere in the exec path. Ctrl-C, a timeout, or a dropped Wi-Fi
connection all leave the cell running on the VM.

So after an unexpected disconnect:

```bash
colab status -s laguna05        # BUSY => still training, just keep waiting
```

Watch the Hub for new checkpoints. When it finishes, the notebook pushes the final adapter
to `ART.sft_adapter/final` on its own — you do not need to be attached for that.

If you want to take back control, or if `status` shows the session is gone:

```bash
colab stop -s laguna05
colab --auth=oauth2 new -s laguna05b --gpu A100
colab upload -s laguna05b ~/.config/laguna/hf_token /content/.hf_token
python3 tools/colab_cli_prelude.py --repo "$(git remote get-url origin)" \
  | colab exec -s laguna05b --timeout 900
caffeinate -dims colab exec -s laguna05b -f notebooks/05_sft.ipynb --timeout 86400
```

`latest_hub_checkpoint()` finds the newest checkpoint and training continues from it.

### Hardening option (untested)

A fully disconnect-proof variant is to launch training as a **detached OS process on the
VM** (`start_new_session=True` via `subprocess.Popen`, output tee'd to `/content/logs/`),
leaving the kernel IDLE so you can poll the logfile with short `colab exec` calls. The gap:
a detached process gets a **new** kernel, so the `userdata` patch from step 5 — which lives
in the CLI's kernel — does not apply, and cell 1 of the notebook would fail on
`userdata.get`. Closing that requires an IPython startup file or a `sitecustomize.py` on
`PYTHONPATH`. **I could not test any of this** (it needs your Google account), so it is
described, not recommended. Validate with `notebooks/01_smoke_test.ipynb` before trusting
it with 150 compute units.

## 8. Getting files in and out

Your pipeline already pushes everything through the Hub, which remains the right channel —
the CLI's file transfer is a thin Jupyter Contents API wrapper and is **single-request,
whole-file, base64-in-JSON**. Fine for tokens and configs, unsuitable for GB-scale weights.
There is no recursive transfer; `download` on a directory raises `IsADirectoryError`.

```bash
colab ls -s laguna05 /content
colab upload   -s laguna05 ./local.json /content/local.json
colab download -s laguna05 /content/logs/train.log ./train.log
colab rm       -s laguna05 /content/.hf_token        # before stopping, if you like
```

For a whole directory, tar it on the VM first, then download the single file.

`colab log -s laguna05 -o run.ipynb` exports the **locally recorded** history as a notebook
(`.md`, `.txt`, `.jsonl` also work). Remember it contains your executed code verbatim.

## 9. Compute units

CLI runtimes are ordinary managed Colab runtimes obtained through the same `/tun/m/assign`
endpoint the browser uses, so **the same compute-unit metering and the same 12 h/24 h caps
apply** — this is inference from the architecture, not a documented statement; no official
source addresses CLI-specific accounting either way.

There is **no CLI command for balance or burn rate**
([#47](https://github.com/googlecolab/google-colab-cli/issues/47)); `colab pay` just opens
the subscription page. Track spend in the web UI. Note `src/project_config.py` assumes
`A100_UNITS_PER_HOUR = 8.5`; third-party write-ups quote ~15 CU/h for A100, and Colab
deliberately does not publish rates ("Colab does not publish these limits, in part because
they can vary over time"). Worth re-checking against your own meter after the profile pass,
since an 18 h run is the difference between ~150 and ~270 units of a 300-unit budget.

If your balance hits zero mid-run you revert to free-tier policy and the backend terminates.

## 10. Known limitations

| Issue | Impact here |
|---|---|
| Unpinned `jupyter-kernel-client` breaks a fresh install ([#94](https://github.com/googlecolab/google-colab-cli/issues/94)) | Blocking. Pin `==0.9.0` (step 1). |
| Colab Secrets unavailable ([#48](https://github.com/googlecolab/google-colab-cli/issues/48)) | Blocking. Solved by the step-5 prelude. |
| `--timeout` default 30 s; 0.9.0 spins a core instead of raising ([#82](https://github.com/googlecolab/google-colab-cli/issues/82)) | Always pass a large `--timeout`. |
| No reattach / follow / tail; `colab log` is local-only | Log stream lost on disconnect. Absorbed by Hub checkpointing. |
| Keep-alive daemon is local and dies on Mac sleep; hard 24 h cap | Use `caffeinate`. 24 h is the absolute ceiling. |
| Runtime lost across idle gaps between discrete `exec` calls | Reported on 0.6.0/macOS/Pro+ after the [#14](https://github.com/googlecolab/google-colab-cli/issues/14) fix, unacknowledged: gaps of 10–15 min gave `Session appears to be lost (404/401)`. Prefer one long `exec`. |
| Failing cells don't stop `.ipynb` runs or change the exit code | Grep logs for `Traceback`. |
| High-RAM shape not selectable ([#47](https://github.com/googlecolab/google-colab-cli/issues/47)) | Use the browser if a phase needs it. |
| `colab url` opens a disconnected scratchpad, can start a 2nd billable VM ([#24](https://github.com/googlecolab/google-colab-cli/issues/24)) | Don't use it. |
| `colab run` mis-executes `.ipynb` as JSON ([#74](https://github.com/googlecolab/google-colab-cli/issues/74)) | Use `colab exec -f`. |
| Repo docs describe unreleased `main` | `colab ssh` and `--env` are documented but absent from 0.6.0. |
| Pro+ runtimes reported terminating in 40–60 min *in the browser too* ([colabtools #5939](https://github.com/googlecolab/colabtools/issues/5939), open) | Platform-level risk, not CLI-specific. Another reason the Hub-checkpoint design matters. |

Colab's FAQ also states it "prioritizes interactive compute" and lists "remote control such
as SSH shells" and "bypassing the notebook UI" among things that may be terminated without
warning **on runtimes without a positive compute unit balance**. With a paid balance the
CLI is an official, supported Google tool; on a depleted balance, headless use is
explicitly at risk.

## 11. Teardown

```bash
colab stop -s laguna05      # terminates the VM and the keep-alive daemon
colab sessions              # confirm nothing is left running
```

An unstopped session bills until the 24 h cap. Check `colab sessions` after every run;
orphans with no local record show as `[?]`.

---

## What is verified vs inferred

**Verified** — reproduced locally on this Mac against an installed 0.6.0, or read directly
from its source / an official doc: package name and version; Python >= 3.12; macOS support;
the `jupyter-kernel-client` 1.0.0 breakage and that `==0.9.0` restores `KernelClient`; the
absent timeout guard in 0.9.0 vs 1.0.0; `--auth` actually defaulting to `oauth2` while the
docs say `adc`; the OAuth copy/paste flow and inlined client config; the full command
surface (and that `ssh`/`--env` are absent); `--timeout 30.0` defaults and the
set-once deadline; that no `interrupt()` exists and `_own_kernel = False`, so remote
execution survives client death; keep-alive being a locally detached 60 s ping loop with a
`24 * 3600` cap; `.ipynb` cell-by-cell execution, `_output.ipynb` writing, and no
error-driven abort or non-zero exit; executed code being logged verbatim to local history
while `upload` logs only paths; single-request base64 file transfer with no directory
support; the ~90 min idle timeout and 12 h/24 h tier caps from Colab's FAQ; and — from the
CLI's own issue tracker — Secrets being unavailable with maintainer confirmation.

I also verified `tools/colab_cli_prelude.py` end to end locally: it emits parseable Python,
clones and fast-forwards a real git repo, loads and `chmod 600`s the token file, errors
cleanly when the token is missing, and — against a stub that raises `TimeoutException` the
way the real `userdata` does — successfully patches `HF_TOKEN` while still delegating
unknown keys.

**Inferred, not stated by any source:** that CLI runtimes bill against the same compute-unit
pool and inherit the same caps (follows from the shared `/tun/m/assign` endpoint and the
repo's own "real, billable assignments" language); that Mac sleep kills keep-alive and
therefore the VM after the idle window; and that A100 via CLI gets a shape with enough
system RAM for this pipeline, since high-RAM is not selectable.

**Not verified at all:** anything requiring an authenticated session. I did not authenticate
or start a session, per instructions. So actual A100 allocation, real streaming behaviour,
the prelude running on a real Colab VM, end-to-end 18 h stability, and the §7 hardening
option are all untested. Do a cheap dry run — `colab new --gpu T4`, prelude,
`notebooks/01_smoke_test.ipynb`, `colab stop` — before committing the SFT budget.

## Sources

- [Introducing the Google Colab CLI](https://developers.googleblog.com/introducing-the-google-colab-cli/) (announcement, 2026-06-05)
- [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli) — repo, `docs/`, and issues [#14](https://github.com/googlecolab/google-colab-cli/issues/14), [#24](https://github.com/googlecolab/google-colab-cli/issues/24), [#47](https://github.com/googlecolab/google-colab-cli/issues/47), [#48](https://github.com/googlecolab/google-colab-cli/issues/48), [#73](https://github.com/googlecolab/google-colab-cli/issues/73), [#74](https://github.com/googlecolab/google-colab-cli/issues/74), [#81](https://github.com/googlecolab/google-colab-cli/issues/81), [#82](https://github.com/googlecolab/google-colab-cli/issues/82), [#94](https://github.com/googlecolab/google-colab-cli/issues/94), [PR #61](https://github.com/googlecolab/google-colab-cli/pull/61), [PR #65](https://github.com/googlecolab/google-colab-cli/pull/65), [discussion #77](https://github.com/googlecolab/google-colab-cli/discussions/77)
- [PyPI: google-colab-cli](https://pypi.org/project/google-colab-cli/) — 0.6.0, 2026-06-16
- Installed 0.6.0 source, plus its bundled `colab readme` / `colab skill`
- [Colab FAQ](https://research.google.com/colaboratory/faq.html) — runtime caps, idle timeouts, compute units
- [googlecolab/colabtools](https://github.com/googlecolab/colabtools) — [`userdata.py`](https://github.com/googlecolab/colabtools/blob/main/google/colab/userdata.py), [#5939](https://github.com/googlecolab/colabtools/issues/5939), [#5950](https://github.com/googlecolab/colabtools/issues/5950)
