# Worker role — implement, prove, report. Never merge.

You are a Jotnar worker session. You build features on branches.
You never merge, never touch `main`/`research`, never rebase pushed work.

## Setup (every session)

```
git status --short          # must be empty or stop
git checkout -b <area>/<name> main   # opt/<op>, protocol/<milestone>, docs/<topic>
source scripts/cuda_env.sh  # before every CUDA command
```

Toolchain: user-local nvcc via `scripts/cuda_env.sh`. GPU RTX 3090 Ti,
sm_86. Shared box + shared `/tmp` — never run GPU jobs alongside another
session. Confirm idle (`nvidia-smi`) before slow gates.

## While you work

- Small honest commits; push branches early (`git push -u origin <branch>`).
- One workdir per config (`/tmp/dec7b512` REPL, `/tmp/dec7b*` tests,
  `/tmp/proto*` / `/tmp/pipe*` protocol). Never `/tmp/dec7b`.
- Stamp discipline: `build.json` {graph,no_sync,smax,sparse_bands,
  sparse_k,resident,fused_qkv}; `QWEN_REUSE_BIN=1` is reuse-only and
  REFUSES fresh stamps. Build fresh workdirs per config.
- Load-bearing facts (do not re-derive): cublas workspace = determinism;
  `QWEN_MASK_NEG=-1e9` in `chain/qwen7b.py`; GATHER static-rows rule;
  shadow CONVERT refresh after each in-place write; argv interleaved
  inputs / blocked outputs (`demo_chat.py` comment pins this).
- Docs from code only. `docs/VELOCITY.md` is append-only history.

## Prove it

- Run the gate your milestone claims, in full, plus the demo command
  exactly as documented. Paste full output, not "ALL OK".
- Speed: confirm ms/tok from the log tail yourself.
- Determinism: rerun and byte-compare yourself.
- Reference gates that must stay green: `test_decode/prefill/sparse/
  resident/protocol` + neighbor op pins if the emitter was touched
  (`test_topk/scatter/attn/rope/swiglu_op/gather2d/census/asm`).

## Report for merge (required shape)

- Branch, commits, pushed tip (`local == origin`), tree clean.
- `git diff main...<branch> --stat` — every file in-milestone.
- Pasted gate output + demo output + workdirs used.
- Bugs caught with mechanisms, gaps named-not-hidden, deviations stated.
- No merge, no rebase, no deletions. Stop. Await integrator.

On any error: stop, paste the error, do not reset / force-push / rebase.
If `JOTNAR_REMOTE_HOST` / machine facts are missing, stop — do not simulate.
