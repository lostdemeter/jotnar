# Roles — how Jotnar work flows

Two roles, one repo. No session does both jobs on the same checkout.

- **Worker** (`roles/worker.md`): implements on a feature branch, proves by
  running, reports for merge. Never merges.
- **Integrator** (`roles/integrator.md`): verifies by running, merges,
  smokes, pushes, reports. Never implements features.

Repo: `/home/balrog/Documents/OpenCode/holographic_enhancement`,
remote `origin` (`github.com/lostdemeter/jotnar.git`), gold `main`.

## Flow

1. Worker: `git checkout -b <area>/<name> main` from a clean tree.
   Work, small commits, push early. REPORT with pastes — never merge.
2. Integrator: checklist in `roles/integrator.md` in order.
   Fail closed at the first red. REPORT back.
3. User: merges `main` only via integrator report, or approves deletions /
   gate changes. No direct pushes to `main` from workers.

## Session start (both roles, every session)

```
git status --short      # must be empty, else stop and reconcile
git log --oneline -3
git branch --show-current
git branch -a           # check for expected branch before creating
```

State branch + status + intent before doing anything. Wait for confirmation
if it touches `main`, deletes files, or changes gates/bars.

## Standing rules (both roles)

- On any error: stop and report. No `reset --hard`, no force-push,
  no rebasing pushed branches. Paste the actual error.
- Separate checkouts: integrator checks out branches — never share a
  checkout with an active worker.
- History append-only. No squash/rebase of pushed work, ever.
- `research` branch exists, never merge it.
- GPU box + `/tmp` shared: one GPU session at a time, distinct workdirs
  per config. Never `/tmp/dec7b` (contaminated graveyard).
- Reviews are read-only unless the task says otherwise. Slow gates by
  dated record (`docs/VELOCITY.md`), never quote old logs as current.

## House contracts (bars both roles enforce)

- Missing ops fail loud (`NoPattern`). Never silent wrong bytes.
- Float = eps, integer = bit-exact, serving grades decisions
  (argmax-rank / fluency), never exact strings.
- Determinism legs 0.0000 where claimed (workspace-pinned cuBLAS).
- `test_emit_c.py` fails on missing system gsl headers — pre-existing,
  environmental, ignore it. Do not "fix".
- Docs sourced from code (mnemonic table from REGISTRY/SIGS +
  `asm-lang-coverage` gate). `docs/VELOCITY.md` append-only.
