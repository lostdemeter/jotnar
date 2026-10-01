# research/ — one-shot research scripts (moved from root, v1.8 reorg)

Not tests (no bars), not demos (no audience): dated probes whose outputs
live in docs/ as measured rows. Each file states its question up top;
each run writes under /tmp (never the repo). Re-run to verify, not to
gate (evidence is committed; reruns reproduce it).

| script | question | evidence |
|---|---|---|
| read_model.py | Qwen down_proj directional readout (112 dirs)? | MODEL_READ.md + readout npz |
| loop1.py / loop_turn.py | labeling loop turns (dir/token/prompt/pos)? | MODEL_READ.md loops 1-3 |
| dd_mine.py | DDColor query vote maps? | LIB-061 |
| dd_footprint.py | causal footprints per query (+hunt mode)? | LIB-061/062 |
| dd_catalog.py | full 100-query palette table? | PALETTE_CATALOG.csv |
| dd_modify.py | palette ADD + disentangle modes? | LIB-066/067 |
| dd_resonant.py | phase-indexing over engrams? | LIB-067 (scrambles) |
| dd_score.py | DDColor cross-attn scoremax per layer? | T_TRANSFORM.md table |

Needs: local HF cache (Qwen/DDColor weights) + ddcolor_reverse checkout
as sibling for dd_* (absolute paths inside, documented per file).
