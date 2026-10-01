# v2.0: the constructed-LLM release (definition of done)

v2.0 is the version that answers the handoff's GO verdict with a running
system, defined by ONE question: "can we build a working LLM from scratch
in assembly, with every design decision priced by measured law?" Everything
below serves that question. It proves the point by example: a transformer
that runs exactly, knows counts, retrieves facts, orders structurally --
no gradients, no training, every claim a gate.

Out of scope for v2.0 (named, not forgotten): vocabulary reach (512-word
frozen; BPE frozen but piece prediction needs its fitting arc), output
quality past bigram-coil (top1 0.472, ppl ~95 class), fluency competitions.

## Acceptance gates (all hold, measured 2026-10-01)

1. **Whole-system float parity.** Depth-4 causal transformer (H=2 heads,
   storebank MLPs, WIDE attention) holds >=40dB vs independent torch
   mirrors at every growth step (single 71-84dB, depth-2 71-84dB,
   SVD 60dB, bankhn 53-55dB, depth-4 44-45dB). The only values-level
   proof at composition scale, never negotiable.
2. **Counts-built behavior.** SVD spectral ends (7.3x decaying) +
   storebank MLPs over next-word stores + identity-V + similarity-QK:
   top1 0.0 (random) -> 0.472 (depth-4, BEATS word-bigram 0.406),
   twin_dist 1.72 -> 0.70 / full-8 1.35 -> 0.91. Construction only.
3. **Order as structure.** Causal mask (past bit-exact), RoPE geometry,
   reversal sensitivity everywhere, active/passive as two execution
   orders over one store (zero template strings), per-head temperatures
   (47th mnemonic TBETA, extension drill logged).
4. **Retrieval bridge.** 55 edges -> 68 keys, recall 1.00/1.00/0.99;
   word-cues 16/16 tag-exact, hidden-cues 14/16 via structural cosine
   (assoc_cos, norm-bias diagnosed); retrieve-then-generate demos run
   end-to-end (demo_fusion.py).
5. **Joint-rule discipline.** 9 rejections with numbers kept (gains,
   seeds, MID-bank, wgate); 8 writes each beating primary without
   regressing constraint. Negative results load-bearing in manifests.
6. **Full suite green.** 61/61 ALL OK (17 new gates this arc); README
   ledger + VELOCITY clocked; phi-core bridge merged to origin/main.

## Deliberately deferred (v2.x)

Vocabulary reach (BPE piece-fitting arc: bank mass, per-path refit,
S=16 standard, cascade listing pending shrink-geometry design);
output quality (ppl gap, v07 ambiguity, v02 UNK, no-repeat/guidance
boundaries); per-block M-dicts (global raises held with measured
cover-tax); depth/width scale-up past 4xH=2 (budgets proven, engineering).
