# Per-layer lens on geometric Qwen2-7B (2026-10-06, 3 prompts)

Same 28-layer program, all residuals read through final-norm +
unembed (fp64 host). scripts/lens_7b.py. Rank = prompt-end rank of
the final pick; entropy over the full distribution.

## Factual prompts ("The capital of France/Germany is")

- Rank collapses smoothly 114k/117k -> 1 across all 28 layers
  (708/~700 at L16, ~5 at L22-24, 1 at L27). No discrete
  funnel/filter boundary; late acceleration.
- Tops: code fragments L0-18 (elseif, Drawabl, :white, typeali),
  blanks L19-24 (entropy 7.6 -> 0.88), native surface L25
  (巴黎), English L26 (' Paris'), final pick L27.
- Entropy non-monotonic: mid peak ~7.8, collapse to 0.88 at L24,
  RISE to 1.5-2.8 at L26-27 (last layer spreads mass among
  near-ties -- the filter signature, if any lives anywhere).
- Picks lock at L27 (fp64/fp16 disagree on near-tie winners:
  fork physics inside the lens; rank curves are the finding,
  not the winner).

## Code prompt ("def fibonacci(n):")

- Same shape (rank falls throughout, locks L27 with a rank-2
  wobble at L26); final pick ' \n' (newline -- correct continuation).
- Early tops are code-flavored here too (.toLocal x6) -- but that
  IS content for this prompt, so the early-code texture may be
  prompt-colored, not pure substrate. Unseparated; stated.

## Reading (funnel/filter verdict)

Both, gradual: convergence runs the whole depth (funnel-like),
the decision lands in the last 1-2 layers (filter-like). No
staged hand-off visible on these prompts. For the siphon: facts
and decisions live LATE (L22+ selects/sharpens); early layers
are shared substrate. Take late, skip early -- and the L22
single-donor precedent rhymes (different scale, same address).
