# Generation-quality battery (measured 2026-10-06, 12 prompts x 8 tokens)

HF (bf16 transformers) vs geometric serve (fp16, bmmv+graph+nosync),
same prompts, greedy both sides. Reproduce:
`source scripts/cuda_env.sh`
`python3 scripts/eval_battery.py --hf-only --out /tmp/b_hf.jsonl`
`python3 scripts/eval_battery.py --geo-only --out /tmp/b_geo.jsonl`
`python3 scripts/eval_battery.py --compare /tmp/b_hf.jsonl /tmp/b_geo.jsonl`
(Geo side reuses /tmp/gen7b artifacts when the build stamp matches.)

## Results

| prompt | fork | top5 overlap |
|---|---|---|
| The capital of France is | fork@2 | 3/5 |
| The capital of Germany is | fork@2 | 3/5 |
| Alexander the Great founded | IDENTICAL | 5/5 |
| The Roman Empire fell in | fork@4 | 3/5 |
| Water boils at | fork@4 | 3/5 |
| The quick brown fox | fork@5 | 3/5 |
| Once upon a time | IDENTICAL | 5/5 |
| def fibonacci(n): | IDENTICAL | 4/5 |
| The president of the United States is | fork@1 | 4/5 |
| In the beginning | fork@3 | 4/5 |
| Machine learning is | fork@1 | 5/5 |
| The Eiffel Tower is located in | fork@0 | 4/5 |

**identical: 3/12; mean top-5 overlap: 3.83/5.**

## S=32 rerun (2026-10-06, same 12 prompts)

identical: 4/12 ("Machine learning is" joins the identical set);
mean top-5 overlap: 3.83/5; minimum 3/5. Floor holds (S changes
padding rows, so the sample shifts while the distribution holds --
expected, not drift).

## 24-token horizon, S=64 (2026-10-06)

identical: 0/12 (longer paths fork more -- expected compounding);
mean top-5 overlap: 3.83/5 (SAME as 8-token); minimum 3/5; several
prompts hold 9-19 tokens. Lesson: overlap is the horizon-robust
metric, identity is not. Floor, horizon-indexed: 8-tok identical
>= 3/12; 24-tok identical >= 0/12 + overlap >= 3.5/5 + min 3/5.

## Reading

Same distribution, different samples: greedy amplifies sub-noise
diffs (parity 4.84 abs on hot logits) into different paths, mostly
late (forks @2-5); overlap never drops below 3/5. The two early
forks (Eiffel fork@0, president/machine fork@1) with high overlap
are the attribution set for future lens work (which layer flips
the pick?).

## Floor (future optimizations must hold)

- identical >= 3/12, mean top-5 overlap >= 3.5/5, no prompt below 3/5.
- Any speed change that moves these moves quality, not just pace.
