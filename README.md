# Jotnar: geometric assembly for bespoke AI on the phi lattice

Jotnar grew out of `holographic_enhancement` (true-amplitude holographic
enhancement, below) into an assembly language for composing geometric AI
structures — every operation an explicit integer op on lattice values or
a frozen LUT built offline, every program a listing, every claim a gate.

One-line history: old `holographic_enhancement` claimed `A=sqrt(I)` but never took a
square root; this repo actually does `A=sqrt(Y)` via exponent-halve, blurs in
amplitude domain, boosts, and squares back with exact `tmul`.

## Results

Check | Result
---|---
`test_core.py` | ALL OK (sqrt-real, tmul, conv-orderfree, rescale-exact, tag-mismatch, hue basis, c-vectors incl. sig)
`test_parity.py` | ALL OK (synthetic 68/65dB, foreman 65dB, hue-preserved 0.000, alpha-ablation gated)
`test_c.py` | ALL OK (shared table + conv + splat file-exchange + v4 gates + float trap)
`test_c_conv.py` | ALL OK (4 cases bit-exact, not dB: fig/flat/corner/noise)
`test_matmul_c.py` | ALL OK (5 cases bit-exact: small/batched/broadcast/big-m/zeros)
`test_matmul_cuda.py` | ALL OK (3-way numpy/C/CUDA bit-exact, same 5 cases; SKIPs without nvcc)
`test_units_cuda.py` | ALL OK (k_conv_rep 4 cases + k_mux/clamp bit-exact vs C twins; SKIPs without nvcc)
`test_flagship_cuda.py` | ALL OK (full splat_soft+v5 A->YENH on CUDA bit-exact: synth/real/big-m)
`test_emit.py` | ALL OK (substrate matrix: flagship numpy/C/CUDA exact + dispatch log; xf numpy + loud C/CUDA refusals; bogus names refused)
`test_census.py` | ALL OK (planted fan-out exact, control empty, 42/42 classified, flagship/xf/stabilize maps pinned incl. STATE versions)
`test_probe.py` | ALL OK (sham exact, D 38dB in [20,45], COH falsified 44dB then held-out 43dB, xf O/DOWN, 11-stream sweep in 4 exact classes, stabilize W static 27dB + motion exact; probe table)
`test_modify.py` | ALL OK (modification demo: iso variant bit-exact vs chain, differs 45dB from v5, census shows new structure)
`test_edits.py` | ALL OK (weight-edit routes: matrix ordering V>MLP>Q, channel spread 15.6dB, scale-linearity, rank-monotone, direction spread 51.4dB basis-independent; standard process)
`test_realw.py` | ALL OK (Qwen2-0.5B L0 MLP on real weights+embeddings: parity 51.82dB, spectrum 17x, direction spread 33dB, planted recovery exact, blind predict 0.22dB; SKIPs without HF cache)
`test_implant.py` | ALL OK (rank-1 implant with functional aim: target token rewritten at negative dB, leads field by 26-35dB on target + held-out; structure transfer 36.8dB quantum-tax bounds; SKIPs without HF cache)
`test_store.py` | ALL OK (directional stores: sham 99.8dB, listing-vs-surgery 91.2dB, bank-vs-matmul 96.1dB toy, stdlib DEFs resolve)
`test_storebank.py` | ALL OK (native banks on real weights: parity 55.8dB, pruned 42.7 vs 42.9 predicted; SKIPs without HF cache)
`dd_mine.py` / `dd_footprint.py` | research one-shots (DDColor 100-query mining: vote maps diffuse, footprints bite -- q39 dark-region 3.7x; needs HF cache + ddcolor checkout)
`dd_catalog.py` | full palette catalog in 28s (docs/PALETTE_CATALOG.csv: hue + mass + causal dB per query; statics beat dynamics as predictor)
`dd_modify.py` | ADD demo + disentangle modes (query-only 36dB / refine-only 19dB / both 14.5dB: slot = place+color pair; see #LIB-067)
`dd_resonant.py` | resonant-phase probe: scalar + H=8 signatures both scramble (~chance); bridge restated for scalar attributes, not key vectors
`test_read.py` | ALL OK (read instrument: table shape, query plumbing, token-effect spread, shelf-map logic, factor form, predictor logic; real 112-dir readout in docs/MODEL_READ.md)
`test_prune.py` | ALL OK (CRUD Q1+Q2: 27-dir removal 42.8 vs 42.9 predicted; gain-x2 == ablation at 22.1dB; SKIPs without HF cache)
`test_create.py` | ALL OK (CRUD Q3+Q4: label-to-store France rank 1/8 at 8.4dB gap 22.6; shelf-null 57.8dB; SKIPs without HF cache)
`test_projbank.py` | ALL OK (v1.4 gate 1: 6/6 projection banks 59-63dB; prune harmless 76.1dB with floor hypothesis; SKIPs without HF cache)
`test_select.py` | ALL OK (v1.4 gate 2: S08 first instance -- P-invariance bit-exact, recompose 82.9dB, top-beats-bottom ordering)
`docs/T_TRANSFORM.md` | v1.4 gate 4: full-range attention specified (fold obstruction, 9-layer ranges, one-assert phi-core fix + our composition; DDColor L0 needs m_of(~16))
`test_layer1.py` | ALL OK (CRUD Q5: layer-1 pattern reproduces -- spectrum 12x, giant 14dB, tracking -0.54; SKIPs without HF cache)
`test_engram.py` | ALL OK (native storage: freeze/load roundtrip 6.4e-16, deterministic bytes, store-IO bit-exact; DDColor query/refine stores frozen; SKIPs without HF cache)
`docs/EDIT_RECEIPT.md` | edit-receipt standard + filled receipt #001 (q56 hue-write SPLIT)
`test_splat_c.py` | ALL OK (full splat_blur bit-exact incl. buckets, both C modes: 6 compositional + 2 fused cases)
`test_splat.py` | ALL OK (4 orientations 1.00, halo<=iso, blur parity 41dB, agreement 0.99, e2e 43dB, fusion-exact bit-identical)
`test_ctrl.py` | ALL OK (file validity, hash, real-frame parity ~43-50dB, rotation-invariant gap 0.04, flat identity)
`test_depth.py` | ALL OK (cache determinism, parity 45.68dB, near p99 3.6x far, toggle-clean)
`test_v4.py` | ALL OK (noise 40.27dB BARRED, real 52dB, structured 55dB, rotation 0.04, flat, sharpens)
`test_v5.py` | ALL OK (grain 47.95dB barred, real 47.61, structured 50.43, rotation 0.04, detail-order, noise measured 37.10)
`test_substitute.py` | ALL OK (sigmoid swaps numpy<->torch/CUDA exact on 2007 triples; chain-swap bit-identical live 2x; replicate-vs-zero refusal characterized)
`test_asm.py` | ALL OK (flagship listing bit-exact vs chain + 4 discipline gates)
`test_stabilize.py` | ALL OK (flicker -26%, no smear, parity 64dB linear, static-exact, frame0-still)
`test_motion.py` | ALL OK (compass 14, rescue-V 0.36->0.81, zeroflow-exact bytes, pan-preserve, parity 42.94dB, pan-e2e ratio 0.504, rotation 0.04)
`test_temporal.py` | ALL OK (mix-frozen, warp-parity 73dB, warp-identity, static-converged + firststep bounds, firstframe-still, memory, step-settles, seq 64dB)
`test_router.py` | ALL OK (decisions incl. boundary, seamless-static exact, flicker-wins, sharpness-bounded, routed parity 60-63dB)
`test_xf_block.py` | ALL OK (v1.0 Gate 1: whole-block 84dB vs torch in-contract S=8/D=16/Dff=32; out-of-contract measured 14.6dB, mechanism stated)
`test_stranger.py` | ALL OK (v1.0 Gate 6: docs-only 5-line listing green first try + self-rescue errors; frictions -> docs/V1_1_BACKLOG.md)
`demo.py --selftest` | GO 68.5dB

## Quick start

```
pip install -r requirements.txt
python3 chain/calibrate.py          # offline, freezes chain/M.json (m_acc + m_cov)
python3 scripts/fit_ctrl.py --write # offline, freezes chain/CTRL.json (5 params)
python3 test_core.py                # expect ALL OK
python3 test_parity.py              # expect ALL OK (>=40dB)
python3 test_c.py                   # expect ALL OK (needs gcc; incl. conv+splat exchange, v4)
python3 test_splat.py               # expect ALL OK (splats-lite)
python3 test_ctrl.py                # expect ALL OK (beta-field controller)
python3 test_depth.py               # expect ALL OK (needs DAV2 checkout + weights)
python3 test_v4.py                  # expect ALL OK (continuity: noise barred)
python3 test_v5.py                  # expect ALL OK (learned gate, grain barred)
python3 test_substitute.py          # expect ALL OK (needs torch + rife checkout; cross-model swap)
python3 test_asm.py                  # expect ALL OK (assembly fidelity + discipline)
python3 test_stabilize.py            # expect ALL OK (generative proof: stabilizer listing)
python3 test_motion.py               # expect ALL OK (motion side-channel consensus)
python3 test_temporal.py             # expect ALL OK (feedback: detail IIR + warp)
python3 test_router.py               # expect ALL OK (routing: still/temporal per frame)
python3 demo.py input.png output.png --beta 0.5 --blur splat --ctrl on
python3 demo.py input.png output.png --beta 0.5 --blur splat_soft --ctrl soft  # v4
python3 demo.py input.png output.png --beta 0.5 --blur splat_soft --ctrl v5    # v5
python3 showcase.py input.png /tmp/showcase  # all modes + internal states
```

## How it works

phi pipeline: sRGB --encode--> linear Y triples -> `sqrt_trip` (the missing
op) -> `conv_trip` integer blur @ m_acc -> `rescale_` -> `binop` detail @
m_cov -> `tmul` boost -> `tmul` square (`I=|A|^2`) -> `tdiv_trip` gain ->
`tmul` chroma-preserve -> --decode--> sRGB. See phi-core `IR.md`. Float only
at boundaries + offline.

Math executed: `A=sqrt(Y); As=blur(A); Aenh=A+beta_eff*(A-As); Ienh=Aenh^2`,
where blur is iso-Gaussian, the splat bank (hard mux / fused / relu-blend),
and beta_eff is scalar, the v3 decision field, the v4 sigmoid field, or the
v5 learned gate (1x1 on coh + detail magnitude, fitted offline, frozen).
Library notes live in docs/LIBRARY_NOTES.md (kept while using the library:
#LIB-001..028). Language reference: docs/LANGUAGE.md (every mnemonic +
tutorial + contribution process). Velocity log: docs/VELOCITY.md. Specs: docs/SPLAT_OP.md, docs/BETA_CTRL.md, docs/COMPOSE.md
(step 3a depth built L1; temporal + L2 stay spec), docs/PROGRESS.md.

## Showcase (seeing every state)

`showcase.py` runs iso / splat / splat+ctrl / v4-soft / v5-gate / strong on
one image, checks parity per mode, asserts the modes actually differ, and saves:

* `out_*.png` -- one output per mode,
* `st_amplitude/structure/detail/buckets/coherence.png` -- internals of the
  flagship config (wave, structure, ripple, 5-color orientation map,
  coherence heat),
* `sheet_states.png`, `sheet_modes.png` -- labeled contact sheets.

On f_012: input sharp 714 -> iso 1350 -> splat 1178 -> splat+ctrl 1092 ->
v4-soft 1108 -> v5-gate 1294 -> strong 1520. v5 recovers sharpness selectively
(1.36 LSB mean change vs iso's 1.90): the detail gate spends boost where
detail lives.

## Debt log (addressed earlier rounds)

1. Two scales: `m_acc` (products) + `m_cov` (values), `rescale_` is the only
   scale changer (fixed@m1->fixed@m2, mirrors `fq_rescale`). Mixing without
   it raises. Fixed one real saturation bug found at 13dB.
2. Alpha removed: parabola was hiding noise/clipping; amplitude math +
   integer gain-clip [0.5,2] + out-clip [0,1] handle both honestly.
   `--alpha` keeps the ablation, still parity-gated.
3. Boundary hardened: Rec.709 luma fingerprinted (sum 1.0), hue-preservation
   gated off-rail (0.000 drift).
4. C lowering: `c_chain/` sqrt/mul + `holo_conv` (replicate-edge, accum
   @ m_acc, `rescale_` to m_cov, schedule oy-ox-ky-kx recorded in header) +
   shared vector table incl. the floor-div edge (d=-3 -> -2) +
   file-exchange parity (4 cases BIT-EXACT, not dB) + `make trap` clean.

## License

GPLv3.
