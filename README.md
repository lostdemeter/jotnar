# holo-phi: true-amplitude holographic enhancement on the phi lattice

One-line: old `holographic_enhancement` claimed `A=sqrt(I)` but never took a
square root; this repo actually does `A=sqrt(Y)` via exponent-halve, blurs in
amplitude domain, boosts, and squares back with exact `tmul`.

## Results

Check | Result
---|---
`test_core.py` | ALL OK (sqrt-real, tmul, conv-orderfree, rescale-exact, tag-mismatch, hue basis, c-vectors incl. sig)
`test_parity.py` | ALL OK (synthetic 68/65dB, foreman 65dB, hue-preserved 0.000, alpha-ablation gated)
`test_c.py` | ALL OK (shared table + conv + splat file-exchange + v4 gates + float trap)
`test_c_conv.py` | ALL OK (4 cases bit-exact, not dB: fig/flat/corner/noise)
`test_splat_c.py` | ALL OK (full splat_blur bit-exact incl. buckets, both C modes: 6 compositional + 2 fused cases)
`test_splat.py` | ALL OK (4 orientations 1.00, halo<=iso, blur parity 41dB, agreement 0.99, e2e 43dB, fusion-exact bit-identical)
`test_ctrl.py` | ALL OK (file validity, hash, real-frame parity ~43-50dB, rotation-invariant gap 0.04, flat identity)
`test_depth.py` | ALL OK (cache determinism, parity 45.68dB, near p99 3.6x far, toggle-clean)
`test_v4.py` | ALL OK (noise 40.27dB BARRED, real 52dB, structured 55dB, rotation 0.04, flat, sharpens)
`test_v5.py` | ALL OK (grain 47.95dB barred, real 47.61, structured 50.43, rotation 0.04, detail-order, noise measured 37.10)
`test_substitute.py` | ALL OK (sigmoid swaps numpy<->torch/CUDA exact on 2007 triples; chain-swap bit-identical live 2x; replicate-vs-zero refusal characterized)
`test_motion.py` | ALL OK (compass 14, rescue-V 0.36->0.81, zeroflow-exact bytes, pan-preserve, parity 42.94dB, pan-e2e ratio 0.504, rotation 0.04)
`test_temporal.py` | ALL OK (mix-frozen, warp-parity 73dB, warp-identity, static-converged + firststep bounds, firstframe-still, memory, step-settles, seq 64dB)
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
python3 test_motion.py               # expect ALL OK (motion side-channel consensus)
python3 test_temporal.py             # expect ALL OK (feedback: detail IIR + warp)
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
#LIB-001..015). Specs: docs/SPLAT_OP.md, docs/BETA_CTRL.md, docs/COMPOSE.md
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
