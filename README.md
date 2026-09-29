# holo-phi: true-amplitude holographic enhancement on the phi lattice

One-line: old `holographic_enhancement` claimed `A=sqrt(I)` but never took a
square root; this repo actually does `A=sqrt(Y)` via exponent-halve, blurs in
amplitude domain, boosts, and squares back with exact `tmul`.

## Results

Check | Result
---|---
`test_core.py` | ALL OK (sqrt-real, tmul, conv-orderfree, rescale-exact, tag-mismatch, hue basis, c-vectors)
`test_parity.py` | ALL OK (synthetic 68/65dB, foreman 65dB, hue-preserved 0.000, alpha-ablation gated)
`test_c.py` | ALL OK (shared table + conv + splat file-exchange + float trap)
`test_c_conv.py` | ALL OK (4 cases bit-exact, not dB: fig/flat/corner/noise)
`test_splat_c.py` | ALL OK (full splat_blur bit-exact incl. buckets, both C modes: 6 compositional + 2 fused cases)
`test_splat.py` | ALL OK (4 orientations 1.00, halo<=iso, blur parity 41dB, agreement 0.99, e2e 43dB, fusion-exact bit-identical)
`test_ctrl.py` | ALL OK (file validity, hash, real-frame parity ~43-50dB, rotation-invariant gap 0.04, flat identity)
`demo.py --selftest` | GO 68.5dB

## Quick start

```
pip install -r requirements.txt
python3 chain/calibrate.py          # offline, freezes chain/M.json (m_acc + m_cov)
python3 scripts/fit_ctrl.py --write # offline, freezes chain/CTRL.json (2 params)
python3 test_core.py                # expect ALL OK
python3 test_parity.py              # expect ALL OK (>=40dB)
python3 test_c.py                   # expect ALL OK (needs gcc; incl. conv exchange)
python3 test_splat.py               # expect ALL OK (splats-lite)
python3 test_ctrl.py                # expect ALL OK (beta-field controller)
python3 demo.py input.png output.png --beta 0.5 --blur splat --ctrl on
python3 showcase.py input.png /tmp/showcase  # all modes + internal states
```

## How it works

phi pipeline: sRGB --encode--> linear Y triples -> `sqrt_trip` (the missing
op) -> `conv_trip` integer blur @ m_acc -> `rescale_` -> `binop` detail @
m_cov -> `tmul` boost -> `tmul` square (`I=|A|^2`) -> `tdiv_trip` gain ->
`tmul` chroma-preserve -> --decode--> sRGB. See phi-core `IR.md`. Float only
at boundaries + offline.

Math executed: `A=sqrt(Y); As=blur(A); Aenh=A+beta_eff*(A-As); Ienh=Aenh^2`,
where blur is iso-Gaussian or the splat bank and beta_eff is scalar or the
controller field. Library notes live in docs/LIBRARY_NOTES.md (kept while
using the library: #LIB-001..009). Specs: docs/SPLAT_OP.md, docs/BETA_CTRL.md,
docs/COMPOSE.md (step 3: depth + temporal, specified not built).

## Showcase (seeing every state)

`showcase.py` runs iso / splat / splat+ctrl / strong on one image, checks
parity per mode, asserts the modes actually differ, and saves:

* `out_*.png` -- one output per mode,
* `st_amplitude/structure/detail/buckets/coherence.png` -- internals of the
  flagship config (wave, structure, ripple, 5-color orientation map,
  coherence heat),
* `sheet_states.png`, `sheet_modes.png` -- labeled contact sheets.

On f_012: input sharp 714 -> iso 1350 -> splat 1178 -> splat+ctrl 1077 ->
strong 1465, mean change shrinking along the intelligence axis
(1.83/1.40/0.95 LSB). Each step visibly restrains itself for stated reasons.

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
