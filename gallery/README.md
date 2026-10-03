# Gallery: one geometric program, three backends

`demo_img.py` builds a 43-op image pipeline (`programs/img_demo.asm`,
generated -- luma, invert, brightness, contrast, 3x3 box blur, unsharp
mask, before/after strip) and runs the SAME text on three targets:

| backend | arithmetic | agreement (vs C) |
|---|---|---|
| C (cc, libc+libm) | float64 | exact (reference) |
| CUDA (nvcc, 3090 Ti) | float32 + kernels | maxabs ~1e-4 |
| non-FPU (cc, no libm, no float in source) | lattice fixed-point triples | PSNR 56-69 dB (quantum floor) |

Files: `input.png` source (352x288); `c_<effect>.png` per effect;
`cu_us.png` / `nf_us.png` backend comparison; `cu_strip.png`
crop|blur strip. Regenerate: `python3 demo_img.py`. Gates:
`tests/test_img_demo.py`.
