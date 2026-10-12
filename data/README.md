# data/ — runtime artifacts live here, ignored (see ../.gitignore)

Only the 9 small emission-gate fixtures are tracked; every kept test that
needs one SKIPs without it, so coverage silently drops if they go missing:

- `lm_bigrams.npz` — bigram counts (test_emit_c, test_emit_cuda,
  test_emit_nonfpu, test_serve)
- `lm_svd_IvoQ.npz`, `bankhn.npz` — SVD weights + bank (test_builder_block,
  test_emit_cuda, test_emit_nonfpu, test_emit_headt)
- `lm_piece32_fit.npz`, `bankp32_64.npz` — D32 refit + bank (test_emit_cuda,
  test_emit_widths)
- `lm_piece64_fit.npz`, `bankpiece64_d64.npz` — D64 refit + bank (same)
- `bpe_vocab.json`, `bpe_merges.json` — BPE tables for the D32/D64 prompt
  encoding in test_emit_widths (via scripts/d64_lib.py)

Everything else (ingest scale artifacts, teacher ports, 7B freezes) is
rebuilt or downloaded, never committed. Full history lives on the
`research` branch, never merged into main.
