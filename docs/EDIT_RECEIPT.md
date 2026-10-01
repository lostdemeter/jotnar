# Edit receipt (standard): making a model edit transferable

Every weight edit ships with a receipt. No receipt = no handoff. Fields:

1. **Change**: what bytes moved (slot, old -> new, mechanism: surgery /
   listing / store rebuild). Enough detail to replay.
2. **Prediction**: stated BEFORE running (band, not point): expected dB
   + expected locus (global? which tokens/pixels?) + mechanism (which law:
   factorization, norm-fit triage, measured analog?).
3. **Measurement**: rerun numbers (global dB, locus check, sham/controls).
4. **Footprint**: where it bit (per-token deltas / spatial map / hue
   histogram -- the representation that fits the model).
5. **Verdict**: CONFIRM / FALSIFY / SPLIT per field, with the residue
   (what the miss teaches) stated, never trimmed.

Bands follow the predictor's honesty grade: factorization-backed (+/-3dB),
measured-analog (+/-10dB triage), novel mechanism (wide or reported-only).
A receipt that hides a miss is green wallpaper; re-read LIB-015.

## Receipt #001: DDColor q56 hue-write (dd_modify.py, f_014)

1. **Change**: q56 slot -- query rows <- q41 copy; refine row <- 135deg @
   norm 0.2 (weight_orig; hooks recompute). Mechanism: surgery.
2. **Prediction**: hue135 mass >3x (new hue appears); global < 35dB
   (norm-fit triage grade).
3. **Measurement**: hue135 mass 0.164 -> 0.158 (x1.0); global 14.5dB.
4. **Footprint**: vote mass flips sign (-14k -> +28k); post-write q56
   silence moves 21.4dB (slot live); hue histogram flat (unproven).
5. **Verdict**: SPLIT -- hue-appears FALSIFIED (query-hue sparsity !=
   output absence: 135deg already at 16%), bounded CONFIRMED, slot-live
   CONFIRMED (follow-up). Residue: disentangle open (refine vs query
   attribution) -- closed later in LIB-067 as place+color pair.
