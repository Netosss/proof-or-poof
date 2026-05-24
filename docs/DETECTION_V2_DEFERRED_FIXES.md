# Deferred fixes from the multi-expert review (`feat/detection-v2-forensic-rewrite`)

These four items came out of the multi-expert review but **touch the prompt or the voting rule**, so they can shift accuracy in either direction. Apply them one at a time after baseline accuracy is measured on a 5-image (or larger) test set, and re-verify accuracy after each.

## Verification protocol for this phase

1. Pick a fixed 5-image test set (mix of AI and real). At minimum: `gym`, `bibi`, `bbq` + 2 from the original gold set.
2. Run each image **5 times** at the current `temp=0.5` to characterise variance — record `(verdict, confidence, voter breakdown)` for each run.
3. Compute baseline: AI-catch rate per image, FP rate on real images.
4. Apply ONE deferred fix below.
5. Re-run the same 5 images × 5 times. Compare.
6. Keep the fix if it improves OR holds the baseline; revert if it regresses any catch by more than 1/5 runs.
7. Repeat for the next fix.

---

## D1 — Tighten the asymmetric-vote rule (currently most likely to fire spurious AI flags)

**Where:** `app/detection/ensemble_engine.py:_asymmetric_vote()` + `app/config.py:ensemble_ai_threshold`

**Current:** Any single voter with `confidence > 0.7` flips the verdict to AI.

**Risk identified:** With 3 voters at `temp=0.5` (non-deterministic), if each voter has ~5% chance of a spurious `>0.7` on a real image, `P(at least one FP) ≈ 1 − 0.95³ = 14.3%`. We measured AI-catch rate but never measured FP rate on the gold set under the ensemble engine.

**Proposed change:** require **two voters at ≥ 0.55** OR **one voter at ≥ 0.85**. Preserves the gym-style "single high-conviction signal" catch path while killing the "one noisy 0.71 dominates three 0.05s" pathology.

**Possible regressions to watch for:**
- bibi: anatomy fired at 0.90 → still wins under "one ≥ 0.85" branch. Should hold.
- bbq: physics fired at 0.92 → still wins. Should hold.
- gym: anatomy fired at 0.92 (1 in ~20 runs) → still wins on the rare lucky run.
- Any gold-set AI image where exactly one voter was firing at 0.70–0.84 with the others < 0.55. These will now MISS. **Measure before deciding.**

---

## D2 — Re-partition the three prompts to be genuinely orthogonal

**Where:** `app/integrations/gemini/prompts_ensemble.py`

**Current state, with duplication:**
- `get_anatomy_prompt` (line 52): *"Background subjects' hands collapsed into shapeless blobs"*
- `get_composition_prompt` (line 97): *"Distant faces in a crowd... blobs"* + *"Background subjects' hands collapsed into shapeless blobs"* (same text)
- `get_physics_prompt` (line 76) and `get_composition_prompt` (line 102) both mention "uniformly polished casual context"

**Risk identified:** the three voters aren't actually routing different attention — they're three near-copies of the v1 prompt. Race-to-AI cancellation throws away ~zero unique signal because composition was going to flag the same anatomy artifact anyway. Catch-rate data confirms this: bibi → anatomy, bbq → physics. Composition is contributing noise.

**Proposed clean partition:**
- **Anatomy** owns ALL human anatomy (foreground + background hands, faces, fists, muscle insertions).
- **Physics** owns ALL light/shadow/geometry/reflection/falloff.
- **Composition** owns ONLY in-scene text + repetition/cloning + watermark detection. Its `confidence` should NOT be allowed to flip the verdict alone (it's the noisiest signal in practice).

**Possible regressions to watch for:**
- An AI image currently caught by composition's anatomy-overlap rules that the anatomy voter happens to miss in a specific run. Measure.

---

## D3 — Replace prose banned-phrase list with structural enforcement

**Where:** `app/integrations/gemini/prompts_ensemble.py:_BASE_GUARDS` and `app/schemas/detection.py:EnsembleSubResult`

**Current:** `_BASE_GUARDS` lists banned phrases (`"asymmetric vascularity"`, `"natural micro-textures"`, etc.) as prose.

**Risk identified:** listing banned phrases in the prompt is a known anti-pattern — naming them in the system instruction **primes the model to produce them**. Empirically observed: the model used "asymmetric vascularity" anyway.

**Proposed change:**
1. Add `region_anchor: str = Field(description="Specific named region the finding is anchored to, e.g. 'left temple', 'upper-left corner', or 'none'")` to `EnsembleSubResult`.
2. In `client_ensemble.analyze_with_prompt`, after parsing the response, reject sub-calls where the `findings` text references skin/texture/symmetry semantics but `region_anchor == "none"` — treat as `ok=False`.
3. Once enforcement is structural, DROP the prose ban list from the prompt entirely (no priming).

**Possible regressions to watch for:**
- AI cases caught by a voter that DID use a banned-list phrase. Re-running may produce different findings that don't trip the structural rejection — measure if catch rate holds.

---

## D4 — Bump voter timeout from 7s → 10s

**Where:** `app/config.py:ensemble_voter_timeout_s`

**Risk identified:** anatomy voter returns the winning 0.95 on bibi at ~4.6s, with run-to-run variance that pushes past 7s on some runs. Hard timeout is killing the strongest voter.

**Proposed change:** raise to **10s**. Race-to-AI cancellation keeps wall-clock down on AI cases (early-exits at ~1.6s typical). REAL cases will sit at 10s instead of 7s on the slow tail.

**Possible regressions to watch for:**
- Tail latency: REAL-image p95 climbs from ~7s → ~10s. Acceptable per the latency budget but should be confirmed on the gold set.
- Catch-rate improvement on bibi-class cases where the winning voter is slow. Likely a gain, not a regression.

---

## Order of operations (recommended)

1. **D4 first** (timeout bump) — lowest accuracy risk, most likely to be a strict win.
2. **D2** (re-partition prompts) — biggest expected accuracy gain on the gold set if the orthogonality hypothesis holds.
3. **D3** (structural anti-anchoring) — depends on D2 since both touch prompts.
4. **D1 last** (vote rule change) — biggest accuracy risk; do it only after the prompts are clean so we're tuning against the right baseline.

After all four, re-run the **full 25-case gold set** to make sure nothing in the easy cases regressed.
