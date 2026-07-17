---
name: gold-set-child-grip-blindspot
description: The detection gold set has no child/infant-gripping-object photos; the prompt false-positived on babies holding cups until GripAndChildGuard was added
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e7fde0e-6771-4547-b8e1-c5cef72c7e8b
---

The forensic gold set (`test_prompt_gold.json`, 25 images) contains **no photos of infants/toddlers gripping an object** — a major real-world blind spot.

Because of this gap, the baseline prompt scored 92% on the gold set while **falsely flagging every real baby-holding-a-cup photo as AI at 0.85–0.95** ("fused fingers / fingers melt into the cup / jointless flesh block"). The baseline's PERSPECTIVE 1 anatomy rule + `<StrictLiability>` guard actively hunt for "fingers melting into cups," and a chubby toddler hand wrapped around a cup looks exactly like that to gemini-3-flash.

Fix shipped in PR #36 (`feat/detection-content-plausibility-grip-guard`): a `<GripAndChildGuard>` — a gripping/occluded/motion-blurred/infant hand that cannot be traced finger-by-finger is **authentic by default**; only flag a fully-resolvable hand with a positively-described impossible digit. Verified: all 3 baby photos → REAL 0.00–0.10; full `/detect` route → "Likely Authentic" 0.99; gold set unchanged at 92%.

**How to apply:** when evaluating detection FP risk, test child/hand-gripping-object cases explicitly — the gold set will not catch regressions there. Run the real `/detect` route (via `detect_ai_media`), not just the bare Gemini call, and watch for stale `forensic:*` Redis cache masking prompt changes. Related: [[detection-prompt-overfit-lesson]].
