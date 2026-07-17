---
name: detection-prompt-overfit-lesson
description: Always validate detection-prompt changes against the gold set; tuning on a few images overfits and silently regresses recall
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7e7fde0e-6771-4547-b8e1-c5cef72c7e8b
---

When iterating on the Gemini forensic detection prompt (`app/integrations/gemini/prompts_combined.py` + `app/schemas/detection.py`), tuning to catch a couple of specific images badly overfits.

In one session, piling on skeptical scaffolding (scene-plausibility + physics/fidelity + rendering-signature fields + many FP fences) to catch 2 images regressed the gold set from **92% → 64%** (added 7 false negatives) — the extra "is this plausible?" fields and "do NOT flag X" guards made the model rationalize real AI as authentic.

**Why:** more skeptical fields decode first and mostly resolve to "plausible/coherent," priming the verdict toward REAL; accumulated fences teach the model to explain away genuine artifacts.

**How to apply:** keep prompt changes minimal and principle-based (not example lists, which overfit in whichever direction the examples lean). ALWAYS run `python scripts/eval_gold_set.py` (the 25-image labeled set, `test_prompt_gold.json`) before shipping — and know it has blind spots, see [[gold-set-child-grip-blindspot]]. The winning change was one semantic field + one targeted guard, see [[detection-content-plausibility]].
