---
name: detection-content-plausibility
description: "The content_plausibility field catches AI parody/concept-ad images via a semantic two-clause test (deliverability + referent), judging the offer not the seller"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e7fde0e-6771-4547-b8e1-c5cef72c7e8b
---

`content_plausibility` (first field in `CombinedDetectionResult`, shipped PR #36) is the detection prompt's semantic catch for high-fidelity AI "concept ad" images whose tell is what the scene DEPICTS, not the pixels.

It judges **the offer's own stated terms, never the seller or category** (this is the key anti-overfit principle — judging by "who sells what" overfit in both directions: a "grocery sells aircraft = absurd" rule false-positived on real Costco/Aldi out-of-category retail; a "win-this-plane is real" fence then lost the catch). THREE clauses, all example-light/principle-based:
- **clause1 DELIVERABILITY** — can a real seller physically hand over this good via the depicted channel? An intangible/digital-only service (LLM tokens, cloud-storage tiers) sold as a physical-shelf SKU cannot → FALSE.
- **clause2 REFERENT** — do all named products/versions/specs exist with coherent units? A non-existent version, an impossible unit → FALSE.
- **clause3 RETAILABILITY** (added later) — could a private consumer COMPLETE this as a real retail purchase? A functional weapon-of-war / military munition (missiles, launchers, air-defense systems, tanks, ordnance) — or any good no retailer can lawfully sell a civilian (restricted hazmat, organs, protected wildlife) — offered for sale with a consumer price → FALSE. Fenced HARD to the conjunction of *functional weapon-of-war + genuine consumer retail offer with a price*: firearms, toys/replicas, deactivated/surplus/collectible militaria, props, fireworks, and display/museum/expo/B2B/news items are all PLAUSIBLE. Caught the "supermarket selling a home missile-interceptor for 250k" parody at 0.95; gold set held 24/25, 0 FP.

ABSURD_CONTENT (any clause FALSE) forces confidence >= 0.70. Caught the parody "OsherAI / Claude Sonnet 4.8 tokens + Osher4Ever cloud storage" sign at 0.95; left real surreal/novelty/promo photography authentic.

**Honest limit:** this only catches AI images that ADVERTISE an impossible offer in readable text. AI with plausible/no text and only physical tells is not caught — see [[gemini-flash-vision-ceiling]]. Validation method and gold-set caveat: [[detection-prompt-overfit-lesson]], [[gold-set-child-grip-blindspot]].
