---
name: gemini-flash-vision-ceiling
description: "gemini-3-flash cannot perceive subtle physical AI artifacts on photoreal plausible scenes; the catchable axis is semantic (content_plausibility), not pixel"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e7fde0e-6771-4547-b8e1-c5cef72c7e8b
---

`gemini-3-flash-preview` (the production detection model) has a hard perceptual ceiling: it does **not** perceive subtle physical AI artifacts (impossible windshield reflections, material clipping, background texture attrition, warped mechanical symmetry) on a photorealistic scene that depicts a plausible situation.

Confirmed exhaustively on the "Osher Ad airplane in a supermarket" AI image: forced schema fields, holistic gestalt, forced per-marker tagging, the user's exact standalone 5-domain forensic prompt, thinking_level MEDIUM/HIGH (8x latency + timeouts, no change), and a model swap to gemini-3.5-flash — ALL still called it authentic. A different model the user tried DID catch it, so it's a vision-backbone difference, not a prompt gap.

**Implication / how to apply:**
- The reliably catchable axis on this model is **semantic** — read the signage and judge whether the offer is real-world possible. That is the `content_plausibility` check shipped in PR #36 (see [[detection-content-plausibility]]). It caught the parody "store selling Claude Sonnet 4.8 tokens / cloud storage" sign at 0.95.
- The "photoreal + plausible scene + only-subtle-physical-tells" class (e.g. the airplane) is OUT OF REACH for flash and should NOT be force-caught (forcing it reintroduces false positives on real promo stunts). To catch that class, add a second-opinion pass on a stronger vision model (`gemini-3-pro-preview`) — not more prompt surgery.
- Lowering temperature below ~0.2 (esp. temp 0.0 + top_k=1) historically collapsed visual judgments toward "authentic" on polished AI; production uses temp 0.2.
