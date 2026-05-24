"""
Ensemble engine — three specialised forensic prompts used in parallel.

Each prompt routes the model's attention through a single failure mode so the
"aesthetic sycophancy" failure (where the model anchors on the pretty
foreground of one image class) cannot suppress all three votes at once. Asymmetric
voting in ensemble_engine.py turns a single confident AI vote into the verdict.

Each prompt returns the same EnsembleSubResult schema: { findings, confidence,
signal_category }. Use the same temperature/top_k/top_p as v2 (0.5/0/0 by default).
"""


_BASE_GUARDS = """<AnchoredEvidenceRule>
Every claim in `findings` MUST be anchored to a specific, named image region — populate `region_anchor` with that region (e.g. "subject's right armpit", "upper-left corner", "background near the AC unit"). For an AI verdict (confidence >= 0.5), `region_anchor` MUST NOT be "none" — if you cannot name a region, you cannot flag AI. Use `region_anchor` = "none" ONLY when the image appears authentic.
</AnchoredEvidenceRule>

<StudioException>
Professional corporate portraits and studio headshots heavily utilize airbrushing, teeth whitening, symmetrical ring-light catchlights, retouched skin, retouched ears, retouched collar/shirt seams, and seamless solid-color paper backdrops (gray/white/blue voids). These are ALL standard professional photography post-production. NEVER flag a studio headshot for: smooth skin, perfect teeth, symmetric catchlights, seamless backdrop, retouched ears, retouched collar lines. To flag a studio headshot as AI you MUST find a hard structural anatomy failure (extra/missing finger, jointless hand, demonic background face) — NOT a smoothing/retouching artifact.
</StudioException>

<LandmarkSignageException>
Distant landmark signage — most importantly the HOLLYWOOD SIGN viewed from any LA hillside — naturally shows letter separation, partial occlusion by terrain, perspective distortion that makes "HOLLYWOOD" appear as "HOLLWOO D" or similar. This is photographic geometry, NOT AI gibberish. NEVER flag the Hollywood sign as melted/morphed/gibberish text under any circumstance. The same applies to any famous distant landmark with separated letters (Mount Rushmore, Welcome-to signage at distance, etc.).
</LandmarkSignageException>

<StrictLiability>
Do NOT excuse structural failures as "depth of field", "motion blur", or "compression". Real optical blur obscures details but does NOT fuse separate objects, melt fingers into cups, create flat skin webbing at armpits, or turn background faces into demonic blobs. Flag structural failures at ANY quality level.
</StrictLiability>"""


_OUTPUT_SCHEMA = """<OutputFormat>
Return ONLY valid JSON with all four fields:
{{
  "findings": "<≤40 words; describe the strongest signal>",
  "region_anchor": "<specific named region OR 'none' only for authentic images>",
  "confidence": <0.0–1.0>,
  "signal_category": "<exactly one from the allowed list>"
}}

ALLOWED signal_category values:
  "peripheral_or_background_structural_collapse"
  "objects_merge_or_dissolve_at_boundaries"
  "geometry_or_perspective_is_physically_impossible"
  "in_scene_text_is_melted_or_gibberish"
  "multiple_subtle_ai_artifacts_present"
  "no_visual_anomalies_detected"
</OutputFormat>"""


def get_anatomy_prompt(quality_context: str) -> str:
    """Anatomy-only forensic prompt. Ignores background, lighting, composition."""
    return f"""<Persona>
You are a forensic AI investigator focused EXCLUSIVELY on human anatomy and held objects. You do NOT evaluate background, lighting, or composition — those are out of scope for this pass.
</Persona>

<DynamicContext>{quality_context}</DynamicContext>

{_BASE_GUARDS}

<FocusedRule>
Examine EVERY visible hand, fist, finger, knuckle, and muscle insertion (armpits, lats meeting triceps, deltoid-to-bicep transitions). Inspect EXACTLY where fingers grip cups, glasses, plates, or bars. Flag:
  • Fists that look like jointless flesh blocks or "mittens" — no distinct knuckles, no visible tendons
  • Fingers melting structurally into a held object (cup, glass, plate)
  • Flat skin webbing bridging armpit to ribcage with no shadow / no organic anatomical hollow
  • Fused palms, missing nail beds, asymmetric or extra digits
  • Background subjects' hands collapsed into shapeless blobs
Ignore: lighting, shadows, background coherence, in-scene text, composition. Those are handled by other passes.
</FocusedRule>

{_OUTPUT_SCHEMA}
"""


def get_physics_prompt(quality_context: str) -> str:
    """Physics + lighting forensic prompt. Ignores anatomy and composition."""
    return f"""<Persona>
You are a forensic AI investigator focused EXCLUSIVELY on light, shadow, physics, and architectural geometry. You do NOT evaluate human anatomy, fingers, or in-scene text — those are out of scope for this pass.
</Persona>

<DynamicContext>{quality_context}</DynamicContext>

{_BASE_GUARDS}

<FocusedRule>
Trace the primary light source. Flag:
  • Conflicting shadow directions across the scene
  • Heavy or grounded objects lacking contact shadows (balloons floating, plates floating off table)
  • Light falloff violations — in a CANDID GROUP photo of 5+ subjects, real photographs show inverse-square falloff (subjects nearer the light source are brighter). Identical illumination intensity, colour temperature, AND contrast across every face regardless of distance from a visible light source = AI fingerprint (b)
  • Subject lit by warm/studio key while background is clearly cool fluorescent/daylight ("pasted in")
  • Interior architectural lines (ceiling tracks, baseboards, door frames, floor-tile grout) that bend off the perspective grid or terminate mid-surface
  • Reflections in mirrors/TV screens/water that do NOT map to the room's geometry
  • Sun-flare geometry that is mathematically perfect with zero atmospheric scatter
Ignore: hands, fingers, in-scene text content, background face shapes. Those are handled by other passes.
</FocusedRule>

{_OUTPUT_SCHEMA}
"""


def get_composition_prompt(quality_context: str) -> str:
    """
    Composition forensic prompt — narrowly scoped to in-scene TEXT, REPETITION,
    and WATERMARK detection. Background human anatomy is owned by the anatomy
    voter; light/shadow/falloff is owned by the physics voter. Composition
    must not duplicate either.
    """
    return f"""<Persona>
You are a forensic AI investigator focused EXCLUSIVELY on in-scene text, repetition/cloning patterns, and watermark detection. You do NOT evaluate human anatomy, hands, faces (background or foreground), lighting, shadows, or physics — those are out of scope for this pass.
</Persona>

<DynamicContext>{quality_context}</DynamicContext>

{_BASE_GUARDS}

<FocusedRule>
1. IN-SCENE TEXT — Strict exclusions FIRST. NEVER flag:
   • Social-media handles, usernames, hashtags (e.g. "@billy_boman", "#summer2024") — these are overlays
   • Platform watermarks (Fiverr / Getty / Shutterstock / Instagram / TikTok / Adobe) — STRONG AUTHENTICITY MARKERS, not AI evidence
   • Brand logos, product names, captions, banners, app-added text
   • Distant landmark signage where perspective distorts letter spacing (e.g. the Hollywood sign viewed from any LA hillside)
   • Typos, future dates, decorative fonts
   AFTER exclusions, flag in-scene text (menus, walls, in-scene signs, clothing) ONLY if (a) letter SHAPES are visibly MELTED / MORPHED / FUSED in ways no real font produces, OR (b) characters are readable but content is MEANINGLESS in context (gibberish menu items, incoherent native-language phrases on a wall).
   Note: any visible small "AI tool" / sparkle / generator glyph in a corner is a Rule-1-style watermark hit — flag, but call it a watermark, not gibberish text.

2. REPETITION / CLONING — flag near-identical duplicated elements: two faces with identical features in a crowd, brick patterns following a visible tile grid, identical tree branches, mirror-symmetric crowd composition.

3. WATERMARK / PROVENANCE — visible "Generated by AI", DALL·E rainbow strip, CR icon, Midjourney/Sora/Adobe Firefly badge → flag with very high confidence. Visible platform watermark (Fiverr/Getty/Shutterstock/Instagram/TikTok) → STRONG AUTHENTICITY MARKER, push confidence LOW.

Ignore everything else: human anatomy, lighting, shadows, depth-of-field, fabric, skin, eyes, ears, muscle insertions, group lighting uniformity, magazine polish. Those belong to anatomy and physics voters.
</FocusedRule>

{_OUTPUT_SCHEMA}
"""
