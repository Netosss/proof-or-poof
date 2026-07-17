# Gold set — detection eval

The gold set is the 31-image accuracy check for the AI-detection pipeline, run via
`scripts/eval_gold_set.py` (production combined client). See the detection memories in
`.claude/memory/` for the lessons baked into these cases (child-grip blind spot,
validated levers, prompt-overfit).

## What's in git
- `labels.json` — the source of truth: `{ "images/<file>": { "is_ai": bool, "note"? } }`
  (31 entries — 18 AI / 13 real). Repo-relative keys, so it's portable.

## What's NOT in git (and why)
- `images/` — the 31 actual photos (~18 MB). **Intentionally gitignored**: the set
  contains real people's photos, including a real LinkedIn profile and several
  photos of babies/children. Publishing identifiable/minor photos to a git remote is
  permanent (survives in history) and outward-facing, so they are carried out-of-band.

## Restore the images on a new machine
Pick one:
1. **Manual transfer (recommended):** copy the `gold_set_images.tar.gz` archive
   (AirDrop / USB / private cloud) and extract into place:
   ```bash
   tar -xzf gold_set_images.tar.gz -C backend-python/tests/gold_set/
   ```
   (creates `tests/gold_set/images/…`)
2. **Point at an existing folder:** if the photos already live somewhere on the new
   machine, run with an override — no copy needed:
   ```bash
   GOLD_SET_DIR=/path/to/gold_set python scripts/eval_gold_set.py
   ```
   (that folder must contain `labels.json` + `images/`)

## Run
```bash
cd backend-python
python scripts/eval_gold_set.py
```
Needs `GEMINI_API_KEY` in `.env`. Baseline reference: 22/25 (88%) on the older
25-case set; the current set is 31 images.

## Re-label / extend
Edit `labels.json` (add `"images/<newfile>": {"is_ai": true|false, "note": "..."}`),
drop the image in `images/`, re-run. Keep the "note" field for tricky cases — it's
what the detection memories reference.
