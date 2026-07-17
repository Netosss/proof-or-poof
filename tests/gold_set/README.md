# Gold set — detection eval

The gold set is the 31-image accuracy check for the AI-detection pipeline, run via
`scripts/eval_gold_set.py` (production combined client). See the detection memories in
`.claude/memory/` for the lessons baked into these cases (child-grip blind spot,
validated levers, prompt-overfit).

## What's in git
- `labels.json` — the source of truth: `{ "images/<file>": { "is_ai": bool, "note"? } }`
  (31 entries — 18 AI / 13 real). Repo-relative keys, so it's portable.

## `images/` — committed for a one-time laptop transfer (REMOVE AFTER FETCH)
The 31 photos (~18 MB) are committed here **only to move them to the new machine via
this private repo**. The set contains real people's photos (a LinkedIn profile,
photos of babies/children), so once fetched on the new laptop, remove them from git
going forward:
```bash
git rm -r --cached tests/gold_set/images && git commit -m "chore: drop gold images from git after transfer"
```
The files stay on disk locally (the `.gitignore` rule prevents them being
re-committed by accident). Note: they remain in the private repo's *history*
even after that — acceptable per the owner's decision for this private repo.

Alternatively (no git): carry `gold_set_images.tar.gz` out-of-band and
`tar -xzf gold_set_images.tar.gz -C backend-python/tests/gold_set/`, or point at an
existing folder with `GOLD_SET_DIR=/path python scripts/eval_gold_set.py`.

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
