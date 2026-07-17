# Claude project memory (portable copy)

FauxLens **backend** Claude Code memory notes, vendored into the repo so they
survive a laptop switch (the live files live outside git under
`~/.claude/projects/<project-key>/memory/`).

## Files
Detection lessons (hard-won — read before touching the prompt/scoring):
- `MEMORY.md` — index
- `detection-prompt-overfit-lesson.md` — always run the gold set; minimal
  principle-based prompt changes only (a heavy rewrite regressed 92%→64%)
- `detection-validated-levers.md` — what helped/hurt; extended gold set = 31 imgs
- `gold-set-child-grip-blindspot.md` — baseline false-positived babies with cups
  until GripAndChildGuard; gold set gap
- `detection-content-plausibility.md` — two-clause semantic test (judge the offer)
- `gemini-flash-vision-ceiling.md` — flash can't see subtle physical artifacts;
  airplane-class needs gemini-3-pro

Cross-cutting:
- `fauxlens-kmp-migration.md` — Android→KMP + mobile App Check (backend OR-gate,
  PR #44). See also the mobile repo `docs/KMP_PLAYSTORE_PLAN.md` §14.
- `git-commit-blocked-autonomous.md` — personal-repo commit conventions.

## Gold set (detection eval)
The gold eval (`scripts/eval_gold_set.py`) needs the 31 labeled images. Labels are
committed at `tests/gold_set/labels.json`; the images themselves are NOT in git (see
`tests/gold_set/README.md` for how to restore them on a new machine).

## Restore on a new machine
Copy these files into `~/.claude/projects/<your-project-key>/memory/` (key derived
from the home path, e.g. `-Users-<username>-Desktop-fauxlens-backend-python`), keeping
`MEMORY.md` as the index. If the path differs they still work as reference docs.
Not committed: global `~/.claude` config, settings, credentials, or other projects.
