---
name: git-commit-blocked-autonomous
description: Direct git commit/push is blocked in autonomous mode on the fauxlens repos; commits need interactive user approval
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 776c872e-67d3-4577-bf72-8bc519b9dd93
---

On the fauxlens repos, autonomous `git commit`/`git push` is blocked by TWO
layers: the Fiverr `block-direct-git` PreToolUse hook (marketplace plugin) AND
the auto-mode permission classifier. The hook allows a `FIVERR_COMMIT_ACTIVE=1`
prefix, but the auto-mode classifier explicitly DENIES using that flag to tunnel
the guard — so it is not a valid autonomous workaround.

**Why:** `mobile-android` (Netosss/FauxLens-mobile) and `backend-python`
(Netosss/proof-or-poof) are personal repos; the user wants plain-git commits, NOT
the Alan/fiverr-commit workflow, and NOT the work email.

**How to apply:** When working autonomously (user asleep / unattended), do NOT
try to commit — implement + build/test-verify, then save ready-to-apply patches
(`git diff` / `git diff --cached` are allowed) and leave changes staged for the
user to commit interactively. Personal git identity is
`Netosss` / `92423203+Netosss@users.noreply.github.com` (set as local repo
config; global config has the work email — never commit with it). Related:
[[fauxlens-kmp-migration]].
