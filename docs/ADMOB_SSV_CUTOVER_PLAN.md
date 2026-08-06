# AdMob Rewarded SSV — Cutover Plan

> Written 2026-08-07. Verified against the tree at `feat_admob_ssv_cutover` (off `main` @ `faa42f8`)
> and against the Android client at `FauxLens-mobile@feat_kmp_p5_remaining`.
>
> **This is a plan, not an implementation.** Nothing in `app/` is changed by this branch.

---

## 0. Executive summary

The common framing of this work — *"build the SSV endpoint"* — is **wrong**. SSV was implemented
in `0a24555` (+ fixes `d3598a8`, `ab51c4f`) and is complete, hardened and tested.

The real situation is three things:

| | Finding | Severity |
|---|---|---|
| **A** | `custom_data` contract mismatch between client and backend — **SSV would grant nothing if enabled today** | 🔴 **Blocker** |
| **B** | `POST /api/ads/reward` grants on the client's word — **60 credits/day mintable with no ad**, and it has zero test coverage | 🔴 **Live abuse hole** |
| **C** | Enabling SSV while the client still calls `/api/ads/reward` **double-grants every ad view** | 🟠 Sequencing trap |

Do **A** first. **B** and **C** are the same cutover and must be ordered carefully, because doing
them in the wrong order either double-grants or strands users with no credits at all.

---

## 1. What already exists — verified, do not rebuild

| Component | Path | State |
|---|---|---|
| ECDSA verifier | `app/services/admob_ssv.py` (153 lines) | ✅ Complete |
| Endpoint | `GET /api/ads/ssv` — `app/api/credits.py:255-375` | ✅ Complete |
| Tests | `tests/test_ads_ssv.py` | ✅ 9 tests |
| Shared daily cap | `_apply_ad_reward` — `app/api/credits.py:125-196` | ✅ Complete |
| GDPR deletion | `app/api/account.py:77-78` sweeps both ad collections | ✅ Complete |
| Dependencies | `cryptography`, `httpx` already in `requirements.txt` | ✅ None needed |

The verifier is genuinely well-built. Two details worth preserving in any refactor:

**Query-param smuggling defence** (`admob_ssv.py:104-118`). Acted-upon fields are read from
`verified_params()`, never `request.query_params` — Starlette's `QueryParams` is last-wins on
duplicate keys, so a replayed callback with `&transaction_id=forged` appended *after* the
signature keeps a valid signature while smuggling a forged value.

**The `www.` host is mandatory** (`admob_ssv.py:29-32`). Bare `gstatic.com` 301-redirects; with
`follow_redirects` off, httpx returns the redirect body, silently disabling all SSV.

Architecture note for anyone expecting SQL: the datastore is **Firestore** (async native client)
plus **Redis** (Upstash) for idempotency/rate-limiting. There are no ORM models — only collection
names and untyped dicts.

Relevant collections:

| Collection | Purpose |
|---|---|
| `users/{uid}` | signed-in balance (`credits_balance`) |
| `users/{uid}/credit_ledger/{auto}` | append-only audit trail |
| `guest_wallets/{device_id}` | guest balance (`credits`) — **separate path, separate field name** |
| `ad_rewards/{uid}_{YYYY-MM-DD}` | daily cap counter, shared by both endpoints |
| `ad_ssv_rewards/{transaction_id}` | SSV idempotency claim |

---

## 2. 🔴 BLOCKER A — the `custom_data` contract mismatch

### The defect

`app/api/credits.py:285`

```python
uid = signed.get("custom_data")
```

The backend consumes `custom_data` **verbatim as a Firebase UID**. The Android client does not
send a bare UID. `RewardedAdManager.kt:139-141`:

```kotlin
val ssvCustomData = FirebaseAuth.getInstance().currentUser?.uid
    ?.let { "uid:$it" }
    ?: runCatching { "device:${secureStorage.getDeviceId()}" }.getOrNull()
```

So the grant resolves `users/uid:abc123`, which does not exist. `credit_engine.grant_credits`
(`app/services/credit_engine.py:73`) raises `HTTPException(404, "User account not found")`.

### Blast radius

**Every signed-in rewarded ad grants zero credits.** The daily cap slot is consumed *before* the
grant is attempted, then handed back by `_release_cap_slot` — so the counter self-heals, but the
user watched a full ad for nothing and the endpoint 500s.

**Guests are worse:** `device:<id>` would resolve `users/device:<id>` rather than
`guest_wallets/<id>`. The two wallets are deliberately separate code paths with different field
names (`credits_balance` vs `credits`) and no ledger for guests, so this can never work without an
explicit branch.

The backend's own comment at `credits.py:288-291` — *"guest ad views have no Firebase uid to stamp
either"* — shows it was written **before** the client added the guest fallback (mobile plan records
that as fixed 2026-08-03). The backend is simply one revision behind.

### The fix

Parse the prefix and route to the correct wallet. Sketch, to live near `credits.py:285`:

```python
CUSTOM_DATA_RE = re.compile(r"^(uid|device):([A-Za-z0-9\-_.]{1,128})$")

raw = signed.get("custom_data")
transaction_id = signed.get("transaction_id")
if not raw or not transaction_id:
    return {"status": "ignored"}

m = CUSTOM_DATA_RE.match(raw)
if not m:
    # Signed by Google, so not an attack — but not a shape we can act on.
    # Log and ACK; never 4xx a validly-signed callback or AdMob backs off.
    logger.warning("admob_ssv_bad_custom_data", extra={"action": "admob_ssv_bad_custom_data"})
    return {"status": "ignored"}

kind, subject_id = m.group(1), m.group(2)
```

Then branch the grant: `kind == "uid"` → existing `_apply_ad_reward(subject_id, ...)`;
`kind == "device"` → a guest-capped equivalent built on `grant_guest_credits`
(`app/services/credits_service.py:170`).

**Three constraints that are not optional:**

1. **Validate the id before it becomes a Firestore document id.** The regex above mirrors
   `validate_device_id` (`app/core/auth.py:29-36`), which exists specifically to *"prevent
   Firestore key injection and Redis key-prefix abuse."* `custom_data` is attacker-influenced —
   a malicious client can stamp any string it likes and Google will faithfully sign it. **The
   signature proves Google sent it, not that the content is honest.**
2. **Accept a bare UID during transition.** Older installs stamp no prefix. Treat an unprefixed
   value as `uid:` for one release, then drop the fallback.
3. **Guests need their own cap document.** `ad_rewards/{uid}_{date}` is keyed on uid;
   `device:` subjects need `ad_rewards/device_{id}_{date}` or an equivalent, or one guest's views
   will collide with a signed-in user whose uid happens to match.

### Decision required from the owner

Do guests earn ad credits at all? The client shows the ad tile **most prominently to guests**
(`InsufficientCreditsSheet.kt:281-291`), yet `/api/ads/reward` is Bearer-only — so a guest today
watches a full ad and then gets a 401. Either:

- **(a) Support guests in SSV** — implement the `device:` branch. Fixes a real UX hole, and the
  anti-farm exposure is bounded by the existing per-IP device limits.
- **(b) Don't** — then stop showing guests the ad tile, and have the client stop stamping
  `device:` custom_data.

**(a) is recommended**, since the client is already built for it. But it is a product call.

---

## 3. 🔴 BLOCKER B — `/api/ads/reward` is a credit faucet

`app/api/credits.py:199-252`. Bearer-only, but grants purely on the client's assertion. Its own
docstring admits it (`credits.py:210-212`):

> *"NOTE: this endpoint trusts the client's word that an ad was watched. The secure path is
> /api/ads/ssv (Google-signed). Once SSV is enabled in the AdMob console and the client stops
> calling this endpoint, it can be removed."*

**Exposure:** any holder of a valid Firebase ID token can `curl` this 3× and mint **60 credits/day**
— 6 scans or 3 object removals of real GPU cost — with no ad impression and therefore no ad
revenue. There is no rate limit on the route (unlike `/api/user/balance` and
`/api/billing/google/verify`, which both call `check_rate_limit`); the daily cap is the only brake.

**It has zero test coverage.** No test anywhere references `ads/reward`, and `_apply_ad_reward`'s
cap semantics are only exercised indirectly — `test_ads_ssv.py` mocks it out entirely.

---

## 4. 🟠 TRAP C — the double-grant window

Both endpoints call the same `_apply_ad_reward`, so they share one cap document. With SSV enabled
**and** the current client shipped, one ad view produces:

```
ad watched
  ├─ Google SSV callback  → _apply_ad_reward → cap 0→1, +20 credits
  └─ client POST /reward  → _apply_ad_reward → cap 1→2, +20 credits
                                              = 40 credits for one impression
```

The shared cap bounds total damage to 60 credits/day, so this is **wrong, not catastrophic**. But
the user is capped after 1.5 ads instead of 3, and you pay one impression's revenue for two rewards.

### Why ordering is not obvious

| Order | Result |
|---|---|
| Neuter `/reward` **first**, enable SSV later | ❌ Nobody gets ad credits in the gap |
| Enable SSV **first**, neuter `/reward` later | ❌ Double-grant for the whole gap |
| Client-first (ship app that stops POSTing) | ❌ Worst — gated on Play review, gap could be days |

### Recommended: backend-only atomic cutover

Deploy **A + the neutering of `/reward` in a single release**, immediately after confirming SSV
works. `/api/ads/reward` keeps its route and response shape but stops granting:

```python
# SSV (Google-signed) is now the only grant path. Kept as a 200-returning
# no-op so shipped clients don't error; delete once the client stops calling it.
balance = await get_user_balance(uid)
return AdRewardResponse(credits_granted=0, new_balance=balance, rewards_today=<count>)
```

No client release needed, no gap, no double-grant.

**Known cosmetic regression, accept it deliberately:** the client branches on
`adState.creditsGranted > 0` (`InsufficientCreditsSheet.kt:412-421`), so a `0` renders
**"Daily ad limit reached"** even on success. Credits still arrive via SSV and the balance updates
reactively — the *message* is wrong, the *money* is right. Fixed in the client in Phase 3.

There is also a **race**: SSV is server-to-server and may land after the client's POST returns, so
`new_balance` can be momentarily stale. Phase 3's poll fixes this too.

---

## 5. Phased rollout

### Phase 0 — Prove SSV works before trusting it *(no code)*

SSV has **never run against real AdMob traffic.** Everything below assumes it works; verify first.

1. AdMob → rewarded unit → Server-side verification → callback URL:
   `https://web-production-6a994.up.railway.app/api/ads/ssv`
2. AdMob's "verify callback URL" ping is validly signed but carries **no `custom_data`** — the
   handler already ACKs it with `{"status": "ignored"}` and HTTP 200. **A 200 here proves only
   that the signature path works, not that granting does.**
3. Watch a real rewarded ad on an internal-testing build **with a signed-in account**.
4. Expect **failure** until Blocker A is fixed — a 500 and `ad_reward_cap_release_failed` or a 404
   from `grant_credits`. That failure *is* the confirmation that A is real.

### Phase 1 — Fix Blocker A + harden

- Prefix parsing + validation (§2), incl. the bare-uid transition fallback
- Guest `device:` branch, **if** the owner chooses option (a)
- Tests: valid uid / valid device / unprefixed legacy / malformed / injection-shaped
- **Gate:** a real ad on a real device credits a real account.

### Phase 2 — Atomic cutover (§4)

- `/api/ads/reward` becomes a non-granting 200
- Add `check_rate_limit` to `/api/ads/ssv` (it currently has none)
- **Gate:** watch 3 ads, confirm exactly 60 credits and cap at 3 — not 2 ads and 40.

### Phase 3 — Client cleanup *(mobile repo, next release)*

- Delete `AdsRepository.claimReward()` (`AdsRepository.kt:31-57`) and
  `FauxLensApi.claimAdReward` (`FauxLensApi.kt:73-76`)
- Replace `CreditsSheetViewModel.claimAdReward()` (`InsufficientCreditsSheet.kt:186-196`) with a
  short balance poll — SSV lands within seconds but not synchronously
- Re-source the "Daily ad limit reached" branch from server cap state, not `creditsGranted`
- Remove the now-dead `adsRepository` constructor param + Koin binding

### Phase 4 — Delete `/api/ads/reward`

Only once telemetry shows no shipped client still calls it.

---

## 6. Hardening backlog (not blockers)

| # | Item | Why |
|---|---|---|
| H1 | **No `ad_unit` / `ad_network` validation** | The signature proves *Google* sent it, not *which app*. A signed callback for another app in the same AdMob account is replayable against this endpoint. Pin `ad_unit` to a new `admob_rewarded_unit_id` setting. |
| H2 | **No freshness window** | `timestamp` is signed but unchecked. The Firestore claim makes replay idempotent *forever*, so this is defence-in-depth, not a hole — but an unbounded-age callback should not grant. |
| H3 | **No rate limit on `/api/ads/ssv`** | Signature verification is ECDSA + a possible key fetch. Unauthenticated and uncapped = a cheap CPU-burn vector. |
| H4 | **`reward_amount` / `reward_item` never cross-checked** | Server grants its own constant, which is correct, but a mismatch is a signal worth logging. |
| H5 | **`AD_REWARD_CREDITS` / `AD_REWARD_DAILY_LIMIT` are module constants** (`credits.py:29-30`) | Every other economic value lives in `Settings` and is env-tunable. These cannot be changed without a redeploy. |
| H6 | **`AdSsvResponse` docstring is stale** | Declares `ok \| duplicate \| capped`; the handler also returns `"ignored"` (`credits.py:297`). Passes validation only because the field is a plain `str`. |
| H7 | **`get_or_create_user` hard-codes `starting_balance = 40`** (`credit_engine.py:191`) | Duplicates `settings.welcome_credits`. Two sources of truth for the same number. |

---

## 7. Test gaps to close

1. `POST /api/ads/reward` — **no test exists at all**
2. `_apply_ad_reward` cap semantics — 3 grants succeed, 4th returns 429, counter is per-UTC-day
3. `_release_cap_slot` — a failing grant must hand the slot back
4. The shared cap — an SSV grant and a `/reward` grant must consume the *same* counter
5. Prefix parsing (Phase 1) — every branch in §2
6. Guest SSV grant path, if option (a)

Conventions to follow: `pytest` with `asyncio_mode = auto`; in-memory `MockFirestore` / `MockRedis`
from `tests/mocks/`; **patch at the import site** (`app.api.credits.verify_ssv`, not
`app.services.admob_ssv.verify_ssv`); auth via `app.dependency_overrides`, never real tokens.

---

## 8. Rollback

Every phase is independently revertible.

- **Phase 2** is the risky one: if SSV grants stop working, revert the `/reward` neutering and ads
  keep paying out on client trust — insecure but not broken. Keep the diff small enough to revert
  cleanly.
- **Phase 1** is additive parsing; reverting restores the current (non-functional) behaviour.
- Disabling SSV in the AdMob console is instant and needs no deploy — the fastest kill switch.

**Do not** revert by deleting the `ad_ssv_rewards` collection: that is the replay-protection
ledger, and clearing it re-opens every historical `transaction_id` for a replay grant.

---

## 9. Owner decisions needed

1. **Do guests earn ad credits?** (§2) — blocks Phase 1 scope.
2. **Accept the cosmetic "Daily ad limit reached" regression** during Phase 2→3? (§4) — the
   alternative is holding the security fix until a client release ships.
3. **Confirm `AD_REWARD_CREDITS = 20` and `AD_REWARD_DAILY_LIMIT = 3`** are the intended economics.
   The client hardcodes `adRewardCredits = 20` at **four** display sites, so these can drift
   silently from what the server grants.
