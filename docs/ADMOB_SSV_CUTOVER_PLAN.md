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
| **0** | `/api/ads/ssv` accepts validly-signed callbacks from **any AdMob publisher on earth** — an unauthenticated credit mint | 🔴🔴 **CRITICAL** |
| **A** | `custom_data` contract mismatch between client and backend — **SSV would grant nothing if enabled today** | 🔴 **Blocker** |
| **B** | `POST /api/ads/reward` grants on the client's word — **60 credits/day mintable with no ad**, and it has zero test coverage | 🔴 **Live abuse hole** |
| **C** | Enabling SSV while the client still calls `/api/ads/reward` **double-grants every ad view** | 🟠 Sequencing trap |
| **D** | Guest welcome bonus is re-minted on every unseen device id — **already live on the billing path** | 🔴 **Live** |
| **E** | The per-IP new-device limit is **dead code and can never fire** | 🔴 **Live** |

**Do 0 first, and do not enable SSV in the AdMob console until it is done.** A and B/C follow.
B and C are the same cutover and must be ordered carefully — the wrong order either
double-grants or strands users with no credits at all.

---

## 0. 🔴🔴 CRITICAL — SSV trusts every AdMob publisher on earth

**This was originally filed below as hardening item H1, "not a blocker". That was wrong, and
the reasoning behind it was wrong.** Recorded here in full so the mistake is not repeated.

`VERIFIER_KEYS_URL` (`app/services/admob_ssv.py:32`) is
`https://www.gstatic.com/admob/reward/verifier-keys.json` — a **single global key set shared by
every AdMob publisher**. There is no per-account or per-app key. So a valid signature proves
only *"some AdMob server sent this"*, never *"your app sent this"*.

The one field that binds a callback to your inventory is `ad_unit`, and `ads_ssv()` never reads
it — it reads `custom_data` and `transaction_id` only (`credits.py:285-286`).

**Exploit — no modified client, no reverse engineering, ~15 minutes:**

1. Attacker creates their own AdMob account and a rewarded ad unit.
2. In *their* console, they point the SSV callback URL at `https://<your-host>/api/ads/ssv`.
3. Their throwaway app stamps `custom_data` with any wallet they choose.
4. They watch an ad on their own device. Google signs it with the global key and calls your
   endpoint. `verify_ssv` returns `True`. You grant 20 credits.

Their cost per grant is one ad impression **on their own account — which Google pays them for**.
This is not merely free credits; it is an arbitrage funded by your GPU spend. Daily caps do not
contain it, because the cap key is derived from the attacker-chosen `custom_data` (see A/H2).

**Fix — must land before SSV is switched on:**

```python
if signed.get("ad_unit") != settings.admob_rewarded_ad_unit_id:
    logger.warning("admob_ssv_foreign_ad_unit", extra={"ad_unit": signed.get("ad_unit")})
    return {"status": "ignored"}   # 200, so AdMob's own callback-verify ping still passes
```

Add `admob_rewarded_ad_unit_id` to `Settings` (the production unit is
`ca-app-pub-2844061727637796/6754427834`). Pair it with the `timestamp` freshness window from
H2 — `ad_unit` pinning plus freshness is what actually binds the endpoint to your app.

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

### ✅ DECIDED 2026-08-07 — guests DO earn ad credits, with a daily cap

Option **(a)**: implement the `device:` branch. This closes a real UX hole — the client shows
the ad tile **most prominently to guests** (`InsufficientCreditsSheet.kt:281-291`), yet
`/api/ads/reward` is Bearer-only, so a guest today watches a full ad and then gets a 401.

**Guest branch requirements — all mandatory:**

1. **Validate before use as a document id.** Reuse `validate_device_id`
   (`app/core/auth.py:29-36`, `[a-zA-Z0-9\-_.]{1,128}`) on the parsed subject. It exists
   specifically to *"prevent Firestore key injection and Redis key-prefix abuse."*
2. **Separate cap namespace.** `ad_rewards/{uid}_{date}` is keyed on uid alone. A guest subject
   must key `ad_rewards/device_{device_id}_{date}` (or equivalent) so a `device:` subject can
   never collide with a signed-in uid — including the case where a malicious client stamps
   `device:<someone-elses-firebase-uid>`.
3. **Grant through `grant_guest_credits`** (`app/services/credits_service.py:170`), not
   `grant_credits` — guest wallets are a separate collection with a different balance field
   (`credits` vs `credits_balance`) and no ledger.
4. **Consider a lower cap for guests.** Signed-in users are rate-limited by account creation;
   a guest is limited only by device-id rotation. 3/day per device is defensible, but the
   economics are worth an explicit decision rather than inheriting the signed-in number.

**The threat to keep in mind throughout:** `custom_data` is attacker-influenced. A modified
client can stamp any string it likes and Google will faithfully sign it. **The signature proves
Google sent the callback, not that the subject is honest.** Every guest grant must therefore be
treated as an unauthenticated write path with a validated key, not as trusted input.

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
- **Guest `device:` branch — confirmed in scope 2026-08-07.** All four requirements in §2:
  `validate_device_id` on the parsed subject, a separate `device_` cap namespace,
  `grant_guest_credits` rather than `grant_credits`, and an explicit guest cap value.
- Tests: valid uid / valid device / unprefixed legacy / malformed / injection-shaped /
  `device:<a-real-firebase-uid>` (must NOT touch that user's wallet or cap)
- **Gate:** a real ad on a real device credits a real account, **and** a real guest install.

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

## 5b. Live abuse holes found 2026-08-07 — independent of SSV

These exist **today**, on `main`, whether or not SSV is ever enabled. Two of them are larger
than anything the guest SSV branch would introduce, so fix them before tuning guest ad caps.

### D. 🔴 The guest welcome bonus is re-minted on every unseen device id

`app/services/credits_service.py:126-132` — `_apply_guest_topup` creates a missing wallet with
`settings.welcome_credits + amount`, not `amount`:

```python
transaction.set(ref, {"credits": settings.welcome_credits + amount, ...})
```

`grant_guest_credits` routes through it, so the *first* grant to any never-seen device id pays
**60 credits, not 20**. This is **already live on the guest billing path**
(`billing.py:166`) — a guest purchase from a fresh device id grants `credits + 40` — and the
guest SSV branch would inherit it.

**Fix:** split creation from top-up. `_apply_guest_topup` creates with `{"credits": amount}`;
the welcome bonus is issued exactly once, by `get_guest_wallet`. If the organic path needs it,
add an explicit `include_welcome: bool = False` and pass `True` only from there.

### E. 🔴 The per-IP new-device limit is dead code

`app/core/auth.py:184`, called from `detection.py:162` and `inpainting.py:111`:

```python
if current_count >= limit:
    if not token_already_verified:     # ← both call sites pass True
        ...raise 403 / verify turnstile...
write_pipe.sadd(ip_key, device_id)     # always runs
```

**Both and only call sites pass `token_already_verified=True`**, so the enforcement branch is
unreachable. The function degrades to "record this device id in a Redis set", and it fails open
when Redis is down (`auth.py:164-165`).

Combined with the fact that `GET /api/user/balance` mints a wallet, a fresh `X-Device-ID` on a
plain GET yields 40 credits, throttled only per-IP.

> **Correction, 2026-08-07.** The original text continued "*— roughly 400 free credits per
> minute per IP, or 40 free scans*". The credits part is right; **the scans part is not.**
> Minting is free, but *spending* is not: the guest branch of `/detect` and `/inpaint` requires
> a valid Turnstile token on every call. `passes_app_check_gate` returns `False` when there is
> no token **and** unconditionally in `monitor` mode (`app_check.py:131-137`), so it cannot
> currently substitute. Each scan therefore costs one Turnstile solve.
>
> That makes this a *bounded* faucet priced at bulk CAPTCHA-solving rates, not a free one. Still
> worth closing — solve services are cheap — but it is not the cheapest path in the system, and
> planning against the wrong number leads to fixing the wrong thing.

**Why this is still open after the 2026-08-07 pass.** Both "fixes" are riskier than they look:

- Flipping to `token_already_verified=False` demands a Turnstile solve once an IP crosses
  `max_new_devices_per_ip` (**3**). That 403s real users behind shared egress — offices, campus
  and carrier CGNAT, which is most of mobile. The gate is redundant *as written* anyway: the
  caller has **already** passed Turnstile or App Check by the time it runs, so one solve
  satisfies it and a farmer paying for solves is unaffected. It would cost legitimate users
  more than attackers.
- Not minting from a read endpoint was already tried and reverted — see the comment at
  `credits.py:117-123`. That gate ran *before* `get_guest_wallet`, so new guests behind a shared
  IP (and anyone reinstalling, since each reinstall is a fresh device id) got 0 credits instead
  of 40.

The real fix is a **volume** limit that one CAPTCHA solve cannot satisfy — e.g. cap distinct new
device ids per IP per 24h and return 429 rather than a solvable challenge, with an allowlist for
known-shared egress. That is a live-traffic behaviour change on the busiest guest path and needs
the owner's call plus a metrics window first. **Do not switch enforcement on blind.**

### ~~F. `is_banned` is written but never read~~ ✅ DONE 2026-08-07

Enforced in `deduct_guest_credits`, which is the single choke point every guest spend goes
through (`/detect` and `/inpaint` both land there). A banned wallet now gets
`403 {"code": "WALLET_SUSPENDED"}`.

**Not** wired as a `check_ban_status()` call at the routes, which is what this entry originally
suggested. That would cost a second Firestore read on every request and would leave a TOCTOU
window — a ban landing between the check and the deduct still gets served. Reading `is_banned`
and `credits` from one snapshot inside the existing transaction has neither problem.
`check_ban_status()` is left in place for callers that want a non-spending read.

Behaviour change today: **none**, and that is checked rather than assumed — nothing in the
codebase writes `is_banned=True` (only `False`, at wallet creation). This ships the lever; it
does not pull it. Wallets predating the field keep spending, since a missing flag reads as not
banned — covered by a test, because failing closed there would have taken out the oldest users.

---

## 6. Hardening backlog (not blockers)

| # | Item | Why |
|---|---|---|
| ~~H1~~ | ~~No `ad_unit` validation~~ | **PROMOTED TO §0 CRITICAL.** The original entry said "another app in the same AdMob account", which understated it by an order of magnitude — the verifier keys are global to *all* AdMob publishers, so it is any account anywhere. See §0. |
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

1. ~~**Do guests earn ad credits?**~~ ✅ **DECIDED 2026-08-07 — yes, with a daily cap.** Phase 1
   scope now includes the `device:` branch. See §2 for the four mandatory requirements.
2. **Accept the cosmetic "Daily ad limit reached" regression** during Phase 2→3? (§4) — the
   alternative is holding the security fix until a client release ships.
3. **Confirm `AD_REWARD_CREDITS = 20` and `AD_REWARD_DAILY_LIMIT = 3`** are the intended economics,
   and whether guests should get a *lower* cap than signed-in users (§2, requirement 4). The
   client hardcodes `adRewardCredits = 20` at **four** display sites, so these can drift
   silently from what the server grants.
