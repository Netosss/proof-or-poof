# Enterprise Operator Runbook

> **Audience:** the human operator (founder today, on-call engineer tomorrow) who handles enterprise applications, approvals, and partner support.
> **Partner-facing docs** live in [`ENTERPRISE_API.md`](./ENTERPRISE_API.md). Don't link partners here — this is internal-only.

The enterprise surface has a **fully self-serve partner experience** (apply → dashboard → checkout → integrate) plus **one human-in-the-loop step**: approving or rejecting sandbox applications. This document covers everything you need to do that step, plus all common admin operations.

---

## 1. One-time setup

The CLI talks directly to Firestore. You need:

```bash
# Either FIREBASE_SERVICE_ACCOUNT (raw JSON, what Railway uses):
export FIREBASE_SERVICE_ACCOUNT='{"type":"service_account",...}'

# OR GOOGLE_APPLICATION_CREDENTIALS (path to a JSON file):
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/fauxlens/firebase-admin.json
```

Plus the environment toggle:

```bash
export APP_ENV=prod   # use 'dev' to operate against the staging/test surface
```

You can put both in `backend-python/.env` — the CLI auto-loads it via `python-dotenv`.

For prod work, get the service-account JSON from the Firebase Console → Project Settings → Service accounts → Generate new private key. Save it outside the repo (`~/.config/fauxlens/` is fine). Never commit it.

---

## 2. Sandbox application lifecycle

```
                                 ┌───────────────┐
                                 │   pending     │   user submitted /enterprise/apply
                                 │ (in Firestore │   firebase_uid + form data captured
                                 │ application   │
                                 │  record)      │
                                 └───────┬───────┘
                                         │
                            ┌────────────┴────────────┐
                            ▼                         ▼
                  ┌──────────────────┐      ┌──────────────────┐
                  │   provisioned    │      │     rejected     │
                  │  (partner + 50   │      │  (status flipped,│
                  │   credits + key  │      │   no partner     │
                  │   issued by user)│      │   created)       │
                  └────────┬─────────┘      └────────┬─────────┘
                           │                         │
                  partner sees the                partner sees the
                  ready dashboard with             RejectedView panel
                  welcome banner                   (neutral, talk-to-us
                  → clicks "New API key"           CTA + pricing CTA)
                  → integrates
```

**Key invariant:** the partner record only exists in the `provisioned` branch. A pending or rejected application has no partner record yet — only the application document.

**The operator never sees a plaintext signing secret** in this flow. The partner self-serves the API key from their dashboard.

---

## 3. Listing pending applications

```bash
cd backend-python
.venv/bin/python scripts/enterprise_admin.py list-applications
```

Output:

```
id=1ol4YlBFRDcb50SL7BfI  tier=sandbox   vol=2k_10k    ITZIK     ops@acme.com  [FREE_EMAIL]
    notes: Want to evaluate against our trust&safety queue, ~200 images/day initial.
```

What to look at:

| Signal | What it means |
|---|---|
| `[FREE_EMAIL]` flag | Applied from gmail / outlook / etc. instead of a company domain. Not auto-rejected — eyeball it. |
| `tier` | Should be `sandbox` here. If you ever see `starter`/`pro`/`scale`, that's an operator-typed override; treat carefully. |
| `vol` | Their expected volume — pair with use_case to decide if sandbox is a fit. |
| `notes` | Read every word. Best signal of seriousness. |

If you see no pending applications, the CLI prints `(no pending applications)`. The endpoint only returns `status='pending'` — already-approved or rejected applications won't appear.

---

## 4. Approving an application

```bash
.venv/bin/python scripts/enterprise_admin.py approve-application \
  --id <application_id> \
  --credits 50              # optional, defaults to 50
```

What this does (in order):

1. Reads the application from Firestore
2. **Creates** an `enterprise_partners/{partner_id}` doc with the application's company_name, contact_email, firebase_uid
3. **Grants** the credits via an `enterprise_credit_engine.grant_credit` call (writes to the partner's credit_ledger subcollection)
4. **Marks** the application `status='provisioned'` and links it to the partner_id
5. **Logs** `enterprise_application_approved` to Axiom — the umbrella event you'll grep later
6. **Prints** a pre-built `mailto:` URL the partner can be invited with

What this does **NOT** do:

- Issue an API key. The partner clicks "New API key" themselves on `/enterprise/dashboard` — that's by design (operator never sees a plaintext secret).
- Send an email automatically. You paste the mailto link into your browser; Mail.app / Gmail opens with the message pre-filled; you hit send.

### Output

```
======================================================================
Sandbox provisioned
======================================================================
  application_id: 1ol4YlBFRDcb50SL7BfI
  partner_id:     f9a80d26-1e9f-4824-b86e-5200fe295a75
  company:        ITZIK
  contact_email:  ops@acme.com
  credits:        50
  firebase_uid:   ifjIk2G4lyXqF1NemqfxfWUiHFg1

No API credential was issued. The partner self-serves their first
key from the dashboard — same flow paying customers use.

Next step — send the partner their sign-in link:
----------------------------------------------------------------------
mailto:ops@acme.com?subject=Your%20Faux%20Lens%20sandbox%20is%20ready...
----------------------------------------------------------------------

Paste that URL into your browser; your default mail client opens
with the message pre-filled. Hit send.
```

### When to override `--credits`

Default 50 is enough for a real evaluation (≈30-40 scans + signing-mistake retries). Override when:

- The applicant is a serious large-volume prospect (newsroom, T&S platform): `--credits 100`
- They're testing video specifically (1 credit per video, but you might want fewer overall): `--credits 25`
- Internal smoke tests / your own employee: `--credits 500`

There's no upper bound enforced in code. Use judgment — at Gemini cost (~$0.003/scan), even 1000 credits is $3 of your money.

---

## 5. Rejecting an application

```bash
.venv/bin/python scripts/enterprise_admin.py reject-application \
  --id <application_id> \
  --reason "use_case_off_fit: hobbyist requesting commercial sandbox"
```

What this does:

1. Flips the application `status='rejected'` with `rejected_at` timestamp
2. Stores the `--reason` (optional) on the doc for your audit trail
3. Logs `enterprise_application_rejected` to Axiom

**No automatic notification is sent.** The partner only sees the rejection when they next sign in to `/enterprise/dashboard` — the `RejectedView` panel renders with a neutral "we couldn't approve this" message and two CTAs: `Talk to us` (mailto to you) and `See paid plans instead`.

If you want to soften it with a personal email, the application doc has their `contact_email` — write something kind. Most rejections are "off fit", not "you're a bad person".

### Common rejection reasons (free-text, stored for your audit)

- `use_case_off_fit` — they want it for something we don't serve well
- `volume_mismatch` — they need way more than sandbox or way more than we want to give free
- `couldnt_verify` — the company doesn't appear to exist, or the application notes are suspicious

---

## 6. After approval — what the partner sees

When they sign in to `https://fauxlens.com/enterprise/dashboard` with the **same Google account they applied with**:

1. ✅ **Real dashboard renders** (not the PreparingView checklist)
2. ✅ **Cyan welcome banner** at the top: *"Your enterprise account is ready. Issue your first API key to start scanning."* with a `[New API key →]` button
3. ✅ **KPI cards** show: Balance `50`, Status `active`, Rate limit `60/min`
4. ✅ **Empty keys section**: *"No credentials yet. Click New API key above to issue your first pair."*
5. They click **"New API key"** → `EnterpriseKeyNew.tsx` shows both `fxl_test_…` and `fxs_test_…` **exactly once** → they copy + store → start scanning

The welcome banner dismissal is persisted in localStorage keyed by `partner.id`. They see it once.

If they sign in with the **wrong** Google account (different from the one used to apply), they see the "No partner account yet" empty state with links to pricing + apply. The `firebase_uid` linkage is exact — switching accounts means switching identities to us.

---

## 7. Other common operations

### Grant a partner additional credits (comp / refund / make-good)

```bash
.venv/bin/python scripts/enterprise_admin.py grant-credits \
  --partner-id <uuid> \
  --amount 100 \
  --reason "make_good_pipeline_outage_2026_05_21"   # optional
```

Always include a `--reason` — it lands in the ledger and is your only audit trail for "why did this partner get extra credits".

### Bump a partner's rate limit ceiling (manual override)

There's no dedicated CLI for this yet. Do it directly in the Firebase Console: open `enterprise_partners/{partner_id}` → edit `rate_limit_per_min` field. The rate limiter picks up the new value on the next request (no restart needed).

The LS webhook auto-bumps on tier purchases (see `ENTERPRISE_VARIANT_RATE_LIMITS`), so manual edits are usually only needed for special cases (custom enterprise deals, abuse remediation, etc.).

### Suspend / unsuspend a partner

```bash
.venv/bin/python scripts/enterprise_admin.py set-status \
  --partner-id <uuid> \
  --status suspended           # or: active, frozen
```

`suspended` blocks all `/v1/analyze` calls from that partner at the credit-engine layer (they get `403 enterprise_partner_not_active`). Use for abuse cases, payment disputes in flight, etc. Reverse with `--status active`.

### Revoke a single credential

The partner can revoke their own keys from the dashboard. If you need to do it for them (e.g., they emailed you saying "leaked secret, revoke now"):

```bash
.venv/bin/python scripts/enterprise_admin.py revoke-key \
  --partner-id <uuid> \
  --credential-id <uuid>
```

Old key stops working on the next request (no cache lag — the credential lookup hits Firestore each time).

### Check a partner's credit balance

```bash
.venv/bin/python scripts/enterprise_admin.py balance --partner-id <uuid>
```

### List all partners

```bash
.venv/bin/python scripts/enterprise_admin.py list-partners --limit 50
```

### Mint an LS checkout URL for a paid partner (assisted onboarding)

Sometimes a large customer wants you to send them a direct checkout link instead of pointing them at the pricing page. The CLI generates the JSON body to POST to LS:

```bash
.venv/bin/python scripts/enterprise_admin.py mint-checkout \
  --partner-id <uuid> \
  --variant-id <ls_variant_id>
```

The output is a curl command you paste-and-run with your LS API key — the script intentionally never holds the key itself.

---

## 8. Axiom queries (audit trail)

All enterprise events go to the `backend-logs` Axiom dataset (or whatever `AXIOM_DATASET` is set to). Run these in the Axiom query UI:

### Show me every approval in the last 30 days

```
['backend-logs']
| where action == 'enterprise_application_approved'
| project _time, partner_id, application_id, company_name, contact_email, credits_granted, tier
| order by _time desc
```

### Show me every rejection + the reason

```
['backend-logs']
| where action == 'enterprise_application_rejected'
| project _time, application_id, company_name, contact_email, reason
| order by _time desc
```

### Activation funnel — how many approved partners issued a first key

Cross-reference `enterprise_application_approved` (backend) with `enterprise_key_issued` PostHog events (frontend). Approved-but-no-key = ghost partners.

### Top partners by scan volume (last 24h)

```
['backend-logs']
| where action == 'enterprise_scan_completed'
| summarize scans = count(), credits = sum(credits_consumed), avg_ms = avg(duration_ms) by partner_id
| order by scans desc
```

### Suspicious activity — partners with high signature-mismatch rate

```
['backend-logs']
| where action == 'enterprise_signature_mismatch'
| summarize mismatches = count() by partner_id, bin(_time, 1h)
| where mismatches > 10
```

---

## 9. Troubleshooting

### "Application not found" when running approve-application

The `--id` doesn't exist or doesn't match. Run `list-applications` to see current pending IDs. Note: only `status=pending` shows up; if you already approved it, it won't appear.

### "Application status is 'approved', not pending"

You already approved this one. To grant more credits to the resulting partner, use `grant-credits --partner-id <uuid>` instead.

### Partner says "I signed in and don't see anything / it says no partner account"

Most likely: they signed in with a different Google account than the one they used to apply. Confirm the firebase_uid in the application matches the one in their current sign-in. Have them sign out, then sign in with the email shown in the application's `contact_email`.

If the firebase_uid genuinely matches but no partner record exists: re-run `approve-application` (it'll error if already approved) or directly create a partner with `create-partner`.

### Partner says "I lost my signing secret"

By design — secrets are shown once. They issue a new key pair from their dashboard (`/enterprise/dashboard` → New API key), update their integration with the new pair, then revoke the old one from the dashboard. The dashboard's lost-secret recovery copy explains this inline.

### Approve-application succeeded but partner sees the pending state

Cache lag — `enterpriseApi.me()` is fetched on dashboard mount. They need to refresh the dashboard once. If it persists more than a minute, check that the `mark_application_approved` call actually set `status='provisioned'` (Firebase Console → `enterprise_applications/{id}` → status field).

### CLI complains about "Cloud Firestore API has not been used in project X"

Your `GOOGLE_APPLICATION_CREDENTIALS` env var points at a different Firebase project than the one this code targets. Unset it or point it at the correct service account JSON. The CLI prefers `FIREBASE_SERVICE_ACCOUNT` (raw JSON env var) when both are present.

### Webhook arrived but partner credits didn't update

Check Axiom for `lemonsqueezy_enterprise_payload_skipped` events — most commonly `wrong_env_for_prod_server` (test-mode webhook hit prod) or `wrong_env_for_dev_server` (live webhook hit staging). Verify the LS webhook URL points at the right environment.

---

## 10. Pre-prod checklist (one-time)

For the first prod deploy, verify on Railway prod env:

- [ ] `APP_ENV=prod` (or unset, defaults to prod)
- [ ] `FIREBASE_SERVICE_ACCOUNT` set
- [ ] `LEMONSQUEEZY_API_KEY` set to the **live** key
- [ ] `LEMONSQUEEZY_STORE_ID` set to the **live** store ID
- [ ] `LEMONSQUEEZY_WEBHOOK_SECRET` set to the **live** webhook secret
- [ ] `ENTERPRISE_LS_VARIANTS` set to `{"<live_starter>":2000,"<live_pro>":10000,"<live_scale>":25000}`
- [ ] `ENTERPRISE_VARIANT_RATE_LIMITS` set to `{"<live_starter>":60,"<live_pro>":120,"<live_scale>":300}`
- [ ] `AXIOM_TOKEN` set (otherwise logs go to stdout only)
- [ ] `SENTRY_DSN` set
- [ ] Live LS webhook in the LS dashboard points at `https://web-production-6a994.up.railway.app/webhooks/lemonsqueezy`
- [ ] Live products published in LS, variant IDs match what's in `ENTERPRISE_LS_VARIANTS`
- [ ] Frontend `tiers.ts` has the correct live variant IDs in the `LIVE` map
- [ ] Test a real card purchase on Starter ($99.99) end-to-end — refund from LS after success to clean up

---

## Reference

- **CLI source:** `scripts/enterprise_admin.py`
- **Partner-facing docs:** `docs/ENTERPRISE_API.md`
- **Application logic:** `app/services/enterprise_applications.py`
- **Partner / credit engine:** `app/services/enterprise_partners.py`, `app/services/enterprise_credit_engine.py`
- **Webhook handler:** `app/api/enterprise/webhooks_enterprise.py`
- **Dashboard UI:** `frontend-react/frontend/src/pages/enterprise/EnterpriseDashboard.tsx`
- **API code:** `app/api/enterprise/`
