# Credit System Repricing — Deployment Checklist

All backend code and test changes are complete. The items below are **external actions** that must be performed manually before/during deployment.

---

## 1. Railway Environment Variables

Update the Lemon Squeezy variant-to-credits mapping on Railway to reflect the new credit amounts.

### Production

```
LEMON_SQUEEZY_VARIANTS={"<starter_variant_id>": 500, "<pro_variant_id>": 2000, "<max_variant_id>": 5000}
```

### Dev / Staging (test-mode variants)

```
LEMON_SQUEEZY_TEST_VARIANTS={"<test_starter_id>": 500, "<test_pro_id>": 2000, "<test_max_id>": 5000}
```

Replace the `<..._id>` placeholders with your actual Lemon Squeezy variant IDs. The variant IDs themselves do not change — only the credit amounts in the JSON values.

### Optional overrides (defaults are already set in code)

These can be set if you ever need to hot-fix pricing without a deploy:

- `DETECT_CREDIT_COST=10`
- `INPAINT_CREDIT_COST=20`
- `WELCOME_CREDITS=40`

---

## 2. Lemon Squeezy Dashboard

Update variant names/descriptions to match the new credit amounts. **Prices stay the same.**

| Variant | Old Description | New Description | Price |
| ------- | --------------- | --------------- | ----- |
| Starter | 100 credits     | **500 credits** | $2.99 |
| Pro     | 400 credits     | **2,000 credits** | $9.99 |
| Max     | 1,000 credits   | **5,000 credits** | $19.99 |

**Recommended approach:** Keep the same variant IDs and update descriptions only. Credit delivery is controlled server-side by `LEMON_SQUEEZY_VARIANTS`, not by Lemon Squeezy.

After updating, run a test purchase in dev/staging to verify the full flow: checkout → webhook → credits granted.

---

## 3. Frontend App (External Repo)

The following changes must be made in the mobile/web app codebase:

### Credit Pack Display

Update the UI text shown on the purchase screen:

| Pack            | Old             | New               |
| --------------- | --------------- | ----------------- |
| Starter ($2.99) | "100 credits"   | **"500 credits"** |
| Pro ($9.99)     | "400 credits"   | **"2,000 credits"** |
| Max ($19.99)    | "1,000 credits" | **"5,000 credits"** |

### Ad Reward Button

- Old: `"WATCH AD (+5 CREDITS)"`
- New: **`"WATCH AD (+20 CREDITS)"`**

### Inpainting Cost Display

If the frontend shows cost before confirming inpainting:

- Old: `"30 credits"`
- New: **`"20 credits"`**

### No Balance Logic Changes

The frontend reads balance from the API (`/api/user/balance` or `X-User-Balance` response header). No client-side balance math needs to change.

---

## Summary of Code Changes (Already Applied)

| File | Change |
| ---- | ------ |
| `app/config.py` | `inpaint_credit_cost`: 30 → 20, `default_recharge_amount`: 5 → 20 |
| `app/api/credits.py` | `AD_REWARD_CREDITS`: 5 → 20, docstring updated |
| `app/services/credits_service.py` | `deduct_guest_credits` default `cost`: 5 → 10 |
| `tests/test_credits_route.py` | Recharge `amount`: 5 → 20 |
| `tests/test_credits_service.py` | Deduct `cost`: 5 → 10, recharge `amount`: 5 → 20, expected balances adjusted |
