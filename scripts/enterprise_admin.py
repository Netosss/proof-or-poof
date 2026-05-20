"""
Enterprise partner administration CLI.

Usage (run with `python -m scripts.enterprise_admin <command> [...]` from the
backend-python directory, with .env loaded):

    list-partners
    create-partner --name "Acme Corp" --email "ops@acme.com" [--credits 0] [--rpm 60]
    set-status     --partner-id <uuid> --status active|suspended|frozen
    create-key     --partner-id <uuid> [--allowed-ip 1.2.3.0/24] [--expires-days 365]
    revoke-key     --partner-id <uuid> --credential-id <uuid>
    balance        --partner-id <uuid>
    mint-checkout  --partner-id <uuid> --variant-id <ls_variant> [--store-id <ls_store>]

Secrets are printed to stdout EXACTLY ONCE — copy them to your password manager
immediately. They cannot be retrieved later (only re-issued via `revoke-key` +
`create-key`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Make `app.*` imports work when invoked as `python -m scripts.enterprise_admin`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env BEFORE importing any app modules so settings pick up the values.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


async def _bootstrap():
    from app.integrations import firebase as firebase_module
    firebase_module.initialize()


async def cmd_list_partners(args):
    from app.services.enterprise_partners import list_partners
    partners = await list_partners(limit=args.limit)
    for p in partners:
        print(json.dumps(p, default=str))
    if not partners:
        print("(no partners)")


async def cmd_create_partner(args):
    from app.services.enterprise_partners import create_partner
    p = await create_partner(
        company_name=args.name,
        contact_email=args.email,
        initial_credits=args.credits,
        rate_limit_per_min=args.rpm,
    )
    print("Created partner:")
    print(json.dumps(p, indent=2, default=str))
    print()
    print(f"PARTNER_ID = {p['id']}")
    print("Next step: scripts/enterprise_admin.py create-key --partner-id "
          f"{p['id']}")


async def cmd_set_status(args):
    from app.services.enterprise_partners import set_partner_status
    await set_partner_status(args.partner_id, args.status)
    print(f"OK: partner {args.partner_id} status -> {args.status}")


async def cmd_create_key(args):
    from app.services.api_credentials import create_credential
    cred = await create_credential(
        args.partner_id,
        allowed_ips=args.allowed_ip,
        expires_in_days=args.expires_days,
    )
    print("=" * 70)
    print("STORE THESE IMMEDIATELY — THEY ARE NOT RECOVERABLE")
    print("=" * 70)
    print(f"  api_key:    {cred['api_key']}")
    print(f"  secret_key: {cred['secret_key']}")
    print()
    print(f"  credential_id: {cred['credential_id']}")
    print(f"  api_key_prefix: {cred['api_key_prefix']}")
    print("=" * 70)


async def cmd_revoke_key(args):
    from app.services.api_credentials import revoke_credential
    await revoke_credential(args.partner_id, args.credential_id)
    print(f"OK: credential {args.credential_id} revoked")


async def cmd_balance(args):
    from app.services.enterprise_credit_engine import get_partner_balance
    balance = await get_partner_balance(args.partner_id)
    print(f"partner_id={args.partner_id}  credit_balance={balance}")


async def cmd_grant_credits(args):
    """Manually grant credits to a partner (smoke tests, gifts, comp credits)."""
    from app.services.enterprise_credit_engine import grant_credit
    reason = args.reason or "manual_grant"
    reference = args.reference or f"cli:{int(__import__('time').time())}"
    new_balance = await grant_credit(
        args.partner_id, amount=args.amount, reason=reason, reference_id=reference,
    )
    print(f"OK: granted {args.amount} credits to {args.partner_id}")
    print(f"    new_balance={new_balance}  reason={reason}  reference_id={reference}")


async def cmd_list_applications(args):
    """List pending enterprise applications awaiting operator review."""
    from app.services.enterprise_applications import list_pending_applications
    apps = await list_pending_applications(limit=args.limit)
    if not apps:
        print("(no pending applications)")
        return
    for a in apps:
        flag = "  [FREE_EMAIL]" if a.get("free_email") else ""
        print(f"id={a['id']}  tier={a['tier']:8}  vol={a['expected_volume']:8}  "
              f"{a['company_name']:30}  {a['contact_email']}{flag}")
        if a.get("notes"):
            print(f"    notes: {a['notes'][:120]}")


async def cmd_approve_application(args):
    """Approve a sandbox application: provision partner + grant credits + link records."""
    from app.services.api_credentials import create_credential
    from app.services.enterprise_applications import (
        get_application,
        mark_application_approved,
    )
    from app.services.enterprise_credit_engine import grant_credit
    from app.services.enterprise_partners import create_partner

    app_doc = await get_application(args.id)
    if not app_doc:
        print(f"ERROR: application {args.id} not found")
        return
    if app_doc.get("status") != "pending":
        print(f"ERROR: application status is {app_doc.get('status')!r}, not pending")
        return
    if app_doc.get("tier") != "sandbox":
        print(f"NOTE: this is a {app_doc.get('tier')!r} application. Sandbox flow grants credits;")
        print("      for paid tiers prefer Lemon Squeezy checkout. Continue anyway? [y/N]")
        if input().strip().lower() != "y":
            print("Aborted.")
            return

    partner = await create_partner(
        company_name=app_doc["company_name"],
        contact_email=app_doc["contact_email"],
        initial_credits=0,  # granted via ledger entry below
        firebase_uid=app_doc.get("firebase_uid"),
    )
    partner_id = partner["id"]

    credits = args.credits
    await grant_credit(
        partner_id, amount=credits, reason="sandbox_grant",
        reference_id=f"application:{app_doc['id']}",
    )

    await mark_application_approved(app_doc["id"], partner_id)

    print("=" * 70)
    print("Sandbox provisioned")
    print("=" * 70)
    print(f"  application_id: {app_doc['id']}")
    print(f"  partner_id:     {partner_id}")
    print(f"  company:        {app_doc['company_name']}")
    print(f"  contact_email:  {app_doc['contact_email']}")
    print(f"  credits:        {credits}")
    print()
    print("Issuing first API key now:")

    cred = await create_credential(partner_id)
    print("-" * 70)
    print(f"  api_key:    {cred['api_key']}")
    print(f"  secret_key: {cred['secret_key']}")
    print("-" * 70)
    print()
    print("Action items:")
    print(f"  1. Email partner with sign-in instructions (Firebase UID linked: {app_doc.get('firebase_uid')})")
    print( "  2. They sign in at https://fauxlens.com/enterprise/dashboard")
    print(f"  3. Or send them the api_key + secret_key above directly")


async def cmd_mint_checkout(args):
    """Print the JSON body to POST to Lemon Squeezy's `checkouts` API.

    This intentionally does NOT call Lemon Squeezy directly — we keep the LS
    API key out of this script. Paste the JSON into a curl call against:
        POST https://api.lemonsqueezy.com/v1/checkouts
    """
    store_id = args.store_id or os.getenv("LEMONSQUEEZY_STORE_ID", "<STORE_ID>")
    env_tag = "dev" if os.getenv("APP_ENV", "prod") == "dev" else "prod"
    body = {
        "data": {
            "type": "checkouts",
            "attributes": {
                "checkout_data": {
                    "custom": {
                        "account_type": "enterprise",
                        "partner_id": args.partner_id,
                        "env": env_tag,
                    }
                }
            },
            "relationships": {
                "store": {"data": {"type": "stores", "id": str(store_id)}},
                "variant": {"data": {"type": "variants", "id": str(args.variant_id)}},
            },
        }
    }
    print(json.dumps(body, indent=2))
    print()
    print("Send via:")
    print("  curl https://api.lemonsqueezy.com/v1/checkouts \\")
    print("    -H 'Accept: application/vnd.api+json' \\")
    print("    -H 'Content-Type: application/vnd.api+json' \\")
    print("    -H \"Authorization: Bearer $LEMONSQUEEZY_API_KEY\" \\")
    print("    -d @-  # paste the JSON above")


def _add_common(p, parser_fn):
    parser_fn(p)


def main():
    parser = argparse.ArgumentParser(description="Enterprise partner administration")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list-partners")
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(func=cmd_list_partners)

    p = sub.add_parser("create-partner")
    p.add_argument("--name", required=True)
    p.add_argument("--email", required=True)
    p.add_argument("--credits", type=int, default=0)
    p.add_argument("--rpm", type=int, default=None,
                   help="Per-minute rate limit override (defaults to global)")
    p.set_defaults(func=cmd_create_partner)

    p = sub.add_parser("set-status")
    p.add_argument("--partner-id", required=True)
    p.add_argument("--status", required=True, choices=("active", "suspended", "frozen"))
    p.set_defaults(func=cmd_set_status)

    p = sub.add_parser("create-key")
    p.add_argument("--partner-id", required=True)
    p.add_argument("--allowed-ip", action="append", default=[],
                   help="CIDR allowlist (repeatable). Omit for unrestricted.")
    p.add_argument("--expires-days", type=int, default=None)
    p.set_defaults(func=cmd_create_key)

    p = sub.add_parser("revoke-key")
    p.add_argument("--partner-id", required=True)
    p.add_argument("--credential-id", required=True)
    p.set_defaults(func=cmd_revoke_key)

    p = sub.add_parser("balance")
    p.add_argument("--partner-id", required=True)
    p.set_defaults(func=cmd_balance)

    p = sub.add_parser("grant-credits")
    p.add_argument("--partner-id", required=True)
    p.add_argument("--amount", type=int, required=True)
    p.add_argument("--reason", default=None,
                   help="Ledger reason string (default: manual_grant)")
    p.add_argument("--reference", default=None,
                   help="Idempotency reference id (default: cli:<timestamp>)")
    p.set_defaults(func=cmd_grant_credits)

    p = sub.add_parser("list-applications")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_list_applications)

    p = sub.add_parser("approve-application")
    p.add_argument("--id", required=True, help="enterprise_applications document id")
    p.add_argument("--credits", type=int, default=100,
                   help="Sandbox credits to grant (default: 100)")
    p.set_defaults(func=cmd_approve_application)

    p = sub.add_parser("mint-checkout")
    p.add_argument("--partner-id", required=True)
    p.add_argument("--variant-id", required=True)
    p.add_argument("--store-id", default=None)
    p.set_defaults(func=cmd_mint_checkout)

    args = parser.parse_args()

    async def runner():
        await _bootstrap()
        await args.func(args)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
