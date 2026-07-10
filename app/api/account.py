"""
Account deletion — Google Play "Delete account" data-safety compliance.

Google requires a way for users to request deletion of their account and
associated data that is reachable WITHOUT the app. We serve both halves here:

  GET  /delete-account   Public, self-contained web page. The user signs in with
                         Google (the app's only auth method) right on the page and
                         deletes their account. Served from this backend so the
                         delete call below is same-origin (no CORS).
  DELETE /api/account    Authenticated hard-delete: removes the user's Firestore
                         document, credit ledger, ad-reward records, any linked
                         guest wallets, and finally the Firebase Auth user.

Device-local scan history (Room DB on the phone) is removed when the user
uninstalls the app; it never leaves the device, so there is nothing to delete
server-side for it.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from firebase_admin import auth as fb_auth

from app.core.firebase_auth import get_current_user
from app.integrations import firebase as firebase_module
from app.logging_config import user_id_var

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Account"])

# Fallback channel for users who are locked out / no longer have the app.
SUPPORT_EMAIL = "support@fauxlens.com"


async def _delete_all(coll, field: str | None = None, value: str | None = None) -> None:
    """
    Delete every document in `coll` (optionally filtered by field == value).

    IDs are collected first, then deleted, so we never mutate the result set
    while iterating it. Works against both the real async Firestore client and
    the in-memory test mock.
    """
    query = coll.where(field, "==", value) if field else coll
    ids = [snap.id async for snap in query.stream()]
    for doc_id in ids:
        await coll.document(doc_id).delete()


@router.delete("/api/account")
async def delete_account(user: dict = Depends(get_current_user)):
    """
    Permanently delete the authenticated user's account and associated data.

    Requires: Authorization: Bearer <firebase_id_token>.
    """
    uid = user["uid"]
    user_id_var.set(uid)

    db = firebase_module.db
    if not db:
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    try:
        user_ref = db.collection("users").document(uid)

        # Guest wallets this account was ever linked to (for support lookups).
        snap = await user_ref.get()
        device_ids = (snap.to_dict() or {}).get("known_device_ids", []) if snap.exists else []

        # Immutable credit ledger (subcollection of the user doc).
        await _delete_all(user_ref.collection("credit_ledger"))
        # Ad-reward bookkeeping keyed to this user.
        await _delete_all(db.collection("ad_rewards"), "user_id", uid)
        await _delete_all(db.collection("ad_ssv_rewards"), "user_id", uid)
        # Any guest wallets migrated/linked to this account.
        for did in device_ids:
            await db.collection("guest_wallets").document(did).delete()
        # The user document itself.
        await user_ref.delete()
    except Exception as e:
        logger.error(
            "account_delete_firestore_failed",
            extra={"action": "account_delete_firestore_failed", "error": str(e)},
        )
        raise HTTPException(status_code=500, detail="Account deletion failed") from e

    # Firebase Auth user (blocking Admin SDK call -> run off the event loop).
    try:
        await asyncio.to_thread(fb_auth.delete_user, uid)
    except fb_auth.UserNotFoundError:
        pass  # Already gone — deletion is idempotent.
    except Exception as e:
        logger.error(
            "account_delete_auth_failed",
            extra={"action": "account_delete_auth_failed", "error": str(e)},
        )
        raise HTTPException(status_code=500, detail="Account deletion failed") from e

    logger.info("account_deleted", extra={"action": "account_deleted"})
    return {"status": "deleted"}


# --------------------------------------------------------------------------- #
# Public deletion web page (self-service, no app required)                    #
# --------------------------------------------------------------------------- #

_DELETE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Delete your FauxLens account</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; display: flex; align-items: center;
    justify-content: center; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0B0E14; color: #E6EAF0;
  }
  .card {
    width: 100%; max-width: 480px; background: #141922; border: 1px solid #232B38;
    border-radius: 20px; padding: 32px; box-shadow: 0 20px 60px rgba(0,0,0,.45);
  }
  h1 { font-size: 22px; margin: 0 0 6px; letter-spacing: -0.01em; }
  .brand { color: #60A5FA; font-weight: 600; }
  p { color: #9AA5B4; line-height: 1.55; font-size: 14px; margin: 12px 0; }
  ul { color: #9AA5B4; font-size: 14px; line-height: 1.6; padding-left: 18px; }
  li { margin: 4px 0; }
  strong { color: #E6EAF0; }
  button {
    width: 100%; margin-top: 20px; padding: 14px 16px; border: 0; border-radius: 12px;
    background: #2563EB; color: #fff; font-size: 15px; font-weight: 600; cursor: pointer;
    transition: background .15s ease;
  }
  button:hover { background: #1D4ED8; }
  button:disabled { background: #33405A; cursor: default; }
  .danger { background: #B4232E; }
  .danger:hover { background: #911a23; }
  .status { margin-top: 16px; font-size: 14px; min-height: 20px; }
  .ok { color: #34D399; } .err { color: #F87171; }
  .muted { color: #6B7688; font-size: 12px; margin-top: 20px; }
  a { color: #60A5FA; }
</style>
</head>
<body>
  <div class="card">
    <h1>Delete your <span class="brand">FauxLens</span> account</h1>
    <p>This permanently deletes your account and all data associated with it. This
       cannot be undone.</p>
    <p><strong>What is deleted:</strong></p>
    <ul>
      <li>Your account and sign-in (Google) identity</li>
      <li>Your email on record and credit balance</li>
      <li>Your credit history and ad-reward records</li>
      <li>Any guest wallet linked to your account</li>
    </ul>
    <p><strong>What is kept:</strong> minimal purchase/transaction records we are
       legally required to retain for tax and accounting, stored without your
       account identity. Scan history stored on your phone is removed when you
       uninstall the app.</p>

    <button id="signin">Sign in with Google to continue</button>
    <button id="delete" class="danger" style="display:none">Permanently delete my account</button>
    <div class="status" id="status"></div>

    <p class="muted">No longer have the app or can't sign in? Email
      <a id="mailto" href="#">SUPPORT_EMAIL</a> from your account's email address
      with the subject "Delete my account" and we will remove it within 30 days.</p>
  </div>

<script type="module">
  import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
  import { getAuth, GoogleAuthProvider, signInWithPopup }
    from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

  const app = initializeApp({
    apiKey: "FB_API_KEY",
    authDomain: "FB_AUTH_DOMAIN",
    projectId: "FB_PROJECT_ID",
    appId: "FB_APP_ID",
  });
  const auth = getAuth(app);

  const $ = (id) => document.getElementById(id);
  const status = $("status");
  const signinBtn = $("signin");
  const deleteBtn = $("delete");
  let idToken = null;

  $("mailto").href = "mailto:SUPPORT_EMAIL?subject=Delete%20my%20account";

  function setStatus(msg, cls) { status.textContent = msg; status.className = "status " + (cls || ""); }

  signinBtn.addEventListener("click", async () => {
    setStatus("Opening Google sign-in…");
    try {
      const cred = await signInWithPopup(auth, new GoogleAuthProvider());
      idToken = await cred.user.getIdToken();
      signinBtn.style.display = "none";
      deleteBtn.style.display = "block";
      setStatus("Signed in as " + (cred.user.email || "your account") + ". Confirm below.");
    } catch (e) {
      setStatus("Sign-in failed: " + (e && e.message ? e.message : e), "err");
    }
  });

  deleteBtn.addEventListener("click", async () => {
    if (!idToken) { setStatus("Please sign in first.", "err"); return; }
    if (!confirm("Permanently delete your FauxLens account? This cannot be undone.")) return;
    deleteBtn.disabled = true;
    setStatus("Deleting your account…");
    try {
      const res = await fetch("/api/account", {
        method: "DELETE",
        headers: { "Authorization": "Bearer " + idToken },
      });
      if (res.ok) {
        deleteBtn.style.display = "none";
        setStatus("Your account and data have been permanently deleted.", "ok");
      } else {
        deleteBtn.disabled = false;
        setStatus("Deletion failed (" + res.status + "). Please try again or email us.", "err");
      }
    } catch (e) {
      deleteBtn.disabled = false;
      setStatus("Network error. Please try again or email us.", "err");
    }
  });
</script>
</body>
</html>"""


@router.get("/delete-account", response_class=HTMLResponse)
async def delete_account_page():
    """Public, self-contained account-deletion page (linked from the Play listing)."""
    from app.config import settings

    html = (
        _DELETE_PAGE.replace("FB_API_KEY", settings.firebase_web_api_key)
        .replace("FB_AUTH_DOMAIN", settings.firebase_web_auth_domain)
        .replace("FB_PROJECT_ID", settings.firebase_web_project_id)
        .replace("FB_APP_ID", settings.firebase_web_app_id)
        .replace("SUPPORT_EMAIL", SUPPORT_EMAIL)
    )
    return HTMLResponse(html)
