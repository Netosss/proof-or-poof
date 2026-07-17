---
name: fauxlens-kmp-migration
description: State of the FauxLens Android→KMP migration and mobile App Check anti-abuse work
metadata: 
  node_type: memory
  type: project
  originSessionId: 776c872e-67d3-4577-bf72-8bc519b9dd93
---

Migrating the FauxLens Android app to Kotlin Multiplatform + Compose
Multiplatform (iOS later, compile-proven only). Authoritative plan:
`mobile-android/docs/KMP_PLAYSTORE_PLAN.md`; running log:
`docs/OVERNIGHT_STATUS_2026-07-17.md`. Primary goal = a Play-Store-shippable
Android app; `main`/`composeApp` must always build a shippable Android release.

**As of 2026-07-17 — Phases 0-3 DONE + on-device verified, committed to PRs:**
- Mobile PR: https://github.com/Netosss/FauxLens-mobile/pull/1 (branch
  `feat_kmp_migration_rc`). Backend PR #44 (Netosss/proof-or-poof, branch
  `feat_mobile_app_check_rc`).
- Phase 0 toolchain (Gradle 8.14/AGP 8.12.2/Kotlin 2.2.20/KSP2/Room 2.7.2).
- Phase 1 GPS EXIF strip + SecureStorage seam + first unit tests.
- App Check anti-abuse: backend OR-gate (monitor default) + Android client
  (Play Integrity/debug provider, X-Firebase-AppCheck, Turnstile now
  best-effort/App-Check-primary). **Proven E2E on Pixel vs staging(enforce):**
  guest scan minted App Check token, no Turnstile, backend 200.
- Phase 2 Hilt→Koin (frameworkModule + appModule; Koin verify() JVM test).
- Phase 3 KMP module: `:app`→`:composeApp`, kotlin("multiplatform") +
  org.jetbrains.compose, androidTarget + iosSimulatorArm64; commonMain
  platformName() expect/actual; `compileKotlinIosSimulatorArm64` green.

**As of 2026-07-17 (later) — Phase 4a runtime-confirmed + Phase 4b (network/auth) DONE + committed:**
- Phase 4a Ktor rewrite runtime-confirmed on Pixel (real inpaint round-trip:
  multipart upload + raw-ByteArray response + X-User-Balance/X-Op-Ref headers).
- Phase 4b network/auth slice moved to `commonMain` + committed (mobile PR #1,
  commit `0c179a5`, pushed): `FauxLensApi` (Ktor + `UploadFile` seam),
  `HttpClientFactory` (`httpClientEngine()` expect/actual = OkHttp android / Darwin
  iOS), wire DTOs, ApiResponse/AppJson/DetectionStage/Exceptions,
  DetectionProgressClient, `TokenProvider` (+`AuthTokenSource` seam), SecureStorage
  iface, analytics iface/events, `SessionRepository` (+`AuthGateway` seam — sign-in
  takes a Google ID token, not Firebase AuthCredential), AdsRepository,
  ReportsRepository. New common seams: `util/AppLog` (expect), `platform/Time`
  (nowMillis/randomUuid expect). Android actuals: FirebaseAuthTokenSource,
  FirebaseAuthGateway, AndroidUploadFile.
- Verified green: compileDebugKotlin, **compileKotlinIosSimulatorArm64**,
  testDebugUnitTest (Koin verify), assembleRelease (R8).
- Phase 6 prep: proguard-rules.pro rewritten for Ktor/serialization/Koin stack
  (dropped stale Moshi/Retrofit/Hilt/WorkManager keeps).
- Build env: gradle needs `JAVA_HOME=~/.sdkman/candidates/java/17.0.9-tem`; the
  sandbox blocks gradle's file-lock socket + non-fiverr GitHub push → run gradle
  and `git push` (Netosss remote) with the sandbox disabled.

**Phase 4b DEFERRED (local persistence):** Room + DataStore + ScanHistoryRepository
+ HistoryImageStore + entity stay in androidMain. Room-KMP over live history data
needs fixture-migration tests on device (risk register) — not moved blind.

**Phase 5 (UI → commonMain) IN PROGRESS — batches 1-3 done + device-verified,
committed to PR #1:** batch 1 `ce4090d` (CMP-accessor switch + theme + font seam),
batch 2 `6daa9fd` (12 pure components → commonMain), batch 3 `9e23fde` (Coil2→Coil3:
SingletonImageLoader.Factory + coil-network-okhttp + coil3.* across 11 files). Each
installed + launched on Pixel, no crash. Remaining batches: (4) coil-only components
→ commonMain (add coil3-compose to commonMain, network fetcher stays androidMain);
(5) ViewModels → commonMain (KMP lifecycle-viewmodel, Bitmap→ImageBitmap in
analysis/remover/history); (6) screens → commonMain (media/share/EXIF/FileProvider/
haptics/canvas behind expect/actual); (7) androidx nav → JetBrains nav, MainActivity
stays androidMain calling common FauxLensApp, Login last.

**Phase 5 setup notes:** user chose the device-in-loop approach (Pixel
`5A270DLCQ000YL`; migrate + smoke-test per screen). User prefers a REGULAR git commit
(not the fiverr-commit skill) for this personal repo; still needs
FIVERR_COMMIT_ACTIVE=1 to pass the block-direct-git hook, + sandbox disabled for the
Netosss push. The compose-accessor switch is low-risk on Android (compose-multiplatform
plugin maps `compose.material3` etc. to the SAME androidx artifacts on the Android
target, so Coil2/google-fonts/androidx-nav keep working in androidMain; only files
moved to commonMain must be Android-free). Friction: removing the Compose BOM
un-pins `ui-text-google-fonts` (pin it, or drop it for bundled-TTF composeResources —
no TTFs on hand); Coil2→Coil3 has real API breaks (SingletonImageLoader, separate
`coil-network-okhttp` artifact, PlatformContext); ~60 UI files; Bitmap→ImageBitmap in
analysis/remover/history VMs; androidx nav→JetBrains nav + androidx lifecycle→KMP
lifecycle. Batch order: theme+fonts seam → pure components → VMs → screens → nav
(Login last). No @Preview anywhere (0 files). Color/Shape/Spacing/Theme are pure
Compose; Type.kt needs the font seam.

**KMP gotchas learned:** KMP androidMain compiles `src/androidMain/kotlin` only
(NOT the AGP `java/` dir) — move all sources to kotlin/ or classes silently
drop from the APK (ClassNotFound at runtime, build still "succeeds"). KSP →
`kspAndroid`. BOMs need `project.dependencies.platform(...)` in sourceSets.
Compose compiler needs `compose.runtime` on commonMain so iOS compiles.
Committing needs the FIVERR_COMMIT_ACTIVE=1 flag (see [[git-commit-blocked-autonomous]]).

**Remaining:** Phase 4b local persistence (Room-KMP + DataStore-mp + history repo —
deferred pending migration tests), Phase 5 (UI → commonMain, full CMP — see plan
above), Phase 6 (Play release). Owner to-dos: deploy backend PR #44 to prod
(monitor→enforce), Play Console + release keystore, register Play-App-Signing SHA
in Firebase App Check.
