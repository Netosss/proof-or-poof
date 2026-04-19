#!/bin/bash
# Backend Stop hook: runs pytest + mypy + print() check at session end.
# Async — does not block Claude's response.

BACKEND_DIR="/Users/netanel.ossi/Desktop/fauxlens/backend-python"
cd "$BACKEND_DIR" || exit 0

ISSUES=0

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║   Backend — Session End Verification  ║"
echo "╚═══════════════════════════════════════╝"

# ── 1. Ruff lint ──────────────────────────────────────────────────────────────
echo ""
echo "▶ ruff check"
RUFF_OUT=$(/opt/homebrew/bin/ruff check app/ --output-format=concise 2>&1 | head -15)
if [ -n "$RUFF_OUT" ]; then
    echo "$RUFF_OUT"
    ISSUES=$((ISSUES + 1))
else
    echo "  ✅ Clean"
fi

# ── 2. Mypy type check ────────────────────────────────────────────────────────
echo ""
echo "▶ mypy type check"
MYPY_OUT=$(/opt/homebrew/bin/mypy app/ --ignore-missing-imports --no-error-summary --no-pretty 2>&1 \
    | grep -E "^app/.*error:" | head -10)
if [ -n "$MYPY_OUT" ]; then
    echo "$MYPY_OUT"
    ISSUES=$((ISSUES + 1))
else
    echo "  ✅ No type errors"
fi

# ── 3. print() check ─────────────────────────────────────────────────────────
PRINTS=$(grep -rn "^\s*print(" app/ --include="*.py" 2>/dev/null | grep -v "#" | head -5)
if [ -n "$PRINTS" ]; then
    echo ""
    echo "⚠️  print() found in production code (use logger instead):"
    echo "$PRINTS"
    ISSUES=$((ISSUES + 1))
fi

# ── 4. pytest (fast: fail-fast, no external services) ────────────────────────
echo ""
echo "▶ pytest (unit tests only)"
PYTHON="${BACKEND_DIR}/venv/bin/python"
[ ! -x "$PYTHON" ] && PYTHON="${BACKEND_DIR}/.venv/bin/python"
[ ! -x "$PYTHON" ] && PYTHON="python3"
TEST_OUT=$("$PYTHON" -m pytest tests/ -x -q --tb=line \
    --ignore=tests/test_detection_route.py \
    --ignore=tests/test_inpainting_route.py \
    -k "not integration" 2>&1)
PYTEST_EXIT=$?
echo "$TEST_OUT" | tail -15
if [ $PYTEST_EXIT -ne 0 ]; then
    ISSUES=$((ISSUES + 1))
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
if [ $ISSUES -eq 0 ]; then
    echo "✅  Backend session-end checks passed."
else
    echo "⚠️  $ISSUES issue(s) found. Run 'git push' to trigger the full pre-push gate."
fi

exit 0
