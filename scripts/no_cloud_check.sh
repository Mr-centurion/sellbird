#!/usr/bin/env bash
# scripts/no_cloud_check.sh
# Verifies zero cloud SDK imports, API key patterns, and banned provider names in the repo.
# Exits 0 on PASS, exits non-zero on FAIL.

set -euo pipefail

# Find repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Running No-Cloud & Air-Gap Compliance Check ==="

# Patterns to detect
CLOUD_IMPORTS_REGEX="(import[[:space:]]+(anthropic|openai|boto3|cohere|azure\.ai|google\.generativeai)|from[[:space:]]+(anthropic|openai|boto3|cohere|azure\.ai|google\.generativeai))"
API_KEYS_REGEX="(ANTHROPIC_API_KEY|OPENAI_API_KEY|CLAUDE_API_KEY|GEMINI_API_KEY|COHERE_API_KEY|sk-[a-zA-Z0-9]{20,})"
BANNED_PROVIDERS_REGEX="\b(anthropic|openai|claude|codex|bedrock|chatgpt)\b"

COMBINED_REGEX="(${CLOUD_IMPORTS_REGEX}|${API_KEYS_REGEX}|${BANNED_PROVIDERS_REGEX})"

# Run git grep or ripgrep/grep ignoring known legacy migration schema and specified fixtures
MATCHES=$(git grep -i -E "$COMBINED_REGEX" -- ':!docs/AUDIT.md' ':!docs/no_cloud_check.txt' ':!tests/fixtures/**' ':!*.sql' ':!scripts/no_cloud_check.sh' 2>/dev/null || true)

# Filter out legacy SQLite migration SQL lines in state.py permitted by specification
FILTERED_MATCHES=$(echo "$MATCHES" | grep -v -E "sellbird/state\.py:.*(claude_calls|codex_calls|v11:|Claude call|provider TEXT DEFAULT 'claude')" | grep -v "^$" || true)

if [ -n "$FILTERED_MATCHES" ]; then
    echo "FAIL: Cloud dependencies, API keys, or banned provider names detected:"
    echo "--------------------------------------------------"
    echo "$FILTERED_MATCHES"
    echo "--------------------------------------------------"
    exit 1
else
    echo "PASS: Zero cloud SDK imports, API keys, or banned provider references found."
    exit 0
fi
