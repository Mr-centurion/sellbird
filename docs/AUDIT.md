# Audit Notes: Legacy Database Columns and Migrations

Sellbird operates 100% offline-first and locally via Ollama.
For backward compatibility with existing SQLite databases, existing migration SQL and schema definitions are preserved untouched:

- `usage_log.claude_calls`: Legacy column name in SQLite schema representing daily model calls (now exclusively Ollama). Retained to avoid breaking existing databases and migrations.
- `usage_log.codex_calls`: Legacy column name retained for schema compatibility, no longer read or incremented.
- `campaigns.instantly_campaign_id`: Legacy column name retained in schema migrations.
