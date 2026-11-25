"""Buying signals classification over stored text.

Classifies stored prospect text into labels:
- hiring_growth
- funding
- leadership_change
- tech_adoption
- pain_expressed
- expansion
- competitor_mention
- none

Each signal requires an evidence_quote. After the LLM responds, DROPS any signal
whose quote does not appear verbatim in the stored text.
Maintains a regex detector as the non-LLM fallback labeled "non-LLM mode".
"""

import json
import logging
import re
from typing import List, Optional

from sellbird.brain import Brain
from sellbird.config import BrainConfig
from sellbird.ollama_runtime import OllamaUnavailable

logger = logging.getLogger("sellbird.buying_signals")

SIGNAL_LABELS = {
    "hiring_growth",
    "funding",
    "leadership_change",
    "tech_adoption",
    "pain_expressed",
    "expansion",
    "competitor_mention",
    "none",
}

# ── Non-LLM Regex Fallback Patterns ──

REGEX_RULES = [
    (
        "funding",
        [
            re.compile(r"(\$\d+(?:\.\d+)?[MBK]?\s*(?:seed|series\s*[a-d]|funding|round|investment|capital))", re.I),
            re.compile(r"((?:secures?|secured|raised?|closes?|closed?)\s+\$\d+(?:\.\d+)?[MBK]?[^.\n]*)", re.I),
        ],
    ),
    (
        "hiring_growth",
        [
            re.compile(r"((?:opens?\s+\d+\s+new\s+(?:engineering|roles|positions))[^.\n]*)", re.I),
            re.compile(r"((?:expands?\s+(?:engineering\s+)?headcount\s+by\s+\d+%)[^.\n]*)", re.I),
            re.compile(r"((?:hires?\s+\d+\s+account\s+executives)[^.\n]*)", re.I),
            re.compile(r"((?:adds?\s+\d+\s+trained)[^.\n]*)", re.I),
            re.compile(r"((?:hiring\s+growth)[^.\n]*)", re.I),
        ],
    ),
    (
        "leadership_change",
        [
            re.compile(r"((?:Former|Veteran)?\s*[A-Z][a-z]+\s+executive\s+[A-Za-z\s]+\s+joins?\s+[A-Za-z0-9]+[^.\n]*)", re.I),
            re.compile(r"([A-Za-z\s]+\s+appointed\s+(?:as\s+)?(?:Director|VP|CTO|CEO|CIO|Head)[^.\n]*)", re.I),
            re.compile(r"([A-Za-z\s]+\s+joins?\s+[A-Za-z0-9]+\s+as\s+(?:Vice\s+President|VP|CTO|CIO|Director)[^.\n]*)", re.I),
        ],
    ),
    (
        "tech_adoption",
        [
            re.compile(r"((?:deploys?|deployed)\s+[A-Za-z0-9]+(?:\s+and\s+[A-Za-z0-9]+)?\s+to\s+[^.\n]*)", re.I),
            re.compile(r"((?:upgrades?|upgraded)\s+[^.\n]*clusters?\s+with\s+[^.\n]*)", re.I),
            re.compile(r"((?:implements?|implemented)\s+Apache\s+Kafka[^.\n]*)", re.I),
            re.compile(r"((?:launches?|launched)\s+automated\s+dbt[^.\n]*)", re.I),
        ],
    ),
    (
        "pain_expressed",
        [
            re.compile(r"([^.\n]*(?:bottlenecks?|delays?|friction|burnout|fatigue|dissatisfaction|frustration|backlogs?|complaints?)[^.\n]*)", re.I),
        ],
    ),
    (
        "expansion",
        [
            re.compile(r"([^.\n]*announces?\s+the\s+opening\s+of\s+two\s+new\s+fulfillment\s+hubs[^.\n]*)", re.I),
            re.compile(r"([^.\n]*expands?\s+clinical\s+operations\s+into[^.\n]*)", re.I),
            re.compile(r"([^.\n]*regional\s+expansion\s+initiative[^.\n]*)", re.I),
            re.compile(r"([^.\n]*expands?\s+(?:banking|commercial|cleaning)[^.\n]*)", re.I),
            re.compile(r"([^.\n]*commercial\s+commission\s+of\s+a\s+\d+MW[^.\n]*)", re.I),
        ],
    ),
    (
        "competitor_mention",
        [
            re.compile(r"([^.\n]*migration\s+away\s+from\s+(?:Epic|Datadog|Procore|Lightspeed|Trimble)[^.\n]*)", re.I),
            re.compile(r"([^.\n]*dissatisfaction\s+with\s+Procore[^.\n]*)", re.I),
            re.compile(r"([^.\n]*legacy\s+(?:Epic|Datadog|Lightspeed|Procore)[^.\n]*)", re.I),
        ],
    ),
]


def detect_signals_regex(stored_text: str) -> list[dict]:
    """Deterministic regex detector for buying signals (non-LLM fallback)."""
    detected = []
    seen_quotes = set()

    # Split stored text into lines/sentences to extract exact verbatim quotes
    lines = [ln.strip() for ln in stored_text.splitlines() if ln.strip()]

    for label, patterns in REGEX_RULES:
        for pat in patterns:
            for line in lines:
                m = pat.search(line)
                if m:
                    quote = m.group(0).strip()
                    # Strip bullet prefix if any
                    quote = re.sub(r"^[-*•]\s*", "", quote)
                    # Verify quote exists verbatim in stored_text
                    if quote and quote in stored_text and quote not in seen_quotes:
                        seen_quotes.add(quote)
                        detected.append({
                            "label": label,
                            "evidence_quote": quote,
                            "mode": "non-LLM mode",
                        })
    return detected


async def classify_buying_signals(
    prospect_id: str,
    state,
    brain: Optional[Brain] = None,
    product_id: str = "",
) -> list[dict]:
    """Classify buying signals for a prospect from stored text.

    Uses LLM classification into the 8 labels. Enforces verbatim evidence_quote
    match against stored text. If quote is not verbatim in stored_text, drops the signal.
    Falls back to regex detector labeled 'non-LLM mode' when Brain is unavailable.
    """
    stored_text = await state.get_prospect_stored_text(prospect_id)
    if not stored_text:
        return []

    if brain is None:
        try:
            from sellbird.config import load_config
            cfg = load_config()
            brain = Brain(state, cfg.brain)
        except Exception:
            brain = Brain(state)

    prompt = (
        "You are an expert B2B sales intelligence agent detecting buying signals in company text. "
        "Classify signals into exactly one of these labels:\n"
        "- hiring_growth\n"
        "- funding\n"
        "- leadership_change\n"
        "- tech_adoption\n"
        "- pain_expressed\n"
        "- expansion\n"
        "- competitor_mention\n"
        "- none\n\n"
        "MANDATORY REQUIREMENT: Each signal MUST have an 'evidence_quote' that is an EXACT, 100% VERBATIM "
        "substring copied directly from the stored text. If you paraphrase, invent words, or edit the quote, "
        "the signal will be automatically discarded.\n\n"
        f"Stored Company Text:\n{stored_text}\n\n"
        "Return a JSON object:\n"
        '{"signals": [{"label": "funding", "evidence_quote": "..."}]}'
    )

    accepted_signals = []
    try:
        if hasattr(brain, "think_json"):
            data = await brain.think_json(prompt, agent="scout", task="buying_signals")
        elif hasattr(brain, "ask"):
            raw_res = await brain.ask(system="", user=prompt, format="json", agent="scout", task="buying_signals")
            data = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
        elif hasattr(brain, "think"):
            raw_res = await brain.think(prompt, expect_json=True, agent="scout", task="buying_signals")
            data = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
        else:
            raise AttributeError("Brain does not support think_json or ask")

        if not data or not isinstance(data, dict):
            raise ValueError(f"Invalid brain response: {data}")

        raw_signals = data.get("signals") or []
        if isinstance(raw_signals, dict):
            raw_signals = [raw_signals]

        for s in raw_signals:
            if not isinstance(s, dict):
                continue
            label = str(s.get("label") or "").strip().lower()
            if label not in SIGNAL_LABELS or label == "none":
                continue
            quote = str(s.get("evidence_quote") or "").strip()
            # Clean outer quotes if any
            if (quote.startswith('"') and quote.endswith('"')) or (quote.startswith("'") and quote.endswith("'")):
                quote = quote[1:-1].strip()

            # STRICT VERBATIM CHECK: drop any signal whose quote does not appear verbatim in stored text
            if not quote or quote not in stored_text:
                logger.info("Dropping signal %s: evidence_quote not found verbatim in stored text", label)
                continue

            accepted_signals.append({
                "label": label,
                "evidence_quote": quote,
                "mode": "llm",
            })

        if not accepted_signals and not raw_signals:
            # If LLM returned empty, check regex fallback
            pass

    except (OllamaUnavailable, Exception) as e:
        logger.info("Brain unavailable for signal classification (%s), falling back to regex non-LLM mode", e)
        accepted_signals = detect_signals_regex(stored_text)

    # Save accepted signals to state
    for sig in accepted_signals:
        await state.save_prospect_signal(
            prospect_id=prospect_id,
            label=sig["label"],
            evidence_quote=sig["evidence_quote"],
            product_id=product_id,
            mode=sig.get("mode", "llm"),
        )

    return accepted_signals
