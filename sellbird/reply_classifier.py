"""sellbird/reply_classifier.py — 8-label B2B reply intent classifier.

Labels:
- interested
- objection
- not_now
- not_interested
- unsubscribe
- out_of_office
- referral
- unclear

Classifies into {label, confidence, reasoning (max 25 words), suggested_next_action}.
Provides keyword heuristic fallback as "non-LLM mode".
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from sellbird.brain import Brain
from sellbird.ollama_runtime import OllamaUnavailable
from sellbird.state import StateManager

logger = logging.getLogger("sellbird.reply_classifier")

VALID_LABELS = [
    "interested",
    "objection",
    "not_now",
    "not_interested",
    "unsubscribe",
    "out_of_office",
    "referral",
    "unclear",
]

SUGGESTED_ACTIONS = {
    "interested": "Send demo calendar booking link and overview",
    "objection": "Acknowledge concern with case study proof point",
    "not_now": "Snooze contact and schedule follow-up next quarter",
    "not_interested": "Mark as not interested and close cadence",
    "unsubscribe": "Add to suppression list immediately and purge",
    "out_of_office": "Pause outreach until return date mentioned",
    "referral": "Create new prospect record for referred colleague",
    "unclear": "Send short clarifying question confirming context",
}


def classify_reply_heuristic(reply_text: str) -> Dict[str, Any]:
    """Deterministic keyword-based classification for all 8 labels (non-LLM mode)."""
    text = (reply_text or "").lower().strip()

    # 1. Unsubscribe
    unsub_patterns = [
        r"\bunsubscribe\b",
        r"\bremove (me|my email|our domain)\b",
        r"\btake me off\b",
        r"\bstop (emailing|contacting|sending)\b",
        r"\bopt[- ]out\b",
        r"\bnever (contact|email)\b",
        r"\bdelete my (info|data|records)\b",
        r"\bdo not email\b",
        r"\bscrub (my|our)\b",
        r"\bpurge\b",
        r"\bmailing lists?\b",
    ]
    if any(re.search(p, text) for p in unsub_patterns):
        return {
            "label": "unsubscribe",
            "confidence": 0.99,
            "reasoning": "Explicit opt-out request detected in text (non-LLM mode).",
            "suggested_next_action": SUGGESTED_ACTIONS["unsubscribe"],
            "mode": "non-LLM mode",
        }

    # 2. Out of Office
    ooo_patterns = [
        r"\bout of (the )?office\b",
        r"\baway from (my )?desk\b",
        r"\bannual leave\b",
        r"\bmaternity leave\b",
        r"\bauto[- ]reply\b",
        r"\bon vacation\b",
        r"\blimited access to email\b",
        r"\breturning (in|on)\b",
        r"\bspotty (email )?connectivity\b",
        r"\btraveling for\b",
        r"\bupon my return\b",
        r"\bon-sites?\b",
    ]
    if any(re.search(p, text) for p in ooo_patterns):
        return {
            "label": "out_of_office",
            "confidence": 0.95,
            "reasoning": "Automated out-of-office absence notice detected (non-LLM mode).",
            "suggested_next_action": SUGGESTED_ACTIONS["out_of_office"],
            "mode": "non-LLM mode",
        }

    # 3. Referral
    ref_patterns = [
        r"\bnot the right person\b",
        r"\bwrong person\b",
        r"\breach out to\b",
        r"\btalk to\b",
        r"\bcontact (my colleague|our)\b",
        r"\btransitioned out\b",
        r"\bloop in\b",
        r"\brefer you to\b",
        r"\bno longer oversee\b",
        r"\bpass this over\b",
        r"\bforward this (proposal|email)?\b",
        r"\bwho (manages|heads|leads)\b",
        r"\bfalls under\b",
    ]
    if any(re.search(p, text) for p in ref_patterns):
        return {
            "label": "referral",
            "confidence": 0.90,
            "reasoning": "Prospect referred conversation to a colleague (non-LLM mode).",
            "suggested_next_action": SUGGESTED_ACTIONS["referral"],
            "mode": "non-LLM mode",
        }

    # 4. Not Now
    not_now_patterns = [
        r"\bnot right now\b",
        r"\bcircle back\b",
        r"\bnext quarter\b",
        r"\bfollow up in\b",
        r"\bnext summer\b",
        r"\bin a few months\b",
        r"\bmid(dle)? of our\b",
        r"\breach back out\b",
        r"\bheads[- ]down\b",
        r"\bblackout period\b",
        r"\bping me (after|again|in)\b",
        r"\bresurface this\b",
        r"\bcan revisit\b",
        r"\bfreeze lifts\b",
    ]
    if any(re.search(p, text) for p in not_now_patterns):
        return {
            "label": "not_now",
            "confidence": 0.88,
            "reasoning": "Timing is poor but prospect welcomes future contact (non-LLM mode).",
            "suggested_next_action": SUGGESTED_ACTIONS["not_now"],
            "mode": "non-LLM mode",
        }

    # 5. Not Interested
    not_int_patterns = [
        r"\bnot interested\b",
        r"\bno thank(s| you)\b",
        r"\bpass on this\b",
        r"\bzero need\b",
        r"\bdo not buy\b",
        r"\bdon't send\b",
        r"\bhandle everything in-house\b",
        r"\bcompletely content\b",
        r"\bzero (desire|appetite|interest)\b",
    ]
    if any(re.search(p, text) for p in not_int_patterns):
        return {
            "label": "not_interested",
            "confidence": 0.92,
            "reasoning": "Prospect expressed clear disinterest in outreach (non-LLM mode).",
            "suggested_next_action": SUGGESTED_ACTIONS["not_interested"],
            "mode": "non-LLM mode",
        }

    # 6. Objection
    obj_patterns = [
        r"\btoo expensive\b",
        r"\balready (use|have|run)\b",
        r"\bdisruptive\b",
        r"\bfroze\b",
        r"\bunbudgeted\b",
        r"\bno budget\b",
        r"\blacks the strict\b",
        r"\bcompliance guidelines\b",
        r"\bmigrating (off|is)\b",
        r"\bbuilt our (own|custom)\b",
        r"\binternal stack\b",
        r"\bproprietary pipeline\b",
        r"\bnever sign off\b",
        r"\bcompliance officers?\b",
        r"\bon-premises\b",
    ]
    if any(re.search(p, text) for p in obj_patterns):
        return {
            "label": "objection",
            "confidence": 0.85,
            "reasoning": "Pricing, competitor or compliance objection raised (non-LLM mode).",
            "suggested_next_action": SUGGESTED_ACTIONS["objection"],
            "mode": "non-LLM mode",
        }

    # 7. Interested
    interested_patterns = [
        r"\bpricing\b",
        r"\bdemo\b",
        r"\bwalkthrough\b",
        r"\bsounds good\b",
        r"\bintro call\b",
        r"\b(schedule|set up|book)( a)? call\b",
        r"\bshare details\b",
        r"\bexperiencing dispatch\b",
        r"\bhow does\b.*\bwork\b",
        r"\bwould love to see\b",
        r"\bcalendar link\b",
        r"\bdeck\b",
        r"\bmind sending\b",
        r"\b10 minutes\b",
        r"\bwalk through\b",
        r"\bdealing with high\b",
    ]
    if any(re.search(p, text) for p in interested_patterns):
        return {
            "label": "interested",
            "confidence": 0.92,
            "reasoning": "Prospect expressed explicit interest or meeting request (non-LLM mode).",
            "suggested_next_action": SUGGESTED_ACTIONS["interested"],
            "mode": "non-LLM mode",
        }

    # 8. Unclear
    return {
        "label": "unclear",
        "confidence": 0.50,
        "reasoning": "Ambiguous message requiring manual review or clarification (non-LLM mode).",
        "suggested_next_action": SUGGESTED_ACTIONS["unclear"],
        "mode": "non-LLM mode",
    }


async def classify_reply(
    reply_text: str,
    brain: Optional[Brain] = None,
    prospect: Optional[Any] = None,
    state: Optional[StateManager] = None,
    agent: str = "handler",
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    """Classify reply text into one of 8 labels with confidence and next action."""
    clean_text = reply_text.strip()
    if not clean_text:
        return {
            "label": "unclear",
            "confidence": 0.0,
            "reasoning": "Empty reply text received.",
            "suggested_next_action": SUGGESTED_ACTIONS["unclear"],
            "mode": "non-LLM mode",
            "model": "",
            "latency_ms": 0,
        }

    # If brain is unavailable, check allow_fallback
    if brain is None or (hasattr(brain, "unavailable") and brain.unavailable):
        if not allow_fallback:
            reason = getattr(brain, "unavailable", "Brain unavailable") if brain else "No Brain instance provided"
            raise OllamaUnavailable(f"Ollama is unreachable: {reason}")
        res = classify_reply_heuristic(clean_text)
        if state and prospect and hasattr(prospect, "id"):
            await state.save_reply_classification(
                prospect_id=prospect.id,
                thread_text=clean_text,
                label=res["label"],
                confidence=res["confidence"],
                reasoning=res["reasoning"],
                suggested_next_action=res["suggested_next_action"],
                mode="non-LLM mode",
            )
        return res

    prompt = (
        f"Classify this sales email reply into exactly ONE of these 8 labels:\n"
        f"Labels: interested, objection, not_now, not_interested, unsubscribe, out_of_office, referral, unclear.\n\n"
        f"<untrusted source=\"prospect_reply\">\n"
        f"{clean_text}\n"
        f"</untrusted>\n\n"
        f"Return JSON strictly with:\n"
        f"{{\n"
        f'  "label": "<one of the 8 labels>",\n'
        f'  "confidence": <float 0.0 to 1.0>,\n'
        f'  "reasoning": "<concise explanation under 25 words>",\n'
        f'  "suggested_next_action": "<actionable next step>"\n'
        f"}}"
    )

    try:
        data = await brain.think_json(
            prompt=prompt,
            agent=agent,
            task="classify_reply",
            num_predict=300,
        )

        if isinstance(data, dict) and data.get("label"):
            raw_label = str(data.get("label", "")).lower().strip()
            # Normalize common synonyms or mapping
            if raw_label in ("ooo", "vacation", "leave"):
                raw_label = "out_of_office"
            elif raw_label in ("wrong_person", "forward"):
                raw_label = "referral"
            elif raw_label in ("opt_out", "optout"):
                raw_label = "unsubscribe"
            elif raw_label not in VALID_LABELS:
                raw_label = "unclear"

            confidence = float(data.get("confidence", 0.85))
            raw_reasoning = str(data.get("reasoning", "")).strip()
            words = raw_reasoning.split()
            if len(words) > 25:
                raw_reasoning = " ".join(words[:25])

            action = str(data.get("suggested_next_action", "")).strip() or SUGGESTED_ACTIONS.get(raw_label, "Review reply")

            result = {
                "label": raw_label,
                "confidence": confidence,
                "reasoning": raw_reasoning,
                "suggested_next_action": action,
                "mode": "llm",
                "model": getattr(brain, "last_model", brain.model),
                "latency_ms": getattr(brain, "last_latency_ms", 0),
            }

            if state and prospect and hasattr(prospect, "id"):
                await state.save_reply_classification(
                    prospect_id=prospect.id,
                    thread_text=clean_text,
                    label=raw_label,
                    confidence=confidence,
                    reasoning=raw_reasoning,
                    suggested_next_action=action,
                    mode="llm",
                )
            return result

        raise ValueError("Invalid JSON reply classification payload")

    except Exception as e:
        if not allow_fallback:
            raise OllamaUnavailable(f"Ollama classification failed: {e}") from e
        logger.info(f"Ollama unavailable for reply classification, falling back to heuristic: {e}")
        res = classify_reply_heuristic(clean_text)
        if state and prospect and hasattr(prospect, "id"):
            await state.save_reply_classification(
                prospect_id=prospect.id,
                thread_text=clean_text,
                label=res["label"],
                confidence=res["confidence"],
                reasoning=res["reasoning"],
                suggested_next_action=res["suggested_next_action"],
                mode="non-LLM mode",
            )
        return res
