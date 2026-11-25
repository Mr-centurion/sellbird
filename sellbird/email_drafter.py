"""sellbird/email_drafter.py — Validated cold email drafting service.

Inputs:
- product (name, description, value_props)
- ICP (buyer persona, pain points)
- top signal (label, quote)
- problem statement (statement, why_now, evidence)
- prospect (contact name, title, company, stored notes)

Validation Rules:
1. Body under 110 words
2. Mentions the top buying signal
3. Exactly one call to action
4. Contains no number that does not appear in the stored inputs

Retries once on validation failure. If Ollama is down, queues task in SQLite tasks table.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from sellbird.brain import Brain
from sellbird.models.prospect import Prospect
from sellbird.ollama_runtime import OllamaUnavailable
from sellbird.state import StateManager

logger = logging.getLogger("sellbird.email_drafter")


def extract_numbers(text: str) -> Set[str]:
    """Extract all individual numbers (including percentages or decimals) from text."""
    if not text:
        return set()
    # Normalize common punct like commas in numbers (e.g. 1,000 -> 1000)
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text)
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    return set(matches)


def mentions_top_signal(body: str, top_signal: Dict[str, Any]) -> bool:
    """Check if the draft mentions the top buying signal."""
    if not top_signal:
        return True  # No signal to verify against

    b_lower = body.lower()
    label = (top_signal.get("label") or "").lower().strip()
    quote = (top_signal.get("evidence_quote") or "").lower().strip()

    if label and label != "none":
        label_clean = label.replace("_", " ")
        if label_clean in b_lower or label in b_lower:
            return True

        label_synonyms = {
            "hiring_growth": ["hiring", "roles", "headcount", "recruiting", "team growth", "expanding"],
            "funding": ["funding", "round", "series b", "series a", "capital", "investment"],
            "leadership_change": ["joined", "appointed", "vp", "chief", "executive", "leadership"],
            "tech_adoption": ["migration", "migrating", "stack", "deploy", "adopting", "modules", "platform"],
            "expansion": ["expansion", "expanding", "hubs", "locations", "centers", "facilities", "growth"],
            "pain_expressed": ["bottleneck", "challenge", "delay", "friction", "pain", "manual"],
            "competitor_mention": ["competitor", "alternative", "legacy"],
        }
        for syn in label_synonyms.get(label, []):
            if syn in b_lower:
                return True

    # Check key words from evidence quote
    quote_words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", quote) if w not in {"with", "from", "that", "this", "have", "into", "their"}]
    matches = [w for w in quote_words if w in b_lower]
    if len(matches) >= 2:
        return True

    return False


def count_calls_to_action(body: str) -> int:
    """Count calls to action in email body (questions or explicit CTA sentences)."""
    # Count questions
    q_count = body.count("?")
    if q_count > 0:
        return q_count

    # Fallback to imperative CTA patterns
    cta_patterns = [
        r"\blet me know if\b",
        r"\bopen to a quick\b",
        r"\bworth a brief\b",
        r"\bwould love to connect\b",
        r"\bwould you be open\b",
    ]
    matches = sum(1 for p in cta_patterns if re.search(p, body, re.IGNORECASE))
    return matches


def validate_email_draft(
    body: str,
    top_signal: Dict[str, Any],
    stored_inputs_text: str,
) -> Tuple[bool, str]:
    """Validate email body against the 4 strict requirements."""
    words = body.strip().split()
    if not words:
        return False, "Email body is empty"

    # 1. Under 110 words
    if len(words) >= 110:
        return False, f"Email body has {len(words)} words; must be under 110 words"

    # 2. Mentions top signal
    if top_signal and not mentions_top_signal(body, top_signal):
        return False, f"Email body does not mention the top signal: {top_signal.get('label', '')}"

    # 3. Exactly one call to action
    cta_count = count_calls_to_action(body)
    if cta_count == 0:
        return False, "Email body has no call to action"
    if cta_count > 1:
        return False, f"Email body has {cta_count} calls to action (must have exactly one)"

    # 4. Contains no numbers that do not appear in stored inputs
    body_nums = extract_numbers(body)
    input_nums = extract_numbers(stored_inputs_text)
    hallucinated = body_nums - input_nums
    if hallucinated:
        return False, f"Email body contains numbers not found in input data: {sorted(hallucinated)}"

    return True, ""


async def draft_validated_email(
    prospect_id: str,
    product_id: str,
    state: StateManager,
    brain: Brain,
) -> Optional[Dict[str, Any]]:
    """Generate, validate, and store a product-specific email draft for a prospect."""
    prospect = await state.get_prospect(prospect_id)
    if not prospect:
        raise ValueError(f"Prospect not found: {prospect_id}")

    product = await state.get_product(product_id)
    if not product:
        product = {"id": product_id, "name": product_id, "description": "", "value_props": ""}

    icp = await state.get_icp(product_id) or {}
    if hasattr(state, "get_prospect_signals"):
        signals = await state.get_prospect_signals(prospect_id=prospect_id, product_id=product_id)
    elif hasattr(state, "get_signals"):
        signals = await state.get_signals(prospect_id=prospect_id, product_id=product_id)
    else:
        signals = []
    top_signal = signals[0] if signals else {}

    problem_stmt_obj = await state.get_problem_statement(prospect_id=prospect_id, product_id=product_id) or {}
    prob_text = problem_stmt_obj.get("problem_statement") or problem_stmt_obj.get("statement") or ""
    why_now = problem_stmt_obj.get("why_now") or ""

    stored_prospect_text = await state.get_prospect_stored_text(prospect_id)

    p_size = getattr(prospect, "size", None) or getattr(prospect, "company_size", "") or ""
    p_name = getattr(prospect, "contact_name", None) or (f"{prospect.first_name} {prospect.last_name}".strip()) or "there"

    # Compile all input text to form the ground-truth facts for number verification
    inputs_text = (
        f"{prospect.company} {p_name} {prospect.title} {getattr(prospect, 'industry', '')} {p_size}\n"
        f"{stored_prospect_text}\n"
        f"Product: {product.get('name', '')} {product.get('description', '')} {product.get('value_props', '')}\n"
        f"ICP: {json.dumps(icp)}\n"
        f"Signal: {json.dumps(top_signal)}\n"
        f"Problem: {prob_text} Why now: {why_now}"
    )

    contact_name = p_name
    first_name = contact_name.split()[0] if contact_name != "there" else "there"

    base_prompt = (
        f"Write a personalized cold email (first touch only) to {contact_name} ({prospect.title}) at {prospect.company}.\n"
        f"Product: {product.get('name')} — {product.get('description')}\n"
        f"Key Value Prop: {product.get('value_props')}\n"
        f"Problem Statement: {prob_text}\n"
        f"Why Now: {why_now}\n"
        f"Top Buying Signal: {top_signal.get('label', '')} ({top_signal.get('evidence_quote', '')})\n\n"
        f"Strict Requirements:\n"
        f"1. Body MUST be under 110 words\n"
        f"2. MUST mention the top buying signal\n"
        f"3. Exactly ONE call to action (a single question at the end)\n"
        f"4. Do NOT invent any numbers. Only use numbers from the provided facts above.\n\n"
        f"Return JSON strictly with:\n"
        f"{{\"subject\": \"Relevant subject line\", \"body\": \"Email body text\"}}"
    )

    max_attempts = 2
    last_err = ""
    current_prompt = base_prompt

    model_name = getattr(brain, "last_model", "") or getattr(brain, "model", "qwen3:1.7b")

    for attempt in range(max_attempts):
        try:
            if hasattr(brain, "unavailable") and brain.unavailable:
                raise OllamaUnavailable(brain.unavailable)

            res = await brain.think_json(
                prompt=current_prompt,
                agent="writer",
                task="email_draft",
                num_predict=300,
            )

            if not isinstance(res, dict) or not res.get("body"):
                raise ValueError("LLM returned empty or non-dictionary email response")

            subject = str(res.get("subject", f"Quick question for {prospect.company}")).strip()
            body = str(res.get("body", "")).strip()

            valid, err = validate_email_draft(body, top_signal, inputs_text)
            if valid:
                model_used = getattr(brain, "last_model", "") or model_name
                saved = await state.save_email_draft(
                    product_id=product_id,
                    prospect_id=prospect_id,
                    subject=subject,
                    body=body,
                    model=model_used,
                    mode="llm",
                    status="draft",
                )
                return saved

            last_err = err
            logger.warning(f"Draft validation attempt {attempt+1} failed for {prospect.company}: {err}")

            # Prepare retry prompt with specific corrective instruction
            current_prompt = (
                f"{base_prompt}\n\n"
                f"[CORRECTION NEEDED]: Your previous draft failed validation with: {err}.\n"
                f"Please rewrite the email to strictly fix this issue while keeping body under 110 words, "
                f"mentioning the signal, exactly one call to action, and no ungrounded numbers."
            )

        except OllamaUnavailable as e:
            logger.error(f"Ollama unavailable during email drafting for {prospect.company}: {e}")
            await state.queue_task(
                task_type="email_draft",
                payload={"prospect_id": prospect_id, "product_id": product_id},
                error=str(e),
            )
            return None
        except Exception as e:
            last_err = str(e)
            logger.error(f"Error during email drafting: {e}")

    # If both attempts failed validation, queue the task
    logger.error(f"Email drafting failed after {max_attempts} attempts: {last_err}")
    await state.queue_task(
        task_type="email_draft",
        payload={"prospect_id": prospect_id, "product_id": product_id, "error": last_err},
        error=last_err,
    )
    return None
