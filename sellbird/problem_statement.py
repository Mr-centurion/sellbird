"""Problem statement generation per prospect and per product.

Derives targeted problem statements from ICP, research, and buying signals:
{problem_statement (max 40 words), why_now (max 25 words), evidence[]}
Stores result with product_id and prospect_id.
Supports deterministic non-LLM fallback labeled "non-LLM mode".
"""

import json
import logging
from typing import Any, List, Optional

from sellbird.brain import Brain
from sellbird.buying_signals import classify_buying_signals
from sellbird.config import BrainConfig
from sellbird.ollama_runtime import OllamaUnavailable
from sellbird.stored_research import build_research_summary

logger = logging.getLogger("sellbird.problem_statement")


def _truncate_words(text: str, max_words: int) -> str:
    """Ensure text has at most max_words words."""
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(",;:-") + "."


async def generate_problem_statement(
    prospect_id: str,
    product_id: str,
    state,
    brain: Optional[Brain] = None,
) -> dict:
    """Generate problem statement and why_now rationale for prospect and product.

    Constraints:
    - problem_statement: max 40 words
    - why_now: max 25 words
    - evidence: list of facts / quotes
    """
    prospect = await state.get_prospect(prospect_id)
    if not prospect:
        raise ValueError(f"Prospect {prospect_id} not found")

    product = await state.get_product(product_id)
    if not product:
        product = {"id": product_id, "name": "Outreach & Pipeline Automation", "description": "Automated sales pipeline platform"}

    icp = await state.get_icp(product_id) or {}

    # Research
    research = await state.get_prospect_research(prospect_id)
    if not research:
        research = await build_research_summary(prospect_id, state=state, brain=brain)

    # Signals
    signals = await state.get_prospect_signals(prospect_id, product_id=product_id)
    if not signals:
        signals = await classify_buying_signals(prospect_id, state=state, brain=brain, product_id=product_id)

    if brain is None:
        try:
            from sellbird.config import load_config
            cfg = load_config()
            brain = Brain(state, cfg.brain)
        except Exception:
            brain = Brain(state)

    company = prospect.get("company") or "The company"
    title = prospect.get("title") or "Decision Maker"
    first_name = prospect.get("first_name") or ""
    last_name = prospect.get("last_name") or ""

    prod_name = product.get("name") or "Product"
    prod_desc = product.get("description") or ""
    pain_points = icp.get("pain_points") or ""
    if isinstance(pain_points, list):
        pain_points = ", ".join(pain_points)

    signals_summary = [f"{s.get('label')}: \"{s.get('evidence_quote')}\"" for s in signals[:3]]

    prompt = (
        "You are an enterprise sales strategist analyzing how a specific B2B product solves a prospect's challenges. "
        "Formulate a problem statement, a why-now urgency statement, and supporting evidence.\n\n"
        "STRICT LENGTH CONSTRAINTS:\n"
        "- problem_statement: MAXIMUM 40 WORDS. Focus on the core friction or pain this product relieves.\n"
        "- why_now: MAXIMUM 25 WORDS. Focus on recent trigger events, changes, or expansion creating urgency.\n"
        "- evidence: a list of 1 to 3 direct quotes or facts from research/signals.\n\n"
        f"Product:\n- Name: {prod_name}\n- Description: {prod_desc}\n- Target Pain Points: {pain_points}\n\n"
        f"Prospect:\n- Company: {company}\n- Contact: {first_name} {last_name} ({title})\n"
        f"- Research Summary: {research.get('summary', '')}\n"
        f"- Recent Events: {json.dumps(research.get('recent_events', []))}\n"
        f"- Buying Signals: {json.dumps(signals_summary)}\n\n"
        "Return a JSON object with keys: problem_statement, why_now, evidence."
    )

    mode = "llm"
    try:
        if hasattr(brain, "think_json"):
            data = await brain.think_json(prompt, agent="writer", task="problem_statement")
        elif hasattr(brain, "ask"):
            raw_res = await brain.ask(system="", user=prompt, format="json", agent="writer", task="problem_statement")
            data = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
        elif hasattr(brain, "think"):
            raw_res = await brain.think(prompt, expect_json=True, agent="writer", task="problem_statement")
            data = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
        else:
            raise AttributeError("Brain does not support think_json or ask")

        if not data or not isinstance(data, dict):
            raise ValueError(f"Invalid brain response: {data}")

        ps = _truncate_words(str(data.get("problem_statement") or ""), 40)
        wn = _truncate_words(str(data.get("why_now") or ""), 25)
        ev = data.get("evidence") or []
        if isinstance(ev, str):
            ev = [ev]

        if not ps:
            raise ValueError("Brain returned empty problem statement")

    except (OllamaUnavailable, Exception) as e:
        logger.info("Brain unavailable for problem statement (%s), falling back to non-LLM mode", e)
        mode = "non-LLM mode"
        # Deterministic non-LLM synthesis
        pains = icp.get("pain_points_list") or []
        chosen_pain = pains[0] if pains else "manual workflow bottlenecks and operational friction"

        ps = f"{company} struggles with {chosen_pain}, restricting execution speed across team operations. (non-LLM mode)"
        ps = _truncate_words(ps, 40)

        trigger = "Recent organizational expansion and market momentum"
        if signals:
            top_sig = signals[0]
            label_clean = top_sig.get("label", "growth").replace("_", " ")
            trigger = f"Recent {label_clean}"

        wn = f"{trigger} makes resolving operational drag critical for executive leadership today. (non-LLM mode)"
        wn = _truncate_words(wn, 25)

        ev = [s.get("evidence_quote") for s in signals[:2] if s.get("evidence_quote")]
        if not ev and research:
            ev = research.get("recent_events", [])[:2]

    # Save to database
    ps_id = await state.save_problem_statement(
        prospect_id=prospect_id,
        product_id=product_id,
        problem_statement=ps,
        why_now=wn,
        evidence=ev,
        mode=mode,
    )

    return {
        "id": ps_id,
        "prospect_id": prospect_id,
        "product_id": product_id,
        "problem_statement": ps,
        "why_now": wn,
        "evidence": ev,
        "mode": mode,
    }
