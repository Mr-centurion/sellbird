"""sellbird/scoring.py — Deterministic product-specific lead scoring with LLM rationale.

Calculates score 0-100 purely in code from:
- rule weights: title match, industry match, company size match, buying signals
- embedding similarity (all-MiniLM-L6-v2, cached) between stored text and product ICP

The LLM is invoked solely to generate:
{rationale (max 40 words), top_reasons[3], main_risk}
If Ollama is unavailable, produces a deterministic rule-based rationale labeled "non-LLM mode".
"""

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional
import numpy as np

from sellbird.brain import Brain
from sellbird.models.prospect import Prospect
from sellbird.ollama_runtime import OllamaUnavailable
from sellbird.state import StateManager
from sellbird.stored_research import compute_embedding

logger = logging.getLogger("sellbird.scoring")


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two numerical vectors."""
    if not vec_a or not vec_b:
        return 0.0
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Fallback text similarity when sentence transformers are absent."""
    words_a = set(re.findall(r"\w+", text_a.lower()))
    words_b = set(re.findall(r"\w+", text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


async def calculate_score_breakdown(
    prospect: Prospect,
    product: Dict[str, Any],
    icp: Dict[str, Any],
    signals: List[Dict[str, Any]],
    state: Optional[StateManager] = None,
) -> Dict[str, int]:
    """Calculate deterministic component points totaling 0-100."""
    # 1. Title Match (0-20 points)
    title_pts = 0
    prospect_title = (prospect.title or "").lower().strip()
    target_titles = icp.get("titles") or []
    if isinstance(target_titles, str):
        target_titles = [t.strip() for t in target_titles.split(",") if t.strip()]

    target_titles_lower = [t.lower() for t in target_titles]
    if prospect_title and target_titles_lower:
        if any(t == prospect_title or t in prospect_title or prospect_title in t for t in target_titles_lower):
            title_pts = 20
        elif any(word in prospect_title for t in target_titles_lower for word in t.split() if len(word) > 3):
            title_pts = 10

    # 2. Industry Match (0-20 points)
    industry_pts = 0
    prospect_ind = (prospect.industry or "").lower().strip()
    target_industries = icp.get("industries") or []
    if isinstance(target_industries, str):
        target_industries = [i.strip() for i in target_industries.split(",") if i.strip()]
    target_ind_lower = [i.lower() for i in target_industries]

    if prospect_ind and target_ind_lower:
        if any(i == prospect_ind or i in prospect_ind or prospect_ind in i for i in target_ind_lower):
            industry_pts = 20
        elif any(word in prospect_ind for i in target_ind_lower for word in i.split() if len(word) > 3):
            industry_pts = 10
    elif not target_ind_lower:
        industry_pts = 10

    # 3. Company Size Match (0-15 points)
    size_pts = 0
    prospect_size = str(getattr(prospect, "size", None) or getattr(prospect, "company_size", "") or "").lower().strip()
    target_size = str(icp.get("company_size") or "").lower().strip()

    if prospect_size and target_size:
        # Check digit overlap
        p_nums = [int(n) for n in re.findall(r"\d+", prospect_size)]
        t_nums = [int(n) for n in re.findall(r"\d+", target_size)]
        if p_nums and t_nums:
            p_val = p_nums[0]
            if len(t_nums) >= 2:
                if t_nums[0] <= p_val <= t_nums[1]:
                    size_pts = 15
                elif 0.5 * t_nums[0] <= p_val <= 2.0 * t_nums[1]:
                    size_pts = 5
                else:
                    size_pts = 0
            elif p_val >= t_nums[0]:
                size_pts = 15
            else:
                size_pts = 0
        elif any(w in target_size for w in prospect_size.split() if len(w) > 3):
            size_pts = 15
        else:
            size_pts = 0
    else:
        size_pts = 0

    # 4. Buying Signals (0-20 points)
    signals_pts = 0
    if signals:
        valid_signals = [s for s in signals if s.get("label") and s.get("label") != "none"]
        # Up to 20 pts (7 pts per active verified buying signal)
        signals_pts = min(20, len(valid_signals) * 7)

    # 5. Embedding Similarity (0-25 points)
    embedding_pts = 0
    stored_txt = getattr(prospect, "stored_text", "")
    if not stored_txt and state and hasattr(prospect, "id"):
        try:
            stored_txt = await state.get_prospect_stored_text(prospect.id)
        except Exception:
            stored_txt = ""
    prospect_text = (stored_txt or f"{prospect.company} {getattr(prospect, 'industry', '')} {prospect_size}").strip()
    icp_text = f"{product.get('name', '')} {product.get('description', '')} {icp.get('buyer_persona', '')} {' '.join(icp.get('pain_points', []) if isinstance(icp.get('pain_points'), list) else [str(icp.get('pain_points', ''))])}".strip()

    try:
        emb_prospect = await compute_embedding(prospect_text, state=state)
        emb_icp = await compute_embedding(icp_text, state=state)
        if any(emb_prospect) and any(emb_icp):
            sim = _cosine_similarity(emb_prospect, emb_icp)
        else:
            sim = _jaccard_similarity(prospect_text, icp_text)
    except Exception as e:
        logger.debug(f"Falling back to jaccard similarity for scoring: {e}")
        sim = _jaccard_similarity(prospect_text, icp_text)

    # Scale 0.0-1.0 to 0-25 points
    normalized_sim = max(0.0, min(1.0, float(sim)))
    embedding_pts = int(round(normalized_sim * 25.0))

    total = min(100, title_pts + industry_pts + size_pts + signals_pts + embedding_pts)

    return {
        "title_match": title_pts,
        "industry_match": industry_pts,
        "company_size_match": size_pts,
        "buying_signals": signals_pts,
        "embedding_similarity": embedding_pts,
        "total": total,
    }


async def score_prospect_for_product(
    prospect_id: str,
    product_id: str,
    state: StateManager,
    brain: Brain,
) -> Dict[str, Any]:
    """Score a prospect for a specific product and generate rationale."""
    prospect = await state.get_prospect(prospect_id)
    if not prospect:
        raise ValueError(f"Prospect not found: {prospect_id}")

    product = await state.get_product(product_id)
    if not product:
        product = {"id": product_id, "name": product_id, "description": ""}

    icp = await state.get_icp(product_id) or {}
    if hasattr(state, "get_prospect_signals"):
        signals = await state.get_prospect_signals(prospect_id=prospect_id, product_id=product_id)
    elif hasattr(state, "get_signals"):
        signals = await state.get_signals(prospect_id=prospect_id, product_id=product_id)
    else:
        signals = []

    breakdown = await calculate_score_breakdown(
        prospect=prospect,
        product=product,
        icp=icp,
        signals=signals,
        state=state,
    )
    total_score = breakdown["total"]

    # Generate rationale via LLM or non-LLM fallback
    mode = "llm"
    rationale = ""
    top_reasons = []
    main_risk = ""

    signal_labels = [s.get("label") for s in signals if s.get("label")]
    signals_str = ", ".join(signal_labels) if signal_labels else "none"

    size_str = getattr(prospect, "size", None) or getattr(prospect, "company_size", "") or ""
    prompt = (
        f"Score this prospect for {product.get('name', product_id)}.\n"
        f"Prospect: {prospect.company}, Title: {prospect.title}, Industry: {prospect.industry}, Size: {size_str}\n"
        f"Score: {total_score}/100 (Breakdown: Title={breakdown['title_match']}, Industry={breakdown['industry_match']}, "
        f"Size={breakdown['company_size_match']}, Signals={breakdown['buying_signals']}, Similarity={breakdown['embedding_similarity']})\n"
        f"Signals: {signals_str}\n\n"
        f"Return JSON strictly with:\n"
        f"{{\"rationale\": \"Concise rationale in under 40 words\", "
        f"\"top_reasons\": [\"Reason 1\", \"Reason 2\", \"Reason 3\"], "
        f"\"main_risk\": \"Main risk or challenge\"}}"
    )

    try:
        if hasattr(brain, "unavailable") and brain.unavailable:
            raise OllamaUnavailable(brain.unavailable)

        res = await brain.think_json(
            prompt=prompt,
            agent="scout",
            task="score_rationale",
            num_predict=300,
        )

        if isinstance(res, dict) and res.get("rationale"):
            raw_rat = str(res.get("rationale", "")).strip()
            # Enforce max 40 words
            words = raw_rat.split()
            if len(words) > 40:
                raw_rat = " ".join(words[:40])
            rationale = raw_rat
            top_reasons = res.get("top_reasons") if isinstance(res.get("top_reasons"), list) else [str(res.get("top_reasons", ""))]
            top_reasons = top_reasons[:3]
            main_risk = str(res.get("main_risk", "")).strip()
            mode = "llm"
        else:
            raise ValueError("Invalid LLM response for scoring rationale")
    except (OllamaUnavailable, Exception) as e:
        logger.info(f"Using non-LLM rationale fallback for {prospect.company}: {e}")
        mode = "non-LLM mode"
        rationale = (
            f"{prospect.company} scored {total_score}/100 via rule weights "
            f"(title={breakdown['title_match']}, industry={breakdown['industry_match']}, "
            f"signals={breakdown['buying_signals']}) and semantic similarity (non-LLM mode)."
        )
        words = rationale.split()
        if len(words) > 40:
            rationale = " ".join(words[:40])

        top_reasons = [
            f"Role match: {prospect.title or 'Relevant title'}",
            f"Industry match: {prospect.industry or 'Target market'}",
            f"Active signals: {len(signals)} detected",
        ]
        main_risk = "Rule-based score requires manual outreach validation (non-LLM mode)."

    saved = await state.save_product_score(
        product_id=product_id,
        prospect_id=prospect_id,
        score=total_score,
        breakdown=breakdown,
        rationale=rationale,
        top_reasons=top_reasons,
        main_risk=main_risk,
        mode=mode,
    )
    return saved
