"""Offline research service from stored prospect data.

Uses sentence-transformers (all-MiniLM-L6-v2) to embed and retrieve the top-k
relevant snippets from stored text (no HTTP/network calls), then queries the Brain
for structured insights: {summary, key_facts[], recent_events[], gaps[]}.
Caches snippet and query embeddings in SQLite embedding_cache.
"""

import hashlib
import json
import logging
import math
import re
from typing import Any, List, Optional
import numpy as np

from sellbird.brain import Brain
from sellbird.config import BrainConfig
from sellbird.ollama_runtime import OllamaUnavailable

logger = logging.getLogger("sellbird.stored_research")

_MODEL_INSTANCE = None


def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Get or initialize the sentence-transformers model."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL_INSTANCE = SentenceTransformer(model_name)
        except Exception as e:
            logger.warning("Could not load SentenceTransformer(%s): %s", model_name, e)
            return None
    return _MODEL_INSTANCE


def _hash_text(text: str, model_name: str) -> str:
    """Generate SHA256 key for embedding cache."""
    return hashlib.sha256(f"{model_name}:{text.strip()}".encode("utf-8")).hexdigest()


async def compute_embedding(text: str, state=None, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    """Compute embedding for a snippet, checking and updating SQLite cache."""
    if not text.strip():
        return [0.0] * 384

    h = _hash_text(text, model_name)
    if state and hasattr(state, "get_cached_embedding"):
        cached = await state.get_cached_embedding(h, model=model_name)
        if cached is not None:
            return cached

    model = get_sentence_transformer(model_name)
    if model is not None:
        emb = model.encode(text)
        if hasattr(emb, "tolist"):
            emb_list = emb.tolist()
        else:
            emb_list = list(emb)
    else:
        # Deterministic offline fallback embedding vector (384 dimensions)
        h_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(b / 255.0) - 0.5 for b in h_bytes]
        # Repeat to 384 dimensions
        expanded = (raw * (384 // len(raw) + 1))[:384]
        norm = math.sqrt(sum(x * x for x in expanded)) or 1.0
        emb_list = [x / norm for x in expanded]

    if state and hasattr(state, "set_cached_embedding"):
        await state.set_cached_embedding(h, emb_list, model=model_name)

    return emb_list


def chunk_stored_text(stored_text: str) -> list[str]:
    """Break stored prospect text into meaningful sentence/bullet snippets."""
    snippets = []
    lines = stored_text.splitlines()
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        # Remove bullet prefix
        cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
        if len(cleaned) < 15:
            continue
        if cleaned.upper() == "[SAMPLE DATA]" or cleaned.upper() == "SAMPLE DATA":
            continue
        # If line contains multiple sentences, split
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) >= 20 and s_clean not in snippets:
                snippets.append(s_clean)
    if not snippets and stored_text.strip():
        snippets = [stored_text.strip()]
    return snippets


async def retrieve_top_k_snippets(
    snippets: list[str],
    query: str = "company business model operations recent news growth challenges pain points",
    k: int = 5,
    state=None,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[str]:
    """Retrieve the top-k most relevant snippets for a query using embeddings."""
    if len(snippets) <= k:
        return snippets

    q_emb = np.array(await compute_embedding(query, state=state, model_name=model_name))
    q_norm = np.linalg.norm(q_emb) or 1.0

    scored = []
    for s in snippets:
        s_emb = np.array(await compute_embedding(s, state=state, model_name=model_name))
        s_norm = np.linalg.norm(s_emb) or 1.0
        sim = float(np.dot(q_emb, s_emb) / (q_norm * s_norm))
        scored.append((sim, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:k]]


async def build_research_summary(
    prospect_id: str,
    state,
    brain: Optional[Brain] = None,
    k: int = 5,
    query: str = "company business model operations recent news growth challenges pain points",
) -> dict:
    """Take a prospect already in DB and build research summary using ONLY stored text (no HTTP)."""
    stored_text = await state.get_prospect_stored_text(prospect_id)
    if not stored_text:
        prospect = await state.get_prospect(prospect_id)
        if not prospect:
            raise ValueError(f"Prospect {prospect_id} not found")
        stored_text = f"Company: {prospect.get('company', '')}\nIndustry: {prospect.get('industry', '')}"

    snippets = chunk_stored_text(stored_text)
    top_snippets = await retrieve_top_k_snippets(snippets, query=query, k=k, state=state)

    if brain is None:
        try:
            from sellbird.config import load_config
            cfg = load_config()
            brain = Brain(state, cfg.brain)
        except Exception:
            brain = Brain(state)

    full_prompt = (
        "You are a sales intelligence researcher. Analyze the following verified snippets from stored company text. "
        "Return a JSON object with strictly these keys:\n"
        "- summary: string (2-3 concise sentences summarizing company operations and business)\n"
        "- key_facts: list of strings (concrete operational facts)\n"
        "- recent_events: list of strings (recent dated news, milestones, or leadership changes)\n"
        "- gaps: list of strings (missing information or questions to investigate)\n"
        "Do NOT invent facts not present in the snippets.\n\n"
        f"Stored Snippets:\n"
        + "\n".join(f"- {s}" for s in top_snippets)
        + "\n\nProvide the research summary JSON:"
    )

    mode = "llm"
    try:
        if hasattr(brain, "think_json"):
            data = await brain.think_json(full_prompt, agent="researcher", task="stored_research")
        elif hasattr(brain, "ask"):
            raw_res = await brain.ask(system="", user=full_prompt, format="json", agent="researcher", task="stored_research")
            data = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
        elif hasattr(brain, "think"):
            raw_res = await brain.think(full_prompt, expect_json=True, agent="researcher", task="stored_research")
            data = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
        else:
            raise AttributeError("Brain does not support think_json or ask")

        if not data or not isinstance(data, dict):
            raise ValueError(f"Invalid brain response: {data}")

        summary = str(data.get("summary") or "").strip()
        key_facts = data.get("key_facts") or []
        if isinstance(key_facts, str):
            key_facts = [key_facts]
        recent_events = data.get("recent_events") or []
        if isinstance(recent_events, str):
            recent_events = [recent_events]
        gaps = data.get("gaps") or []
        if isinstance(gaps, str):
            gaps = [gaps]

        if not summary:
            raise ValueError("Brain returned empty summary")

    except (OllamaUnavailable, Exception) as e:
        logger.info("Brain unavailable for research summary (%s), falling back to non-LLM mode", e)
        mode = "non-LLM mode"
        # Deterministic non-LLM extraction from stored snippets
        comp_match = re.search(r"Company:\s*([^\n\[]+)", stored_text)
        ind_match = re.search(r"Industry:\s*([^\n]+)", stored_text)
        company = comp_match.group(1).strip() if comp_match else "Company"
        industry = ind_match.group(1).strip() if ind_match else "Business Services"

        summary = (
            f"{company} operates in {industry}. Key business capabilities derived from stored profiles. (non-LLM mode)"
        )
        key_facts = [s for s in top_snippets if "recent news" not in s.lower() and "sample data" not in s.lower()][:3]
        recent_events = [s for s in top_snippets if any(yr in s for yr in ("2024", "2025", "2026", "2027"))]
        if not recent_events:
            recent_events = top_snippets[:2]
        gaps = ["Pricing structure details", "Contract renewal dates", "Current software tooling stack"]

    await state.save_prospect_research(
        prospect_id=prospect_id,
        summary=summary,
        key_facts=key_facts,
        recent_events=recent_events,
        gaps=gaps,
    )

    return {
        "prospect_id": prospect_id,
        "summary": summary,
        "key_facts": key_facts,
        "recent_events": recent_events,
        "gaps": gaps,
        "snippets_used": top_snippets,
        "mode": mode,
    }
