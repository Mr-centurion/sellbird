"""Task queue processor for draining and retrying inference tasks queued while Ollama was unavailable."""

import json
import logging
from typing import Any, Optional

from sellbird.brain import Brain
from sellbird.config import load_config
from sellbird.ollama_runtime import OllamaClient, OllamaUnavailable
from sellbird.state import StateManager

logger = logging.getLogger("sellbird.task_queue")


async def drain_task_queue(
    state: StateManager,
    brain: Optional[Brain] = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Drain queued tasks if Ollama is reachable.
    
    Returns a status dict:
      {
        "success": bool,
        "message": str,
        "drained": int,
        "failed": int,
        "remaining": int,
      }
    """
    config = load_config()
    brain_cfg = getattr(config, "brain", None)
    base_url = getattr(brain_cfg, "ollama_url", "http://localhost:11434")
    model = getattr(brain_cfg, "model", "qwen3:1.7b")

    client = OllamaClient(base_url=base_url, model=model)
    reason = await client.preflight()
    if reason:
        remaining = await state.count_queued_tasks()
        return {
            "success": False,
            "message": f"Ollama is unreachable: {reason}",
            "drained": 0,
            "failed": 0,
            "remaining": remaining,
        }

    if brain is None:
        brain = Brain(state, brain_cfg)

    tasks = await state.get_queued_tasks(limit=limit)
    if not tasks:
        return {
            "success": True,
            "message": "Task queue is empty.",
            "drained": 0,
            "failed": 0,
            "remaining": 0,
        }

    drained = 0
    failed = 0

    for task in tasks:
        task_id = task["id"]
        attempts = int(task.get("attempts", 0) or 0) + 1
        try:
            payload = json.loads(task["payload"]) if isinstance(task["payload"], str) else task["payload"]
        except Exception as e:
            await state.update_task_status(task_id, "failed", attempts=attempts, error=f"Bad payload: {e}")
            failed += 1
            continue

        prompt = payload.get("prompt", "")
        agent = payload.get("agent", "")
        task_name = payload.get("task", "")
        expect_json = bool(payload.get("expect_json", False))

        try:
            if expect_json:
                result = await brain.think_json(prompt, agent=agent, task=task_name)
            else:
                result = await brain.think(prompt, agent=agent, task=task_name)

            if result:
                await state.update_task_status(task_id, "completed", attempts=attempts)
                drained += 1
            else:
                if attempts >= 3:
                    await state.update_task_status(task_id, "failed", attempts=attempts, error="Empty response after 3 attempts")
                    failed += 1
                else:
                    await state.update_task_status(task_id, "queued", attempts=attempts, error="Empty response, will retry")
        except OllamaUnavailable as e:
            # Ollama went down during draining; stop processing
            await state.update_task_status(task_id, "queued", attempts=attempts, error=str(e))
            remaining = await state.count_queued_tasks()
            return {
                "success": False,
                "message": f"Ollama became unavailable during drain: {e}",
                "drained": drained,
                "failed": failed,
                "remaining": remaining,
            }
        except Exception as e:
            if attempts >= 3:
                await state.update_task_status(task_id, "failed", attempts=attempts, error=str(e))
                failed += 1
            else:
                await state.update_task_status(task_id, "queued", attempts=attempts, error=str(e))

    remaining = await state.count_queued_tasks()
    return {
        "success": True,
        "message": f"Drained {drained} task(s).",
        "drained": drained,
        "failed": failed,
        "remaining": remaining,
    }
