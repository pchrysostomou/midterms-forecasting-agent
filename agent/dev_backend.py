"""DEVELOPMENT ONLY: send model calls to the Anthropic API with your own key.

Used by `agent.py --dev` to test plumbing, prompts and stability without spending the
event's OpenRouter credit. It never touches the event's run log (.ftf/) and agent.py
refuses to submit in dev mode. The judged run must use ftf.ask() through OpenRouter.

Setup:  pip install anthropic   and   export ANTHROPIC_API_KEY=...
Optional: DEV_MODEL (default claude-sonnet-5-5), DEV_WEB_SEARCH_TOOL (default web_search_20250305),
DEV_EFFORT (default low, like the event run's reasoning effort)

temperature is not forwarded: current Claude models and the anthropic 2.x SDK reject it, so dev
reruns sample with the model's defaults, as Claude does in the event run (OpenRouter lists temperature only
for Gemini 4 Argon among our models).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

MODEL = os.environ.get("DEV_MODEL", "claude-sonnet-5-5")
WEB_TOOL = os.environ.get("DEV_WEB_SEARCH_TOOL", "web_search_20250305")
EFFORT = os.environ.get("DEV_EFFORT", "low")  # thinking counts against max_tokens, so keep it low
MAX_RESUMES = 3  # a long web-search turn can pause (stop_reason pause_turn) and must be resumed
LOG = Path(__file__).resolve().parent.parent / "out" / "dev_calls.jsonl"
_client, _lock = None, threading.Lock()
COST = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "searches": 0}


def _anthropic():
    global _client
    with _lock:
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic(max_retries=3)
        return _client


def ask(question_id, role, messages, model, run, online=False, **kwargs):
    """Same signature as ftf.ask (the event model name is recorded but MODEL is used)."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    params = {"model": MODEL, "max_tokens": max(kwargs.get("max_tokens", 2500), 1500), "messages": messages,
              "output_config": {"effort": EFFORT}}
    if online:
        params["tools"] = [{"type": WEB_TOOL, "name": "web_search", "max_uses": 5}]
    blocks, n_in, n_out = [], 0, 0
    for _ in range(MAX_RESUMES + 1):
        resp = _anthropic().messages.create(**params)
        blocks += resp.content
        n_in += resp.usage.input_tokens
        n_out += resp.usage.output_tokens
        if resp.stop_reason != "pause_turn":
            break
        params["messages"] = messages + [{"role": "assistant", "content": blocks}]
    if resp.stop_reason in ("max_tokens", "refusal", "pause_turn"):
        print(f"[dev] WARNING {question_id} {role}: stop_reason={resp.stop_reason}, the reply may be cut off")
    text = "".join(b.text for b in blocks if getattr(b, "type", "") == "text")
    searches = sum(1 for b in blocks if getattr(b, "type", "") == "server_tool_use")
    with _lock:
        COST["calls"] += 1
        COST["input_tokens"] += n_in
        COST["output_tokens"] += n_out
        COST["searches"] += searches
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "run": str(run), "question": question_id, "role": role,
                                "event_model": model, "dev_model": MODEL, "online": online,
                                "searches": searches, "in": n_in, "out": n_out,
                                "stop_reason": resp.stop_reason}) + "\n")
    print(f"[dev] {question_id} {role} as {model.split('/')[-1]} -> {MODEL} "
          f"in={n_in} out={n_out} searches={searches} stop={resp.stop_reason}")
    return text
