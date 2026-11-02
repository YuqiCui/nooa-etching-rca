"""Audit actual journal traffic and data provenance without re-running science."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .autoresearch_agent_v2.database import create_conversation_audit_service
from .io_utils import write_json


def audit_session(run: Path) -> dict:
    session = json.loads((run / "session.json").read_text())
    cells_path = run / "execution.jsonl"
    cells = (
        [json.loads(s) for s in cells_path.read_text().splitlines()] if cells_path.exists() else []
    )
    blocks, calls, wire = {}, [], []
    for path in sorted((run / "traces").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a live journal can be between append and newline
            j = row.get("nooaJournal", {})
            if j.get("type") == "blocks":
                blocks.update({b["hash"]: b["content"] for b in j["blocks"]})
            elif j.get("type") == "call":
                calls.append(j["call"])
            for rs in row.get("resourceSpans", []):
                for scope in rs.get("scopeSpans", []):
                    for span in scope.get("spans", []):
                        attrs = {
                            a["key"]: a["value"].get("stringValue")
                            for a in span.get("attributes", [])
                        }
                        raw = attrs.get("llm.invocation_parameters")
                        if raw:
                            p = json.loads(raw)
                            wire.append(
                                {
                                    "model": p.get("model"),
                                    "output_cap": p.get("max_tokens"),
                                    "reasoning_effort": p.get("reasoning_effort"),
                                    "thinking": p.get("extra_body", {}).get("thinking"),
                                    "tool_names": [
                                        t["function"]["name"] for t in p.get("tools", [])
                                    ],
                                }
                            )
    if calls:
        system = calls[0]["input_skeleton"][0]
        (run / "system-prompt.txt").write_text(
            "".join(
                blocks.get(p.get("block_hash"), "") if "block_hash" in p else p.get("text", "")
                for p in system.get("parts", [])
            )
        )
    unchanged = all(
        hashlib.sha256((run / "input" / name).read_bytes()).hexdigest() == digest
        for name, digest in session["public_sha256"].items()
    )
    event_counts, summaries, active_event_count = create_conversation_audit_service(run).summary()
    conversation = [json.loads(s) for s in (run / "conversation.jsonl").read_text().splitlines()]
    inputs = [c for c in conversation if "message" in c]
    for i, entry in enumerate(inputs):
        end = inputs[i + 1]["time"] if i + 1 < len(inputs) else float("inf")
        candidates = [c for c in calls if entry["time"] <= c["ts_start"] < end]
        if candidates:
            first = min(candidates, key=lambda c: c["ts_start"])
            text = "".join(
                blocks.get(p.get("block_hash"), "") if "block_hash" in p else p.get("text", "")
                for p in first["input_skeleton"][0].get("parts", [])
            )
            (run / "rounds" / f"round_{entry['round']:03d}" / "system-prompt.txt").write_text(text)
    result = {
        "provider_calls": len(calls),
        "max_single_input_tokens": max((c["tokens"].get("prompt", 0) for c in calls), default=0),
        "cumulative_input_tokens": sum(c["tokens"].get("prompt", 0) for c in calls),
        "cumulative_output_tokens": sum(c["tokens"].get("completion", 0) for c in calls),
        "max_single_output_tokens": max(
            (c["tokens"].get("completion", 0) for c in calls), default=0
        ),
        "code_cells": len(cells),
        "worker_cell_errors": sum(not c["ok"] for c in cells),
        "subprocess_nonzero_returns": sum(
            c.get("kind") == "script" and c.get("returncode") not in (None, 0) for c in cells
        ),
        "data_export_unchanged": unchanged,
        "completed_rounds": sum(c["speaker"] == "agent" for c in conversation),
        "expert_feedback_turns": sum(c["speaker"] == "simulated_expert" for c in conversation),
        "event_counts": event_counts,
        "active_event_count": active_event_count,
        "native_summaries": [
            {
                "tag": s["tag"],
                "status": s["status"],
                "children_preserved": len(s["children_tags"]),
                "summary_chars": len(s["summary_text"]),
            }
            for s in summaries
        ],
        "wire_settings_variants": list({json.dumps(w, sort_keys=True): w for w in wire}.values()),
        "discovery_target_runs": session["discovery_target_runs"],
        "discovery_control_runs": session["discovery_control_runs"],
    }
    write_json(run / "audit.json", result)
    return result
