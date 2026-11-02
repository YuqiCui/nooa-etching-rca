"""Compare saved NOOA and Claude Code trajectories without another model call."""

import argparse
import json
from pathlib import Path

from factor_demo.io_utils import write_json


def _records(path):
    """Read complete journal lines, tolerating the last line of an active stream."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _baseline_metrics(run):
    """Extract authoritative completed-response usage and scientific outcomes."""
    spans = {}
    for path in (run / "traces").glob("*.jsonl"):
        for record in _records(path):
            for resource in record.get("resourceSpans", []):
                for scope in resource.get("scopeSpans", []):
                    for span in scope["spans"]:
                        attrs = {
                            item["key"]: next(iter(item["value"].values()))
                            for item in span.get("attributes", [])
                        }
                        if "llm.token_count.completion" in attrs:
                            spans[span["spanId"]] = (span, attrs)
    cells = _records(run / "execution.jsonl")[1:]
    times = [
        (int(span["startTimeUnixNano"]) / 1e9, int(span["endTimeUnixNano"]) / 1e9)
        for span, _ in spans.values()
    ]
    return {
        "framework": "NOOA CodeActV2",
        "model": json.loads((run / "provider.json").read_text())["model"],
        "status": json.loads((run / "status.json").read_text())["status"],
        "completed_model_responses": len(spans),
        "output_tokens": sum(int(a["llm.token_count.completion"]) for _, a in spans.values()),
        "total_tokens": sum(int(a["llm.token_count.total"]) for _, a in spans.values()),
        "reasoning_tokens": sum(
            int(a.get("llm.token_count.completion_details.reasoning", 0)) for _, a in spans.values()
        ),
        "scientific_executions": len(cells),
        "failed_scientific_executions": sum(not cell["ok"] for cell in cells),
        "scientific_elapsed_s": sum(cell["elapsed_s"] for cell in cells),
        "observed_model_span_wall_s": max(t[1] for t in times) - min(t[0] for t in times),
        "final_response_present": (run / "latest_reply.json").exists(),
        "run": str(run),
    }


def _claude_metrics(run):
    """Exclude all probes and count only completed formal responses and tools."""
    directory = run / "formal"
    usages = [row["usage"] for row in _records(directory / "upstream-usage.jsonl")]
    tools = _records(directory / "tool-calls.jsonl")
    execution_refs = {
        row["result"]["execution_ref"]
        for row in tools
        if isinstance(row.get("result"), dict) and row["result"].get("execution_ref")
    }
    cells = [
        row
        for row in _records(run / "execution.jsonl")
        if row.get("execution_ref") in execution_refs
    ]
    manifest = json.loads((run / "manifest.json").read_text())
    scientific_tools = [
        row for row in tools if row["tool"] in {"run_code", "code_run", "run_script", "write_file"}
    ]
    archived_refs = {row.get("execution_ref") for row in cells}
    unarchived_tools = [
        row
        for row in scientific_tools
        if (row.get("result") or {}).get("execution_ref") not in archived_refs
    ]
    summary = (
        json.loads((directory / "summary.json").read_text())
        if (directory / "summary.json").exists()
        else {}
    )
    files = sorted(
        str(path.relative_to(run / "workspace"))
        for path in (run / "workspace/artifacts").rglob("*")
        if path.is_file()
    )
    terminal = summary.get("result") or {}
    terminal_reason = terminal.get("terminal_reason", manifest["status"])
    if "Experiment budget exhausted" in terminal.get("result", ""):
        terminal_reason = "local_experiment_budget_exhausted"
    return {
        "framework": "Claude Code",
        "model": manifest["model"],
        "status": manifest["status"],
        "terminal_reason": terminal_reason,
        "completed_model_responses": len(usages),
        "output_tokens": sum(usage.get("completion_tokens", 0) for usage in usages),
        "total_tokens": sum(usage.get("total_tokens", 0) for usage in usages),
        "reasoning_tokens": sum(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            for usage in usages
        ),
        "business_tool_calls": len(tools),
        "failed_business_tool_calls": sum(not row["ok"] for row in tools),
        "scientific_executions": len(scientific_tools),
        "failed_scientific_executions": sum(not row["ok"] for row in scientific_tools),
        "scientific_elapsed_s": sum(cell["elapsed_s"] for cell in cells)
        + sum(row["elapsed_s"] for row in unarchived_tools),
        "unreferenced_scientific_tool_results": len(unarchived_tools),
        "formal_wall_s": summary.get("elapsed_s"),
        "final_response_present": (run / "final-response.md").exists(),
        "artifacts": files,
        "artifact_correctness": "Requires independent review of the saved tables and code",
        "run": str(run),
    }


def main() -> None:
    """Write a local JSON comparison of one baseline and one CLI experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "One task, one attempt; probe and compatibility attempts excluded",
        "token_semantics": "Output tokens include reasoning tokens; do not add the two counts",
        "comparability": "Framework, system prompt, context management and tool protocol differ",
        "experiments": [
            _baseline_metrics(args.baseline.resolve()),
            _claude_metrics(args.run.resolve()),
        ],
    }
    write_json(args.run / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
