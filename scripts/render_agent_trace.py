"""将带时间戳的 Agent 审计轨迹导出为离线并行时间线。"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ASSETS = Path(__file__).with_name("agent_trace_viewer")


def read_json(path: Path, default=None):
    """读取已有快照；文件未保存时返回明确的缺省值。"""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_events(path: Path, warnings: list[str]) -> list[dict]:
    """读取 JSONL，允许仍在写入的最后一行暂不完整。"""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if number != len(lines):
                raise ValueError(f"{path}:{number} 不是有效 JSON") from exc
            warnings.append(f"{path.name} 最后一行未完成，已跳过。")
    return result


def timestamp(value: str) -> float:
    """将带时区的 ISO 时间转换为毫秒，拒绝猜测无时区时间。"""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"时间缺少时区：{value}")
    return parsed.timestamp() * 1000


def end_time(start: float, event: dict) -> float:
    """优先使用起始时间加耗时，兼容旧日志在结束事件中重复起始时间。"""
    duration = event.get("elapsed_s")
    if isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0:
        return start + duration * 1000
    return max(start, timestamp(event["at"]))


class BlobStore:
    """去重保存完整消息和结果，保持每次请求的消息顺序与内容。"""

    def __init__(self):
        self.values: list[str] = []
        self._indices: dict[str, int] = {}

    def put(self, value) -> int:
        """将相同 JSON 值保存一次，并返回可复用索引。"""
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if encoded not in self._indices:
            self._indices[encoded] = len(self.values)
            self.values.append(encoded)
        return self._indices[encoded]


def outcome(result, feedback=None) -> str:
    """区分接口失败、输出中出现异常和正常返回，不把返回零当作语义成功。"""
    if isinstance(feedback, dict) and feedback.get("error"):
        return "error"
    if isinstance(result, dict):
        if result.get("error") or result.get("ok") is False:
            return "error"
        if result.get("returncode") not in (None, 0):
            return "error"
        if isinstance(result.get("result"), dict) and result["result"].get("outcome") in {
            "failed", "blocked", "cancelled",
        }:
            return "error"
        output = str(result.get("stdout", "")) + str(result.get("stderr", ""))
        if "Traceback (most recent call last)" in output or "SyntaxError:" in output:
            return "warning"
    if isinstance(result, str) and result.startswith(("TypeError:", "ValueError:", "Error:")):
        return "error"
    return "success"


def session_metadata(root: Path) -> tuple[dict, dict]:
    """只读提取会话身份和交付校验回执，不将最终会话冒充历史请求快照。"""
    actors, feedback = {}, {}
    for folder in [root / "case/.autoresearch", root / ".autoresearch"]:
        if not folder.is_dir():
            continue
        for path in folder.rglob("session.json"):
            state = read_json(path, {})
            if state.get("agent_id"):
                actors[state["agent_id"]] = state.get("parent_agent_id")
            for message in state.get("messages", []):
                content = message.get("content")
                if message.get("role") != "tool" or not isinstance(content, str):
                    continue
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and parsed.get("error"):
                    feedback[(state.get("agent_id"), message.get("tool_call_id"))] = parsed
    return actors, feedback


def build_trace(source: Path, *, title: str | None = None,
                timezone: str = "Asia/Shanghai") -> dict:
    """从任意运行目录或 audit 目录重建共享时钟和跨会话交接关系。"""
    ZoneInfo(timezone)
    source = source.expanduser().resolve()
    audit = source / "audit" if (source / "audit").is_dir() else source
    root = audit.parent if audit.name == "audit" else source
    warnings: list[str] = []
    models = read_events(audit / "model_events.jsonl", warnings)
    tools = read_events(audit / "tool_events.jsonl", warnings)
    if not models and not tools:
        raise ValueError("未找到带时间的 model_events.jsonl 或 tool_events.jsonl；"
                         "仅 session.json 无法还原真实时序。")
    blobs = BlobStore()
    parents, feedback = session_metadata(root)
    actors: dict[str, dict] = {}
    events: list[dict] = []
    links: list[dict] = []

    def actor(identity, at, parent=None):
        """登记首次观察时间；只有实际启动回执才设置创建时间。"""
        if identity not in actors:
            actors[identity] = {"id": identity, "first": at, "created": None,
                                "parent": parents.get(identity, parent)}
        actors[identity]["first"] = min(actors[identity]["first"], at)
        if parent is not None:
            actors[identity]["parent"] = parent

    def add(kind, identity, at, label, **fields):
        """添加完整事件，展示用的摘要不替代原始内容。"""
        actor(identity, at)
        event = {"id": f"e{len(events)}", "kind": kind, "actor": identity,
                 "start": at, "end": None, "title": label, "status": "pending", **fields}
        events.append(event)
        return event

    model_ends = {(row["agent_id"], row["request"]): row for row in models
                  if row["event"] in {"end", "error"}}
    for row in models:
        if row["event"] != "start":
            continue
        identity, number, at = row["agent_id"], row["request"], timestamp(row["at"])
        actor(identity, at, row.get("parent_agent_id"))
        stem = f"{int(number):04d}-{identity}"
        request = read_json(audit / f"{stem}-request.json")
        response = read_json(audit / f"{stem}-response.json")
        message_ids = [blobs.put(m) for m in request.get("messages", [])] if request else []
        system = " ".join(str(m.get("content", "")) for m in request.get("messages", [])
                          if m.get("role") == "system") if request else ""
        summary = "上下文压缩" in system or "context compaction" in system.lower()
        label = f"{'上下文摘要' if summary else '模型请求'} #{number}"
        event = add("model", identity, at, label, request=number,
                    assignment=row.get("assignment_id"), messages=message_ids,
                    tools=blobs.put(request.get("tools")) if request else None,
                    response=blobs.put(response) if response is not None else None,
                    snapshot=bool(request), input=blobs.put(row))
        if ending := model_ends.get((identity, number)):
            event.update(end=end_time(at, ending),
                         status="error" if ending["event"] == "error" else "success",
                         output=blobs.put(ending))

    pending: dict[tuple, deque] = defaultdict(deque)
    for row in tools:
        identity, at = row["agent_id"], timestamp(row["at"])
        key = (identity, row.get("assignment_id"), row.get("call_id"), row["name"])
        if row["event"] == "start":
            event = add("tool", identity, at, row["name"],
                        assignment=row.get("assignment_id"), input=blobs.put(row.get("arguments")))
            pending[key].append(event)
            continue
        if row["event"] not in {"end", "error", "rejected"}:
            continue
        event = pending[key].popleft() if pending[key] else add(
            "tool", identity, at, row["name"], assignment=row.get("assignment_id"),
            input=blobs.put(row.get("arguments")))
        result = row.get("result", {"error": row.get("error_type", "执行异常")})
        validation = feedback.get((identity, row.get("call_id")))
        event.update(end=end_time(event["start"], row), output=blobs.put(result),
                     status=outcome(result, validation))
        if row["event"] in {"error", "rejected"}:
            event["status"] = "error"
        if validation:
            event["feedback"] = blobs.put(validation)

    assignments = {}
    tool_calls = [event for event in events if event["kind"] == "tool" and "output" in event]
    for event in tool_calls:
        result = json.loads(blobs.values[event["output"]])
        if not isinstance(result, dict) or not result.get("ok"):
            continue
        worker = result.get("worker_id")
        if not worker:
            continue
        at = event["end"]
        if event["title"] == "StartWorker":
            actor(worker, at, event["actor"])
            actors[worker]["created"] = at
            child = add("lifecycle", worker, at, "Worker 启动", status="success",
                        end=at, output=event["output"])
            links.append({"from": event["id"], "to": child["id"], "label": "启动"})
        elif event["title"] == "AssignTask":
            actor(worker, at, event["actor"])
            args = json.loads(blobs.values[event["input"]]) or {}
            assignment = result.get("assignment_id")
            child = add("assignment", worker, at, f"任务 {len(assignments) + 1}",
                        assignment=assignment, input=blobs.put(args), output=event["output"])
            assignments[assignment] = child
            links.append({"from": event["id"], "to": child["id"], "label": "派发"})

    for event in tool_calls:
        if event["title"] != "CollectWorkerResult":
            continue
        result = json.loads(blobs.values[event["output"]]) if "output" in event else {}
        if not isinstance(result, dict) or not result.get("ok") or result.get("replayed"):
            continue
        task = assignments.get(result.get("assignment_id"))
        if not task:
            continue
        terminals = [candidate for candidate in events if candidate.get("end") is not None
                     and candidate.get("assignment") == task["assignment"]
                     and ((candidate["kind"] == "tool" and candidate["title"] == "finish"
                           and candidate["status"] == "success")
                          or (candidate["kind"] == "model" and candidate["status"] == "error"))]
        ready = max(candidate["end"] for candidate in terminals) if terminals else event["start"]
        task.update(end=max(task["start"], ready), status=outcome(result),
                    result=blobs.put(result.get("result")), end_inferred=not bool(terminals))
        child = add("lifecycle", task["actor"], event["end"], "结果已收取",
                    status=task["status"], end=event["end"], assignment=task["assignment"],
                    output=event["output"])
        links.append({"from": child["id"], "to": event["id"], "label": "收取"})

    if not events:
        raise ValueError("日志没有事件开始记录，无法还原真实时序。")
    for identity, parent in parents.items():
        if identity not in actors:
            actors[identity] = {"id": identity, "first": None, "created": None, "parent": parent}
    events.sort(key=lambda event: (event["start"], event["id"]))
    ordered = sorted(actors.values(), key=lambda item: (
        item["parent"] is not None, item["created"] or item["first"] or math.inf, item["id"]))
    worker_count = router_count = 0
    for item in ordered:
        if item["parent"] is None:
            router_count += 1
            item["name"] = "Router" if router_count == 1 else f"Router {router_count}"
        else:
            worker_count += 1
            item["name"] = f"Worker {worker_count}"
    status = read_json(root / "status.json", {})
    start = min(event["start"] for event in events)
    end = max(event["end"] or event["start"] for event in events)
    if status.get("started_at"):
        start = min(start, timestamp(status["started_at"]))
    if status.get("finished_at"):
        end = max(end, timestamp(status["finished_at"]))
    missing = sum(event["kind"] == "model" and not event["snapshot"] for event in events)
    if missing:
        warnings.append(f"{missing} 次模型请求未保存请求快照，不能展示其原始 prompt。")
    return {"title": title or f"{root.name} · Agent 轨迹", "timezone": timezone,
            "start": start, "end": end, "status": status.get("status", "unknown"),
            "actors": ordered, "events": events, "links": links,
            "blobs": blobs.values, "warnings": warnings}


def render_html(trace: dict) -> str:
    """内嵌全部资源；原始消息只作为 JSON 数据，不允许注入 HTML 或脚本。"""
    data = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = (ASSETS / "index.html").read_text(encoding="utf-8")
    return (template.replace("__TITLE__", html.escape(trace["title"]))
            .replace("__STYLE__", (ASSETS / "viewer.css").read_text(encoding="utf-8"))
            .replace("__SCRIPT__", (ASSETS / "viewer.js").read_text(encoding="utf-8"))
            .replace("__TRACE_DATA__", data))


def main(argv=None) -> int:
    """执行只读导出，只写入目标 HTML 文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="包含 audit 的运行目录，或 audit 目录本身")
    parser.add_argument("-o", "--output", type=Path, help="输出的单文件 HTML 路径")
    parser.add_argument("--title", help="页面标题")
    parser.add_argument("--timezone", default="Asia/Shanghai", help="显示时区，默认 Asia/Shanghai")
    args = parser.parse_args(argv)
    try:
        trace = build_trace(args.source, title=args.title, timezone=args.timezone)
        output = args.output or args.source / "agent-trace.html"
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_html(trace), encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"导出失败：{exc}\n")
    print(f"已生成 {output}\n{len(trace['actors'])} 个会话，{len(trace['events'])} 个事件")
    for warning in trace["warnings"]:
        print(f"提示：{warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
