"""验证离线轨迹导出的真实时序、请求快照和内容隔离。"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_agent_trace.py"
SPEC = importlib.util.spec_from_file_location("render_agent_trace", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def at(seconds):
    """构造确定的 UTC 时间。"""
    return f"2026-11-17T00:00:{seconds:02d}+00:00"


def write_events(folder, name, rows):
    """保存与真实审计记录相同的 JSONL。"""
    (folder / name).write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def tool(name, start, duration, arguments, result, *, actor="main", assignment=None, call="call"):
    """复现旧审计格式：结束事件仍带开始时间，耗时另存。"""
    common = {"at": at(start), "agent_id": actor, "assignment_id": assignment,
              "call_id": call, "name": name, "arguments": arguments}
    return [{**common, "event": "start"},
            {**common, "event": "end", "elapsed_s": duration, "result": result}]


def fixture_run(tmp_path):
    """建立两个并行会话及一个尚无任务的 Worker。"""
    audit = tmp_path / "audit"
    audit.mkdir()
    models = []
    for number, actor, parent, start, end in [
        (1, "main", None, 0, 1), (2, "w1", "main", 20, 40), (3, "main", None, 21, 25),
    ]:
        common = {"request": number, "agent_id": actor}
        models.extend([{**common, "event": "start", "at": at(start),
                        "parent_agent_id": parent, "assignment_id": "a1" if parent else None},
                       {**common, "event": "end", "at": at(end), "elapsed_s": end - start}])
        request = {"messages": [{"role": "system", "content": "共同 system"},
                                {"role": "user", "content": "</script><img onerror=alert(1)>"}],
                   "tools": []}
        (audit / f"{number:04d}-{actor}-request.json").write_text(json.dumps(request))
        (audit / f"{number:04d}-{actor}-response.json").write_text(json.dumps(
            {"content": "合法文字响应", "tool_calls": None}))
    write_events(audit, "model_events.jsonl", models)
    tools = tool("StartWorker", 1, 9, {}, {"ok": True, "worker_id": "w1"}, call="start1")
    tools += tool("StartWorker", 12, 1, {}, {"ok": True, "worker_id": "w2"}, call="start2")
    tools += tool("AssignTask", 18, .5, {"task": "检查数据", "deliver_list": ["结果"]},
                  {"ok": True, "worker_id": "w1", "assignment_id": "a1"}, call="assign")
    tools += tool("finish", 40, 1, {}, "交付", actor="w1", assignment="a1", call="finish")
    tools += tool("CollectWorkerResult", 45, 2, {}, {
        "ok": True, "worker_id": "w1", "assignment_id": "a1",
        "result": {"outcome": "completed", "summary": "已完成"}}, call="collect")
    write_events(audit, "tool_events.jsonl", tools)
    return tmp_path


def test_parallel_lanes_keep_inactive_time_empty_and_recover_tool_end(tmp_path):
    trace = renderer.build_trace(fixture_run(tmp_path))
    actors = {actor["id"]: actor for actor in trace["actors"]}
    assert [actor["name"] for actor in trace["actors"]] == ["Router", "Worker 1", "Worker 2"]
    assert actors["w1"]["created"] == renderer.timestamp(at(10))
    assert all(event["start"] >= actors["w1"]["created"] for event in trace["events"]
               if event["actor"] == "w1")
    assert [event["title"] for event in trace["events"] if event["actor"] == "w2"] == [
        "Worker 启动"]
    requests = {event["request"]: event for event in trace["events"] if event["kind"] == "model"}
    assert requests[2]["start"] < requests[3]["start"] < requests[3]["end"] < requests[2]["end"]
    task = next(event for event in trace["events"] if event["kind"] == "assignment")
    assert task["start"] == renderer.timestamp(at(18)) + 500
    assert task["end"] == renderer.timestamp(at(41))
    assert task["status"] == "success"
    assert len(trace["links"]) == 4
    collected = next(event for event in trace["events"] if event["title"] == "结果已收取")
    assert collected["start"] == renderer.timestamp(at(47))


def test_prompt_snapshots_are_exact_deduplicated_and_safe_to_embed(tmp_path):
    trace = renderer.build_trace(fixture_run(tmp_path))
    requests = [event for event in trace["events"] if event["kind"] == "model"]
    assert requests[0]["messages"] == requests[1]["messages"] == requests[2]["messages"]
    message = json.loads(trace["blobs"][requests[0]["messages"][1]])
    assert message["content"] == "</script><img onerror=alert(1)>"
    page = renderer.render_html(trace)
    data = page.split('<script id="trace-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "</script>" not in data
    assert "<img" not in data
    assert json.loads(data) == trace
    assert page.count("</script>") == 2


def test_reused_tool_call_ids_are_matched_per_assignment(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    rows = tool("read_file", 1, 2, {}, "first", actor="w1", assignment="a1")
    rows += tool("read_file", 10, 3, {}, "second", actor="w1", assignment="a2")
    write_events(audit, "tool_events.jsonl", rows)
    trace = renderer.build_trace(tmp_path)
    assert [json.loads(trace["blobs"][event["output"]]) for event in trace["events"]] == [
        "first", "second"]
    assert [event["end"] - event["start"] for event in trace["events"]] == [2000, 3000]


def test_request_numbers_are_scoped_to_each_agent(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    rows = []
    for actor, start, end in [("main", 1, 3), ("w1", 2, 10)]:
        rows.extend([{"event": "start", "request": 1, "agent_id": actor, "at": at(start)},
                     {"event": "end", "request": 1, "agent_id": actor, "at": at(end)}])
    write_events(audit, "model_events.jsonl", rows)
    trace = renderer.build_trace(tmp_path)
    assert [event["end"] for event in trace["events"]] == [
        renderer.timestamp(at(3)), renderer.timestamp(at(10))]


def test_worker_without_any_recorded_activity_keeps_an_empty_lane(tmp_path):
    root = fixture_run(tmp_path)
    session = root / ".autoresearch/children/unused/session.json"
    session.parent.mkdir(parents=True)
    session.write_text(json.dumps({"agent_id": "unused", "parent_agent_id": "main",
                                   "messages": []}))
    trace = renderer.build_trace(root)
    assert any(actor["id"] == "unused" for actor in trace["actors"])
    assert not any(event["actor"] == "unused" for event in trace["events"])


def test_missing_snapshot_is_not_replaced_with_final_session(tmp_path):
    root = fixture_run(tmp_path)
    (root / "audit/0002-w1-request.json").unlink()
    trace = renderer.build_trace(root / "audit")
    request = next(event for event in trace["events"] if event.get("request") == 2)
    assert request["snapshot"] is False and request["messages"] == []
    assert trace["warnings"]


def test_cli_writes_standalone_html(tmp_path):
    root = fixture_run(tmp_path)
    output = root / "output/trace.html"
    assert renderer.main([str(root), "-o", str(output)]) == 0
    page = output.read_text()
    assert "System" in page and "User" in page and "Worker" in page
    assert "__TRACE_DATA__" not in page
    assert "<script src=" not in page and '<link rel="stylesheet"' not in page


def test_delivery_rejection_is_not_used_as_successful_finish(tmp_path):
    root = fixture_run(tmp_path)
    session = root / ".autoresearch/children/w1/session.json"
    session.parent.mkdir(parents=True)
    session.write_text(json.dumps({"agent_id": "w1", "parent_agent_id": "main", "messages": [
        {"role": "tool", "tool_call_id": "finish", "content": json.dumps({
            "ok": False, "error": {"code": "invalid_delivery"}})},
    ]}))
    trace = renderer.build_trace(root)
    finish = next(event for event in trace["events"] if event["title"] == "finish")
    assert finish["status"] == "error"
    assert "feedback" in finish
    task = next(event for event in trace["events"] if event["kind"] == "assignment")
    assert task["end_inferred"] is True
    assert task["end"] == renderer.timestamp(at(45))


def test_pending_request_and_unfinished_last_line_remain_visible(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    write_events(audit, "model_events.jsonl", [{
        "event": "start", "at": at(1), "request": 1, "agent_id": "main",
    }])
    with (audit / "model_events.jsonl").open("a") as handle:
        handle.write('{"event":')
    trace = renderer.build_trace(tmp_path)
    assert trace["events"][0]["status"] == "pending"
    assert trace["events"][0]["end"] is None
    assert any("最后一行未完成" in warning for warning in trace["warnings"])


def test_without_timestamps_refuses_to_invent_order(tmp_path):
    with pytest.raises(ValueError, match="无法还原真实时序"):
        renderer.build_trace(tmp_path)
