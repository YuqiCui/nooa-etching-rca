"""验证工具返回值的字节阈值、完整落盘、首尾预览及消息接入。"""

import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from factor_demo.autoresearch_agent_v2.agent.loop import AgentLoop
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.agent.typing import SessionState
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.tools import result_guard
from factor_demo.autoresearch_agent_v2.tools.result_guard import (
    MAX_TOOL_RESULT_BYTES,
    PREVIEW_CHARS,
    TRUNCATION_MARKER,
    protect_tool_result,
)


@pytest.mark.parametrize("content", ["", "short", "中🙂", "x" * MAX_TOOL_RESULT_BYTES])
def test_small_and_exact_limit_strings_are_unchanged_without_files(tmp_path, content):
    output_dir = tmp_path / "not-created"
    assert protect_tool_result(content, output_dir) == content
    assert not output_dir.exists()


@pytest.mark.parametrize("result", [None, 42, [1, "中文"], {"value": "🙂"}])
def test_structured_results_use_unicode_json(tmp_path, result):
    content = protect_tool_result(result, tmp_path / "not-created")
    assert content == json.dumps(result, ensure_ascii=False)
    assert json.loads(content) == result
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize(
    "content",
    [
        "x" * (MAX_TOOL_RESULT_BYTES + 1),
        "首" * 3500 + "中" * 1000 + "尾🙂" * 1500,
        "head\r\n" + "x" * MAX_TOOL_RESULT_BYTES + "\r\ntail\x00",
    ],
)
def test_large_results_save_exact_bytes_and_preview_both_ends(tmp_path, content):
    protected = protect_tool_result(content, tmp_path / "results")
    envelope = json.loads(protected)
    assert envelope["truncated"] is True
    assert envelope["original_size_bytes"] == len(content.encode("utf-8"))
    preview = envelope["preview"]
    assert len(preview) == PREVIEW_CHARS
    head, tail = preview.split(TRUNCATION_MARKER)
    assert content.startswith(head)
    assert content.endswith(tail)
    assert abs(len(head) - len(tail)) <= 1
    saved_path = Path(envelope["file_path"])
    assert saved_path.is_absolute()
    assert saved_path.parent == (tmp_path / "results").resolve()
    assert saved_path.read_bytes() == content.encode("utf-8")
    assert len(protected.encode("utf-8")) < MAX_TOOL_RESULT_BYTES


def test_multibyte_threshold_counts_bytes_instead_of_characters(tmp_path):
    content = "中" * (MAX_TOOL_RESULT_BYTES // 3)
    assert protect_tool_result(content, tmp_path) == content
    assert json.loads(protect_tool_result(content + "中", tmp_path))["truncated"] is True


def test_large_structured_results_are_recoverable_and_files_never_overwrite(tmp_path):
    result = {"data": "中文🙂" * 4000}
    paths = [Path(json.loads(protect_tool_result(result, tmp_path))["file_path"]) for _ in range(2)]
    assert paths[0] != paths[1]
    for path in paths:
        assert json.loads(path.read_text(encoding="utf-8")) == result


def test_storage_failure_propagates_without_returning_large_content(tmp_path):
    output_dir = tmp_path / "file"
    output_dir.write_text("keep", encoding="utf-8")
    with pytest.raises(OSError):
        protect_tool_result("x" * (MAX_TOOL_RESULT_BYTES + 1), output_dir)
    assert output_dir.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("nested", [False, True])
def test_result_storage_rejects_symlink_at_every_directory_level(tmp_path, nested):
    workdir, outside = tmp_path / "work", tmp_path / "outside"
    workdir.mkdir()
    outside.mkdir()
    (workdir / "results").symlink_to(outside, target_is_directory=True)
    target = workdir / "results" / "nested" if nested else workdir / "results"
    with pytest.raises(OSError):
        protect_tool_result("x" * (MAX_TOOL_RESULT_BYTES + 1), target, workspace_root=workdir)
    assert list(outside.iterdir()) == []


def test_result_storage_rejects_path_outside_workspace(tmp_path):
    workdir, outside = tmp_path / "work", tmp_path / "outside"
    workdir.mkdir()
    outside.mkdir()
    with pytest.raises(ValueError):
        protect_tool_result("x" * (MAX_TOOL_RESULT_BYTES + 1), outside, workspace_root=workdir)
    assert list(outside.iterdir()) == []


def test_result_storage_rejects_directory_swapped_after_creation(tmp_path, monkeypatch):
    workdir, outside = tmp_path / "work", tmp_path / "outside"
    workdir.mkdir()
    outside.mkdir()
    original_open = os.open

    def swap_directory(path, flags, *args, **kwargs):
        if path == "results" and flags & os.O_DIRECTORY:
            (workdir / "results").rename(workdir / "moved")
            (workdir / "results").symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(result_guard.os, "open", swap_directory)
    with pytest.raises(OSError):
        protect_tool_result(
            "x" * (MAX_TOOL_RESULT_BYTES + 1), workdir / "results", workspace_root=workdir
        )
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("tool_name", ["inspect", "finish"])
def test_loop_protects_all_tool_results_before_history_and_model_replay(
    monkeypatch, tmp_path, tool_name, capsys
):
    large_content = "首" + "x" * MAX_TOOL_RESULT_BYTES + "尾"
    result = {"data": large_content} if tool_name == "inspect" else large_content
    requests = []

    async def completion(self, messages, *, tools):
        requests.append(deepcopy(messages))
        name = tool_name if len(requests) == 1 else "finish"
        arguments = (
            {}
            if name == "inspect"
            else {"content": large_content if len(requests) == 1 else "Done"}
        )
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call-{len(requests)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }

    monkeypatch.setattr(Model, "completion", completion)
    state = SessionState(session_id="s", agent_id="a", task_id="t", workdir=tmp_path)
    loop = AgentLoop(Model(ModelConfig(model="openai/fixture", api_key="fixture")), state)
    loop.tools.register("inspect", lambda: result, description="返回大结果", parameters={})

    assert asyncio.run(loop.run("Start")) == (large_content if tool_name == "finish" else "Done")
    tool_message = state.messages[2]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call-1"
    envelope = json.loads(tool_message["content"])
    saved = Path(envelope["file_path"]).read_text(encoding="utf-8")
    assert (json.loads(saved) if tool_name == "inspect" else saved) == result
    if tool_name != "finish":
        assert requests[1][-1] == tool_message
    capsys.readouterr()
