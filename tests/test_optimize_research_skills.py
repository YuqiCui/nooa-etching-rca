"""验证技能优化脚本的配置隔离、成对输出与真实文件保护。"""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "optimize_research_skills.py"
spec = importlib.util.spec_from_file_location("optimize_research_skills", SCRIPT)
optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimizer)


def fixture_result():
    """提供两个最小候选稿，用于接口边界测试。"""
    return {
        "skills": [
            {
                "name": name,
                "optimized_markdown": (
                    f"---\nname: {name}\ndescription: 执行指定研究工作\n---\n"
                    "研究约定见 references/research-state.jsonc。\n"
                ),
                "removed": [],
                "rewritten": [],
                "added": [],
                "preserved_contracts": [],
                "risks": [],
            }
            for name in optimizer.SKILLS
        ],
        "pair_review": "主会话维护状态，子会话交回检查。",
    }


def test_config_uses_selected_file_without_environment_fallback(tmp_path, monkeypatch):
    """缺少文件配置时不能静默使用当前进程中的其他凭据。"""
    monkeypatch.setenv("FACTOR_API_KEY", "unrelated-process-key")
    env = tmp_path / ".env"
    env.write_text("FACTOR_API_BASE=https://example.test/v1\n")
    with pytest.raises(ValueError, match="必须设置"):
        optimizer.load_config(env)
    env.write_text("FACTOR_API_BASE=https://example.test/v1\nFACTOR_API_KEY=file-key\n")
    assert optimizer.load_config(env)["api_key"] == "file-key"


@pytest.mark.parametrize(
    ("style", "base", "suffix"),
    [
        ("openai", "https://example.test/v1", "/v1/chat/completions"),
        ("anthropic", "https://example.test", "/v1/messages"),
        ("anthropic", "https://example.test/v1/", "/v1/messages"),
    ],
)
def test_request_preserves_explicit_model_and_protocol(style, base, suffix):
    """指定模型和协议不被服务默认值替换，凭据只进入请求头。"""
    url, headers, payload = optimizer.make_request(
        {"base_url": base, "api_key": "private-key", "style": style},
        "claude-opus-5-5", "待编辑资料", 16000,
    )
    assert url == "https://example.test" + suffix
    assert payload["model"] == "claude-opus-5-5"
    assert "private-key" not in json.dumps(payload)
    assert "private-key" in str(headers)


def serialize_result(result):
    """生成模型承诺的纯 Markdown 分段格式。"""
    first, second = result["skills"]
    return (
        "<<<AUTORESEARCH>>>\n" + first["optimized_markdown"]
        + "<<<RESEARCH_EXECUTOR>>>\n" + second["optimized_markdown"]
        + "<<<COMPARISON>>>\n" + result["pair_review"] + "\n<<<END>>>"
    )


def test_rejects_incomplete_pair_and_incompatible_metadata():
    """缺失第二份技能或改错技能标识时不能生成有效候选稿。"""
    result = fixture_result()
    originals = {item["name"]: item["optimized_markdown"] for item in result["skills"]}
    text = serialize_result(result)
    validated = optimizer.validate_result(text, originals)
    assert len(validated["skills"]) == 2
    with pytest.raises(ValueError, match="缺少完整分段边界"):
        optimizer.validate_result(text[:text.index("<<<RESEARCH_EXECUTOR>>>")], originals)
    with pytest.raises(ValueError, match="元数据不完整"):
        optimizer.validate_result(text.replace("name: autoresearch", "name: unknown"), originals)


@pytest.mark.parametrize(
    ("style", "response"),
    [
        ("openai", {"choices": [{"finish_reason": "length"}]}),
        ("anthropic", {"stop_reason": "max_tokens"}),
    ],
)
def test_rejects_truncated_model_output(style, response):
    """被截断的响应不生成看似完整的候选稿。"""
    with pytest.raises(ValueError, match="未正常结束"):
        optimizer.response_text(response, style)


@pytest.mark.parametrize(
    ("http_status", "returned_model"),
    [(200, "claude-opus-5-5"), (401, "claude-opus-5-5"), (200, "other-model")],
)
def test_cli_preserves_sources_and_secrets(tmp_path, monkeypatch, http_status, returned_model):
    """通过模拟 HTTP 传输检查单次调用、源文件保护和凭据不落盘。"""
    root = tmp_path / "project"
    root.mkdir()
    result = fixture_result()
    originals = {item["name"]: item["optimized_markdown"] for item in result["skills"]}
    paths = {name: Path(name) / "SKILL.md" for name in originals}
    for name, path in paths.items():
        (root / path).parent.mkdir()
        (root / path).write_text(originals[name])
    env = tmp_path / ".env"
    env.write_text(
        "FACTOR_API_BASE=https://example.test/v1\n"
        "FACTOR_API_KEY=do-not-save-this-key\nFACTOR_MODEL=other-model\n"
    )
    output = tmp_path / "output"
    calls = []

    def handler(request):
        """记录调用参数并返回固定响应，不连接真实服务。"""
        calls.append(json.loads(request.content))
        return httpx.Response(http_status, json={
            "model": returned_model,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": serialize_result(result)},
            }],
        })

    client_type = httpx.Client
    monkeypatch.setattr(optimizer, "ROOT", root)
    monkeypatch.setattr(optimizer, "SKILLS", paths)
    monkeypatch.setattr(optimizer, "CONTEXT", ())
    monkeypatch.setattr(
        optimizer.httpx, "Client",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr("sys.argv", [
        str(SCRIPT), "--env-file", str(env), "--output", str(output),
    ])
    success = http_status == 200 and returned_model == "claude-opus-5-5"
    assert optimizer.main() == (0 if success else 1)
    assert len(calls) == 1
    assert calls[0]["model"] == "claude-opus-5-5"
    for name, path in paths.items():
        assert (root / path).read_text() == originals[name]
    for path in output.rglob("*"):
        if path.is_file():
            assert "do-not-save-this-key" not in path.read_text()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == ("completed" if success else "failed")
    assert (output / "comparison.md").exists() == success
