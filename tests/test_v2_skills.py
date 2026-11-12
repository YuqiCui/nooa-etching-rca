"""检查技能加载、恢复、能力依赖和资源访问边界。"""

import asyncio
from pathlib import Path

import pytest

from factor_demo.autoresearch_agent_v2.errors import UnknownToolError
from factor_demo.autoresearch_agent_v2.skills import SkillContext, SkillRegistry
from factor_demo.autoresearch_agent_v2.utils.file import FileToolError


def create_skill(root: Path, name: str = "analysis", *, tools: str = "[]") -> Path:
    """创建测试技能及参考资料，不执行其中的文本。"""
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: analysis\ndescription: 检查指定任务\nversion: '1.0'\n"
        f"requires_tools: {tools}\n---\n按父任务执行，不要求固定合同。\n"
    )
    (folder / "references").mkdir()
    (folder / "references" / "check.md").write_text("实际验证后再报告结果。")
    return folder


def test_catalog_is_progressive_and_contexts_are_independent(tmp_path):
    """目录只暴露元数据；一个会话加载不会激活另一个会话。"""
    create_skill(tmp_path)
    registry = SkillRegistry([tmp_path])
    first, second = SkillContext(registry), SkillContext(registry)
    assert "按父任务执行" not in first.instructions()
    first.preload(["analysis"])
    assert "按父任务执行" in first.instructions()
    assert "实际验证后" not in first.instructions()
    assert "按父任务执行" not in second.instructions()
    assert first.read_resource("analysis", "references/check.md") == "实际验证后再报告结果。"
    assert len(first.snapshots()) == 1
    assert first.load("analysis").sha256 == first.load("analysis").sha256
    assert len(first.snapshots()) == 1


def test_restore_uses_frozen_content_and_checks_hash(tmp_path):
    """部署目录更新不改变既有快照，篡改检查点会被拒绝。"""
    folder = create_skill(tmp_path)
    registry = SkillRegistry([tmp_path])
    context = SkillContext(registry)
    context.preload(["analysis"])
    saved = context.snapshots()
    (folder / "SKILL.md").write_text("新版部署正文")
    (folder / "references" / "check.md").write_text("新版资料")
    restored = SkillContext(registry, snapshots=saved)
    assert "按父任务执行" in restored.instructions()
    assert restored.read_resource("analysis", "references/check.md") == "实际验证后再报告结果。"
    saved[0]["content"] = "伪造的正文"
    with pytest.raises(ValueError, match="哈希"):
        SkillContext(registry, snapshots=saved)


def test_tool_dependencies_cannot_grant_or_restore_revoked_tools(tmp_path):
    """技能只检查依赖，不能激活当前会话没有的工具。"""
    create_skill(tmp_path, tools="[notebook__exec_cell]")
    registry = SkillRegistry([tmp_path])
    with pytest.raises(ValueError, match="缺少工具"):
        SkillContext(registry).load("analysis")
    context = SkillContext(registry, available_tools=["notebook__exec_cell"])
    context.load("analysis")
    with pytest.raises(ValueError, match="缺少工具"):
        SkillContext(registry, snapshots=context.snapshots())


def test_preload_failure_does_not_leave_partial_selection(tmp_path):
    """未知技能使本次预加载整体失败，已存在的选择保持不变。"""
    create_skill(tmp_path)
    context = SkillContext(SkillRegistry([tmp_path]))
    with pytest.raises(ValueError, match="未知技能"):
        context.preload(["analysis", "missing"])
    assert context.snapshots() == []


@pytest.mark.parametrize("path", ["../secret", "/secret", "a/../secret", "a\\secret", ""])
def test_resource_paths_cannot_escape_snapshot(tmp_path, path):
    """资源读取不接受父目录、绝对路径和平台混合分隔符。"""
    create_skill(tmp_path)
    context = SkillContext(SkillRegistry([tmp_path]))
    context.preload(["analysis"])
    with pytest.raises(ValueError, match="路径"):
        context.read_resource("analysis", path)


def test_symlink_resource_and_skill_directory_are_rejected(tmp_path):
    """宿主批准根目录不允许通过技能文件或目录符号链接越界。"""
    skills = tmp_path / "skills"
    folder = create_skill(skills)
    secret = tmp_path / "secret.txt"
    secret.write_text("不应读到的内容")
    (folder / "link.md").symlink_to(secret)
    registry = SkillRegistry([skills])
    with pytest.raises(ValueError, match="符号链接"):
        registry.load("analysis")
    (folder / "link.md").unlink()
    (skills / "linked").symlink_to(folder, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        SkillRegistry([skills])


def test_symlink_skill_body_is_not_followed(tmp_path):
    """SKILL.md 自身为链接时，目录发现也不能读取目标。"""
    skills = tmp_path / "skills"
    folder = create_skill(skills)
    original = folder / "SKILL.md"
    outside = tmp_path / "outside.md"
    original.rename(outside)
    original.symlink_to(outside)
    with pytest.raises((FileToolError, OSError)):
        SkillRegistry([skills])


def test_duplicate_ids_and_unapproved_restore_are_rejected(tmp_path):
    """技能标识必须唯一，恢复不能重新获得已撤销的目录授权。"""
    create_skill(tmp_path / "first")
    create_skill(tmp_path / "second")
    with pytest.raises(ValueError, match="重复"):
        SkillRegistry([tmp_path / "first", tmp_path / "second"])
    context = SkillContext(SkillRegistry([tmp_path / "first"]))
    context.load("analysis")
    with pytest.raises(ValueError, match="批准目录"):
        SkillContext(SkillRegistry(), snapshots=context.snapshots())


def test_replacing_approved_root_cannot_redirect_skill_reads(tmp_path):
    """批准根被换成外部链接后，读取仍绑定最初的目录句柄。"""
    approved = tmp_path / "approved"
    create_skill(approved)
    registry = SkillRegistry([approved])
    outside = tmp_path / "outside"
    other = create_skill(outside)
    (other / "references" / "check.md").write_text("外部秘密")
    approved.rename(tmp_path / "original")
    approved.symlink_to(outside, target_is_directory=True)
    context = SkillContext(registry)
    context.preload(["analysis"])
    assert context.read_resource("analysis", "references/check.md") == "实际验证后再报告结果。"
    assert "外部秘密" not in str(context.snapshots())


def test_closed_registry_cannot_open_new_skill_snapshots(tmp_path):
    """关闭注册目录后不再允许读取新的部署内容。"""
    create_skill(tmp_path)
    registry = SkillRegistry([tmp_path])
    registry.close()
    with pytest.raises(RuntimeError, match="closed"):
        registry.load("analysis")


def test_session_preload_without_skill_tools_and_snapshot_restore(tmp_path):
    """预加载独立于加载工具授权，恢复使用会话快照而非新版正文。"""
    from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
    from factor_demo.autoresearch_agent_v2.agent.model import Model
    from factor_demo.autoresearch_agent_v2.config import ModelConfig

    folder = create_skill(tmp_path / "skills")
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    registry = SkillRegistry([tmp_path / "skills"])
    state = SessionState(
        session_id="session", agent_id="main", task_id="task", workdir=workdir,
    )

    def create_session(current, skills=()):
        """仅授予 finish，技能上下文仍需在请求前准备就绪。"""
        model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
        return Session(
            model, current, skill_registry=registry, skills=skills, allow_tools=["finish"],
        )

    async def scenario():
        """使用实际会话启动与检查点，不发出模型请求。"""
        async with create_session(state, ["analysis"]) as first:
            assert [item["function"]["name"] for item in first.schemas()] == ["finish"]
            prompt = first.loop.build_system_prompt()
            assert "按父任务执行" in prompt
            assert len(first.state.skill_snapshots) == 1
            with pytest.raises((PermissionError, UnknownToolError)):
                await first.call("skill__load", {"skill_id": "analysis"})
        restored = SessionState.model_validate_json(state.model_dump_json())
        (folder / "SKILL.md").write_text("替换后的部署正文")
        async with create_session(restored) as second:
            assert "按父任务执行" in second.loop.build_system_prompt()
            assert "替换后的部署正文" not in second.loop.build_system_prompt()
            assert len(second.state.skill_snapshots) == 1

    asyncio.run(scenario())


def test_session_skill_load_updates_context_once_and_persists(tmp_path):
    """显式加载更新请求投影和检查点，不重复追加模型消息。"""
    from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
    from factor_demo.autoresearch_agent_v2.agent.model import Model
    from factor_demo.autoresearch_agent_v2.config import ModelConfig

    create_skill(tmp_path / "skills")
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
    session = Session(
        model,
        SessionState(session_id="s", agent_id="main", task_id="t", workdir=workdir),
        skill_registry=SkillRegistry([tmp_path / "skills"]),
    )

    async def scenario():
        """重复调用真实加载工具，状态和上下文均保持幂等。"""
        async with session:
            assert "按父任务执行" not in session.loop.build_system_prompt()
            for _ in range(2):
                await session.call("skill__load", {"skill_id": "analysis"})
            assert session.loop.build_system_prompt().count("按父任务执行") == 1
            assert len(session.state.skill_snapshots) == 1
            assert session.state.messages == []

    asyncio.run(scenario())
