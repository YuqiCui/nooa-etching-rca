"""声明技能目录与可恢复的不可变正文快照。"""

from pydantic import BaseModel, ConfigDict, Field


class SkillDescription(BaseModel):
    """向模型展示技能用途及显式工具依赖。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    skill_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    version: str = Field(default="1", min_length=1)
    requires_tools: list[str] = Field(default_factory=list)


class SkillSnapshot(BaseModel):
    """保留正文与引用文件，使恢复不依赖可变的部署目录。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    description: SkillDescription
    content: str
    resources: dict[str, str] = Field(default_factory=dict)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
