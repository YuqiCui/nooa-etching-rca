"""记录当前会话已经观察到的文件版本和行范围。"""

from pydantic import BaseModel, Field


class FileReadState(BaseModel):
    """读取记录只在工具实例中保存，不写入模型参数或 Session 历史。"""

    digest: bytes
    """已观察内容的 SHA-256 摘要，用于区分版本。"""

    ranges: list[tuple[int, int]] = Field(default_factory=list)
    """同一版本中已经返回给模型的行范围，合并后按起始行排序。"""

    complete: bool = False
    """是否已经完整读取该版本，或完整内容由本会话写入。"""
