"""将已筛选的任务输入复制到文件工具与执行工具共用的工作目录。"""

import shutil
from pathlib import Path
from tempfile import mkdtemp


def stage_workspace_dataset(source: Path, workdir: Path) -> Path:
    """在工作目录内创建独立数据副本，不覆盖已有输入或研究产物。

    Args:
        source: 宿主已经筛选的公开输入或当前登记数据集目录。
        workdir: 已存在的会话工作目录。

    Returns:
        工作目录内数据副本的绝对路径。

    Raises:
        ValueError: 输入包含符号链接，或输入目录包含工作目录。
        OSError: 目录不存在、复制失败或输入不是普通文件与目录。
    """
    if source.is_symlink():
        raise ValueError("Dataset inputs cannot contain symbolic links")
    source = source.resolve(strict=True)
    workdir = workdir.resolve(strict=True)
    if workdir.is_relative_to(source):
        raise ValueError("Dataset source cannot contain the workspace")
    destination = Path(mkdtemp(prefix="dataset-", dir=workdir))
    try:
        shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)
        if any(path.is_symlink() for path in destination.rglob("*")):
            raise ValueError("Dataset inputs cannot contain symbolic links")
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination
