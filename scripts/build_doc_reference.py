"""Build reference pages from source AST/config only; no runtime or secret imports."""

import ast
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/sphinx/_generated"
REPO = "https://github.com/pty819/nooa-etching-rca"
REF = os.environ.get("GITHUB_SHA", "main")


def source(path, line=1):
    return f"{REPO}/blob/{REF}/{path}#L{line}"


def generate():
    OUT.mkdir(exist_ok=True)
    modules = [
        "# 源码模块与入口索引",
        "",
        "本页由AST生成，不导入NOOA、供应商配置或研究会话。文件链接指向本次构建的提交（本地构建为main）。",
        "",
    ]
    for path in sorted((ROOT / "src/factor_demo").rglob("*.py")):
        tree = ast.parse(path.read_text())
        relative = path.relative_to(ROOT).as_posix()
        modules += [
            f"## {path.relative_to(ROOT / 'src/factor_demo').with_suffix('').as_posix().replace('/', '.')}",
            "",
            ast.get_docstring(tree) or "项目模块。",
            "",
            f"[查看源码]({source(relative)})",
            "",
        ]
        for item in tree.body:
            if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "类" if isinstance(item, ast.ClassDef) else "函数"
                modules.append(f"- {kind} [`{item.name}`]({source(relative, item.lineno)})")
                if isinstance(item, ast.ClassDef):
                    methods = [
                        x.name
                        for x in item.body
                        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and not x.name.startswith("_")
                    ]
                    if methods:
                        modules.append("  公开方法：" + "、".join(f"`{x}`" for x in methods) + "。")
        modules.append("")
    (OUT / "modules.md").write_text("\n".join(modules) + "\n")
    # Extract the public agent method surface without importing the runtime
    # or private provider configuration. NOOA now owns tool/proxy visibility.
    agent_tree = ast.parse((ROOT / "src/factor_demo/autoresearch_agent.py").read_text())
    agent = next(
        node
        for node in agent_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AutoresearchAgent"
    )
    tool_lines = [
        "# CodeAct实际工具契约",
        "",
        "本页从AutoresearchAgent公开方法的AST生成；不导入运行时，不读取凭据。参数与说明跟随构建提交，领域与只读权限仍由宿主实际检查。",
        "",
        "说明中的checked/verified只表达各自检查口径，不构成任意科学结论认证。工具分组与使用流程见[agent设计](../agents.md)和[执行运行协议](../repl-runtime.md)。",
        "",
    ]
    for item in agent.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or item.name.startswith(
            "_"
        ):
            continue
        args = ast.unparse(item.args)
        if args.startswith("self, "):
            args = args[6:]
        elif args == "self":
            args = ""
        tool_lines += [
            f"## {item.name}",
            "",
            f"[实现与权限检查]({source('src/factor_demo/autoresearch_agent.py', item.lineno)})",
            "",
            "```python",
            f"{item.name}({args})",
            "```",
            "",
            (ast.get_docstring(item) or "项目工具。").replace("\n", " "),
            "",
        ]
    (OUT / "tools.md").write_text("\n".join(tool_lines))
    lines = [
        "# 实际配置与结构化契约",
        "",
        "配置值直接读取版本化JSON；字段定义来自AST，不运行模型或读取API密钥。",
        "",
    ]
    for name in ["runtime", "capabilities"]:
        lines += [
            f"## {name}.json",
            "",
            f"[源码]({source('configs/' + name + '.json')})",
            "",
            "```json",
            (ROOT / f"configs/{name}.json").read_text().strip(),
            "```",
            "",
        ]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    stack = [
        "nooa",
        "polars",
        "numpy",
        "scipy",
        "scikit-learn",
        "aeon",
        "statsmodels",
        "pandas",
        "matplotlib",
    ]
    lines += [
        "## 科学分析库的锁定版本",
        "",
        "这些版本来自uv.lock，不代表全部算法或可选extras都经过验收。Polars只是表格首选，agent可自由选择、组合其他已安装库。",
        "",
        "| 发行包 | 锁定版本 |",
        "|---|---|",
    ]
    for package in stack:
        versions = sorted({p["version"] for p in lock["package"] if p["name"] == package})
        lines.append(f"| {package} | {', '.join(versions)} |")
    lines += ["", f"[依赖约束]({source('pyproject.toml')}) · [完整锁文件]({source('uv.lock')})", ""]
    for filename, wanted in [
        ("runtime.py", {"RuntimeSettings"}),
        ("capabilities.py", {"Capabilities"}),
        ("research_models.py", None),
        ("formula_models.py", None),
        ("dataset_models.py", None),
        ("web.py", {"NewSession", "NewMessage"}),
    ]:
        relative = "src/factor_demo/" + filename
        tree = ast.parse((ROOT / relative).read_text())
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or (wanted and cls.name not in wanted):
                continue
            lines += [
                f"## {cls.name}",
                "",
                f"[字段定义与校验规则]({source(relative, cls.lineno)})",
                "",
                "```python",
            ]
            for node in cls.body:
                if isinstance(node, ast.AnnAssign):
                    lines.append(ast.unparse(node))
            lines += ["```", ""]
    (OUT / "contracts.md").write_text("\n".join(lines) + "\n")
    tree = ast.parse((ROOT / "src/factor_demo/cli.py").read_text())
    parsers = sorted(
        {
            n.args[0].value
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_parser"
            and n.args
            and isinstance(n.args[0], ast.Constant)
        }
    )
    lines = [
        "# CLI 实现参考",
        "",
        "以下为CLI解析器的实际源码片段，包含循环创建的research/resume与autoresearch入口。完整运行示例见[复现指南](../operations.md)。",
        "",
        f"[源码]({source('src/factor_demo/cli.py')})",
        "",
        "静态字符串入口：" + "、".join(f"`{x}`" for x in parsers) + "。",
        "",
        "```python",
    ]
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    original = (ROOT / "src/factor_demo/cli.py").read_text().splitlines()
    end = next(
        n.lineno
        for n in ast.walk(main)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "args" for t in n.targets)
    )
    lines += original[main.lineno - 1 : end] + ["```", ""]
    (OUT / "cli.md").write_text("\n".join(lines))


if __name__ == "__main__":
    generate()
