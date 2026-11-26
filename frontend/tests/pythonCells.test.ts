import { describe, expect, it } from "vitest";
import {
  extractPythonCells,
  tokenizePythonLines,
} from "../src/utils/pythonCells";

describe("Notebook write_cell / write_cells Python 提取与高亮", () => {
  it("提取 notebook__write_new_cell 的单 cell 源码并从返回结果补充 cell_id", () => {
    const payload = extractPythonCells(
      "notebook__write_new_cell",
      JSON.stringify({
        notebook_name: "etch_analysis",
        source: "import pandas as pd\ndf = pd.read_csv('data.csv')",
        after_cell_id: "c0",
      }),
      JSON.stringify({
        notebook_name: "etch_analysis",
        cell_id: "c1",
        revision: 1,
      }),
    );

    expect(payload).toEqual({
      notebookName: "etch_analysis",
      afterCellId: "c0",
      cells: [
        {
          index: 1,
          source: "import pandas as pd\ndf = pd.read_csv('data.csv')",
          cellId: "c1",
          revision: 1,
        },
      ],
    });
  });

  it("提取 notebook__write_cells 的多 cell 源码并按顺序映射 cell_id", () => {
    const payload = extractPythonCells(
      "notebook__write_cells",
      JSON.stringify({
        notebook_name: "drift_check",
        cells: [
          { source: "x = 1\nprint(x)" },
          { source: "def calc(v):\n    return v * 2" },
        ],
      }),
      JSON.stringify({
        notebook_name: "drift_check",
        count: 2,
        cells: [
          { cell_id: "cell-a", revision: 1 },
          { cell_id: "cell-b", revision: 1 },
        ],
      }),
    );

    expect(payload?.notebookName).toBe("drift_check");
    expect(payload?.cells).toHaveLength(2);
    expect(payload?.cells[0]).toMatchObject({
      index: 1,
      cellId: "cell-a",
      source: "x = 1\nprint(x)",
    });
    expect(payload?.cells[1]).toMatchObject({
      index: 2,
      cellId: "cell-b",
      source: "def calc(v):\n    return v * 2",
    });
  });

  it("提取 write_file 的文件内容并按扩展名区分 Python 与纯文本", () => {
    const pyPayload = extractPythonCells(
      "write_file",
      JSON.stringify({
        path: "scripts/check_drift.py",
        content: "def check(x):\n    return x > 0",
      }),
    );
    expect(pyPayload).toEqual({
      filePath: "scripts/check_drift.py",
      cells: [
        {
          index: 1,
          source: "def check(x):\n    return x > 0",
          label: "scripts/check_drift.py",
          language: "python",
        },
      ],
    });

    const mdPayload = extractPythonCells(
      "write_file",
      JSON.stringify({
        path: "notes/summary.md",
        content: "# Summary\nreturn if true",
      }),
    );
    expect(mdPayload?.cells[0]).toMatchObject({
      label: "notes/summary.md",
      language: "plain",
    });
  });

  it("对非写入工具或无效参数返回 null", () => {
    expect(
      extractPythonCells("run_code", JSON.stringify({ code: "print(1)" })),
    ).toBeNull();
    expect(
      extractPythonCells("notebook__write_cells", "{bad json"),
    ).toBeNull();
    expect(
      extractPythonCells(
        "notebook__edit_cell",
        JSON.stringify({
          notebook_name: "nb",
          cell_id: "c1",
          expected_revision: 1,
          old_string: "a",
          new_string: "b",
        }),
      ),
    ).toBeNull();
  });

  it("正确切分 Python 关键字、函数名、字符串、跨行三引号、注释与数字", () => {
    const code = [
      "@cache",
      "def score(x: float) -> float:",
      '    """多行',
      '    说明"""',
      "    # 计算偏移",
      "    return round(x * 1.5, 2) if x is not None else 0",
    ].join("\n");

    const lines = tokenizePythonLines(code);
    expect(lines).toHaveLength(6);
    expect(lines[0]).toEqual([{ type: "decorator", text: "@cache" }]);
    expect(lines[1]).toContainEqual({ type: "keyword", text: "def" });
    expect(lines[1]).toContainEqual({ type: "def-name", text: "score" });
    expect(lines[2]).toContainEqual({ type: "string", text: '"""多行' });
    expect(lines[3]).toContainEqual({ type: "string", text: '    说明"""' });
    expect(lines[4]).toContainEqual({ type: "comment", text: "# 计算偏移" });
    expect(lines[5]).toContainEqual({ type: "keyword", text: "return" });
    expect(lines[5]).toContainEqual({ type: "builtin", text: "round" });
    expect(lines[5]).toContainEqual({ type: "number", text: "1.5" });
    expect(lines[5]).toContainEqual({ type: "constant", text: "None" });
  });
});
