/** Notebook 写入与 write_file 文件写入工具的代码提取及轻量词法高亮工具。 */

export type PythonTokenType =
  | "plain"
  | "keyword"
  | "constant"
  | "builtin"
  | "def-name"
  | "decorator"
  | "string"
  | "comment"
  | "number"
  | "operator";

export interface PythonToken {
  type: PythonTokenType;
  text: string;
}

export interface PythonCellItem {
  index: number;
  source: string;
  cellId?: string;
  revision?: number;
  label?: string;
  language?: "python" | "plain";
}

export interface PythonCellsPayload {
  notebookName?: string;
  filePath?: string;
  afterCellId?: string | null;
  cells: PythonCellItem[];
}

const SINGLE_CELL_TOOLS = new Set([
  "notebook__write_new_cell",
  "write_new_cell",
  "write_cell",
  "notebook__edit_cell",
  "edit_cell",
]);

const MULTI_CELL_TOOLS = new Set([
  "notebook__write_cells",
  "write_cells",
]);

const WRITE_FILE_TOOLS = new Set(["write_file", "files__write_file"]);

const PY_KEYWORDS = new Set([
  "and",
  "as",
  "assert",
  "async",
  "await",
  "break",
  "case",
  "class",
  "continue",
  "def",
  "del",
  "elif",
  "else",
  "except",
  "finally",
  "for",
  "from",
  "global",
  "if",
  "import",
  "in",
  "is",
  "lambda",
  "match",
  "nonlocal",
  "not",
  "or",
  "pass",
  "raise",
  "return",
  "try",
  "while",
  "with",
  "yield",
]);

const PY_CONSTANTS = new Set(["True", "False", "None"]);

const PY_BUILTINS = new Set([
  "abs",
  "all",
  "any",
  "bool",
  "bytes",
  "callable",
  "dict",
  "dir",
  "enumerate",
  "Exception",
  "filter",
  "float",
  "getattr",
  "hasattr",
  "hash",
  "int",
  "isinstance",
  "issubclass",
  "iter",
  "KeyError",
  "len",
  "list",
  "map",
  "max",
  "min",
  "next",
  "object",
  "open",
  "print",
  "range",
  "repr",
  "reversed",
  "round",
  "RuntimeError",
  "set",
  "setattr",
  "slice",
  "sorted",
  "str",
  "sum",
  "super",
  "tuple",
  "type",
  "TypeError",
  "ValueError",
  "zip",
]);

/** 尝试安全解析 JSON 字符串，失败时返回 null。 */
function safeParseJson(raw?: string): Record<string, any> | null {
  if (!raw || !raw.trim()) return null;
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, any>;
    }
  } catch {}
  return null;
}

/** 判断文件路径是否为 Python 源码文件。 */
function isPythonFilePath(filePath?: string): boolean {
  if (!filePath) return true;
  return /\.pyi?$/i.test(filePath.trim());
}

/** 将纯文本按行转换为 plain token 二维数组，供非 Python 文件安全渲染。 */
export function tokenizePlainLines(source: string): PythonToken[][] {
  return source
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) => (line ? [{ type: "plain", text: line }] : []));
}

/**
 * 从 Notebook 写入工具或 write_file 工具的参数与返回值中提取代码渲染载荷。
 *
 * Args:
 *   toolName: 工具原始名称，如 `notebook__write_new_cell`、`notebook__write_cells`、`write_file`。
 *   rawCode: 工具调用参数字符串（通常为 JSON）。
 *   rawOutput: 工具返回结果字符串，用于补充 `cell_id` 与 `revision`。
 *
 * Returns:
 *   提取到的代码 cell 结构；不匹配或无有效源码时返回 null。
 */
export function extractPythonCells(
  toolName: string,
  rawCode: string,
  rawOutput?: string,
): PythonCellsPayload | null {
  const isSingle = SINGLE_CELL_TOOLS.has(toolName);
  const isMulti = MULTI_CELL_TOOLS.has(toolName);
  const isWriteFile = WRITE_FILE_TOOLS.has(toolName);
  if (!isSingle && !isMulti && !isWriteFile) return null;

  const args = safeParseJson(rawCode);
  if (!args) return null;

  if (isWriteFile) {
    if (typeof args.content !== "string") return null;
    const filePath =
      typeof args.path === "string" && args.path.trim()
        ? args.path.trim()
        : undefined;
    return {
      filePath,
      cells: [
        {
          index: 1,
          source: args.content,
          label: filePath || "file",
          language: isPythonFilePath(filePath) ? "python" : "plain",
        },
      ],
    };
  }

  const output = safeParseJson(rawOutput);
  const notebookName =
    typeof args.notebook_name === "string" && args.notebook_name.trim()
      ? args.notebook_name.trim()
      : undefined;
  const afterCellId =
    typeof args.after_cell_id === "string" && args.after_cell_id.trim()
      ? args.after_cell_id.trim()
      : null;

  if (isSingle) {
    if (typeof args.source !== "string") return null;
    const cellId =
      typeof args.cell_id === "string" && args.cell_id.trim()
        ? args.cell_id.trim()
        : typeof output?.cell_id === "string"
          ? output.cell_id
          : undefined;
    const revision =
      typeof output?.revision === "number" ? output.revision : undefined;
    return {
      notebookName,
      afterCellId,
      cells: [{ index: 1, source: args.source, cellId, revision }],
    };
  }

  if (!Array.isArray(args.cells) || !args.cells.length) return null;
  const outputCells = Array.isArray(output?.cells) ? output.cells : [];
  const cells: PythonCellItem[] = [];
  for (let i = 0; i < args.cells.length; i++) {
    const item = args.cells[i];
    const source =
      typeof item === "string"
        ? item
        : item && typeof item === "object" && typeof item.source === "string"
          ? item.source
          : null;
    if (source === null) return null;
    const outItem = outputCells[i];
    cells.push({
      index: i + 1,
      source,
      cellId: typeof outItem?.cell_id === "string" ? outItem.cell_id : undefined,
      revision:
        typeof outItem?.revision === "number" ? outItem.revision : undefined,
    });
  }

  return {
    notebookName,
    afterCellId,
    cells,
  };
}

/** 追加 token，相邻同类 token 自动合并以减少 DOM 节点数量。 */
function pushToken(line: PythonToken[], type: PythonTokenType, text: string) {
  if (!text) return;
  const prev = line.at(-1);
  if (prev && prev.type === type) {
    prev.text += text;
  } else {
    line.push({ type, text });
  }
}

/**
 * 将 Python 源码按行切分为带语法类型的 token 数组，支持跨行三引号字符串。
 *
 * Args:
 *   source: 原始 Python 源码字符串。
 *
 * Returns:
 *   按行排列的 token 二维数组，可直接由模板安全渲染。
 */
export function tokenizePythonLines(source: string): PythonToken[][] {
  const normalized = source.replace(/\r\n?/g, "\n");
  const lines: PythonToken[][] = [[]];
  let currentLine = lines[0];
  let i = 0;
  const len = normalized.length;
  let expectDefName = false;
  let atLineStart = true;

  const nextLine = () => {
    currentLine = [];
    lines.push(currentLine);
    atLineStart = true;
    expectDefName = false;
  };

  while (i < len) {
    const ch = normalized[i];

    // 换行
    if (ch === "\n") {
      nextLine();
      i++;
      continue;
    }

    // 空白字符
    if (ch === " " || ch === "\t") {
      let j = i + 1;
      while (j < len && (normalized[j] === " " || normalized[j] === "\t")) j++;
      pushToken(currentLine, "plain", normalized.slice(i, j));
      i = j;
      continue;
    }

    // 注释
    if (ch === "#") {
      let j = i + 1;
      while (j < len && normalized[j] !== "\n") j++;
      pushToken(currentLine, "comment", normalized.slice(i, j));
      atLineStart = false;
      expectDefName = false;
      i = j;
      continue;
    }

    // 行首装饰器
    if (ch === "@" && atLineStart) {
      let j = i + 1;
      while (j < len && /[A-Za-z0-9_.]/.test(normalized[j])) j++;
      pushToken(currentLine, "decorator", normalized.slice(i, j));
      atLineStart = false;
      expectDefName = false;
      i = j;
      continue;
    }

    // 字符串（含前缀 r/f/b/u 及其双字符组合，以及单/三引号）
    const prefixMatch = normalized
      .slice(i, i + 4)
      .match(/^([rRbBuUfF]{0,2})('''|"""|'|\")/);
    if (prefixMatch) {
      const prefix = prefixMatch[1];
      const quote = prefixMatch[2];
      const isTriple = quote.length === 3;
      const isRaw = /[rR]/.test(prefix);
      let j = i + prefix.length + quote.length;
      let segStart = i;
      atLineStart = false;
      expectDefName = false;

      while (j < len) {
        if (normalized[j] === "\n") {
          pushToken(currentLine, "string", normalized.slice(segStart, j));
          nextLine();
          j++;
          segStart = j;
          if (!isTriple) break;
          continue;
        }
        if (!isRaw && normalized[j] === "\\" && j + 1 < len) {
          j += 2;
          continue;
        }
        if (normalized.startsWith(quote, j)) {
          j += quote.length;
          pushToken(currentLine, "string", normalized.slice(segStart, j));
          segStart = j;
          break;
        }
        j++;
      }
      if (segStart < j) {
        pushToken(currentLine, "string", normalized.slice(segStart, j));
      }
      i = j;
      continue;
    }

    // 数字字面量
    if (
      /[0-9]/.test(ch) ||
      (ch === "." && i + 1 < len && /[0-9]/.test(normalized[i + 1]))
    ) {
      const numMatch = normalized
        .slice(i)
        .match(
          /^(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?[jJ]?)/,
        );
      if (numMatch) {
        pushToken(currentLine, "number", numMatch[0]);
        atLineStart = false;
        expectDefName = false;
        i += numMatch[0].length;
        continue;
      }
    }

    // 标识符 / 关键字 / 内置函数 / 常量
    if (/[A-Za-z_]/.test(ch)) {
      let j = i + 1;
      while (j < len && /[A-Za-z0-9_]/.test(normalized[j])) j++;
      const word = normalized.slice(i, j);
      let type: PythonTokenType = "plain";
      if (expectDefName) {
        type = "def-name";
        expectDefName = false;
      } else if (PY_KEYWORDS.has(word)) {
        type = "keyword";
        if (word === "def" || word === "class") {
          expectDefName = true;
        }
      } else if (PY_CONSTANTS.has(word)) {
        type = "constant";
      } else if (PY_BUILTINS.has(word)) {
        type = "builtin";
      }
      pushToken(currentLine, type, word);
      atLineStart = false;
      i = j;
      continue;
    }

    // 运算符与标点
    if (/[+\-*/%=&|^~<>!:;,()[\]{}.@]/.test(ch)) {
      pushToken(currentLine, "operator", ch);
      atLineStart = false;
      expectDefName = false;
      i++;
      continue;
    }

    // 其他字符（如中文标识符等）
    pushToken(currentLine, "plain", ch);
    atLineStart = false;
    expectDefName = false;
    i++;
  }

  return lines;
}
