/** apply_patch 与 edit_file 工具的 Git 风格代码差异提取与行级 Diff 计算工具。 */

export type DiffLineKind = "add" | "del" | "ctx" | "hunk";

export interface DiffLine {
  kind: DiffLineKind;
  text: string;
  oldNum?: number;
  newNum?: number;
}

export type DiffFileAction = "add" | "delete" | "update" | "move";

export interface DiffFileBlock {
  path: string;
  destination?: string;
  action: DiffFileAction;
  additions: number;
  deletions: number;
  replaceAll?: boolean;
  lines: DiffLine[];
}

export interface CodeDiffPayload {
  toolKind: "apply_patch" | "edit_file";
  files: DiffFileBlock[];
  totalAdditions: number;
  totalDeletions: number;
}

const EDIT_FILE_TOOLS = new Set(["edit_file", "files__edit_file"]);
const APPLY_PATCH_TOOLS = new Set(["apply_patch", "files__apply_patch"]);

/** LCS 矩阵规模上限，超过时退化为整段删改以避免大文件卡顿。 */
const MAX_LCS_CELLS = 200_000;

/** 尝试安全解析 JSON 对象字符串，失败时返回 null。 */
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

/** 将文本规范化为 LF 并切分为行数组；空字符串返回空数组。 */
function splitLines(text: string): string[] {
  if (text === "") return [];
  return text.replace(/\r\n?/g, "\n").split("\n");
}

/**
 * 计算两段文本的行级差异，生成带行号与上下文的 Git 风格 diff 行列表。
 *
 * Args:
 *   oldText: 替换前文本。
 *   newText: 替换后文本。
 *
 * Returns:
 *   包含上下文、删除行与新增行的 DiffLine 数组。
 */
export function computeLineDiff(oldText: string, newText: string): DiffLine[] {
  const oldLines = splitLines(oldText);
  const newLines = splitLines(newText);

  let prefix = 0;
  const minLen = Math.min(oldLines.length, newLines.length);
  while (prefix < minLen && oldLines[prefix] === newLines[prefix]) {
    prefix++;
  }

  let suffix = 0;
  while (
    suffix < minLen - prefix &&
    oldLines[oldLines.length - 1 - suffix] ===
      newLines[newLines.length - 1 - suffix]
  ) {
    suffix++;
  }

  const result: DiffLine[] = [];
  let oldNum = 1;
  let newNum = 1;

  for (let i = 0; i < prefix; i++) {
    result.push({ kind: "ctx", text: oldLines[i], oldNum: oldNum++, newNum: newNum++ });
  }

  const oldMid = oldLines.slice(prefix, oldLines.length - suffix);
  const newMid = newLines.slice(prefix, newLines.length - suffix);

  if (oldMid.length === 0) {
    for (const line of newMid) {
      result.push({ kind: "add", text: line, newNum: newNum++ });
    }
  } else if (newMid.length === 0) {
    for (const line of oldMid) {
      result.push({ kind: "del", text: line, oldNum: oldNum++ });
    }
  } else if (oldMid.length * newMid.length > MAX_LCS_CELLS) {
    for (const line of oldMid) {
      result.push({ kind: "del", text: line, oldNum: oldNum++ });
    }
    for (const line of newMid) {
      result.push({ kind: "add", text: line, newNum: newNum++ });
    }
  } else {
    const m = oldMid.length;
    const n = newMid.length;
    const cols = n + 1;
    const dp = new Int32Array((m + 1) * cols);

    for (let i = m - 1; i >= 0; i--) {
      for (let j = n - 1; j >= 0; j--) {
        if (oldMid[i] === newMid[j]) {
          dp[i * cols + j] = dp[(i + 1) * cols + (j + 1)] + 1;
        } else {
          const down = dp[(i + 1) * cols + j];
          const right = dp[i * cols + (j + 1)];
          dp[i * cols + j] = down >= right ? down : right;
        }
      }
    }

    let i = 0;
    let j = 0;
    while (i < m && j < n) {
      if (oldMid[i] === newMid[j]) {
        result.push({
          kind: "ctx",
          text: oldMid[i],
          oldNum: oldNum++,
          newNum: newNum++,
        });
        i++;
        j++;
      } else if (dp[(i + 1) * cols + j] >= dp[i * cols + (j + 1)]) {
        result.push({ kind: "del", text: oldMid[i], oldNum: oldNum++ });
        i++;
      } else {
        result.push({ kind: "add", text: newMid[j], newNum: newNum++ });
        j++;
      }
    }
    while (i < m) {
      result.push({ kind: "del", text: oldMid[i++], oldNum: oldNum++ });
    }
    while (j < n) {
      result.push({ kind: "add", text: newMid[j++], newNum: newNum++ });
    }
  }

  for (let i = oldLines.length - suffix; i < oldLines.length; i++) {
    result.push({ kind: "ctx", text: oldLines[i], oldNum: oldNum++, newNum: newNum++ });
  }

  return result;
}

/**
 * 解析 Codex 格式的 apply_patch 补丁文本为按文件分组的差异块列表。
 *
 * Args:
 *   patchText: `*** Begin Patch` ... `*** End Patch` 格式补丁字符串。
 *
 * Returns:
 *   文件差异块列表；未解析出有效文件操作时返回空数组。
 */
export function parseApplyPatch(patchText: string): DiffFileBlock[] {
  const trimmed = patchText.replace(/\r\n?/g, "\n").trim();
  if (!trimmed) return [];

  let lines = trimmed.split("\n");
  if (
    lines.length >= 4 &&
    (lines[0] === "<<EOF" || lines[0] === "<<'EOF'" || lines[0] === '<<"EOF"') &&
    lines[lines.length - 1].endsWith("EOF")
  ) {
    lines = lines.slice(1, -1);
  }

  const files: DiffFileBlock[] = [];
  let current: DiffFileBlock | null = null;
  let addLineNum = 1;

  for (const raw of lines) {
    const stripped = raw.trim();
    if (stripped === "*** Begin Patch") continue;
    if (stripped === "*** End Patch") break;

    if (stripped.startsWith("*** Add File: ")) {
      current = {
        path: stripped.slice("*** Add File: ".length).trim(),
        action: "add",
        additions: 0,
        deletions: 0,
        lines: [],
      };
      addLineNum = 1;
      files.push(current);
      continue;
    }

    if (stripped.startsWith("*** Delete File: ")) {
      current = {
        path: stripped.slice("*** Delete File: ".length).trim(),
        action: "delete",
        additions: 0,
        deletions: 0,
        lines: [],
      };
      files.push(current);
      continue;
    }

    if (stripped.startsWith("*** Update File: ")) {
      current = {
        path: stripped.slice("*** Update File: ".length).trim(),
        action: "update",
        additions: 0,
        deletions: 0,
        lines: [],
      };
      files.push(current);
      continue;
    }

    if (!current) continue;

    if (current.action === "add") {
      if (raw.startsWith("+")) {
        current.lines.push({
          kind: "add",
          text: raw.slice(1),
          newNum: addLineNum++,
        });
        current.additions++;
      }
      continue;
    }

    if (current.action === "update" || current.action === "move") {
      if (stripped.startsWith("*** Move to: ") && !current.lines.length) {
        current.destination = stripped.slice("*** Move to: ".length).trim();
        current.action = "move";
        continue;
      }
      if (stripped === "*** End of File") {
        continue;
      }
      if (stripped === "@@" || stripped.startsWith("@@ ")) {
        current.lines.push({ kind: "hunk", text: stripped });
        continue;
      }
      if (raw.startsWith("+")) {
        current.lines.push({ kind: "add", text: raw.slice(1) });
        current.additions++;
      } else if (raw.startsWith("-")) {
        current.lines.push({ kind: "del", text: raw.slice(1) });
        current.deletions++;
      } else if (raw.startsWith(" ")) {
        current.lines.push({ kind: "ctx", text: raw.slice(1) });
      } else if (raw === "") {
        current.lines.push({ kind: "ctx", text: "" });
      }
    }
  }

  return files;
}

/**
 * 从 apply_patch 或 edit_file 工具调用参数中提取 Git 风格代码差异载荷。
 *
 * Args:
 *   toolName: 工具原始名称，如 `apply_patch`、`edit_file`。
 *   rawCode: 工具调用参数字符串（通常为 JSON）。
 *
 * Returns:
 *   解析后的代码差异结构；不匹配或无有效差异时返回 null。
 */
export function extractCodeDiff(
  toolName: string,
  rawCode: string,
): CodeDiffPayload | null {
  const isEdit = EDIT_FILE_TOOLS.has(toolName);
  const isPatch = APPLY_PATCH_TOOLS.has(toolName);
  if (!isEdit && !isPatch) return null;

  const args = safeParseJson(rawCode);
  if (!args) return null;

  if (isEdit) {
    if (typeof args.old_string !== "string") return null;
    const newText = typeof args.new_string === "string" ? args.new_string : "";
    const filePath =
      typeof args.path === "string" && args.path.trim()
        ? args.path.trim()
        : "file";
    const lines = computeLineDiff(args.old_string, newText);
    let additions = 0;
    let deletions = 0;
    for (const line of lines) {
      if (line.kind === "add") additions++;
      else if (line.kind === "del") deletions++;
    }
    return {
      toolKind: "edit_file",
      files: [
        {
          path: filePath,
          action: "update",
          additions,
          deletions,
          replaceAll: Boolean(args.replace_all),
          lines,
        },
      ],
      totalAdditions: additions,
      totalDeletions: deletions,
    };
  }

  const patchInput =
    typeof args.input === "string"
      ? args.input
      : typeof args.patch === "string"
        ? args.patch
        : null;
  if (!patchInput) return null;

  const files = parseApplyPatch(patchInput);
  if (!files.length) return null;

  let totalAdditions = 0;
  let totalDeletions = 0;
  for (const file of files) {
    totalAdditions += file.additions;
    totalDeletions += file.deletions;
  }

  return {
    toolKind: "apply_patch",
    files,
    totalAdditions,
    totalDeletions,
  };
}
