import { describe, expect, it } from "vitest";
import {
  computeLineDiff,
  extractCodeDiff,
  parseApplyPatch,
} from "../src/utils/codeDiff";

describe("apply_patch 与 edit_file Git 风格差异解析", () => {
  it("计算 edit_file 的行级上下文、删除与新增行", () => {
    const lines = computeLineDiff(
      "a = 1\nb = 2\nc = 3",
      "a = 1\nb = 20\nd = 4\nc = 3",
    );
    expect(lines).toEqual([
      { kind: "ctx", text: "a = 1", oldNum: 1, newNum: 1 },
      { kind: "del", text: "b = 2", oldNum: 2 },
      { kind: "add", text: "b = 20", newNum: 2 },
      { kind: "add", text: "d = 4", newNum: 3 },
      { kind: "ctx", text: "c = 3", oldNum: 3, newNum: 4 },
    ]);
  });

  it("提取 edit_file 调用参数并统计增删行数与 replace_all 标记", () => {
    const payload = extractCodeDiff(
      "edit_file",
      JSON.stringify({
        path: "analysis/model.py",
        old_string: "alpha = 0.1\nbeta = 0.2",
        new_string: "alpha = 0.05\nbeta = 0.2",
        replace_all: true,
      }),
    );
    expect(payload).not.toBeNull();
    expect(payload?.toolKind).toBe("edit_file");
    expect(payload?.totalAdditions).toBe(1);
    expect(payload?.totalDeletions).toBe(1);
    expect(payload?.files[0]).toMatchObject({
      path: "analysis/model.py",
      action: "update",
      additions: 1,
      deletions: 1,
      replaceAll: true,
    });
  });

  it("解析 apply_patch 的新增、更新、移动和删除操作", () => {
    const patch = [
      "*** Begin Patch",
      "*** Update File: src/old_name.py",
      "*** Move to: src/new_name.py",
      "@@ def run():",
      " x = 1",
      "-y = 2",
      "+y = 3",
      "+z = 4",
      "*** Add File: src/helper.py",
      "+def helper():",
      "+    return 42",
      "*** Delete File: src/unused.py",
      "*** End Patch",
    ].join("\n");

    const files = parseApplyPatch(patch);
    expect(files).toHaveLength(3);
    expect(files[0]).toMatchObject({
      path: "src/old_name.py",
      destination: "src/new_name.py",
      action: "move",
      additions: 2,
      deletions: 1,
    });
    expect(files[0].lines).toEqual([
      { kind: "hunk", text: "@@ def run():" },
      { kind: "ctx", text: "x = 1" },
      { kind: "del", text: "y = 2" },
      { kind: "add", text: "y = 3" },
      { kind: "add", text: "z = 4" },
    ]);
    expect(files[1]).toMatchObject({
      path: "src/helper.py",
      action: "add",
      additions: 2,
      deletions: 0,
    });
    expect(files[2]).toMatchObject({
      path: "src/unused.py",
      action: "delete",
      additions: 0,
      deletions: 0,
    });

    const payload = extractCodeDiff(
      "apply_patch",
      JSON.stringify({ input: patch }),
    );
    expect(payload?.toolKind).toBe("apply_patch");
    expect(payload?.totalAdditions).toBe(4);
    expect(payload?.totalDeletions).toBe(1);
  });

  it("对无关工具或无效补丁返回 null", () => {
    expect(
      extractCodeDiff("read_file", JSON.stringify({ path: "a.py" })),
    ).toBeNull();
    expect(extractCodeDiff("apply_patch", "{bad json")).toBeNull();
    expect(
      extractCodeDiff("apply_patch", JSON.stringify({ input: "not a patch" })),
    ).toBeNull();
  });
});
