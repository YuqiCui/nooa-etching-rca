// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import App from "../src/App.vue";

class Source {
  static instances: Source[] = [];
  listener: ((e: any) => void) | null = null;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) {
    Source.instances.push(this);
  }
  addEventListener(_type: string, fn: (e: any) => void) {
    this.listener = fn;
  }
  close() {}
  emit(
    id: number,
    type: string,
    data: any,
    task_id = "task-1",
    agent_id = "main",
    parent_agent_id?: string,
  ) {
    this.listener?.({
      data: JSON.stringify({
        id,
        type,
        data,
        task_id,
        agent_id,
        parent_agent_id,
        time: 1700000000 + id,
      }),
    });
  }
}

let wrapper: ReturnType<typeof mount> | undefined;

async function settle() {
  await flushPromises();
  await new Promise((resolve) => setTimeout(resolve, 20));
  await flushPromises();
}

beforeEach(() => {
  Source.instances = [];
  vi.stubGlobal("EventSource", Source);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string) => {
      const url = String(input);
      let body: any = {};
      if (url === "/api/sessions") {
        body = [
          {
            id: "case-1",
            title: "刻蚀漂移定位",
            targets: 40,
            controls: 10,
            updated: 1,
            active: false,
          },
        ];
      } else if (url === "/api/config") {
        body = { providers: [{ id: "glm", model: "glm-4" }], datasets: ["demo"] };
      } else if (url === "/api/sessions/case-1") {
        body = {
          id: "case-1",
          title: "刻蚀漂移定位",
          targets: 40,
          controls: 10,
          updated: 1,
          active: false,
          artifacts: [],
          branches: [],
          formula_studies: [],
          history: [],
          last_event_id: 0,
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => body,
      } as Response;
    }),
  );
});

afterEach(() => {
  wrapper?.unmount();
  vi.unstubAllGlobals();
});

describe("执行轨迹视图与子 Agent 派生点分屏", () => {
  it("支持双 Tab 切换、隐藏底部输入框、折叠思考与切换工具结果、从派生点展开子 Agent", async () => {
    wrapper = mount(App);
    await settle();

    // 默认在研究对话 Tab，底部输入框可见
    expect(wrapper.find(".composer-wrap").exists()).toBe(true);

    const es = Source.instances[0];
    // 系统提示词与用户首条消息
    es.emit(1, "agent.system_prompt", { content: "你是刻蚀故障定位主 Router Agent。" });
    es.emit(2, "user.message", { message: "排查 RF 反射功率异常原因" });
    // 主 Agent 第 1 步：思考 + 工具调用 + 工具返回
    es.emit(3, "model.start", { call_id: "c1" });
    es.emit(4, "model.reasoning.delta", {
      call_id: "c1",
      text: "先检查 RF 反射功率趋势",
    });
    es.emit(5, "model.stream.end", { call_id: "c1" });
    es.emit(6, "agent.tool.start", {
      tool_call_id: "t1",
      name: "run_code",
      arguments: { code: "print('rf_check')" },
    });
    es.emit(7, "agent.tool.end", {
      tool_call_id: "t1",
      ok: true,
      elapsed_s: 1.2,
      output: "RF 反射功率在 Step 3 升高 18%",
    });

    // 主 Agent 第 2 步：派生子 Agent
    es.emit(8, "model.start", { call_id: "c2" });
    es.emit(9, "model.reasoning.delta", {
      call_id: "c2",
      text: "需要派生子任务排查气体流量",
    });
    es.emit(10, "model.stream.end", { call_id: "c2" });
    es.emit(11, "agent.tool.start", {
      tool_call_id: "t2",
      name: "spawn_research",
      arguments: { title: "气体流量排查" },
    });
    es.emit(
      12,
      "branch.created",
      {
        branch: {
          id: "sub-gas",
          parent_id: "main",
          mission: "排查气体流量",
          title: "气体流量排查",
          task: "检查 MFC 漂移并输出统计表",
          state: "running",
        },
      },
      "task-1",
      "sub-gas",
      "main",
    );
    es.emit(13, "agent.tool.end", {
      tool_call_id: "t2",
      ok: true,
      elapsed_s: 0.3,
      output: "branch sub-gas created",
    });

    // 子 Agent System Prompt 与第 1 步
    es.emit(
      14,
      "agent.system_prompt",
      { content: "你是子任务执行 Worker。" },
      "task-1",
      "sub-gas",
    );
    es.emit(15, "model.start", { call_id: "sc1" }, "task-1", "sub-gas");
    es.emit(
      16,
      "model.reasoning.delta",
      { call_id: "sc1", text: "读取 CF4 流量序列" },
      "task-1",
      "sub-gas",
    );
    es.emit(17, "model.stream.end", { call_id: "sc1" }, "task-1", "sub-gas");
    es.emit(
      18,
      "agent.tool.start",
      {
        tool_call_id: "st1",
        name: "run_code",
        arguments: { code: "check_cf4()" },
      },
      "task-1",
      "sub-gas",
    );
    es.emit(
      19,
      "agent.tool.end",
      {
        tool_call_id: "st1",
        ok: true,
        elapsed_s: 0.8,
        output: "CF4 流量正常",
      },
      "task-1",
      "sub-gas",
    );

    await settle();

    // 切换到「执行轨迹」Tab
    const tabs = wrapper.findAll('.main-tabs button[role="tab"]');
    await tabs[1].trigger("click");
    await settle();

    // 执行轨迹 Tab 隐藏底部输入框，展示 System Prompt 与用户消息节点
    expect(wrapper.find(".composer-wrap").exists()).toBe(false);
    expect(wrapper.find(".traj-view").exists()).toBe(true);
    expect(wrapper.find(".traj-sys-prompt").text()).toContain("System Prompt");
    expect(wrapper.find(".traj-sys-body").text()).toContain("主 Router Agent");
    expect(wrapper.find(".traj-user-card").text()).toContain("排查 RF 反射功率异常原因");

    // 默认子 Agent 折叠，单栏展示主 Agent 2 步
    expect(wrapper.find(".traj-single").exists()).toBe(true);
    expect(wrapper.find(".traj-split").exists()).toBe(false);
    expect(wrapper.findAll(".traj-step")).toHaveLength(2);

    // 模型思考默认折叠为 thinking...
    expect(wrapper.find(".thinking-body").exists()).toBe(false);
    expect(wrapper.find(".thinking-label").text()).toBe("thinking...");
    await wrapper.find(".thinking-toggle").trigger("click");
    expect(wrapper.find(".thinking-body").text()).toContain("先检查 RF 反射功率趋势");

    // 工具卡片标题先展示原始工具名，再展示中文说明；默认显示调用内容，点击“结果”切换显示返回值
    const firstTool = wrapper.find(".traj-tool-card");
    expect(firstTool.find(".tool-raw-name").text()).toBe("run_code");
    expect(firstTool.find(".tool-desc").text()).toBe("执行研究代码");
    expect(firstTool.find(".traj-tool-title").text()).toContain("run_code执行研究代码");
    expect(firstTool.find(".traj-code").text()).toContain("rf_check");
    expect(firstTool.find(".traj-output").exists()).toBe(false);
    const modeButtons = firstTool.findAll(".tool-mode-switch button");
    await modeButtons[1].trigger("click");
    expect(firstTool.find(".traj-output").text()).toContain("RF 反射功率在 Step 3 升高 18%");

    // 点击第 2 步上的折叠子 Agent 卡片，展开左右独立滚动分屏
    await wrapper.find(".traj-spawn-card").trigger("click");
    await settle();

    expect(wrapper.find(".traj-split-panes").exists()).toBe(true);
    expect(wrapper.findAll(".traj-pane-main .traj-step")).toHaveLength(2);
    expect(wrapper.findAll(".traj-pane-main .traj-step")[1].classes()).toContain(
      "spawn-anchor",
    );
    expect(wrapper.findAll(".traj-pane-sub .traj-step")).toHaveLength(1);
    expect(wrapper.find(".traj-pane-sub .traj-sys-body").text()).toContain(
      "子任务执行 Worker",
    );
    expect(wrapper.find(".traj-pane-sub .traj-task-card").text()).toContain(
      "检查 MFC 漂移并输出统计表",
    );
    expect(wrapper.find(".traj-pane-sub").text()).toContain("check_cf4()");

    // 点击收起按钮可折叠回单栏
    await wrapper.find(".sub-close-btn").trigger("click");
    expect(wrapper.find(".traj-single").exists()).toBe(true);
    expect(wrapper.find(".traj-split-panes").exists()).toBe(false);

    // report / finish 工具卡片与对话界面一致按 Markdown 渲染（表格、图片路径）
    es.emit(20, "model.start", { call_id: "c3" });
    es.emit(21, "model.stream.end", { call_id: "c3" });
    es.emit(22, "agent.tool.start", {
      tool_call_id: "t3",
      name: "report",
      arguments: {
        content: "**阶段发现**\n\n|传感器|变化|\n|---|---|\n|RF|+18%|\n\n![图](artifacts/rf.png)",
      },
    });
    es.emit(23, "agent.tool.end", {
      tool_call_id: "t3",
      ok: true,
      elapsed_s: 0.1,
      output: '{"ok": true}',
    });
    await settle();

    const mdBody = wrapper.find(".traj-markdown-body");
    expect(mdBody.exists()).toBe(true);
    expect(mdBody.find("table").exists()).toBe(true);
    expect(mdBody.find("img").attributes("src")).toBe(
      "/api/sessions/case-1/artifacts/artifacts/rf.png",
    );
  });

  it("为 notebook__write_new_cell 和 notebook__write_cells 提供独立代码 Tab 并按 Python 语法渲染", async () => {
    wrapper = mount(App);
    await settle();

    const es = Source.instances[0];
    es.emit(1, "model.start", { call_id: "c1" });
    es.emit(2, "model.stream.end", { call_id: "c1" });
    es.emit(3, "agent.tool.start", {
      tool_call_id: "t-write-one",
      name: "notebook__write_new_cell",
      arguments: {
        notebook_name: "rf_check",
        source: "import numpy as np\n# 单 cell 检查\nprint(np.mean([1, 2, 3]))",
      },
    });
    es.emit(4, "agent.tool.end", {
      tool_call_id: "t-write-one",
      ok: true,
      elapsed_s: 0.2,
      output: JSON.stringify({
        notebook_name: "rf_check",
        cell_id: "cell_01",
        revision: 1,
      }),
    });
    es.emit(5, "agent.tool.start", {
      tool_call_id: "t-write-many",
      name: "notebook__write_cells",
      arguments: {
        notebook_name: "rf_check",
        cells: [
          { source: "a = 10\nb = 20" },
          { source: "def total(x, y):\n    return x + y" },
        ],
      },
    });
    es.emit(6, "agent.tool.end", {
      tool_call_id: "t-write-many",
      ok: true,
      elapsed_s: 0.3,
      output: JSON.stringify({
        notebook_name: "rf_check",
        count: 2,
        cells: [
          { cell_id: "cell_02", revision: 1 },
          { cell_id: "cell_03", revision: 1 },
        ],
      }),
    });
    await settle();

    // 切换到「执行轨迹」Tab
    const tabs = wrapper.findAll('.main-tabs button[role="tab"]');
    await tabs[1].trigger("click");
    await settle();

    const cards = wrapper.findAll(".traj-tool-card");
    expect(cards).toHaveLength(2);

    // 1. notebook__write_new_cell: 具备「代码 / 调用 / 结果」三个 Tab，默认选中「代码」
    const singleCard = cards[0];
    const singleTabs = singleCard.findAll(".tool-mode-switch button");
    expect(singleTabs.map((b) => b.text())).toEqual(["代码", "调用", "结果"]);
    expect(singleTabs[0].classes()).toContain("on");
    expect(singleCard.find(".py-cells-view").exists()).toBe(true);
    expect(singleCard.find(".py-meta-chip").text()).toContain("rf_check.ipynb");
    expect(singleCard.find(".py-cell-id").text()).toBe("cell_01");
    expect(singleCard.find(".py-tok-keyword").text()).toBe("import");
    expect(singleCard.find(".py-tok-comment").text()).toContain("# 单 cell 检查");

    // 切换到「调用」查看原始 JSON，再切换到「结果」查看返回，再切回「代码」
    await singleTabs[1].trigger("click");
    expect(singleCard.find(".py-cells-view").exists()).toBe(false);
    expect(singleCard.find(".traj-code").text()).toContain('"notebook_name": "rf_check"');
    await singleTabs[2].trigger("click");
    expect(singleCard.find(".traj-output").text()).toContain('"cell_id":"cell_01"');
    await singleTabs[0].trigger("click");
    expect(singleCard.find(".py-cells-view").exists()).toBe(true);

    // 2. notebook__write_cells: 渲染多个独立 Python cell 块
    const multiCard = cards[1];
    const multiTabs = multiCard.findAll(".tool-mode-switch button");
    expect(multiTabs.map((b) => b.text())).toEqual(["代码", "调用", "结果"]);
    expect(multiCard.findAll(".py-cell-block")).toHaveLength(2);
    expect(multiCard.findAll(".py-cell-id").map((n) => n.text())).toEqual([
      "cell_02",
      "cell_03",
    ]);
    expect(multiCard.find(".py-tok-def-name").text()).toBe("total");
  });

  it("为 write_file 渲染代码视图，并为 edit_file 与 apply_patch 渲染 Git 风格红绿 Diff", async () => {
    wrapper = mount(App);
    await settle();

    const es = Source.instances[0];
    es.emit(1, "model.start", { call_id: "c1" });
    es.emit(2, "model.stream.end", { call_id: "c1" });
    es.emit(3, "agent.tool.start", {
      tool_call_id: "t-write-file",
      name: "write_file",
      arguments: {
        path: "scripts/fit.py",
        content: "import math\ndef area(r):\n    return math.pi * r * r",
      },
    });
    es.emit(4, "agent.tool.end", {
      tool_call_id: "t-write-file",
      ok: true,
      elapsed_s: 0.1,
      output: "created",
    });
    es.emit(5, "agent.tool.start", {
      tool_call_id: "t-edit-file",
      name: "edit_file",
      arguments: {
        path: "scripts/fit.py",
        old_string: "def area(r):\n    return math.pi * r * r",
        new_string: "def area(r):\n    return round(math.pi * r * r, 4)",
      },
    });
    es.emit(6, "agent.tool.end", {
      tool_call_id: "t-edit-file",
      ok: true,
      elapsed_s: 0.1,
      output: "edited",
    });
    es.emit(7, "agent.tool.start", {
      tool_call_id: "t-apply-patch",
      name: "apply_patch",
      arguments: {
        input: [
          "*** Begin Patch",
          "*** Update File: scripts/fit.py",
          "@@ def area(r):",
          " import math",
          "-return round(math.pi * r * r, 4)",
          "+return round(math.pi * (r ** 2), 4)",
          "*** End Patch",
        ].join("\n"),
      },
    });
    es.emit(8, "agent.tool.end", {
      tool_call_id: "t-apply-patch",
      ok: true,
      elapsed_s: 0.2,
      output: JSON.stringify({ completed: [{ kind: "update", path: "scripts/fit.py" }] }),
    });
    await settle();

    // 切换到「执行轨迹」Tab
    const tabs = wrapper.findAll('.main-tabs button[role="tab"]');
    await tabs[1].trigger("click");
    await settle();

    const cards = wrapper.findAll(".traj-tool-card");
    expect(cards).toHaveLength(3);

    // 1. write_file: 默认选中「代码」，与 notebook cell 一致高亮 Python
    const writeCard = cards[0];
    expect(
      writeCard.findAll(".tool-mode-switch button").map((b) => b.text()),
    ).toEqual(["代码", "调用", "结果"]);
    expect(writeCard.find(".py-cells-view").exists()).toBe(true);
    expect(writeCard.find(".py-cell-label").text()).toContain("scripts/fit.py");
    expect(writeCard.find(".py-tok-keyword").text()).toBe("import");

    // 2. edit_file: 默认选中「变更」，展示红色 - 与绿色 + 差异行
    const editCard = cards[1];
    const editTabs = editCard.findAll(".tool-mode-switch button");
    expect(editTabs.map((b) => b.text())).toEqual(["变更", "调用", "结果"]);
    expect(editTabs[0].classes()).toContain("on");
    expect(editCard.find(".code-diff-view").exists()).toBe(true);
    expect(editCard.find(".diff-line-del").text()).toContain(
      "return math.pi * r * r",
    );
    expect(editCard.find(".diff-line-add").text()).toContain(
      "return round(math.pi * r * r, 4)",
    );

    // 3. apply_patch: 默认选中「变更」，展示 hunk、红色 - 与绿色 + 差异行
    const patchCard = cards[2];
    const patchTabs = patchCard.findAll(".tool-mode-switch button");
    expect(patchTabs.map((b) => b.text())).toEqual(["变更", "调用", "结果"]);
    expect(patchCard.find(".code-diff-view").exists()).toBe(true);
    expect(patchCard.find(".diff-line-hunk").text()).toContain("@@ def area(r):");
    expect(patchCard.find(".diff-line-del .diff-sign").text()).toBe("-");
    expect(patchCard.find(".diff-line-add .diff-sign").text()).toBe("+");

    // 切换到「调用」可查看原始 JSON
    await patchTabs[1].trigger("click");
    expect(patchCard.find(".code-diff-view").exists()).toBe(false);
    expect(patchCard.find(".traj-code").text()).toContain("*** Begin Patch");
  });
});
