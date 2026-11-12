// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import App from "../src/App.vue";
import MarkdownView from "../src/components/MarkdownView.vue";

class Source {
  static instances: Source[] = [];
  listener: ((e: any) => void) | null = null;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  constructor(public url: string) {
    Source.instances.push(this);
  }
  addEventListener(_type: string, fn: (e: any) => void) {
    this.listener = fn;
  }
  close() {
    this.closed = true;
  }
  emit(id: number, type: string, data: any, task_id = "one", agent_id = "main") {
    this.listener?.({
      data: JSON.stringify({ id, type, data, task_id, agent_id, time: 0 }),
    });
  }
}
const sessions = [
  {
    id: "case-a",
    title: "腔室 A",
    targets: 75,
    controls: 15,
    updated: 1,
    active: false,
  },
  {
    id: "case-b",
    title: "腔室 B",
    targets: 75,
    controls: 15,
    updated: 1,
    active: false,
  },
];
let availableBranches: any[] = [];
let requests: { url: string; body: any }[] = [];
let wrapper: ReturnType<typeof mount> | undefined;
async function settle() {
  await flushPromises();
  await new Promise((resolve) => setTimeout(resolve, 15));
  await flushPromises();
}
beforeEach(() => {
  requests = [];
  availableBranches = [];
  Source.instances = [];
  Object.defineProperty(window, "innerWidth", {
    value: 1280,
    configurable: true,
  });
  vi.stubGlobal("EventSource", Source);
  vi.stubGlobal("requestAnimationFrame", (fn: Function) =>
    setTimeout(() => fn(0), 0),
  );
  vi.stubGlobal("cancelAnimationFrame", clearTimeout);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options: any = {}) => {
      if (options.method === "POST")
        requests.push({ url, body: JSON.parse(options.body) });
      let body: any = {};
      if (url === "/api/config")
        body = {
          providers: [{ id: "glm", model: "glm-test" }],
          datasets: ["etching-v2"],
        };
      else if (url === "/api/sessions") body = sessions;
      else if (url.endsWith("/messages")) body = { task_id: "one" };
      else
        body = {
          ...sessions.find((s) => url.endsWith(s.id)),
          branches: availableBranches,
          history: [],
          artifacts: [],
          last_event_id: 0,
        };
      return { ok: true, json: async () => body };
    }),
  );
});
afterEach(() => {
  wrapper?.unmount();
  wrapper = undefined;
  vi.unstubAllGlobals();
});

describe("Vue 3 workspace", () => {
  it("renders safe Markdown tables and local image links", () => {
    wrapper = mount(MarkdownView, {
      props: {
        session: "case-a",
        text: '**证据**\n\n|指标|值|\n|---|---|\n|He|7.9|\n\n![图](artifacts/plot.png)\n<script>alert(1)</script><img src=x onerror="alert(1)">',
      },
    });
    expect(wrapper.find("table").exists()).toBe(true);
    expect(wrapper.find("img").attributes("src")).toBe(
      "/api/sessions/case-a/artifacts/artifacts/plot.png",
    );
    expect(wrapper.find("script").exists()).toBe(false);
    expect(wrapper.html()).not.toContain("onerror");
  });
  it("replays Unicode text once, hides tool argument streams, shows actual tool output", async () => {
    wrapper = mount(App);
    await settle();
    const source = Source.instances.at(-1)!;
    source.emit(1, "task.start", { mode: "research" });
    source.emit(2, "model.start", { call_id: "model" });
    source.emit(3, "model.reasoning.delta", {
      call_id: "model",
      text: "检查压力证据",
    });
    source.emit(4, "tool.arguments.delta", {
      call_id: "model",
      text: "PRIVATE_PARTIAL_ARGUMENTS",
    });
    source.emit(5, "tool.start", { cell: 1, code: 'print("real output")' });
    source.emit(6, "tool.output", { cell: 1, text: "real output" });
    source.emit(7, "tool.end", { cell: 1, ok: true, elapsed_s: 1 });
    source.emit(8, "reply.start", { message_id: "answer" });
    source.emit(9, "reply.delta", {
      message_id: "answer",
      text: "🌡️",
      offset: 0,
    });
    source.emit(10, "reply.delta", {
      message_id: "answer",
      text: " 已验证",
      offset: 2,
    });
    source.emit(10, "reply.delta", {
      message_id: "answer",
      text: " 已验证",
      offset: 2,
    });
    await settle();
    expect(wrapper.find(".message.assistant").text()).toContain("🌡️ 已验证");
    source.emit(11, "reply.commit", {
      message_id: "answer",
      markdown: "**已验证**\n\n|指标|值|\n|---|---|\n|He|7.9|",
    });
    source.emit(12, "task.end", { status: "checkpoint" });
    await settle();
    expect(wrapper.findAll(".message.assistant")).toHaveLength(1);
    expect(wrapper.find("table").exists()).toBe(true);
    expect(wrapper.text()).toContain("real output");
    expect(wrapper.text()).not.toContain("PRIVATE_PARTIAL_ARGUMENTS");
    expect(wrapper.find('[data-framework="vue3"]').exists()).toBe(true);
    expect(wrapper.find(".status-pill").text()).toBe("准备就绪");
    wrapper.unmount();
    wrapper = undefined;
    expect(source.closed).toBe(true);
  });
  it("shows progress during research and keeps it alongside the final result", async () => {
    wrapper = mount(App);
    await settle();
    const source = Source.instances.at(-1)!;
    source.emit(1, "task.start", { mode: "research" });
    source.emit(2, "reply.commit", {
      message_id: "progress-one",
      markdown: "已检查数据\n\n|指标|值|\n|---|---|\n|样本|2|\n\n![图](artifacts/check.png)",
      publication: "progress",
    });
    await settle();
    expect(wrapper.findAll(".message.assistant")).toHaveLength(1);
    expect(wrapper.find(".conversation table").exists()).toBe(true);
    expect(wrapper.find(".conversation img").attributes("src")).toBe(
      "/api/sessions/case-a/artifacts/artifacts/check.png",
    );
    expect(wrapper.find(".activity-line").exists()).toBe(true);
    source.emit(3, "reply.commit", {
      message_id: "progress-one",
      markdown: "已完成数据检查，继续比较",
      publication: "progress",
    });
    source.emit(4, "reply.commit", {
      message_id: "final",
      markdown: "最终研究结果",
      publication: "final",
    });
    source.emit(5, "task.end", { status: "checkpoint" });
    await settle();
    expect(wrapper.findAll(".message.assistant")).toHaveLength(2);
    expect(wrapper.find(".conversation").text()).toContain("已完成数据检查，继续比较");
    expect(wrapper.find(".conversation").text()).toContain("最终研究结果");
  });
  it("shows main and runtime child activity together with distinct prefixes", async () => {
    wrapper = mount(App);
    await settle();
    const source = Source.instances.at(-1)!;
    const child = "32018b59828a47eab86c5896e7b653b0";
    source.emit(1, "task.start", { mode: "research" });
    source.emit(2, "model.start", { call_id: "main-model" });
    source.emit(3, "model.reasoning.delta", { call_id: "main-model", text: "主任务推理" });
    source.emit(4, "model.start", { call_id: "child-model" }, "one", child);
    source.emit(5, "model.reasoning.delta", {
      call_id: "child-model", text: "子任务独立推理",
    }, "one", child);
    source.emit(6, "agent.tool.start", {
      tool_call_id: "same", name: "agent__wait", arguments: {},
    });
    source.emit(7, "agent.tool.start", {
      tool_call_id: "same", name: "sandbox__run_shell", arguments: { command: "child check" },
    }, "one", child);
    source.emit(8, "agent.tool.start", {
      tool_call_id: "same", name: "read_file", arguments: { path: "second-check.txt" },
    }, "one", "another-42fe");
    await settle();
    const panel = wrapper.find(".research-panel");
    expect(wrapper.findAll(".direction-bar option")).toHaveLength(1);
    expect(panel.text()).toContain("主任务推理");
    expect(panel.text()).toContain("子任务独立推理");
    expect(panel.findAll(".thought-card > summary").map((card) => card.text())).toEqual([
      expect.stringContaining("Sub Agent · 53b0"),
      expect.stringContaining("Main Agent"),
    ]);
    expect(panel.findAll(".tool-card > summary").map((card) => card.text())).toEqual([
      expect.stringContaining("Sub Agent · 42fe · 读取研究记录"),
      expect.stringContaining("Sub Agent · 53b0 · 执行文件操作"),
      expect.stringContaining("Main Agent · 等待子 Agent 结果"),
    ]);
    expect(panel.findAll(".live-metrics b").at(-1)?.text()).toBe("3");
    source.emit(9, "agent.tool.end", {
      tool_call_id: "same", name: "sandbox__run_shell", ok: true, output: "子任务真实输出",
    }, "one", child);
    await settle();
    expect(panel.findAll(".tool-card.running")).toHaveLength(2);
    expect(panel.find(".tool-card.ok").text()).toContain("Sub Agent · 53b0");
    expect(panel.find(".tool-card.ok").text()).toContain("子任务真实输出");
    expect(panel.findAll(".live-metrics b").at(-1)?.text()).toBe("2");
  });
  it("keeps older active agents visible while another agent produces recent history", async () => {
    wrapper = mount(App);
    await settle();
    const source = Source.instances.at(-1)!;
    let id = 1;
    source.emit(id++, "model.start", { call_id: "main-waiting" });
    source.emit(id++, "agent.tool.start", { tool_call_id: "main-tool", name: "agent__wait" });
    for (let index = 0; index < 65; index++) {
      source.emit(id++, "model.start", { call_id: `child-${index}` }, "one", "child-1234");
      source.emit(id++, "model.stream.end", { call_id: `child-${index}` }, "one", "child-1234");
      source.emit(id++, "agent.tool.start", {
        tool_call_id: `child-${index}`, name: "read_file",
      }, "one", "child-1234");
      source.emit(id++, "agent.tool.end", {
        tool_call_id: `child-${index}`, ok: true,
      }, "one", "child-1234");
    }
    await settle();
    expect(wrapper.findAll(".thought-card > summary").map((card) => card.text())).toContainEqual(
      expect.stringContaining("Main Agent · 模型正在生成"),
    );
    expect(wrapper.find(".tool-card.running").text()).toContain("Main Agent · 等待子 Agent 结果");
    expect(wrapper.findAll(".live-metrics b").at(-1)?.text()).toBe("1");
    source.emit(id++, "model.stream.end", { call_id: "main-waiting" });
    source.emit(id++, "agent.tool.end", { tool_call_id: "main-tool", ok: true });
    await settle();
    expect(wrapper.findAll(".thought-card")).toHaveLength(12);
    expect(wrapper.findAll(".tool-card")).toHaveLength(40);
    expect(wrapper.findAll(".live-metrics b").at(-1)?.text()).toBe("0");
  });
  it("sends with Vue v-model, respects IME and discards old session callbacks", async () => {
    wrapper = mount(App);
    await settle();
    const source = Source.instances.at(-1)!;
    const input = wrapper.find("textarea");
    await input.setValue("检查异常");
    await input.trigger("keydown", { key: "Enter", isComposing: true });
    expect(requests).toHaveLength(0);
    await input.trigger("keydown", { key: "Enter", isComposing: false });
    await settle();
    expect(requests).toHaveLength(1);
    expect(requests[0].body.message).toBe("检查异常");
    expect((input.element as HTMLTextAreaElement).value).toBe("");
    await wrapper.findAll("nav .session")[1].trigger("click");
    await settle();
    expect(source.closed).toBe(true);
    source.emit(30, "reply.commit", {
      message_id: "old",
      markdown: "STALE_RESULT",
    });
    await settle();
    expect(wrapper.text()).not.toContain("STALE_RESULT");
    expect(wrapper.find(".breadcrumb").text()).toContain("腔室 B");
  });
});

describe("research branch UI", () => {
  it("isolates conversation and guidance while keeping all agents in the live panel", async () => {
    wrapper = mount(App);
    await settle();
    const source = Source.instances.at(-1)!;
    availableBranches = [
      {
        id: "research-a",
        title: "热过程",
        parent_id: "main",
        mission: "one",
        task: "thermal",
        state: "running",
      },
      {
        id: "research-b",
        title: "压力",
        parent_id: "main",
        mission: "one",
        task: "pressure",
        state: "running",
      },
    ];
    source.emit(1, "task.start", { mode: "research", target_agent_id: "main" });
    source.emit(2, "branch.created", { branch: availableBranches[0] });
    source.emit(3, "branch.created", { branch: availableBranches[1] });
    source.emit(4, "reply.commit", {
      message_id: "same",
      markdown: "主研究消息",
      publication: "progress",
    });
    source.listener?.({
      data: JSON.stringify({
        id: 5,
        type: "reply.commit",
        task_id: "one",
        agent_id: "research-a",
        parent_agent_id: "main",
        data: {
          message_id: "same",
          markdown: "热过程独立消息",
          publication: "progress",
        },
        time: 0,
      }),
    });
    source.listener?.({
      data: JSON.stringify({
        id: 6,
        type: "tool.start",
        task_id: "one",
        agent_id: "research-a",
        data: { cell: 1, code: 'print("thermal")' },
        time: 0,
      }),
    });
    source.listener?.({
      data: JSON.stringify({
        id: 7,
        type: "tool.start",
        task_id: "one",
        agent_id: "research-b",
        data: { cell: 1, code: 'print("pressure")' },
        time: 0,
      }),
    });
    await settle();
    expect(wrapper.text()).toContain("主研究消息");
    expect(wrapper.text()).not.toContain("热过程独立消息");
    await wrapper.find(".direction-bar select").setValue("research-a");
    await settle();
    expect(wrapper.text()).toContain("热过程独立消息");
    expect(wrapper.text()).not.toContain("主研究消息");
    expect(wrapper.text()).toContain('print("thermal")');
    expect(wrapper.text()).toContain('print("pressure")');
    await wrapper.find("textarea").setValue("检查热响应");
    await wrapper.find("textarea").trigger("keydown", { key: "Enter" });
    await settle();
    expect(requests.at(-1)?.body.agent_id).toBe("research-a");
  });
  it("shows review failures, bounded reflection and shared budget without claiming causal truth", async () => {
    wrapper = mount(App);
    await settle();
    const source = Source.instances.at(-1)!;
    source.emit(1, "review.start", { revision: 0 });
    source.emit(2, "validation.issue", { message: "saved-table mismatch" });
    source.emit(3, "reflection.verdict", {
      iteration: 2,
      is_satisfactory: false,
    });
    source.emit(4, "review.end", {
      status: "budget_exhausted",
      revision: 2,
      issues: [
        {
          severity: "blocking",
          problem: "证据不足",
          suggested_check: "独立复核",
        },
      ],
    });
    source.emit(5, "budget.snapshot", {
      token_limit: 4000000,
      charged_tokens: 12000,
      reserved_tokens: 0,
    });
    await settle();
    expect(wrapper.text()).toContain("预算耗尽，保留局限");
    expect(wrapper.text()).toContain("证据不足");
    expect(wrapper.text()).toContain("不等于根因认证");
    expect(wrapper.text()).toContain("saved-table mismatch");
    expect(wrapper.text()).toContain("需要修订");
    expect(wrapper.text()).toContain("12,000");
  });
  it("keeps relative branch images inside their own artifact namespace", () => {
    wrapper = mount(MarkdownView, {
      props: {
        session: "case-a",
        agent: "research-a",
        text: "![图](artifacts/plot.png)",
      },
    });
    expect(wrapper.find("img").attributes("src")).toBe(
      "/api/sessions/case-a/artifacts/branches/research-a/artifacts/plot.png",
    );
  });
});
