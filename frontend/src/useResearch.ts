import {
  reactive,
  ref,
  computed,
  watch,
  onMounted,
  onUnmounted,
  nextTick,
} from "vue";
import { api } from "./api";
import { newRequestId } from "./requestId";
import type {
  Evt,
  Session,
  Msg,
  ModelCall,
  Tool,
  Artifact,
  Branch,
  FormulaStudy,
} from "./types";

export function useResearch() {
  const s = reactive({
    sessions: [] as Session[],
    selected: "",
    meta: null as Session | null,
    messages: [] as Msg[],
    calls: [] as ModelCall[],
    tools: [] as Tool[],
    artifacts: [] as Artifact[],
    providers: [] as { id: string; model: string }[],
    datasets: [] as string[],
    provider: "glm",
    mode: "research",
    draft: "",
    active: false,
    phase: "待命",
    connected: false,
    error: "",
    tab: "live",
    panel: window.innerWidth >= 1100,
    newDialog: false,
    newTitle: "设备与工艺数据研究",
    dataset: "",
    eventCount: 0,
    newBelow: false,
    pending: false,
    branches: [] as Branch[],
    formulaStudies: [] as FormulaStudy[],
    selectedAgent: "main",
    activeAgent: "main",
    agentPhases: {} as Record<string, string>,
    reviews: {} as Record<
      string,
      { status: string; revision: number; issues: any[] }
    >,
    notices: {} as Record<string, string[]>,
    budget: null as Record<string, number> | null,
    replaying: false,
    datasetRevision:0,
  });
  const scroll = ref<HTMLDivElement | null>(null),
    textarea = ref<HTMLTextAreaElement | null>(null);
  let source: EventSource | null = null,
    frame = 0,
    queue: Evt[] = [],
    cursor = 0,
    nearBottom = true,
    replayUntil = 0,
    generation = 0,
    artifactRequest = 0;
  let artifactRefresh: ReturnType<typeof setTimeout> | undefined;
  let request: { session: string; agent:string; text: string; id: string } | null = null;
  const latest = computed(() => s.calls.at(-1)),
    thoughtChars = computed(() =>
      s.calls.reduce((n, c) => n + c.thinking.length, 0),
    ),
    activeTools = computed(
      () =>
        s.tools.filter(
          (t) =>
            t.state === "running" && (t.agent_id || "main") === s.selectedAgent,
        ).length,
    );
  const visibleMessages = computed(() =>
    s.messages.filter((m) => (m.agent_id || "main") === s.selectedAgent),
  );
  const visibleCalls = computed(() =>
      s.calls
        .filter((c) => (c.agent_id || "main") === s.selectedAgent)
        .slice(-12)
        .reverse(),
    ),
    visibleTools = computed(() =>
      s.tools
        .filter((t) => (t.agent_id || "main") === s.selectedAgent)
        .slice(-40)
        .reverse(),
    );
  const selectedBranch = computed(() =>
    s.branches.find((b) => b.id === s.selectedAgent),
  );
  const selectedReview = computed(() => s.reviews[s.selectedAgent]);
  const visibleArtifacts = computed(() =>
    s.artifacts.filter((f) => (f.agent_id || "main") === s.selectedAgent),
  );
  const targetAvailable = computed(
    () =>
      !s.active ||
      (s.selectedAgent === "main"
        ? s.activeAgent === "main"
        : ["queued", "starting", "running"].includes(
            selectedBranch.value?.state || "",
          )),
  );
  const canSend = computed(
    () =>
      !!s.draft.trim() &&
      !s.pending &&
      !s.replaying &&
      !!s.selected &&
      targetAvailable.value,
  );
  async function refresh() {
    try {
      s.sessions = await api("/sessions");
      if (!s.selected && s.sessions.length) s.selected = s.sessions[0].id;
    } catch (e: any) {
      s.error = e.message;
    }
  }
  async function loadArtifacts() {
    const name = s.selected;
    if (!name) return;
    const sequence = ++artifactRequest;
    const own = generation;
    try {
      const data = await api("/sessions/" + name);
      if (s.selected === name && own === generation && sequence === artifactRequest) {
        s.artifacts = data.artifacts;
        s.branches = data.branches || [];
        s.formulaStudies = data.formula_studies || [];
      }
    } catch {}
  }
  function scheduleArtifacts() {
    if (artifactRefresh) return;
    artifactRefresh = setTimeout(() => {
      artifactRefresh = undefined;
      void loadArtifacts();
    }, 75);
  }
  function call(id: string, agentId = "main") {
    let c = s.calls.find((c) => c.id === id);
    if (!c) {
      c = { id, agent_id: agentId, thinking: "", content: "", ended: false };
      s.calls.push(c);
      if(s.calls.length>60)s.calls.splice(0,s.calls.length-60);
      c = s.calls.at(-1)!;
    }
    return c;
  }
  function handle(events: Evt[]) {
    s.eventCount += events.length;
    const boundary = replayUntil;
    const replayDone = !!boundary && events.at(-1)!.id >= boundary;
    for (const e of events) {
      const d = e.data,
        agentId = e.agent_id || "main",
        prefix = e.task_id + ":" + (agentId === "main" ? "" : agentId + ":"),
        replay = e.id <= boundary;
      const phase = (v: string) => {
        if (!replay) {
          s.agentPhases[agentId] = v;
          if (agentId === s.selectedAgent) s.phase = v;
        }
      };
      if (e.type.startsWith('formula.') && !replay) {
        scheduleArtifacts();
        if (e.type === 'formula.evaluation.start') phase('正在比较专家公式');
        if (e.type === 'formula.replay.start') phase('正在重算最佳公式');
        if (e.type === 'formula.selection.frozen') phase('最佳候选已重跑核对');
      }
      if(e.type.startsWith('dataset.') && !replay){s.datasetRevision++;scheduleArtifacts();}
      switch (e.type) {
        case "user.message":
          s.messages.push({
            id: "user-" + e.id,
            role: "user",
            agent_id: agentId,
            markdown: d.message,
            time: e.time,
          });
          break;
        case "task.start":
          if (!replay) {
            s.active = true;
            s.error = "";
            void refresh();
          }
          s.activeAgent = d.target_agent_id || "main";
          s.mode = d.mode === "discuss" ? "discuss" : "research";
          phase("研究已开始");
          break;
        case "task.stopping":
          phase("正在保存并停止");
          break;
        case "task.end":
          for (const t of s.tools)
            if (t.state === "running") {
              t.state = "error";
              t.output += "\n本轮已结束，此调用没有成功完成记录。";
            }
          for (const c of s.calls) c.ended = true;
          for (const m of s.messages)
            if (m.provisional && m.id.startsWith(prefix)) {
              m.provisional = false;
              m.markdown += "\n\n> 此段未提交，生成过程已中断。";
            }
          if (!replay) {
            s.active = false;
            void refresh();
            void loadArtifacts();
          }
          phase(
            d.status === "checkpoint"
              ? "本轮完成"
              : d.status === "cancelled"
                ? "已停止"
                : "本轮中断",
          );
          break;
        case "task.error":
          if (!replay) {
            s.error = d.message || "本轮中断";
            phase("正在保存中断记录");
          }
          break;
        case "steering.applied":
          phase("已收到追加指导");
          break;
        case "branch.created":
        case "branch.state": {
          const row = d.branch as Branch;
          const at = s.branches.findIndex((x) => x.id === row.id);
          if (at < 0) s.branches.push(row);
          else s.branches[at] = row;
          if (!replay) void loadArtifacts();
          break;
        }
        case 'branch.stopping':{const branch=s.branches.find(b=>b.id===agentId);if(branch)branch.state='stopping';phase('正在保存并停止');break;}
        case "branch.guidance":
          s.messages.push({
            id: "guidance-" + e.id,
            role: "assistant",
            agent_id: agentId,
            markdown: "主研究指导：" + d.message,
          });
          break;
        case "review.start":
          s.reviews[agentId] = {
            status: "reviewing",
            revision: d.revision,
            issues: [],
          };
          phase("正在审查证据");
          break;
        case "review.end":
          s.reviews[agentId] = {
            status: d.status,
            revision: d.revision,
            issues: d.issues || d.review?.issues || [],
          };
          phase(
            d.status === "passed"
              ? "审查通过"
              : d.status === "needs_revision"
                ? "按审查意见补证"
                : d.status === 'revision_limit' ? '达到修订上限，结论待复核' : d.status === 'not_required' ? '按需交付图表' : d.status === "not_reviewed" ? "尚未完成有效审查" : "审查预算耗尽，保留局限",
          );
          break;
        case "validation.issue":
        case "review.invalid":
        case "loop.guard":
        case "reflection.verdict": {
          const text =
            e.type === "reflection.verdict"
              ? "局部反思 " +
                d.iteration +
                "：" +
                (d.is_satisfactory ? "通过" : "需要修订") + (d.issues?.length ? '；' + d.issues.join('；') : '')
              : d.message;
          (s.notices[agentId] ??= []).push(text);
          if (s.notices[agentId].length > 20) s.notices[agentId].shift();
          break;
        }
        case "reflection.end": {
          (s.notices[agentId] ??= []).push('局部反思结束：' + (d.status === 'passed' ? '检查通过' : d.status === 'budget_exhausted' ? '预算耗尽，未通过' : d.status === 'not_satisfactory' ? '反思轮次结束，仍需补证' : '中断，已有记录保留'));
          break;
        }
        case "budget.usage":
        case "budget.snapshot":
        case "budget.exhausted":
          s.budget = d;
          if (e.type === "budget.exhausted") phase("共享预算已耗尽");
          break;
        case "model.waiting":
          phase("模型正在生成");
          break;
        case "context.summary":
          phase("已整理上下文，继续研究");
          break;
        case "reply.start": {
          const id = prefix + d.message_id;
          let m = s.messages.find((m) => m.id === id);
          if (m) {
            m.markdown = "";
            m.provisional = true;
          } else
            s.messages.push({
              id,
              role: "assistant",
              agent_id: agentId,
              markdown: "",
              provisional: true,
            });
          phase("正在回复");
          break;
        }
        case "reply.delta": {
          const id = prefix + d.message_id;
          let m = s.messages.find((m) => m.id === id);
          if (!m)
            s.messages.push({
              id,
              role: "assistant",
              agent_id: agentId,
              markdown: d.text,
              provisional: true,
            });
          else if (
            d.offset === 0 ||
            Array.from(m.markdown).length === d.offset
          ) {
            m.markdown = d.offset === 0 ? d.text : m.markdown + d.text;
            m.provisional = true;
          }
          phase("正在回复");
          break;
        }
        case "reply.commit": {
          const id = prefix + d.message_id;
          const m = s.messages.find((m) => m.id === id);
          if (m) {
            m.markdown = d.markdown;
            m.provisional = false;
            m.publication = d.publication;
            m.review_status = d.review_status;
          } else
            s.messages.push({
              id,
              role: "assistant",
              agent_id: agentId,
              publication: d.publication,
              review_status: d.review_status,
              markdown: d.markdown,
              provisional: false,
            });
          if (!replay) void loadArtifacts();
          break;
        }
        case "model.start":
          call(d.call_id, agentId);
          break;
        case "model.reasoning.delta":
          call(d.call_id, agentId).thinking += d.text;
          break;
        case "model.content.delta":
          call(d.call_id, agentId).content += d.text;
          break;
        // Historical tool.arguments.delta events are intentionally ignored too.
        case "model.stream.end":
          call(d.call_id, agentId).ended = true;
          break;
        case "tool.start":
          phase("执行研究代码");
          s.tools.push({
            id: prefix + "worker-" + d.cell,
            name: "研究代码 · " + d.cell,
            agent_id: agentId,
            code: d.code,
            output: "",
            state: "running",
          });
          break;
        case "tool.output": {
          const t = s.tools.find((t) => t.id === prefix + "worker-" + d.cell);
          if (t) t.output += d.text;
          break;
        }
        case "tool.end": {
          const t = s.tools.find((t) => t.id === prefix + "worker-" + d.cell);
          if (t) {
            t.state = d.ok ? "ok" : "error";
            t.elapsed = d.elapsed_s;
            t.output = t.output || d.output;
          }
          break;
        }
        case "agent.tool.start":
          s.tools.push({
            id: prefix + d.tool_call_id,
            name: d.name,
            agent_id: agentId,
            code: d.arguments ? JSON.stringify(d.arguments, null, 2) : "",
            output: "",
            state: "running",
          });
          break;
        case "agent.tool.end": {
          const t = s.tools.find((t) => t.id === prefix + d.tool_call_id);
          if (t) {
            t.state = d.ok ? "ok" : "error";
            t.elapsed = d.elapsed_s;
            if (d.output) t.output = d.output;
          }
          break;
        }
        case "tool.cell.start":
          s.tools.push({
            id: prefix + d.tool_call_id,
            name: "模型工具单元",
            agent_id: agentId,
            code: d.code,
            output: "",
            state: "running",
          });
          break;
        case "tool.cell.finished": {
          const t = s.tools.find((t) => t.id === prefix + d.tool_call_id);
          if (t) t.state = d.ok === false ? "error" : "ok";
          break;
        }
        case "tool.cell.end":
          if (!replay && d.error)
            s.error = "工具执行出错，agent 可继续检查与修正。";
          break;
        case "model.http_error":
          if (!replay)
            s.error = "模型接口返回 HTTP " + d.status + "，本轮事件已保留。";
          break;
        case "model.stream.warning":
          if (!replay) s.error = d.message;
          break;
      }
    }
    if(s.tools.length>240){const active=s.tools.filter(t=>t.state==='running');s.tools=[...s.tools.filter(t=>t.state!=='running').slice(-Math.max(0,240-active.length)),...active];}
    if (replayDone) {
      replayUntil = 0;
      nearBottom = true;
      s.replaying = false;
      s.phase = s.active ? "正在研究" : "历史已恢复";
    }
  }
  function disconnect() {
    if (artifactRefresh) clearTimeout(artifactRefresh);
    artifactRefresh = undefined;
    artifactRequest++;
    source?.close();
    source = null;
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    queue = [];
  }
  watch(
    () => s.selected,
    async (name) => {
      const own = ++generation;
      disconnect();
      cursor = 0;
      replayUntil = 0;
      s.eventCount = 0;
      s.calls = [];
      s.tools = [];
      s.messages = [];
      s.meta = null;
      s.artifacts = [];
      s.branches = [];
      s.formulaStudies = [];
      s.datasetRevision++;
      s.selectedAgent = "main";
      s.activeAgent = "main";
      s.agentPhases = {};
      s.reviews = {};
      s.notices = {};
      s.budget = null;
      s.error = "";
      s.connected = false;
      s.active = false;
      s.replaying = true;
      s.phase = "正在载入";
      nearBottom = true;
      if (!name) {
        s.replaying = false;
        return;
      }
      try {
        const x = await api("/sessions/" + name);
        if (own !== generation) return;
        s.meta = x;
        s.branches = x.branches || [];
        s.formulaStudies = x.formula_studies || [];
        s.activeAgent = x.active_agent_id || "main";
        s.artifacts = x.artifacts;
        s.messages = x.history.map((m: Msg) => ({ ...m, legacy: true }));
        s.active = x.active;
        replayUntil = x.last_event_id || 0;
        s.replaying = !!replayUntil;
        s.phase = replayUntil ? "正在恢复历史" : s.active ? "正在研究" : "待命";
        const es = new EventSource("/api/sessions/" + name + "/events");
        source = es;
        es.onopen = () => {
          if (own === generation) s.connected = true;
        };
        es.onerror = () => {
          if (own === generation) s.connected = false;
        };
        es.addEventListener("research", (event) => {
          if (own !== generation) return;
          const e = JSON.parse((event as MessageEvent).data) as Evt;
          if (e.id <= cursor) return;
          cursor = e.id;
          queue.push(e);
          if (!frame)
            frame = requestAnimationFrame(() => {
              frame = 0;
              const batch = queue;
              queue = [];
              if (own === generation) handle(batch);
            });
        });
      } catch (e: any) {
        if (own === generation) {
          s.error = e.message;
          s.replaying = false;
        }
      }
    },
  );
  watch(
    () => visibleMessages.value,
    async () => {
      await nextTick();
      if (!scroll.value) return;
      if (nearBottom) {
        scroll.value.scrollTop = scroll.value.scrollHeight;
        s.newBelow = false;
      } else s.newBelow = true;
    },
    { deep: true, flush: "post" },
  );
  watch(
    () => s.draft,
    async () => {
      await nextTick();
      if (textarea.value) {
        textarea.value.style.height = "auto";
        textarea.value.style.height =
          Math.min(textarea.value.scrollHeight, 160) + "px";
      }
    },
  );
  function onScroll() {
    const box = scroll.value;
    if (box) {
      if (!replayUntil)
        nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 100;
      if (nearBottom) s.newBelow = false;
    }
  }
  function jumpLatest() {
    nearBottom = true;
    scroll.value?.scrollTo({
      top: scroll.value.scrollHeight,
      behavior: "smooth",
    });
    s.newBelow = false;
  }
  async function send() {
    if (!canSend.value) return;
    const value = s.draft.trim(),
      name = s.selected, agent=s.selectedAgent;
    if (request?.session !== name || request?.text !== value || request?.agent!==agent)
      request = { session: name, agent, text: value, id: newRequestId() };
    s.pending = true;
    s.error = "";
    try {
      await api("/sessions/" + name + "/messages", "POST", {
        message: value,
        agent_id:s.selectedAgent,
        provider: s.provider,
        mode: s.mode,
        request_id: request.id,
      });
      request = null;
      if (s.selected === name && s.selectedAgent===agent && s.draft.trim()===value) {
        s.draft = "";
        nearBottom = true;
        textarea.value?.focus();
      }
    } catch (e: any) {
      s.error = e.message;
    } finally {
      s.pending = false;
    }
  }
  function onKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      void send();
    }
  }
  async function create() {
    s.pending = true;
    s.error = "";
    try {
      const session = await api("/sessions", "POST", {
        title: s.newTitle,
        dataset: s.dataset,
      });
      await refresh();
      s.selected = session.id;
      s.newDialog = false;
    } catch (e: any) {
      s.error = e.message;
    } finally {
      s.pending = false;
    }
  }
  async function stopBranch(identity: string) {
    try {
      await api(
        `/sessions/${s.selected}/branches/${identity}/stop`,
        "POST",
        {},
      );
      await loadArtifacts();
    } catch (e: any) {
      s.error = e.message;
    }
  }
  function selectAgent(identity: string) {
    s.selectedAgent = identity;
    nearBottom = true;
    s.phase =
      s.agentPhases[identity] ||
      (identity === "main"
        ? s.active
          ? "研究进行中"
          : "待命"
        : s.branches.find((x) => x.id === identity)?.state || "待命");
    void nextTick(() => {
      if (scroll.value) scroll.value.scrollTop = scroll.value.scrollHeight;
    });
  }
  async function stop() {
    try {
      await api("/sessions/" + s.selected + "/stop", "POST", {});
    } catch (e: any) {
      s.error = e.message;
    }
  }
  onMounted(() => {
    void refresh();
    api("/config")
      .then((x) => {
        s.providers = x.providers;
        s.datasets = x.datasets;
        if (!s.datasets.includes(s.dataset)) s.dataset = s.datasets[0] || "";
        if (x.providers.length && !x.providers.some((p: any) => p.id === "glm"))
          s.provider = x.providers[0].id;
      })
      .catch((e) => {
        s.error = e.message;
      });
  });
  onUnmounted(() => {
    generation++;
    disconnect();
  });
  return {
    s,
    scroll,
    textarea,
    latest,
    thoughtChars,
    activeTools,
    visibleMessages,
    visibleArtifacts,
    selectedBranch,
    selectedReview,
    canSend,
    targetAvailable,
    selectAgent,
    stopBranch,
    visibleCalls,
    visibleTools,
    refresh,
    loadArtifacts,
    handle,
    onScroll,
    jumpLatest,
    send,
    onKeydown,
    create,
    stop,
  };
}
