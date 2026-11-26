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
  TrajectoryStep,
  TrajectoryTool,
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
    mainView: "chat" as "chat" | "trajectory",
    trajectory: {} as Record<string, TrajectoryStep[]>,
    runtimeWorkers: [] as Branch[],
    systemPrompts: {} as Record<string, string>,
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
        s.tools.filter((t) => t.state === "running").length,
    );
  const visibleMessages = computed(() =>
    s.messages.filter((m) => (m.agent_id || "main") === s.selectedAgent),
  );
  const visibleCalls = computed(() =>
      s.calls
        .filter((c, index) => !c.ended || index >= s.calls.length - 12)
        .reverse(),
    ),
    visibleTools = computed(() =>
      s.tools
        .filter((t, index) => t.state === "running" || index >= s.tools.length - 40)
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
      if (s.calls.length > 60)
        s.calls = s.calls.filter((c, index) => !c.ended || index >= s.calls.length - 60);
      c = s.calls.at(-1)!;
    }
    return c;
  }
  /** 获取或初始化指定 Agent 的轨迹步骤列表。 */
  function agentSteps(agentId = "main"): TrajectoryStep[] {
    return (s.trajectory[agentId] ??= []);
  }
  /** 判断轨迹节点是否为 Agent 执行步骤（排除用户消息与任务分配卡片）。 */
  function isAgentStep(step: TrajectoryStep): boolean {
    return !step.kind || step.kind === "agent";
  }
  /** 根据模型调用 ID 获取或新建对应的轨迹步骤。 */
  function stepForCall(callId: string, agentId = "main", time = 0): TrajectoryStep {
    const steps = agentSteps(agentId);
    let step = steps.find((item) => isAgentStep(item) && item.call_id === callId);
    if (!step) {
      const agentIndex = steps.filter(isAgentStep).length + 1;
      step = {
        id: `${agentId}:step:${agentIndex}:${steps.length + 1}`,
        kind: "agent",
        index: agentIndex,
        agent_id: agentId,
        call_id: callId,
        thinking: "",
        content: "",
        modelEnded: false,
        tools: [],
        spawnedBranches: [],
        time,
      };
      steps.push(step);
    }
    return step;
  }
  /** 获取指定 Agent 当前正在进行的轨迹步骤，不存在或最新节点非步骤时自动补建。 */
  function currentStep(agentId = "main", time = 0): TrajectoryStep {
    const steps = agentSteps(agentId);
    const last = steps.at(-1);
    if (last && isAgentStep(last)) return last;
    const agentIndex = steps.filter(isAgentStep).length + 1;
    const step: TrajectoryStep = {
      id: `${agentId}:step:${agentIndex}:${steps.length + 1}`,
      kind: "agent",
      index: agentIndex,
      agent_id: agentId,
      thinking: "",
      content: "",
      modelEnded: true,
      tools: [],
      spawnedBranches: [],
      time,
    };
    steps.push(step);
    return step;
  }
  /** 为子 Agent 写入分配任务卡片；首次写入置于轨迹首位，后续重分配追加在当前位置。 */
  function recordWorkerTask(
    workerId: string,
    taskText: string,
    deliverList?: string[],
    time = 0,
  ) {
    const body = taskText.trim();
    if (!workerId || workerId === "main" || !body) return;
    const deliverBlock =
      Array.isArray(deliverList) && deliverList.length
        ? `\n\n**交付清单**\n${deliverList.map((item) => `- ${item}`).join("\n")}`
        : "";
    const fullText = body + deliverBlock;
    const steps = agentSteps(workerId);
    if (
      steps.some(
        (item) =>
          item.kind === "task" &&
          (item.userMessage?.trim() === fullText.trim() ||
            item.userMessage?.trim() === body),
      )
    ) {
      return;
    }
    const card: TrajectoryStep = {
      id: `${workerId}:task:${steps.length + 1}`,
      kind: "task",
      index: 0,
      agent_id: workerId,
      userMessage: fullText,
      thinking: "",
      content: "",
      modelEnded: true,
      tools: [],
      spawnedBranches: [],
      time,
    };
    const hasTask = steps.some((item) => item.kind === "task");
    if (!hasTask) steps.unshift(card);
    else steps.push(card);
  }
  /** 在指定 Agent 的轨迹步骤中按工具调用 ID 逆序查找工具记录。 */
  function findTrajTool(agentId: string, toolId: string): TrajectoryTool | undefined {
    const steps = agentSteps(agentId);
    for (let i = steps.length - 1; i >= 0; i--) {
      const found = steps[i].tools.find((item) => item.id === toolId);
      if (found) return found;
    }
    return undefined;
  }
  /** 将指定子 Agent 绑定到父 Agent 的派生锚点步骤（仅首次绑定）。 */
  function attachSpawnAnchor(parentId: string, childId: string, step?: TrajectoryStep) {
    const parentSteps = agentSteps(parentId);
    if (parentSteps.some((item) => item.spawnedBranches.includes(childId))) return;
    const anchor =
      step || [...parentSteps].reverse().find((item) => item.kind !== "user");
    if (anchor) anchor.spawnedBranches.push(childId);
  }
  /** 为新版 Router+Worker 运行时自动登记子 Worker，以便在执行轨迹中折叠与分屏展示。 */
  function ensureRuntimeWorker(
    workerId: string,
    parentId = "main",
    patch: Partial<Branch> = {},
    anchorStep?: TrajectoryStep,
  ) {
    if (!workerId || workerId === "main") return;
    if (s.branches.some((b) => b.id === workerId)) return;
    let worker = s.runtimeWorkers.find((w) => w.id === workerId);
    if (!worker) {
      worker = {
        id: workerId,
        parent_id: parentId,
        mission: "",
        title: `Worker[general] · ${workerId.slice(-4)}`,
        task: "",
        state: "running",
        worker_type: "general",
        ...patch,
      };
      s.runtimeWorkers.push(worker);
    } else {
      if (patch.worker_type) worker.worker_type = patch.worker_type;
      if (patch.title) worker.title = patch.title;
      if (patch.task) worker.task = patch.task;
      if (patch.state) worker.state = patch.state;
    }
    attachSpawnAnchor(parentId, workerId, anchorStep);
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
      if (agentId !== "main") {
        ensureRuntimeWorker(agentId, e.parent_agent_id || "main");
      }
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
          agentSteps(agentId).push({
            id: `${agentId}:user:${e.id}`,
            kind: "user",
            index: 0,
            agent_id: agentId,
            userMessage: d.message,
            thinking: "",
            content: "",
            modelEnded: true,
            tools: [],
            spawnedBranches: [],
            time: e.time,
          });
          break;
        case "agent.system_prompt":
          if (typeof d.content === "string" && d.content.trim()) {
            s.systemPrompts[agentId] = d.content;
          }
          break;
        case "agent.task":
          if (typeof d.task === "string" && d.task.trim()) {
            recordWorkerTask(agentId, d.task, undefined, e.time);
          }
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
          for (const steps of Object.values(s.trajectory)) {
            for (const step of steps) {
              step.modelEnded = true;
              for (const t of step.tools) {
                if (t.state === "running") {
                  t.state = "error";
                  t.output += "\n本轮已结束，此调用没有成功完成记录。";
                }
              }
            }
          }
          for (const w of s.runtimeWorkers) {
            if (["queued", "starting", "running"].includes(w.state)) {
              w.state = d.status === "checkpoint" ? "completed" : d.status || "interrupted";
            }
          }
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
          if (e.type === "branch.created") {
            const parentId = row.parent_id || e.parent_agent_id || "main";
            const parentSteps = agentSteps(parentId);
            const anchor = [...parentSteps].reverse().find(isAgentStep);
            if (anchor && !anchor.spawnedBranches.includes(row.id)) {
              anchor.spawnedBranches.push(row.id);
            }
            if (row.task) {
              recordWorkerTask(row.id, row.task, undefined, e.time);
            }
          }
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
          agentSteps(agentId).push({
            id: `${agentId}:guidance:${e.id}`,
            kind: "user",
            index: 0,
            agent_id: agentId,
            userMessage: "主研究指导：" + d.message,
            thinking: "",
            content: "",
            modelEnded: true,
            tools: [],
            spawnedBranches: [],
            time: e.time,
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
          const step = currentStep(agentId, e.time);
          step.report = "";
          step.reportProvisional = true;
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
          const step = currentStep(agentId, e.time);
          const cur = step.report || "";
          if (d.offset === 0 || Array.from(cur).length === d.offset) {
            step.report = d.offset === 0 ? d.text : cur + d.text;
            step.reportProvisional = true;
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
          const step = currentStep(agentId, e.time);
          step.report = d.markdown;
          step.reportProvisional = false;
          if (!replay) void loadArtifacts();
          break;
        }
        case "model.start":
          call(d.call_id, agentId);
          stepForCall(d.call_id, agentId, e.time);
          break;
        case "model.reasoning.delta":
          call(d.call_id, agentId).thinking += d.text;
          stepForCall(d.call_id, agentId, e.time).thinking += d.text;
          break;
        case "model.content.delta":
          call(d.call_id, agentId).content += d.text;
          stepForCall(d.call_id, agentId, e.time).content += d.text;
          break;
        // Historical tool.arguments.delta events are intentionally ignored too.
        case "model.stream.end":
          call(d.call_id, agentId).ended = true;
          stepForCall(d.call_id, agentId, e.time).modelEnded = true;
          break;
        case "tool.start": {
          phase("执行研究代码");
          const toolId = prefix + "worker-" + d.cell;
          const toolItem: TrajectoryTool = {
            id: toolId,
            name: "研究代码 · " + d.cell,
            code: d.code,
            output: "",
            state: "running",
          };
          s.tools.push({ ...toolItem, agent_id: agentId });
          currentStep(agentId, e.time).tools.push(toolItem);
          break;
        }
        case "tool.output": {
          const toolId = prefix + "worker-" + d.cell;
          const t = s.tools.find((t) => t.id === toolId);
          if (t) t.output += d.text;
          const tt = findTrajTool(agentId, toolId);
          if (tt) tt.output += d.text;
          break;
        }
        case "tool.end": {
          const toolId = prefix + "worker-" + d.cell;
          const t = s.tools.find((t) => t.id === toolId);
          if (t) {
            t.state = d.ok ? "ok" : "error";
            t.elapsed = d.elapsed_s;
            t.output = t.output || d.output;
          }
          const tt = findTrajTool(agentId, toolId);
          if (tt) {
            tt.state = d.ok ? "ok" : "error";
            tt.elapsed = d.elapsed_s;
            tt.output = tt.output || d.output;
          }
          break;
        }
        case "agent.tool.start": {
          const toolId = prefix + d.tool_call_id;
          const toolItem: TrajectoryTool = {
            id: toolId,
            name: d.name,
            code: d.arguments ? JSON.stringify(d.arguments, null, 2) : "",
            output: "",
            state: "running",
          };
          s.tools.push({ ...toolItem, agent_id: agentId });
          currentStep(agentId, e.time).tools.push(toolItem);
          break;
        }
        case "agent.tool.end": {
          const toolId = prefix + d.tool_call_id;
          const t = s.tools.find((t) => t.id === toolId);
          if (t) {
            t.state = d.ok ? "ok" : "error";
            t.elapsed = d.elapsed_s;
            if (d.output) t.output = d.output;
          }
          const tt = findTrajTool(agentId, toolId);
          if (tt) {
            tt.state = d.ok ? "ok" : "error";
            tt.elapsed = d.elapsed_s;
            if (d.output) tt.output = d.output;
          }
          if (
            d.ok &&
            (d.name === "StartWorker" ||
              d.name === "AssignTask" ||
              d.name === "CollectWorkerResult") &&
            typeof d.output === "string"
          ) {
            try {
              const parsed = JSON.parse(d.output);
              const wid = parsed?.worker_id || parsed?.agent_id;
              if (typeof wid === "string" && wid) {
                const ownerStep = agentSteps(agentId).find((st) =>
                  st.tools.some((item) => item.id === toolId),
                );
                if (d.name === "CollectWorkerResult") {
                  ensureRuntimeWorker(wid, agentId, { state: "completed" }, ownerStep);
                } else {
                  let taskHint = "";
                  let deliverList: string[] | undefined;
                  let workerType =
                    typeof parsed?.worker_type === "string"
                      ? parsed.worker_type
                      : "";
                  if (tt?.code) {
                    try {
                      const args = JSON.parse(tt.code);
                      if (!workerType && typeof args?.worker_type === "string") {
                        workerType = args.worker_type;
                      }
                      if (d.name === "AssignTask") {
                        if (typeof args?.task === "string") taskHint = args.task.trim();
                        if (Array.isArray(args?.deliver_list)) {
                          deliverList = args.deliver_list.map(String);
                        }
                      }
                    } catch {}
                  }
                  const existing = s.runtimeWorkers.find((w) => w.id === wid);
                  const resolvedType = workerType || existing?.worker_type || "general";
                  const prefixTag = `Worker[${resolvedType}]`;
                  const shortTitle = taskHint
                    ? `${prefixTag} · ${taskHint.slice(0, 18)}${taskHint.length > 18 ? "…" : ""}`
                    : !existing?.task
                      ? `${prefixTag} · ${wid.slice(-4)}`
                      : undefined;
                  ensureRuntimeWorker(
                    wid,
                    agentId,
                    {
                      state: d.name === "AssignTask" ? "running" : "queued",
                      worker_type: resolvedType,
                      ...(shortTitle ? { title: shortTitle } : {}),
                      ...(taskHint ? { task: taskHint } : {}),
                    },
                    ownerStep,
                  );
                  if (d.name === "AssignTask" && taskHint) {
                    recordWorkerTask(wid, taskHint, deliverList, e.time);
                  }
                }
              }
            } catch {}
          }
          break;
        }
        case "tool.cell.start": {
          const toolId = prefix + d.tool_call_id;
          const toolItem: TrajectoryTool = {
            id: toolId,
            name: "模型工具单元",
            code: d.code,
            output: "",
            state: "running",
          };
          s.tools.push({ ...toolItem, agent_id: agentId });
          currentStep(agentId, e.time).tools.push(toolItem);
          break;
        }
        case "tool.cell.finished": {
          const toolId = prefix + d.tool_call_id;
          const t = s.tools.find((t) => t.id === toolId);
          if (t) t.state = d.ok === false ? "error" : "ok";
          const tt = findTrajTool(agentId, toolId);
          if (tt) tt.state = d.ok === false ? "error" : "ok";
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
      s.trajectory = {};
      s.runtimeWorkers = [];
      s.systemPrompts = {};
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
        if (x.system_prompts && typeof x.system_prompts === "object") {
          s.systemPrompts = { ...x.system_prompts };
        }
        if (x.worker_tasks && typeof x.worker_tasks === "object") {
          for (const [wid, taskText] of Object.entries(x.worker_tasks)) {
            if (typeof taskText === "string" && taskText.trim()) {
              recordWorkerTask(wid, taskText);
            }
          }
        }
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
