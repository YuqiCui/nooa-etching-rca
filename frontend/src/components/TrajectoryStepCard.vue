<script setup lang="ts">
import { reactive, ref, computed } from "vue";
import {
  AlertCircle,
  Check,
  ChevronDown,
  ChevronRight,
  ClipboardList,
  GitBranch,
  LoaderCircle,
  MessageSquareText,
  SplitSquareHorizontal,
  Terminal,
  User,
  X,
} from "lucide-vue-next";
import MarkdownView from "./MarkdownView.vue";
import PythonCellsView from "./PythonCellsView.vue";
import CodeDiffView from "./CodeDiffView.vue";
import { extractPythonCells } from "../utils/pythonCells";
import { extractCodeDiff } from "../utils/codeDiff";
import type { Branch, TrajectoryStep, TrajectoryTool } from "../types";

const props = defineProps<{
  session: string;
  step: TrajectoryStep;
  names: Record<string, string>;
  branches?: Branch[];
  trajectory?: Record<string, TrajectoryStep[]>;
  selectedSubId?: string | null;
  isSpawnAnchor?: boolean;
  statusLabel: (value: string) => string;
}>();

const emit = defineEmits<{
  (e: "open-sub", branchId: string): void;
}>();

/** 模型思考默认折叠为 thinking...，按需点击展开。 */
const thinkingOpen = ref(false);

/** 记录每个工具卡片当前激活的视图标签（代码 / 变更 / 调用 / 结果）。 */
const toolTab = reactive<Record<string, "code" | "diff" | "call" | "result">>({});

/** 提取 Notebook 写入工具或 write_file 中的代码列表，非匹配工具返回 null。 */
function toolPython(tool: TrajectoryTool) {
  return extractPythonCells(tool.name, tool.code, tool.output);
}

/** 提取 apply_patch / edit_file 工具中的 Git 风格代码差异，非匹配工具返回 null。 */
function toolDiff(tool: TrajectoryTool) {
  return extractCodeDiff(tool.name, tool.code);
}

/** 计算工具卡片当前生效的视图标签；具备代码或差异视图的工具默认优先展示。 */
function activeToolTab(tool: TrajectoryTool): "code" | "diff" | "call" | "result" {
  const hasCode = !!toolPython(tool);
  const hasDiff = !!toolDiff(tool);
  const selected = toolTab[tool.id];
  if (selected === "code" && !hasCode) return hasDiff ? "diff" : "call";
  if (selected === "diff" && !hasDiff) return hasCode ? "code" : "call";
  if (selected) return selected;
  if (hasCode) return "code";
  if (hasDiff) return "diff";
  return "call";
}

const stepState = computed<"running" | "error" | "ok">(() => {
  if (!props.step.modelEnded || props.step.tools.some((t) => t.state === "running")) {
    return "running";
  }
  if (props.step.tools.some((t) => t.state === "error")) {
    return "error";
  }
  return "ok";
});

const hasThinking = computed(
  () => !!props.step.thinking.trim() || !props.step.modelEnded,
);

/** 提取 report / report_progress / finish 工具中的 Markdown 正文，便于与对话区一致渲染。 */
function toolMarkdown(tool: TrajectoryTool): string | null {
  if (!["report", "report_progress", "finish"].includes(tool.name)) return null;
  if (!tool.code) return null;
  try {
    const parsed = JSON.parse(tool.code);
    const raw =
      parsed?.content ?? parsed?.markdown ?? parsed?.message ?? parsed?.summary;
    if (typeof raw === "string" && raw.trim()) {
      // Worker 的 finish 可能返回结构化 JSON 字符串，保留原始 JSON 视图避免误当正文渲染
      const trimmed = raw.trim();
      if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
        try {
          JSON.parse(trimmed);
          return null;
        } catch {}
      }
      return raw;
    }
  } catch {}
  return null;
}

const reportText = computed(() => {
  const candidate =
    props.step.report ||
    (!props.step.tools.length && props.step.content.trim() ? props.step.content : "");
  if (!candidate) return "";
  // 若同一步的 report/finish 工具卡片已经按 Markdown 渲染了相同正文，避免重复展示
  if (props.step.tools.some((t) => toolMarkdown(t)?.trim() === candidate.trim())) {
    return "";
  }
  return candidate;
});

const spawnedList = computed(() => {
  if (!props.step.spawnedBranches.length) return [];
  return props.step.spawnedBranches.map((id) => {
    const branch = props.branches?.find((b) => b.id === id);
    const count =
      props.trajectory?.[id]?.filter((s) => !s.kind || s.kind === "agent")
        .length ?? 0;
    const wtype = branch?.worker_type || "general";
    const rawTitle = branch?.title || id.slice(-4);
    const label = rawTitle.startsWith("Worker[")
      ? rawTitle
      : `Worker[${wtype}] · ${rawTitle}`;
    return {
      id,
      title: label,
      state: branch?.state || "running",
      steps: count,
      active: props.selectedSubId === id,
    };
  });
});

/** 将秒级时间戳格式化为 HH:mm:ss，方便现场核对执行节奏。 */
function formatTime(ts?: number) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function branchDotClass(state: string) {
  if (["queued", "starting", "running", "stopping"].includes(state)) return "running";
  if (["failed", "interrupted", "cancelled"].includes(state)) return "error";
  return "ok";
}
</script>

<template>
  <article v-if="step.kind === 'user'" class="traj-user-card">
    <header class="traj-user-head">
      <span class="traj-user-badge"><User :size="12" />用户</span>
      <time v-if="step.time" class="step-time">{{ formatTime(step.time) }}</time>
    </header>
    <div class="traj-user-body">{{ step.userMessage }}</div>
  </article>

  <article v-else-if="step.kind === 'task'" class="traj-task-card">
    <header class="traj-user-head">
      <span class="traj-task-badge"><ClipboardList :size="12" />分配任务</span>
      <time v-if="step.time" class="step-time">{{ formatTime(step.time) }}</time>
    </header>
    <div class="traj-task-body">
      <MarkdownView
        :text="step.userMessage || ''"
        :session="session"
        :agent="step.agent_id"
      />
    </div>
  </article>

  <article
    v-else
    :class="[
      'traj-step',
      stepState,
      { 'spawn-anchor': isSpawnAnchor },
    ]"
  >
    <header class="traj-step-head">
      <div class="traj-step-id">
        <span class="step-num">#{{ String(step.index).padStart(2, "0") }}</span>
        <span :class="['step-state-icon', stepState]">
          <LoaderCircle v-if="stepState === 'running'" :size="13" class="spin" />
          <AlertCircle v-else-if="stepState === 'error'" :size="13" />
          <Check v-else :size="13" />
        </span>
      </div>
      <div class="traj-step-meta">
        <span v-if="isSpawnAnchor" class="spawn-tag">
          <GitBranch :size="11" />派生点
        </span>
        <time v-if="step.time" class="step-time">{{ formatTime(step.time) }}</time>
      </div>
    </header>

    <div v-if="hasThinking" class="traj-thinking">
      <button
        type="button"
        class="thinking-toggle"
        :aria-expanded="thinkingOpen"
        @click="thinkingOpen = !thinkingOpen"
      >
        <ChevronDown v-if="thinkingOpen" :size="13" />
        <ChevronRight v-else :size="13" />
        <span class="thinking-label">thinking...</span>
        <span v-if="!step.modelEnded" class="thinking-live" />
        <span v-if="step.thinking.length" class="thinking-len">
          {{ step.thinking.length.toLocaleString() }} 字
        </span>
      </button>
      <div v-if="thinkingOpen" class="thinking-body">
        {{ step.thinking || "等待模型返回推理内容…" }}
      </div>
    </div>

    <div v-if="step.tools.length" class="traj-tools">
      <section
        v-for="tool in step.tools"
        :key="tool.id"
        :class="['traj-tool-card', tool.state]"
      >
        <div class="traj-tool-head">
          <div class="traj-tool-title">
            <LoaderCircle
              v-if="tool.state === 'running'"
              :size="13"
              class="spin"
            />
            <Check v-else-if="tool.state === 'ok'" :size="13" />
            <X v-else :size="13" />
            <Terminal :size="12" class="tool-term-icon" />
            <strong class="tool-raw-name">{{ tool.name }}</strong>
            <span
              v-if="names[tool.name] && names[tool.name] !== tool.name"
              class="tool-desc"
              >{{ names[tool.name] }}</span
            >
          </div>
          <div class="traj-tool-actions">
            <span class="tool-elapsed">
              {{
                tool.elapsed !== undefined
                  ? tool.elapsed.toFixed(1) + "s"
                  : tool.state === "running"
                    ? "运行中"
                    : ""
              }}
            </span>
            <div class="tool-mode-switch" role="tablist">
              <button
                v-if="toolPython(tool)"
                type="button"
                role="tab"
                :aria-selected="activeToolTab(tool) === 'code'"
                :class="{ on: activeToolTab(tool) === 'code' }"
                @click="toolTab[tool.id] = 'code'"
              >
                代码
              </button>
              <button
                v-else-if="toolDiff(tool)"
                type="button"
                role="tab"
                :aria-selected="activeToolTab(tool) === 'diff'"
                :class="{ on: activeToolTab(tool) === 'diff' }"
                @click="toolTab[tool.id] = 'diff'"
              >
                变更
              </button>
              <button
                type="button"
                role="tab"
                :aria-selected="activeToolTab(tool) === 'call'"
                :class="{ on: activeToolTab(tool) === 'call' }"
                @click="toolTab[tool.id] = 'call'"
              >
                调用
              </button>
              <button
                type="button"
                role="tab"
                :aria-selected="activeToolTab(tool) === 'result'"
                :class="{
                  on: activeToolTab(tool) === 'result',
                  'has-error': tool.state === 'error',
                }"
                @click="toolTab[tool.id] = 'result'"
              >
                结果
              </button>
            </div>
          </div>
        </div>
        <div class="traj-tool-body">
          <template v-if="activeToolTab(tool) === 'code' && toolPython(tool)">
            <PythonCellsView :payload="toolPython(tool)!" />
          </template>
          <template v-else-if="activeToolTab(tool) === 'diff' && toolDiff(tool)">
            <CodeDiffView :payload="toolDiff(tool)!" />
          </template>
          <template v-else-if="activeToolTab(tool) === 'call'">
            <div v-if="toolMarkdown(tool)" class="traj-markdown-body">
              <MarkdownView
                :text="toolMarkdown(tool)!"
                :session="session"
                :agent="step.agent_id"
              />
            </div>
            <pre v-else class="traj-code">{{
              tool.code || "（无调用参数）"
            }}</pre>
          </template>
          <pre v-else class="traj-output">{{
            tool.output ||
            (tool.state === "running" ? "工具执行中，等待返回结果…" : "（无返回内容）")
          }}</pre>
        </div>
      </section>
    </div>

    <div v-if="reportText" class="traj-report">
      <div class="traj-report-head">
        <MessageSquareText :size="12" />
        <span>{{ step.reportProvisional ? "阶段汇报（生成中）" : "阶段汇报" }}</span>
      </div>
      <div class="traj-report-body">
        <MarkdownView
          :text="reportText"
          :session="session"
          :agent="step.agent_id"
        />
      </div>
    </div>

    <div v-if="spawnedList.length" class="traj-spawns">
      <button
        v-for="item in spawnedList"
        :key="item.id"
        type="button"
        :class="['traj-spawn-card', { active: item.active }]"
        @click="emit('open-sub', item.id)"
      >
        <span :class="['worker-dot', branchDotClass(item.state)]" />
        <GitBranch :size="13" />
        <span class="spawn-title">{{ item.title }}</span>
        <span class="spawn-meta">
          {{ statusLabel(item.state) }} · {{ item.steps }} 步
        </span>
        <span class="spawn-action">
          <SplitSquareHorizontal :size="12" />
          {{ item.active ? "收起分屏" : "分屏查看" }}
        </span>
      </button>
    </div>
  </article>
</template>
