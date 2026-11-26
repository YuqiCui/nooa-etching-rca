<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import {
  Activity,
  AlertCircle,
  ChevronRight,
  FileText,
  GitBranch,
  LoaderCircle,
  Square,
  X,
} from "lucide-vue-next";
import TrajectoryStepCard from "./TrajectoryStepCard.vue";
import type { Branch, TrajectoryStep } from "../types";

const props = defineProps<{
  session: string;
  trajectory: Record<string, TrajectoryStep[]>;
  systemPrompts?: Record<string, string>;
  branches: Branch[];
  stoppableBranchIds?: string[];
  active: boolean;
  phase: string;
  agentPhases: Record<string, string>;
  names: Record<string, string>;
  statusLabel: (value: string) => string;
}>();

const emit = defineEmits<{
  (e: "stop-branch", branchId: string): void;
}>();

/** 当前右侧分屏展开的子 Agent ID，默认折叠（null）。 */
const selectedSubId = ref<string | null>(null);
const onlyErrors = ref(false);
const autoFollow = ref(true);
const singleScrollBox = ref<HTMLDivElement | null>(null);
const mainScrollBox = ref<HTMLDivElement | null>(null);
const subScrollBox = ref<HTMLDivElement | null>(null);

watch(
  () => props.branches,
  (list) => {
    if (selectedSubId.value && !list.some((b) => b.id === selectedSubId.value)) {
      selectedSubId.value = null;
    }
  },
  { deep: true },
);

function isAgentStep(step: TrajectoryStep) {
  return !step.kind || step.kind === "agent";
}

const rawMainSteps = computed(() => props.trajectory["main"] || []);
const mainAgentStepCount = computed(
  () => rawMainSteps.value.filter(isAgentStep).length,
);

const mainSteps = computed(() => {
  if (!onlyErrors.value) return rawMainSteps.value;
  return rawMainSteps.value.filter(
    (step) => isAgentStep(step) && step.tools.some((t) => t.state === "error"),
  );
});

const selectedBranch = computed(
  () => props.branches.find((b) => b.id === selectedSubId.value) || null,
);

const rawSubSteps = computed(() =>
  selectedSubId.value ? props.trajectory[selectedSubId.value] || [] : [],
);
const subAgentStepCount = computed(
  () => rawSubSteps.value.filter(isAgentStep).length,
);

const subSteps = computed(() => {
  if (!onlyErrors.value) return rawSubSteps.value;
  return rawSubSteps.value.filter(
    (step) => isAgentStep(step) && step.tools.some((t) => t.state === "error"),
  );
});

const mainSystemPrompt = computed(() => props.systemPrompts?.["main"] || "");
const subSystemPrompt = computed(() =>
  selectedSubId.value ? props.systemPrompts?.[selectedSubId.value] || "" : "",
);

/**
 * 计算所选子 Agent 在主 Agent 轨迹中的派生起始索引，用于标记派生点并在展开分屏时定位。
 */
const spawnAnchorIndex = computed(() => {
  if (!selectedSubId.value) return -1;
  const idx = mainSteps.value.findIndex((step) =>
    step.spawnedBranches.includes(selectedSubId.value!),
  );
  return idx >= 0 ? idx : 0;
});

const runningBranchCount = computed(
  () =>
    props.branches.filter((b) =>
      ["queued", "starting", "running"].includes(b.state),
    ).length,
);

const subIsRunning = computed(() =>
  selectedBranch.value
    ? ["queued", "starting", "running"].includes(selectedBranch.value.state)
    : false,
);
const subCanStop = computed(
  () =>
    subIsRunning.value &&
    !!selectedSubId.value &&
    (!props.stoppableBranchIds ||
      props.stoppableBranchIds.includes(selectedSubId.value)),
);

function toggleSub(branchId: string) {
  selectedSubId.value = selectedSubId.value === branchId ? null : branchId;
}

/** 展开分屏时将左栏定位到派生点，并将右栏回到子 Agent 首步。 */
watch(selectedSubId, async (id) => {
  if (!id) return;
  await nextTick();
  const anchor = mainScrollBox.value?.querySelector(".spawn-anchor");
  if (anchor && typeof (anchor as HTMLElement).scrollIntoView === "function") {
    (anchor as HTMLElement).scrollIntoView({ block: "start" });
  }
  if (subScrollBox.value) {
    subScrollBox.value.scrollTop = 0;
  }
});

function workerDisplayName(branch?: Branch | null) {
  if (!branch) return "Worker[general]";
  const wtype = branch.worker_type || "general";
  if (branch.title.startsWith("Worker[")) return branch.title;
  return `Worker[${wtype}] · ${branch.title}`;
}

function branchDotClass(state: string) {
  if (["queued", "starting", "running", "stopping"].includes(state)) return "running";
  if (["failed", "interrupted", "cancelled"].includes(state)) return "error";
  return "ok";
}

function onPaneScroll(event: Event) {
  const el = event.target as HTMLElement | null;
  if (!el) return;
  autoFollow.value = el.scrollHeight - el.scrollTop - el.clientHeight < 90;
}

watch(
  () => [rawMainSteps.value.length, rawMainSteps.value.at(-1)?.tools.length],
  async () => {
    if (!autoFollow.value) return;
    await nextTick();
    const el = selectedSubId.value ? mainScrollBox.value : singleScrollBox.value;
    if (el) el.scrollTop = el.scrollHeight;
  },
);

watch(
  () => [rawSubSteps.value.length, rawSubSteps.value.at(-1)?.tools.length],
  async () => {
    if (!autoFollow.value || !selectedSubId.value) return;
    await nextTick();
    if (subScrollBox.value) {
      subScrollBox.value.scrollTop = subScrollBox.value.scrollHeight;
    }
  },
);
</script>

<template>
  <section class="traj-view">
    <div class="traj-toolbar">
      <div class="traj-summary">
        <span :class="['traj-main-pill', { live: active }]">
          <span :class="['worker-dot', active ? 'running' : 'ok']" />
          <strong>Router</strong>
          <small>{{ mainAgentStepCount }} 步</small>
        </span>
        <span class="traj-phase">{{ statusLabel(phase) }}</span>
      </div>

      <div v-if="branches.length" class="traj-worker-strip" aria-label="Worker 列表">
        <span class="strip-label">
          <GitBranch :size="12" />
          Workers ({{ runningBranchCount }}/{{ branches.length }})
        </span>
        <button
          v-for="branch in branches"
          :key="branch.id"
          type="button"
          :class="['worker-chip', { selected: selectedSubId === branch.id }]"
          @click="toggleSub(branch.id)"
        >
          <span :class="['worker-dot', branchDotClass(branch.state)]" />
          <span class="worker-chip-title">{{ workerDisplayName(branch) }}</span>
          <small>{{
            (trajectory[branch.id] || []).filter(isAgentStep).length
          }} 步</small>
        </button>
      </div>

      <div class="traj-controls">
        <button
          type="button"
          :class="['traj-filter-btn', { on: onlyErrors }]"
          @click="onlyErrors = !onlyErrors"
        >
          <AlertCircle :size="12" />仅异常
        </button>
        <button
          type="button"
          :class="['traj-filter-btn', { on: autoFollow }]"
          @click="autoFollow = !autoFollow"
        >
          <Activity :size="12" />跟随最新
        </button>
      </div>
    </div>

    <div
      v-if="!rawMainSteps.length && !branches.length"
      class="traj-scroll"
    >
      <div class="traj-empty">
        <Activity :size="28" />
        <h3>暂无执行轨迹</h3>
        <p>发起研究后，Router 与 Worker 的每一步思考、工具调用及结果会实时呈现在这里。</p>
      </div>
    </div>

    <div
      v-else-if="!selectedSubId"
      ref="singleScrollBox"
      class="traj-scroll"
      @scroll="onPaneScroll"
    >
      <div class="traj-single">
        <div class="traj-col-head">
          <strong>Router 轨迹</strong>
          <span>{{ mainAgentStepCount }} 步</span>
        </div>
        <details v-if="mainSystemPrompt" class="traj-sys-prompt">
          <summary>
            <ChevronRight :size="13" class="sys-chevron" />
            <FileText :size="12" />
            <strong>System Prompt</strong>
            <span>{{ mainSystemPrompt.length.toLocaleString() }} 字</span>
          </summary>
          <pre class="traj-sys-body">{{ mainSystemPrompt }}</pre>
        </details>
        <div v-if="!mainSteps.length" class="traj-empty-inline">
          当前筛选条件下无匹配步骤
        </div>
        <div class="traj-step-list">
          <TrajectoryStepCard
            v-for="step in mainSteps"
            :key="step.id"
            :session="session"
            :step="step"
            :names="names"
            :branches="branches"
            :trajectory="trajectory"
            :selected-sub-id="selectedSubId"
            :status-label="statusLabel"
            @open-sub="toggleSub"
          />
        </div>
      </div>
    </div>

    <div v-else class="traj-split-panes">
      <div class="traj-pane traj-pane-main">
        <div class="traj-pane-header">
          <strong>Router 轨迹</strong>
          <span>{{ mainAgentStepCount }} 步</span>
        </div>
        <div
          ref="mainScrollBox"
          class="traj-pane-scroll"
          @scroll="onPaneScroll"
        >
          <details v-if="mainSystemPrompt" class="traj-sys-prompt">
            <summary>
              <ChevronRight :size="13" class="sys-chevron" />
              <FileText :size="12" />
              <strong>System Prompt</strong>
              <span>{{ mainSystemPrompt.length.toLocaleString() }} 字</span>
            </summary>
            <pre class="traj-sys-body">{{ mainSystemPrompt }}</pre>
          </details>
          <div v-if="!mainSteps.length" class="traj-empty-inline">
            当前筛选条件下无匹配步骤
          </div>
          <div v-else class="traj-step-list">
            <TrajectoryStepCard
              v-for="(step, idx) in mainSteps"
              :key="step.id"
              :session="session"
              :step="step"
              :names="names"
              :branches="branches"
              :trajectory="trajectory"
              :selected-sub-id="selectedSubId"
              :is-spawn-anchor="idx === spawnAnchorIndex"
              :status-label="statusLabel"
              @open-sub="toggleSub"
            />
          </div>
        </div>
      </div>

      <div class="traj-pane traj-pane-sub">
        <div class="traj-sub-head">
          <div class="traj-sub-title">
            <span
              :class="[
                'worker-dot',
                branchDotClass(selectedBranch?.state || 'running'),
              ]"
            />
            <GitBranch :size="13" />
            <select
              v-if="branches.length > 1"
              :value="selectedSubId"
              aria-label="切换 Worker"
              @change="
                selectedSubId = ($event.target as HTMLSelectElement).value
              "
            >
              <option
                v-for="branch in branches"
                :key="branch.id"
                :value="branch.id"
              >
                {{ workerDisplayName(branch) }} · {{ statusLabel(branch.state) }}
              </option>
            </select>
            <strong v-else>{{ workerDisplayName(selectedBranch) }}</strong>
          </div>
          <div class="traj-sub-actions">
            <span class="sub-status-text">
              {{
                statusLabel(
                  agentPhases[selectedSubId] ||
                    selectedBranch?.state ||
                    "running",
                )
              }}
              · {{ subAgentStepCount }} 步
            </span>
            <button
              v-if="subCanStop"
              type="button"
              class="sub-stop-btn"
              title="停止此 Worker"
              @click="emit('stop-branch', selectedSubId)"
            >
              <Square :size="10" />停止
            </button>
            <button
              type="button"
              class="sub-close-btn"
              aria-label="折叠 Worker 分屏"
              @click="selectedSubId = null"
            >
              <X :size="14" />
            </button>
          </div>
        </div>

        <div
          ref="subScrollBox"
          class="traj-pane-scroll"
          @scroll="onPaneScroll"
        >
          <details v-if="subSystemPrompt" class="traj-sys-prompt">
            <summary>
              <ChevronRight :size="13" class="sys-chevron" />
              <FileText :size="12" />
              <strong>System Prompt</strong>
              <span>{{ subSystemPrompt.length.toLocaleString() }} 字</span>
            </summary>
            <pre class="traj-sys-body">{{ subSystemPrompt }}</pre>
          </details>
          <div v-if="!subSteps.length" class="traj-empty-inline">
            <LoaderCircle v-if="subIsRunning" :size="14" class="spin" />
            {{
              onlyErrors
                ? "该 Worker 无异常步骤"
                : subIsRunning
                  ? "Worker 已派生，等待首步输出…"
                  : "暂无步骤记录"
            }}
          </div>
          <div v-else class="traj-step-list">
            <TrajectoryStepCard
              v-for="step in subSteps"
              :key="step.id"
              :session="session"
              :step="step"
              :names="names"
              :status-label="statusLabel"
            />
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
