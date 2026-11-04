<script setup lang="ts">
import {
  Activity,
  ArrowDown,
  ArrowUp,
  BookOpen,
  Brain,
  Check,
  ChevronRight,
  Download,
  FileCode2,
  FileImage,
  FileSpreadsheet,
  FlaskConical,
  FolderOpen,
  LoaderCircle,
  MessageSquare,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  Radio,
  Square,
  Terminal,
  X,
} from "lucide-vue-next";
import ResearchMessage from "./components/ResearchMessage.vue";
import FormulaStudies from "./components/FormulaStudies.vue";
import DatasetPanel from "./components/DatasetPanel.vue";
import EvaluationPanel from "./components/EvaluationPanel.vue";
import { useResearch } from "./useResearch";
import { asset } from "./api";
const {
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
  loadArtifacts,
  onScroll,
  jumpLatest,
  send,
  onKeydown,
  create,
  stop,
} = useResearch();
const names: Record<string, string> = {
  task_list: "查看研究计划",
  task_get: "读取研究目标",
  task_create: "创建研究轮次",
  task_update: "更新研究进展",
  notebook__create_notebook: "创建分析 Notebook",
  notebook__write_new_cell: "编写分析代码",
  notebook__edit_cell: "修改分析代码",
  notebook__exec_cell: "执行分析与校验",
  notebook__read_cell: "读取代码与结果",
  notebook__list_cells: "查看分析步骤",
  notebook__list_notebooks: "查看 Notebook",
  python__describe_class: "查看工具接口",
  sandbox__run_shell: "执行文件操作",
  read_file: "读取研究记录",
  edit_file: "修改研究记录",
  finish: "提交研究结果",
  run_code: "执行研究代码",
  write_file: "保存研究脚本",
  run_script: "运行研究脚本",
  reply: "推送回复",
  read_artifact: "读取已有证据",
  list_artifacts: "查看研究产物",
  spawn_research: "派生研究分支",
  list_research: "研究分支状态",
  wait_research: "等待研究结果",
  read_branch_result: "读取分支证据",
  guide_research: "指导研究分支",
  cancel_research: "停止研究分支",
  resume_research: "恢复研究分支",
  inspect_candidate: "局部自反思",
  inspect_dataset:'查看数据版本',list_datasets:'查找数据集',read_dataset_lineage:'查看数据血缘',derive_dataset:'登记动态子集',select_dataset:'切换分析范围',return_dataset:'返回父集或主集',request_data:'登记补数请求',
  create_formula_study: "固定公式比较协议",
  materialize_raw_evidence: "核对原始数值",
  materialize_formula_evidence: "核对模型指标",
  begin_formula_comparison: "继续本轮公式比较",
  register_formula: "登记公式版本",
  evaluate_formula_set: "比较公式与模型",
  read_formula_study: "读取公式研究记录",
  retire_formula_set: "记录候选排除理由",
  freeze_formula_selection: "重跑并确认最佳公式",
  review: "结构化证据审查",
  revise: "定向修订",
};
const starters = [
  "比较三个时段，提出带数学证据的怀疑方向。",
  "检查哪些高权重特征只是共模漂移或控制补偿。",
  "展示已有图表，解释维修后仍有哪些残留异常。",
  "定义跨传感器特征，用相同 lot 验证与 Logistic、树模型比较，返回最佳公式。",
];
function fileSize(size: number) {
  return size < 1024 ? size + " B" : (size / 1024).toFixed(1) + " KB";
}
function statusLabel(value: string) {
  return ({ queued: '等待执行', starting: '正在启动', running: '研究中', stopping: '正在停止', completed: '已完成', failed: '执行失败', interrupted: '已中断', cancelled: '已停止', reviewing: '审查中', passed: '检查通过', needs_revision: '需要补证', budget_exhausted: '预算耗尽，保留局限', revision_limit:'达到修订上限，结论待复核', not_required:'图表按需交付', not_reviewed: '尚未完成有效审查' } as Record<string,string>)[value] || value;
}
</script>
<template>
  <div class="app-shell" data-framework="vue3">
    <aside class="sidebar">
      <a class="brand" href="/"
        ><span class="brand-mark"><FlaskConical :size="22" /></span>
        <div>NOOA<span>RESEARCH WORKSPACE</span></div></a
      >
      <button class="new-research" @click="s.newDialog = true">
        <Plus :size="16" />新建研究<span>＋</span>
      </button>
      <div class="section-label">
        研究会话
        <span>{{ s.sessions.length.toString().padStart(2, "0") }}</span>
      </div>
      <nav>
        <button
          v-for="session in s.sessions"
          :key="session.id"
          :aria-label="session.title"
          :class="['session', { selected: session.id === s.selected }]"
          @click="s.selected = session.id"
        >
          <span class="session-icon"><MessageSquare :size="16" /></span>
          <div>
            <strong>{{ session.title }}</strong
            ><small
              >{{ session.targets }} {{session.kind==='dynamic_datasets'?'个样本':'次目标加工'}} ·
              {{
                new Date(session.updated * 1000).toLocaleDateString("zh-CN", {
                  month: "2-digit",
                  day: "2-digit",
                })
              }}</small
            >
          </div>
          <i v-if="session.active" class="live-dot" />
        </button>
      </nav>
      <div class="sidebar-foot">
        <div class="bench-icon"><BookOpen :size="17" /></div>
        <div>
          <strong>设备专家 × 自主研究</strong
          ><span>从证据出发，保留可验证的解释</span>
        </div>
      </div>
    </aside>
    <main class="main">
      <header class="topbar">
        <div class="breadcrumb">
          研究工作台<ChevronRight :size="12" /><strong>{{
            s.meta?.title || "选择一个研究会话"
          }}</strong>
        </div>
        <button
          class="icon-button"
          aria-label="切换研究面板"
          @click="s.panel = !s.panel"
        >
          <PanelRightClose v-if="s.panel" :size="18" /><PanelRightOpen
            v-else
            :size="18"
          />
        </button>
      </header>
      <div class="direction-bar">
        <label
          >当前研究方向
          <select
            :value="s.selectedAgent"
            @change="selectAgent(($event.target as HTMLSelectElement).value)"
          >
            <option value="main">主研究 · 综合结论</option>
            <option
              v-for="branch in s.branches"
              :key="branch.id"
              :value="branch.id"
            >
              {{ branch.title }} · {{ statusLabel(branch.state) }}
            </option>
          </select></label
        ><span v-if="selectedBranch" class="branch-status"
          >{{ statusLabel(selectedBranch.state)
          }}<button
            v-if="
              ['queued', 'starting', 'running'].includes(selectedBranch.state)
            "
            @click="stopBranch(selectedBranch.id)"
          >
            停止此分支
          </button></span
        >
      </div>
      <div class="workspace-heading">
        <div>
          <div class="eyebrow"><span />CHAMBER MATCHING</div>
          <h1>让异常，成为可检验的线索。</h1>
          <p>按问题划分数据，逐步验证工艺与设备线索。由 agent 研究，由专家审查。</p>
        </div>
        <span :class="['status-pill', { working: s.active }]"
          ><span :class="{ 'live-dot': s.active }" />{{
            s.active ? "研究进行中" : "准备就绪"
          }}</span
        >
      </div>
      <div class="facts">
        <span
          ><Activity :size="14" /><b>{{ s.meta?.targets ?? "—" }}</b>
          个研究对象</span
        ><span
          ><FlaskConical :size="14" /><b>{{ s.meta?.controls ?? "—" }}</b>
          个参考对象</span
        ><span><i class="tiny-lock" />版本与血缘留存</span
        ><span class="connection"
          ><i :class="s.connected ? 'connected' : 'disconnected'" />{{
            s.connected ? "实时连接" : "连接恢复中"
          }}</span
        >
      </div>
      <div ref="scroll" class="conversation" @scroll="onScroll">
        <div v-if="!visibleMessages.length && !s.replaying" class="welcome">
          <span class="welcome-icon"><FlaskConical :size="32" /></span>
          <h2>开始一轮有依据的研究</h2>
          <p>提出问题或补充设备经验。代码、计算与图表会随研究实时呈现。</p>
          <div class="starters">
            <button
              v-for="(text, i) in starters"
              :key="text"
              @click="s.draft = text"
            >
              <span>0{{ i + 1 }}</span
              >{{ text }}<ArrowUp :size="14" />
            </button>
          </div>
        </div>
        <ResearchMessage
          v-for="message in visibleMessages"
          :key="message.id"
          :msg="message"
          :session="s.selected"
        />
        <div v-if="s.replaying" class="activity-line">
          <LoaderCircle class="spin" :size="14" />正在恢复研究记录…
        </div>
        <div
          v-else-if="s.active && targetAvailable && !visibleMessages.at(-1)?.provisional"
          class="activity-line"
        >
          <LoaderCircle class="spin" :size="14" />{{ s.phase
          }}<span>工具与推理进度见右侧</span>
        </div>
      </div>
      <button v-if="s.newBelow" class="jump-latest" @click="jumpLatest">
        <ArrowDown :size="13" />回到最新消息
      </button>
      <div class="composer-wrap">
        <div v-if="s.error" class="error-banner" role="alert">
          {{ s.error
          }}<button aria-label="关闭提示" @click="s.error = ''">
            <X :size="13" />
          </button>
        </div>
        <div class="composer">
          <textarea
            ref="textarea"
            v-model="s.draft"
            :placeholder="
              s.replaying
                ? '正在恢复研究记录…'
                : s.active
                  ? '补充指导，会在下一次模型调用时送达…'
                  : '描述你的问题，或质疑上一轮的解释…'
            "
            aria-label="输入研究消息"
            @keydown="onKeydown"
          />
          <div class="composer-bottom">
            <div class="mode-switch">
              <button
                :disabled="s.active || s.replaying"
                :class="{ on: s.mode === 'research' }"
                @click="s.mode = 'research'"
              >
                <FlaskConical :size="13" />自主研究</button
              ><button
                :disabled="s.active || s.replaying"
                :class="{ on: s.mode === 'discuss' }"
                @click="s.mode = 'discuss'"
              >
                <MessageSquare :size="13" />仅讨论
              </button>
            </div>
            <select
              v-model="s.provider"
              aria-label="模型"
              :disabled="s.active || s.replaying"
            >
              <option
                v-for="provider in s.providers"
                :key="provider.id"
                :value="provider.id"
              >
                {{ provider.model }}
              </option>
            </select>
            <div class="composer-actions">
              <button
                v-if="s.active"
                class="stop"
                aria-label="停止研究"
                @click="stop"
              >
                <Square :size="12" /></button
              ><button
                class="send"
                :disabled="
                  !s.draft.trim() || s.pending || !s.selected || s.replaying
                "
                :aria-label="s.active ? '追加指导' : '发送消息'"
                @click="send"
              >
                <LoaderCircle
                  v-if="s.pending"
                  class="spin"
                  :size="16"
                /><ArrowUp v-else :size="18" />
              </button>
            </div>
          </div>
        </div>
        <div v-if="!targetAvailable" class="target-note">
          该方向当前未运行；请选择正在运行的方向追加指导，或等本轮结束后继续此分支。
        </div>
        <div v-if="s.selectedAgent !== 'main' && !s.active" class="target-note">
          发送消息将恢复此分支的历史和文件；执行进程会重新建立。
        </div>
        <div class="composer-note">
          <span>{{
            s.active
              ? "追加指导会保留在本次研究历史中"
              : "Enter 发送 · Shift + Enter 换行"
          }}</span
          ><span>结论以证据和可观测范围为界</span>
        </div>
      </div>
    </main>
    <aside v-if="s.panel" class="research-panel">
      <div class="panel-title">
        <Radio :size="16" /><strong>{{
          selectedBranch?.title || "研究现场"
        }}</strong
        ><span :class="s.active ? 'live-badge' : 'idle-badge'">{{
          s.active ? "LIVE" : "IDLE"
        }}</span>
      </div>
      <div v-if="s.budget" class="budget-box">
        <span>本轮共享预算</span
        ><strong
          >{{ (s.budget.charged_tokens || 0).toLocaleString() }} /
          {{ s.budget.token_limit?.toLocaleString() }} tokens</strong
        ><small
          >预留 {{ (s.budget.reserved_tokens || 0).toLocaleString() }} ·
          未报告用量按上界计入</small
        >
      </div>
      <div class="panel-tabs">
        <button :class="{ on: s.tab === 'live' }" @click="s.tab = 'live'">
          <Activity :size="14" />实时过程</button
        ><button
          :class="{ on: s.tab === 'files' }"
          @click="
            s.tab = 'files';
            loadArtifacts();
          "
        >
          <FolderOpen :size="14" />研究产物
          <span>{{ s.artifacts.length }}</span>
        </button>
        <button :class="{on:s.tab==='datasets'}" @click="s.tab='datasets'"><Activity :size="14" />数据范围</button>
        <button :class="{ on: s.tab === 'formulas' }" @click="s.tab = 'formulas'; loadArtifacts()">
          <FlaskConical :size="14" />公式比较<span>{{ s.formulaStudies.length }}</span>
        </button>
        <button :class="{on:s.tab==='evaluation'}" @click="s.tab='evaluation'"><Check :size="14" />收尾评估</button>
      </div>
      <div v-if="s.tab==='evaluation'" class="panel-scroll"><EvaluationPanel :session="s.selected" :active="s.active" /></div>
      <div v-if="s.tab === 'live'" class="panel-scroll">
        <div class="current-stage">
          <span class="section-label">当前状态</span>
          <h3>
            <LoaderCircle
              v-if="(s.active && targetAvailable) || s.replaying"
              class="spin"
              :size="17"
            /><Check v-else :size="17" /> {{ statusLabel(s.phase) }}
          </h3>
          <div class="live-metrics">
            <span
              ><b>{{ s.eventCount.toLocaleString() }}</b> 事件</span
            ><span
              ><b>{{ thoughtChars.toLocaleString() }}</b> 推理字数</span
            ><span
              ><b>{{ activeTools }}</b> 执行中</span
            >
          </div>
        </div>
        <div
          v-if="selectedReview"
          :class="['review-box', selectedReview.status]"
        >
          <strong>证据审查 · {{ statusLabel(selectedReview.status) }}</strong
          ><small
            >修订 {{ selectedReview.revision }} · 通过检查不等于根因认证</small
          >
          <ul v-if="selectedReview.issues.length">
            <li v-for="(issue, i) in selectedReview.issues" :key="i">
              {{ issue.severity === 'blocking' ? '待解决' : '提醒' }} · {{ issue.problem
              }}<small>{{ issue.suggested_check }}</small>
            </li>
          </ul>
        </div>
        <details v-if="s.notices[s.selectedAgent]?.length" class="review-box">
          <summary>
            校验与反思记录 {{ s.notices[s.selectedAgent].length }}
          </summary>
          <p v-for="(notice, i) in s.notices[s.selectedAgent]" :key="i">
            {{ notice }}
          </p>
        </details>
        <div class="section-label panel-section">
          模型推理 <Brain :size="14" />
        </div>
        <div v-if="!s.calls.length" class="empty-panel">
          <Brain :size="22" />
          <p>模型返回的推理会在这里实时展开。</p>
        </div>
        <details
          v-for="(call, i) in visibleCalls"
          :key="call.id"
          class="thought-card"
          :open="i === 0"
        >
          <summary>
            <span class="thought-icon"
              ><Check v-if="call.ended" :size="12" /><LoaderCircle
                v-else
                :size="12"
                class="spin" /></span
            ><strong>{{
              call.ended ? "本次生成已结束" : "模型正在生成"
            }}</strong
            ><span
              >{{
                (call.thinking.length + call.content.length).toLocaleString()
              }}
              字</span
            >
          </summary>
          <div class="thought-body">
            {{
              call.thinking ||
              (call.ended
                ? "本次接口未返回推理内容。"
                : "等待模型返回推理内容…")
            }}
          </div>
          <details v-if="call.content" class="raw-content">
            <summary>生成内容</summary>
            <pre>{{ call.content }}</pre>
          </details>
        </details>
        <div class="section-label panel-section">
          工具与执行 <Terminal :size="14" />
        </div>
        <div v-if="!s.tools.length" class="empty-panel small">
          <Terminal :size="20" />
          <p>实际调用开始后显示代码与输出。</p>
        </div>
        <details
          v-for="tool in visibleTools"
          :key="tool.id"
          :class="['tool-card', tool.state]"
          :open="tool.state === 'running'"
        >
          <summary>
            <LoaderCircle
              v-if="tool.state === 'running'"
              class="spin"
              :size="13"
            /><Check v-else-if="tool.state === 'ok'" :size="13" /><X
              v-else
              :size="13"
            /><strong>{{ names[tool.name] || tool.name }}</strong
            ><span>{{
              tool.elapsed !== undefined
                ? tool.elapsed.toFixed(1) + "s"
                : tool.state === "running"
                  ? "运行中"
                  : tool.state === "ok"
                    ? "完成"
                    : "错误"
            }}</span>
          </summary>
          <pre v-if="tool.code" class="code-block">{{ tool.code }}</pre>
          <div v-if="tool.output" class="output">
            <small>输出</small>
            <pre>{{ tool.output }}</pre>
          </div>
        </details>
        <div v-if="!s.active && latest" class="end-note">
          本轮记录已保存，可继续提问或切换到仅讨论。
        </div>
      </div>
      <div v-else-if="s.tab === 'datasets'" class="panel-scroll files">
        <DatasetPanel :session="s.selected" :agent="s.selectedAgent" :active="s.active" :supplied="s.datasets" :revision="s.datasetRevision" @changed="loadArtifacts" />
      </div>
      <div v-else-if="s.tab === 'formulas'" class="panel-scroll files">
        <FormulaStudies :studies="s.formulaStudies" :session="s.selected" />
      </div>
      <div v-else class="panel-scroll files">
        <p class="files-note">
          研究代码、数值表与图。点击可查看或下载原始文件。
        </p>
        <div v-if="!visibleArtifacts.length" class="empty-panel">
          <FolderOpen :size="25" />
          <p>agent 保存的研究产物会出现在这里。</p>
        </div>
        <a
          v-for="file in visibleArtifacts"
          :key="file.path"
          :href="asset(s.selected, file.path)"
          target="_blank"
          rel="noreferrer"
          class="artifact"
          ><span
            ><FileImage
              v-if="file.mime.startsWith('image/')"
              :size="18" /><FileSpreadsheet
              v-else-if="file.path.endsWith('.csv')"
              :size="18" /><FileCode2 v-else :size="18"
          /></span>
          <div>
            <strong>{{ file.path.split("/").at(-1) }}</strong
            ><small
              >{{ file.path.split("/").slice(0, -1).join("/") }} ·
              {{ fileSize(file.size) }}</small
            >
          </div>
          <Download :size="13"
        /></a>
      </div>
    </aside>
    <div v-if="s.newDialog" class="modal-backdrop" @click="s.newDialog = false">
      <section class="modal" role="dialog" aria-label="新建研究" @click.stop>
        <button
          class="modal-close icon-button"
          aria-label="关闭"
          @click="s.newDialog = false"
        >
          <X :size="18" /></button
        ><span class="eyebrow">NEW INVESTIGATION</span>
        <h2>开始新的研究会话</h2>
        <p>选择已提供的表格、曲线或图文数据，保留独立研究记录。</p>
        <label>会话名称<input v-model="s.newTitle" maxlength="80" /></label
        ><label
          >数据集<select v-model="s.dataset">
            <option v-for="dataset in s.datasets" :key="dataset">
              {{ dataset }}
            </option>
          </select></label
        ><button
          class="primary-button"
          :disabled="s.pending || !s.newTitle.trim()"
          @click="create"
        >
          <LoaderCircle v-if="s.pending" class="spin" :size="16" /><Plus
            v-else
            :size="16"
          />创建研究
        </button>
      </section>
    </div>
  </div>
</template>
