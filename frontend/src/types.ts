export type Evt = {
  id: number;
  task_id: string;
  agent_id?: string;
  parent_agent_id?: string;
  type: string;
  data: Record<string, any>;
  time: number;
};
export type Session = {
  id: string;
  title: string;
  active: boolean;
  task_id: string | null;
  targets: number;
  controls: number;
  kind?: string;
  updated: number;
};
export type Msg = {
  id: string;
  role: string;
  markdown: string;
  provisional?: boolean;
  legacy?: boolean;
  time?: number;
  agent_id?: string;
  publication?: string;
  review_status?: string;
};
export type ModelCall = {
  id: string;
  agent_id?: string;
  thinking: string;
  content: string;
  ended: boolean;
};
export type Tool = {
  id: string;
  name: string;
  agent_id?: string;
  code: string;
  output: string;
  state: "running" | "ok" | "error";
  elapsed?: number;
};
/** 执行轨迹中的单次工具调用与返回记录。 */
export type TrajectoryTool = {
  id: string;
  name: string;
  code: string;
  output: string;
  state: "running" | "ok" | "error";
  elapsed?: number;
};
/** 执行轨迹中的单个节点：用户消息、分配任务，或模型思考 + 工具调用/阶段汇报 + 工具返回。 */
export type TrajectoryStep = {
  id: string;
  kind?: "agent" | "user" | "task";
  index: number;
  agent_id: string;
  call_id?: string;
  userMessage?: string;
  thinking: string;
  content: string;
  report?: string;
  reportProvisional?: boolean;
  modelEnded: boolean;
  tools: TrajectoryTool[];
  spawnedBranches: string[];
  time: number;
};
export type Artifact = {
  path: string;
  size: number;
  mime: string;
  agent_id?: string;
};
export type Branch = {
  id: string;
  parent_id: string;
  mission: string;
  title: string;
  task: string;
  state: string;
  worker_type?: string;
  review_status?: string;
  error?: string;
};

export interface FormulaStudy {
  id: string;
  owner: string;
  revision: number;
  selection_id: string | null;
  limits: string;
  plan: {
    title: string;
    labels: string[];
    metric: string;
    folds: number;
    input_sha256: string;
    train: string[];
    dev: string[];
    model_params: Record<string, Record<string, unknown>>;
  };
  formulas: {
    ref: string;
    expression: string;
    description: string;
    change_reason?: string;
    sensors: string[];
    steps: number[];
    unit: string;
    params: Record<string, unknown>;
    sha256: string;
  }[];
  sets: {
    id: string;
    formulas: string[];
    excluded: null | { reason: string };
  }[];
  trials: {
    id: string;
    set_id: string;
    formulas: string[];
    model: string;
    status: string;
    dev_score?: number;
    cv_mean?: number;
    cv_std?: number;
    error?: string;
  }[];
  replays?: { id: string; status: string; error?: string }[];
  selections: {
    id: string;
    winner: string;
    ranking: string[];
    replay: { status: string };
  }[];
}

export interface DatasetItem {
  id: string;
  title: string;
  rows: number;
  parents: string[];
  version: number;
  family: string;
  relation: string;
  modality: string;
  reason: string;
}
export interface DatasetState {
  datasets: DatasetItem[];
  total: number;
  next_offset?: number;
  current: DatasetItem;
  scope: { root: string; current: string; stack: string[] };
  lineage: { nodes: DatasetItem[]; edges: { parent: string; child: string }[] };
  acquisition: { status: string; message: string };
}
