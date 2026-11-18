# Agent 轨迹导出

将一次运行导出为可离线打开的单文件 HTML。Router 与各 Worker 并列，共享从上到下的时间顺序。未启动、未分配任务和空闲期间不补造事件。事件较密时展开间距，间距不代表等比例耗时；完整起止时间和耗时可在详情中查看。

```bash
uv run python scripts/render_agent_trace.py runs/你的运行目录 -o runs/你的运行目录/agent-trace.html
```

也可以直接传入 `audit/` 目录。默认输出到输入目录下的 `agent-trace.html`。可选参数：`--title` 设置标题，`--timezone` 设置显示时区（默认 `Asia/Shanghai`）。

## 输入

工具读取运行目录中的审计文件：

```text
audit/
  model_events.jsonl
  tool_events.jsonl
  0001-main-request.json
  0001-main-response.json
  ...
```

- 模型事件使用 `event`（`start`、`end`、`error`）、`at`（带时区 ISO 时间）、`request`、`agent_id`。开始事件可以带 `parent_agent_id` 和 `assignment_id`。
- 工具事件使用 `event`（`start`、`end`、`error`、`rejected`）、`at`、`agent_id`、`assignment_id`、`call_id`、`name`、`arguments`；结束事件带 `result` 和可选的 `elapsed_s`。
- 模型快照命名为 `{request:04d}-{agent_id}-request.json` 和对应 `response.json`。请求包含 `messages` 与 `tools`。每次请求的完整消息顺序保留，重复消息只内嵌一份。
- 可选的 `status.json` 提供运行起止时间；会话文件 `case/.autoresearch/**/session.json` 或 `.autoresearch/**/session.json` 提供身份及交付校验回执。

工具结束时间优先使用匹配的开始时间加 `elapsed_s`，兼容旧日志中结束事件重复开始时间的问题。同一 Worker 复用时，工具调用按会话、Assignment、调用 ID 和工具名配对。任务带从派发到结果就绪显示；没有明确就绪事件时，以收取时间作为上界，并在导出数据中标记 `end_inferred`。

## 查看

- 点击事件查看完整输入、输出和交付校验反馈。
- 点击各列的 `Prompt`，或模型事件详情中的 `System`、`User`，查看该次请求实际发送的消息。下拉框可切换该会话的请求，包括上下文摘要请求。
- 使用搜索、事件类型、异常筛选和间距选项；筛选保留共享时间位置。
- 使用任务下拉框直接跳转到相应时间，调用卡片左侧细线表示其持续区间。
- 启动、派发与收取关系用箭头连接，连线可以关闭。

缺少请求快照时，页面明确显示无法还原 prompt，不使用最终 session 推测历史 system prompt。只有 session、没有带时间的日志时，工具拒绝编造时序。

HTML 内嵌全部样式、脚本和轨迹数据，不请求外部资源，不执行消息中的 HTML。页面包含原始 prompt、响应和工具数据，按运行记录本身的访问范围保管。
