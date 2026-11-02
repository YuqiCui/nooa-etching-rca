# 安装、复现、恢复与文档部署

## 无API的文档构建

文档构建只安装docs组，不导入NOOA运行时，不需要provider、数据集或任何模型key。Python依赖由uv.lock锁定，Mermaid CLI由docs/package-lock.json锁定。

```bash
git clone https://github.com/pty819/nooa-etching-rca.git
cd nooa-etching-rca
npm ci --prefix docs
make docs-check
python3 -m http.server 8877 --directory docs/_build/html
```

打开 `http://127.0.0.1:8877`。首次安装Mermaid CLI会下载其渲染浏览器；在Linux构建机还需Chromium依赖与CJK字体。中文全文检索使用jieba、领域词典与Sphinx搜索语言兼容扩展；生成的stemmer和中文索引由Node实际执行检查。文档渲染browser的CI参数不属于研究worker的安全策略，不能混为一谈。Makefile使用独立 `.venv-docs`，不会卸载正在使用的研究环境。

## 研究环境

当前完整自由研究要求macOS sandbox-exec或Linux Landlock/bubblewrap与seccomp、Python3.12/3.13、uv与Node24。先同步运行依赖和前端：

```bash
uv sync --python 3.12 --frozen
npm ci --prefix frontend
npm run build --prefix frontend
cp .env.example .env
# 在本机编辑.env，填写自己的endpoint、model和key；不提交真实值。
```

Web启动命令显式加载.env，因为provider列表来自服务进程环境；CLI研究入口自身也会加载dotenv。使用custom时配置FACTOR_API_STYLE、FACTOR_API_BASE、FACTOR_MODEL、FACTOR_API_KEY。现成GLM/MiniMax名称需要自行建立私有provider配置，仓库不提供可用key。`probe`会真实调用供应商，不能在不知情时当作普通离线检查。

## 生成并核对数据

```bash
uv run factor-demo generate-etch --data data/etching-v3 --runs 360
uv run factor-demo validate-etch --data data/etching-v3
uv run factor-demo preview-etch --data data/etching-v3
```

生成器拒绝覆盖非空目录；要复现实验应保留seed、sample period和manifest，新数据用新目录。generate-etch生成原始曲线、特征与隐藏状态；validate-etch独立验证，preview-etch单独生成预览。研究会话随后只导出发现范围。仓库未捆绑生成后的二进制数据。

## 启动前端和CLI研究

```bash
uv run --env-file .env factor-demo serve --host 127.0.0.1 --port 8765
```

浏览器新建研究时选择etching-v3与自己的custom模型。也可直接用CLI：

```bash
uv run factor-demo autoresearch --data data/etching-v3 \
  --run runs/my-case --provider custom \
  --message '分别比较三个业务时段，先核对目录和批次，保存可复核证据。'

uv run factor-demo autoresearch-chat --run runs/my-case --provider custom \
  --message '请检验这些高权重是否来自同一个闭环，并区分反证和未测机制。'

uv run factor-demo autoresearch-chat --run runs/my-case --provider custom \
  --mode discuss --message '解释目前已证实什么，尚不能区分什么。'

uv run factor-demo autoresearch-audit --run runs/my-case
```

运行中通过Web向选中方向发送消息会进入该轮指导；空闲时发送会开始新轮。恢复指定分支应在当前主任务结束后进行，不重新派生整套父研究。目录中的Python变量不会跨进程恢复，需重新运行已保存的初始化脚本。

## 更新代码与检查依赖

合并PR、同步本地源码、同步虚拟环境和重启运行中的服务是不同动作。先核对 `git status` 并保存需要保留的本地改动，再 `git fetch origin` / `git merge --ff-only origin/main`；锁文件中的NOOA版本变化后，需要在该运行目录执行 `uv sync --frozen --python 3.12`，不能靠本地Git提交已更新就判断进程已加载新版。不要把一个测试目录的依赖安装当作另一个日常环境已经更新。

当前锁定NOOA `0.0.11.dev473`、上游 `bff0ef4`。独立实验和持续REPL的主流程见[执行运行协议](repl-runtime.md)；本轮Python失败重建内核后应根据宿主观察显式重建输入，不自动重放历史工具副作用。

## 服务重启和异常恢复

```{mermaid} diagrams/recovery.mmd
:caption: HTTP host恢复依靠进程身份与mission核对，不凭同一个PID就认定是原worker。
:alt: 服务启动检查活跃标记，匹配进程则接管，不匹配则清理并标记中断；显式继续时从文件和NOOA快照建立新mission。
```

正常停止优先用界面按钮，让已有输出、快照及预算完成结算。强制退出时不要删除runs来“修复状态”；重启服务后检查主/子状态和错误记录。保留原始任务ID和脚本，有助于判断某一步是否已经产生副作用。旧报告只有mission匹配才会被认作该分支本轮结果。

## GitHub Actions与Pages

```{mermaid} diagrams/publish.mmd
:caption: PR只构建验证；主分支才部署Pages。Actions不运行付费模型请求，也不接触已有研究数据。
:alt: 文档依赖和Mermaid安装后生成引用、严格构建并检查；上传产物，只有主分支非PR允许独立部署job发布静态站。
```

这里指仍启用的 `Sphinx documentation`（`docs.yml`），以push、pull_request和workflow_dispatch触发。运行测试的 `Runtime checks`（`tests.yml`）文件仍在仓库，但已按维护决定停用；当前运行验收在82 x86 Linux完成。两者分别管理，停用运行CI不等于停止文档发布。构建job固定Ubuntu24.04，安装Python3.12、Node22和锁定uv，构建SVG与静态站；权限为contents:read。部署job依赖构建成功，只在本仓库main分支的非PR事件运行，使用pages:write和id-token:write及github-pages环境。PR不能通过此工作流获取生产模型key，因为工作流根本不需要模型secret。

首次仓库需启用Pages的workflow发布源；后续直接推送main即可重新构建。站点在仓库子路径下，必须同时检查首页、章节、CSS与SVG的200状态，不能只看GitHub Actions绿色。部署操作和图源均可由仓库历史审查。

更多CLI解析细节见[自动生成的CLI参考](_generated/cli.md)。

## 动态数据与导出

把已提供的本地文件/目录放在项目data目录，在新建会话中选择；CSV/Parquet可自动发现身份列，复杂模态应提供dataset.json及索引。新会话不会覆盖旧目录。界面“数据范围”可读schema、样本预览、划分源码与血缘，在空闲时切换/回退或登记已提供数据；运行中通过对话指导agent。见[动态数据协议及示例](dynamic-datasets.md)。

当前选择保存在SQLite，重启可恢复；每次切换重建科学worker。不要编辑`.datasets/versions`或对象存储，校验会拒绝哈希不符的文件。公开公式副本按内容增量导出，复制期间不持有SQLite写事务；相同内容的冻结快照去重。科学worker输出有界，截断会明确标记，不应把日志末尾当成完整计算表。

request_data只暴露未来统一拉数接口；未配置时返回unavailable，不会访问外部API或知识图谱。已冻结的旧研究仍绑定原DatasetRef，加入新条目后应创建新比较协议，不能混用分数。

## 旧案例与独立评估

旧90片案例使用显式`generate-etch --data data/old-case --runs 90 --profile legacy_v2`；早期特征表命令在`factor-demo legacy`内，不是默认research入口。新版默认360片会保持recipe，扩展独立lot。

独立审查默认`review_provider=auto`选另一个已配置模型；也可在runtime.json指定provider名，或设置FACTOR_REVIEW_MODEL、FACTOR_REVIEW_API_BASE、FACTOR_REVIEW_API_KEY。没有不同模型就保留not_reviewed，不能当作通过。

研究完成后在Vue“收尾评估”页冻结当前定义，或执行`factor-demo finalize-etch --run runs/你的会话`；`factor-demo evaluation --run ...`只读结果。已冻结报告的变化不允许再次选优。画图和一般探索不强制评估。

`python scripts/run_blind_benchmark.py --root data/一个新目录 --runs 180`准备六种中性命名场景，映射和真值保持在发现输入之外。`--score`只评分已产生的报告，缺失报告不会生成假分数。

## 原生NOOA外层沙箱配置（Linux）

外层固定使用标准CodeActV2/SandboxedExecutor；不根据级别切换到另一个自定义REPL。启动时探测能力、记录选中的配置，doctor的orchestration_sandbox描述原生外层，原backend字段描述科学计算层。

| 环境变量 | 默认与含义 |
|---|---|
| FACTOR_SANDBOX_FILESYSTEM | auto：有Landlock则启用；required：必须具备；off：明确关闭文件隔离 |
| FACTOR_CELL_ALLOW_NETWORK | 0：要求seccomp阻止新建互联网socket；1：允许外层与科学worker直接REST请求 |
| FACTOR_SANDBOX_WORKSPACE | 空：不额外授权写目录；非空必须是已有目录，作为原生外层读写workspace |
| FACTOR_SANDBOX_MEMORY_MB | 2048：原生worker虚拟地址空间基线之上的内存余量；0关闭此项限制 |
| FACTOR_SANDBOX_CPU_SECONDS | 300：原生worker生命周期CPU限制；0关闭此项限制 |
| FACTOR_CELL_TOOL_TIMEOUT | 330秒：原生宿主broker单次工具调用期限，与cell执行时间分开 |

配置始终require=True：只启用已选择的机制，不在失败后偷偷取消已声明的保护。auto在Linux5.0缺Landlock时可选无文件限制级别；不是相同文件安全保证，也不创建PID namespace。网络禁止但seccomp不可用时明确拒绝，管理员可显式设网络允许。采用NOOA默认代码规则；直接REST调用应使用httpx.AsyncClient等异步方式，避免同步阻塞被框架校验拒绝。主机工具访问网络与worker是否允许直接联网是不同路径。

本次仅在82 Linux验证；实际5.0目标机仍需原生能力探测和应用验收。macOS不提供本次原生外层运行支持。科学计算执行层的Landlock/bubblewrap/process选择仍独立存在，不要把两层能力合并成一个安全等级。
