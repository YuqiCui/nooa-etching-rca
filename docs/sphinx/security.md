# 执行、数据与凭据边界

本章描述本地研究程序如何约束生成代码，防止误读真值、改写其他任务或让计算结果串线。这是项目自身的运行边界，不涉及对外部系统操作。

```{mermaid} diagrams/trust.mmd
:caption: 编排持有模型凭据；两类生成代码都在独立受限子进程执行，科学输出由宿主监督。
:alt: 可信主进程运行NOOA和工具代理，原生外层fork worker与另行启动的无凭据科学Python具有不同边界，隐藏测试和真值仅供收尾评分。
```

## 外层 CodeAct 与科学代码

外层生成代码直接使用NOOA `CodeActV2` 和标准 `SandboxedExecutor`。项目不再覆盖执行器、创建自定义代理或改写官方REPL辅助函数。NOOA的fork worker持有其标准namespace，通过原生parent-agent proxy访问宿主；这与旧版只提供工具白名单的新解释器边界不同。`hidden`是文档/可见性机制，不应当作任意Python的权限认证；不再承诺外层进程没有继承宿主内存、凭据或全部私有对象不可达。

启动时实际探测Landlock、seccomp及rlimit，通过 `SandboxConfig(require=True)` 只启用明确选中的机制。`FACTOR_SANDBOX_FILESYSTEM=auto` 有Landlock时启用，否则披露无文件隔离；`required`要求Landlock，`off`明确关闭。`FACTOR_CELL_ALLOW_NETWORK=1`允许直接REST请求；默认0要求原生seccomp可用，否则拒绝启动。没有Landlock的Linux5.0可用无文件限制的原生进程后端，但仍需实际验证目标环境；82上的无Landlock配置测试不等于5.0实机测试。`doctor`单独显示外层原生与科学执行层的能力。

原生外层普通运行异常遵循NOOA规则，可保留已经发生的namespace修改；超时/崩溃按上游恢复配置重建。项目中间件只记录父子执行观察和附加工具证据，不修改captured locals、返回协议、辅助函数或异常状态。工具副作用不自动回滚。

科学代码也使用长驻进程，同一次研究调用共享namespace和事件循环，不通过pickle恢复变量。编译错误不执行，保留现有状态；运行异常使受影响内核失效并停止，下一次执行使用新epoch。脚本使用独立临时内核。超时、取消、崩溃和结束清理进程组；已经发生的工具副作用和文件写入不回滚。新用户请求仍建立新研究调用，磁盘对话存档不代表活对象已恢复。

科学stdout/stderr写入每cell独立文件并由宿主流式读取，与限长、请求关联的完成帧分开；打印JSON不会完成cell。stdout保留首部、stderr保留尾部，各262144字符并继续排空。完成帧来自不可信Python执行进程，不能作为抵御恶意同进程代码伪造的证明，也不证明科学正确性；仍需宿主证据检查。缺失匹配完成帧的退出视为崩溃。CPU资源上限按内核生命周期累计，每cell另外有墙钟截止时间。

macOS使用deny-default的sandbox-exec规则；Linux使用Landlock或bubblewrap的受限挂载视图，再设置seccomp和资源上限；缺少Landlock时会实际验证命名空间后端，而不是依据内核版本猜测。仅允许解释器/依赖、引导代码、选中只读数据版本与本方向workspace；`factor_demo`应用源码从运行库授权中排除。Linux引导仅复制固定NOOA版本的底层约束模块并保留许可头，不在每个科学进程导入整套agent SDK；子进程显式使用本地模型成本表，避免无关初始化。`FACTOR_LINUX_BACKEND=auto`下若Landlock与bubblewrap都不可用（如内核<5.13的Kubernetes容器），会记录警告并回退到`process`后端（仅rlimit+尽力seccomp，文件系统隔离依赖容器，见[Linux部署指南](linux-arm.md)）；设置`FACTOR_STRICT_BACKEND=1`或显式选择`landlock`/`bubblewrap`时仍拒绝执行。以上Landlock/bubblewrap挂载和扩展syscall限制描述科学计算执行层；原生外层只使用NOOA声明的Landlock/seccomp/rlimit，不额外创建PID namespace。Runtime checks现已停用，当前Linux实际验收见[验证记录](validation.md)中的82精确提交结果；文档构建不能代替运行回归。

读写规则不以`/Users`作为唯一边界。科学进程不继承控制FD、模型key、原始会话目录或test/truth路径；通信控制、源码副本、输入哈希和执行账本由宿主保存。外层代理仍可调用已公开的工具，所以每个工具必须自己检查数据范围、产物路径和任务权限。进程约束与回放均不证明任意算法的数学正确性。

## 输入、产物与证据

发现数据只包含train/dev。动态数据集是不可变版本，worker每次仅获得当前版本的DATASET，不能用旧`../input`绕回全集。子集必须保持原值、原结构和选中资产闭包；改值/加文件必须显式登记为transform。产物根目录自身及下载路径均拒绝逃逸符号链接。

科学cell的成功/失败状态、输入和源码哈希均进入私有SQLite账本，变更产物保留副本；证据来源仍要求对应成功执行。独立实验另有冻结输入、源码revision、checks与暂存产物manifest；仅current且checked的实验可由宿主显式发布，源码/输入变化后的stale实验或改写产物不能用作当前来源。该来源证明只赋予探索性可复现等级；raw数值与正式模型指标另由宿主重算，详见[证据设计](evidence-review.md)。

## 本地服务与凭据

服务默认监听127.0.0.1。请求Host必须为localhost、127.0.0.1或[::1]，有Origin时还需与本机来源匹配。CLI与网页共用内核文件锁，运行中不能从另一入口改数据范围、启动第二轮或冻结评估；进程退出自动释放锁。

`.env`和`.secrets/providers.json`仅在本地可信编排中读取。公开源码不含远程SSH取key的个人环境脚本，不向供应商附加内部agent/mission标识请求头。模型服务仍会接收到本轮任务、已选上下文和必要工具结果。公开仓库只包含源码、测试、非秘密配置、锁文件与文档；原始对话、生成数据、运行数据库、真实key和缓存不发布。

## 评估隔离的限度

隐藏test和仿真参考存放在宿主私有快照，所有agent工具均不提供评分入口。收尾评分先冻结报告和已选方法，只允许对同一定义重试。生成器公开，所以拿到全仓权限的人仍能了解故障构造；不能把它当成第三方独立科学发现证明。公开机制ID只是分类词表，不包含哪个场景对应哪个答案。

## Host path integrity and process reaping

- Host writes into the sandbox workspace (`research_cells`, JSON outputs, formula/`state.joblib` copies, font cache seeds) refuse symbolic links (`O_NOFOLLOW` / realpath containment). A cell cannot redirect host I/O outside the workspace by planting symlinks.
- **Bubblewrap backend**: private mount root plus `--unshare-all --die-with-parent` (PID, net, mount, …), **and Landlock rules stacked inside** the mount namespace for defense in depth. Killing the bwrap supervisor tears down the whole namespace.
- **Process backend** (`FACTOR_LINUX_BACKEND=process`, or the `auto` fallback when Landlock and bubblewrap are both unavailable): no privileges and no namespaces. The cell runs as a child process in its own session/process group under `cell_reaper`, with `cwd` set to the scratch workdir and a clean environment. rlimits: CPU 300 s, address space +2 GiB, `RLIMIT_FSIZE` (`FACTOR_CELL_FSIZE_MB`, default 1024), `RLIMIT_NOFILE` (`FACTOR_CELL_NOFILE`, default 1024), `RLIMIT_NPROC` (`FACTOR_CELL_NPROC`); the host wall-clock budget kills the process group. When running as root and `FACTOR_CELL_UID` (optional `FACTOR_CELL_GID`) is set, the scratch dir is chowned and the cell drops to that uid. The same seccomp denylist is installed when possible; failure is logged to stderr instead of refusing. **There is no filesystem confinement: the container must provide it** (read-only root, non-root user, emptyDir workdir, no secrets mounted).
- **Network toggle** `FACTOR_CELL_ALLOW_NETWORK=1` (default off): the seccomp denylist stops refusing `socket`/`socketpair` (connect, send/recv, sockopts were never filtered; everything else in the filter stays), resolver/CA files (`/etc/resolv.conf`, `/etc/hosts`, `/etc/nsswitch.conf`, `/etc/ssl`, …) become readable, and bubblewrap adds `--share-net`. Honored by all three Linux backends (Landlock itself does not filter network here). macOS sandbox-exec is unchanged. Restrict egress with a NetworkPolicy.
- **Landlock backend**: filesystem rules come from Landlock. Process isolation requires a PID namespace via `bwrap --unshare-pid --die-with-parent` whenever bubblewrap is installed. **Landlock without bubblewrap is fail-closed by default**; set `FACTOR_ALLOW_LANDLOCK_WITHOUT_PIDNS=1` only for explicit degraded testing. A trusted `cell_reaper` remains a subreaper with SIGSTOP/SIGKILL fallback.
- **Process caps**: Linux uses `RLIMIT_NPROC` (default 512; override with `FACTOR_CELL_NPROC`). macOS is opt-in: only an explicit `FACTOR_CELL_NPROC` applies the rlimit because XNU counts all processes of the login uid. Cgroup `pids.max` is best effort when `FACTOR_CELL_CGROUP` points at a delegated writable v2 directory; it is not guaranteed to be available.
- **Native outer deadlines**: NOOA manages cell deadlines and recovery. Brokered host tools have a separate `broker_timeout_s` (configured with `FACTOR_CELL_TOOL_TIMEOUT`, default 330s); their waiting time is not charged like the removed executor's overall deadline. There is no project-specific per-cell tool-call-count cap. Research iteration/call/token budgets remain active.
- **Linux seccomp** (both backends): denies sockets/`io_uring_*`, cross-process signals, ptrace/process-vm/pidfd, namespaces (`unshare`/`setns`/clone `NEW*`), `perf_event_open`/`userfaultfd`/`bpf`, keyring, kexec/modules, mount graph syscalls, handle lookups, swap/reboot/acct/quotactl/`personality`, and x86 `iopl`/`ioperm`/`modify_ldt`. `clone3` returns `ENOSYS`. `fork`/`exec` remain for scientific helpers. Wrong arch and x86_64 x32 ABI are rejected.
- **macOS**: `(deny default)` sandbox-exec plus `RLIMIT_AS` / `RLIMIT_CPU`; `RLIMIT_NPROC` is opt-in. There is no Linux-style PID namespace or seccomp. Process-group reaping does not prove that arbitrary detached descendants are contained; Linux additionally checks `/proc` descendants at scientific cell boundaries.
