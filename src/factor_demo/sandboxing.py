"""Fail-closed process policy shared by CodeAct cells and scientific children."""

from __future__ import annotations

import json
import os
import signal
import sys
from functools import lru_cache
from pathlib import Path


def runtime_reads() -> list[Path]:
    import importlib.util

    spec = importlib.util.find_spec("factor_demo")
    hidden = Path(spec.origin).resolve().parent if spec and spec.origin else None
    candidates = [
        Path(p).resolve()
        for p in [
            sys.prefix,
            sys.base_prefix,
            "/System",
            "/usr",
            "/etc/fonts",
            "/var/cache/fontconfig",
            "/bin",
            "/lib",
            "/lib64",
            "/sbin",
            "/Library/Apple",
            "/Library/Fonts",
            "/private/var/db/timezone",
            # Resolver/TLS config, only when cells may use the network.
            *(_NETWORK_READS if allow_network() else []),
        ]
        if Path(p).exists()
    ]
    result = set()

    def grant(path):
        if hidden and (path == hidden or path.is_relative_to(hidden)):
            return
        if hidden and hidden.is_relative_to(path) and path.is_dir():
            for child in path.iterdir():
                grant(child.resolve())
        else:
            result.add(path)

    for candidate in candidates:
        grant(candidate)
    return sorted(result)


_NETWORK_READS = [
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/gai.conf",
    "/etc/services",
    "/etc/ssl",
    "/etc/pki",
    "/etc/ca-certificates",
]


def allow_network() -> bool:
    """Opt-in FACTOR_CELL_ALLOW_NETWORK=1: cells may open sockets (default off)."""
    return _env_flag("FACTOR_CELL_ALLOW_NETWORK")


def landlock_abi():
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    return max(0, int(libc.syscall(444, 0, 0, 1)))


@lru_cache(maxsize=1)
def linux_backend():
    """Select an enforced filesystem backend; never substitute unrestricted execution."""
    import shutil
    import subprocess

    requested = os.environ.get("FACTOR_LINUX_BACKEND", "auto")
    if requested not in {"auto", "landlock", "bubblewrap", "process"}:
        raise RuntimeError("FACTOR_LINUX_BACKEND must be auto, landlock, bubblewrap or process")
    if requested == "process":
        return "process"
    abi = landlock_abi()
    if requested != "bubblewrap" and abi >= 1:
        return "landlock"
    if requested == "landlock":
        raise RuntimeError("This kernel does not provide Landlock; select bubblewrap or auto")
    binary = shutil.which("bwrap")
    if binary:
        probe = subprocess.run(
            [
                binary,
                "--unshare-all",
                "--die-with-parent",
                "--ro-bind",
                "/",
                "/",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "/usr/bin/true",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode == 0:
            return "bubblewrap"
        reason = probe.stderr.strip()[:500]
    else:
        reason = "bubblewrap is not installed"
    if requested == "auto" and not _env_flag("FACTOR_STRICT_BACKEND"):
        # Unprivileged containers (kernel < 5.13, no nested namespaces): degrade to
        # rlimits + best-effort seccomp; filesystem isolation is the container's job.
        import logging

        logging.getLogger(__name__).warning(
            "Landlock (ABI %s) and bubblewrap unavailable (%s); falling back to the "
            "'process' backend: no filesystem isolation beyond the container. Set "
            "FACTOR_STRICT_BACKEND=1 to refuse instead.",
            abi,
            reason,
        )
        return "process"
    raise RuntimeError(
        f"No enforced Linux filesystem backend for {requested} (Landlock ABI {abi}): {reason}"
    )


def execution_info():
    import platform

    mode = (
        linux_backend()
        if sys.platform.startswith("linux")
        else "sandbox-exec"
        if sys.platform == "darwin"
        else "unsupported"
    )
    return {
        "platform": sys.platform,
        "architecture": platform.machine(),
        "backend": mode,
        "landlock_abi": landlock_abi() if sys.platform.startswith("linux") else None,
    }


def dotenv_warnings(roots=None):
    """Warn about .env files that sandboxed cells must never be allowed to read."""
    if roots is None:
        roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    found = []
    for root in roots:
        path = Path(root) / ".env"
        if path.exists() and path not in found:
            found.append(path)
    return [
        f"{path} exists: cells cannot read it (Landlock); keep LITELLM_MODE=PRODUCTION "
        "in the cell environment and never whitelist .env (it holds API keys)"
        for path in found
    ]


def interpreter_links(allowed):
    """Preserve version aliases used by venv interpreter symlinks (not extra mounts)."""
    pending = [Path(sys.executable), Path(getattr(sys, "_base_executable", sys.executable))]
    seen, links = set(), {}
    while pending:
        path = pending.pop()
        for item in [path, *path.parents]:
            if item in seen:
                continue
            seen.add(item)
            if not item.is_symlink():
                continue
            target = Path(os.readlink(item))
            literal = target if target.is_absolute() else item.parent / target
            pending.append(literal)
            if not any(item == root or item.is_relative_to(root) for root in allowed):
                links[item] = os.readlink(item)
    return links


def bubblewrap_command(python_args, reads, writes):
    import shutil

    binary = shutil.which("bwrap")
    if not binary:
        raise RuntimeError("Selected bubblewrap executable disappeared")
    kit = Path(python_args[0]).resolve().parent
    marker = kit / "_mount_root.txt"
    if not marker.is_file():
        raise RuntimeError("Missing trusted mount bootstrap marker")
    command = [
        binary,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        *(["--share-net"] if allow_network() else []),
        "--cap-drop",
        "ALL",
        "--tmpfs",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    # ELF interpreters use literal /lib or /lib64 paths even on usr-merged
    # distributions; binding only their resolved targets is insufficient.
    for alias in [Path("/lib"), Path("/lib64"), Path("/bin"), Path("/sbin")]:
        if alias.is_symlink():
            command.extend(["--symlink", os.readlink(alias), str(alias)])
    extras = [Path(p) for p in ["/etc/ld.so.cache", "/etc/localtime"] if Path(p).exists()]
    allowed = sorted(
        {*runtime_reads(), *(Path(p).resolve() for p in reads), *extras},
        key=lambda p: (len(p.parts), str(p)),
    )
    for path in allowed:
        if path == Path("/"):
            raise RuntimeError("A cell cannot receive the whole host root")
        command.extend(["--ro-bind", str(path), str(path)])
    for alias, target in interpreter_links(allowed).items():
        command.extend(["--symlink", target, str(alias)])
    for path in writes:
        path = Path(path).resolve()
        command.extend(["--bind", str(path), str(path)])
    command.extend(
        [
            "--ro-bind",
            str(marker),
            "/.nooa-cell-backend",
            "--chdir",
            str(Path(writes[0]).resolve()),
            "--remount-ro",
            "/proc",
            "--remount-ro",
            "/",
            "--",
            sys.executable,
            "-I",
            "-u",
            *map(str, python_args),
        ]
    )
    return command


def mac_profile(reads, writes):
    def rules(paths):
        return " ".join(f"(subpath {json.dumps(str(Path(p).resolve()))})" for p in paths)

    return (
        "(version 1)(deny default)"
        "(allow process-exec)(allow process-fork)(allow sysctl-read)"
        "(allow process-info* (target self))"
        '(allow mach-lookup (global-name "com.apple.cfprefsd.agent") (global-name "com.apple.cfprefsd.daemon") (global-name "com.apple.FontObjectsServer"))'
        "(allow ipc-posix-shm)(allow ipc-posix-sem)"
        '(allow file-read-metadata)(allow file-read-data (literal "/"))'
        f"(allow file-read* {rules([*runtime_reads(), *reads, *writes])})"
        f'(allow file-write* {rules(writes)} (literal "/dev/null"))'
        '(allow file-read-data (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random") (literal "/private/etc/localtime"))'
    )


def _pid_namespace_prefix():
    """Best-effort private PID namespace for Landlock cells via bubblewrap.

    Filesystem confinement stays Landlock's job; this only adds `--unshare-pid`
    so killing namespace init reaps setsid/fork-chain grandchildren. Returns []
    when bwrap is missing or the probe fails (documented fallback).
    """
    import shutil
    import subprocess

    binary = shutil.which("bwrap")
    if not binary:
        return []
    probe = subprocess.run(
        [
            binary,
            "--unshare-pid",
            "--die-with-parent",
            "--bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "/usr/bin/true",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if probe.returncode != 0:
        return []
    return [
        binary,
        "--unshare-pid",
        "--die-with-parent",
        "--bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]


def launch(python_args, reads, writes):
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
        return [
            "/usr/bin/sandbox-exec",
            "-p",
            mac_profile(reads, writes),
            sys.executable,
            "-I",
            "-u",
            *map(str, python_args),
        ]
    if sys.platform.startswith("linux"):
        backend = linux_backend()
        if backend == "bubblewrap":
            return bubblewrap_command(python_args, reads, writes)
        # Landlock FS + subreaper; require bwrap PID ns unless explicitly opted out.
        reaper = str(Path(__file__).resolve().with_name("cell_reaper.py"))
        command = [
            sys.executable,
            "-I",
            "-u",
            reaper,
            sys.executable,
            "-I",
            "-u",
            *map(str, python_args),
        ]
        if backend == "process":
            # No namespaces: subreaper + process-group kill (reap) tear down the tree.
            return command
        prefix = _pid_namespace_prefix()
        if not prefix and not _allow_landlock_without_pidns():
            raise RuntimeError(
                "Landlock without bubblewrap PID namespace is disabled by default "
                "(fork storms can exhaust host PIDs). Install bubblewrap, or set "
                "FACTOR_ALLOW_LANDLOCK_WITHOUT_PIDNS=1 to opt into the degraded mode."
            )
        return [*prefix, *command]
    raise RuntimeError("No enforced execution backend on this platform")


def font_cache_seed():
    import hashlib
    from importlib.metadata import version

    explicit = os.environ.get("FACTOR_FONT_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    identity = hashlib.sha256(str(Path(sys.prefix).resolve()).encode()).hexdigest()[:12]
    return Path.home() / ".cache/nooa-etching-rca/fonts" / (version("matplotlib") + "-" + identity)


def prepare_font_cache():
    """Deployment-time warmup; no model calls and no mutable cache shared by cells."""
    cache = font_cache_seed()
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    prior = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(cache)
    try:
        from matplotlib import font_manager

        files = sorted(cache.glob("fontlist-v*.json"))
        return {
            "font_cache": str(cache),
            "cache_files": len(files),
            "fonts": len(font_manager.fontManager.ttflist),
        }
    finally:
        if prior is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = prior


def seed_worker_fonts(cache):
    import re

    from .path_guard import copy_regular_nofollow, ensure_real_dir

    source = font_cache_seed()
    if not source.is_dir() or source.is_symlink():
        return
    target = ensure_real_dir(Path(cache) / "mpl")
    for path in source.iterdir():
        if (
            not re.fullmatch(r"fontlist-v[0-9]+(?:\.[0-9]+)*\.json", path.name)
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        dest = target / path.name
        if dest.exists() or dest.is_symlink():
            continue
        if path.stat().st_size > 8 * 1024 * 1024:
            continue
        copy_regular_nofollow(path, dest, required_root=source)


def child_environment(work: Path, **extra):
    from .path_guard import ensure_real_dir

    cache = ensure_real_dir(Path(work) / ".cache", root=work)
    seed_worker_fonts(cache)
    environment = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "HOME": str(cache),
        "TMPDIR": str(cache),
        "MPLCONFIGDIR": str(cache / "mpl"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "JOBLIB_TEMP_FOLDER": str(cache / "joblib"),
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        # SDK imports must not discover credential files inside guarded children.
        "PYTHON_DOTENV_DISABLED": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        # DEV mode makes `import litellm` call load_dotenv(), which walks up to the
        # project-root .env; Landlock denies that open() and the import crashes.
        "LITELLM_MODE": "PRODUCTION",
        "POLARS_MAX_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        **extra,
    }
    if sys.platform.startswith("linux"):
        environment["NOOA_CELL_BACKEND"] = linux_backend()
    # Tunables read by install_linux / cell_reaper inside the child.
    for key in (
        "FACTOR_CELL_NPROC",
        "FACTOR_CELL_CGROUP",
        "FACTOR_FONT_CACHE_DIR",
        "FACTOR_ALLOW_LANDLOCK_WITHOUT_PIDNS",
        "FACTOR_CELL_UID",
        "FACTOR_CELL_GID",
        "FACTOR_CELL_FSIZE_MB",
        "FACTOR_CELL_NOFILE",
        "FACTOR_CELL_ALLOW_NETWORK",
    ):
        if key in os.environ:
            environment[key] = os.environ[key]
    return environment


def prepare_guard_modules(kit: Path):
    """Copy the pinned, attributed guards without importing the full SDK in children."""
    import importlib.util

    (kit / "_mount_root.txt").write_text("nooa-private-mount-root-v1")
    for module, name in [("config", "_guard_config.py"), ("guards", "_guard_rules.py")]:
        spec = importlib.util.find_spec("nooa.runtime.sandbox." + module)
        if spec is None or not spec.origin:
            raise RuntimeError("Pinned NOOA guard source is unavailable")
        source = Path(spec.origin).read_text()
        if module == "guards":
            original = "from nooa.runtime.sandbox.config import LandlockRule, ResolvedSpec"
            if source.count(original) != 1:
                raise RuntimeError(
                    "Pinned NOOA guard import changed; refuse an unverified bootstrap"
                )
            source = source.replace(
                original, "from _guard_config import LandlockRule, ResolvedSpec"
            )
        (kit / name).write_text(source)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _allow_landlock_without_pidns() -> bool:
    return _env_flag("FACTOR_ALLOW_LANDLOCK_WITHOUT_PIDNS")


def cell_nproc_limit() -> int:
    """Max processes per cell (RLIMIT_NPROC / cgroup pids.max). Default 512."""
    raw = os.environ.get("FACTOR_CELL_NPROC", "512").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError("FACTOR_CELL_NPROC must be an integer") from error
    if value < 8:
        raise RuntimeError("FACTOR_CELL_NPROC must be >= 8")
    return value


def _apply_nproc_rlimit() -> None:
    import resource

    # XNU counts all processes of this login uid; a default cap can break every
    # scientific subprocess on a busy desktop. Explicit operator limits remain.
    if sys.platform == "darwin" and "FACTOR_CELL_NPROC" not in os.environ:
        return
    limit = cell_nproc_limit()
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        new_hard = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(resource.RLIMIT_NPROC, (new_hard, new_hard))
    except (OSError, ValueError):
        pass


def apply_cell_cgroup_pids(max_pids: int | None = None) -> Path | None:
    """Best-effort cgroup v2 pids.max for the current process tree (host side).

    Uses FACTOR_CELL_CGROUP when it points at a writable cgroup directory; otherwise
    returns None. Safe no-op when cgroup delegation is unavailable.
    """
    max_pids = cell_nproc_limit() if max_pids is None else max_pids
    root = os.environ.get("FACTOR_CELL_CGROUP", "").strip()
    if not root:
        return None
    base = Path(root)
    if not base.is_dir():
        return None
    cell = base / f"factor-cell-{os.getpid()}-{uuid_like()}"
    try:
        cell.mkdir(parents=True, exist_ok=False)
        pids_max = cell / "pids.max"
        if pids_max.exists():
            pids_max.write_text(str(max_pids))
        procs = cell / "cgroup.procs"
        procs.write_text(str(os.getpid()))
        return cell
    except OSError:
        return None


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}")
    return value


def _apply_process_rlimits() -> None:
    """Extra caps for the namespace-free 'process' backend (file size, open files)."""
    import resource

    caps = {
        resource.RLIMIT_FSIZE: _env_int("FACTOR_CELL_FSIZE_MB", 1024, 1) * 1024 * 1024,
        resource.RLIMIT_NOFILE: _env_int("FACTOR_CELL_NOFILE", 1024, 64),
    }
    for kind, limit in caps.items():
        try:
            _soft, hard = resource.getrlimit(kind)
            value = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
            resource.setrlimit(kind, (value, value))
        except (OSError, ValueError):
            pass


def _drop_privileges(writes) -> None:
    """When root and FACTOR_CELL_UID is set, hand scratch dirs to that uid and drop to it."""
    raw = os.environ.get("FACTOR_CELL_UID", "").strip()
    if os.geteuid() != 0 or not raw:
        return
    uid = _env_int("FACTOR_CELL_UID", 0, 1)
    gid = _env_int("FACTOR_CELL_GID", uid, 1)
    for root in writes:
        for directory, dirs, files in os.walk(root):
            for name in [directory, *(os.path.join(directory, f) for f in dirs + files)]:
                os.lchown(name, uid, gid)
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    if os.getuid() != uid or os.geteuid() != uid:
        raise RuntimeError("Privilege drop failed; execution refused")


def uuid_like() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


def _apply_host_rlimits():
    """Memory/CPU caps for Linux and macOS children (soft == hard)."""
    import resource

    # Mirror NOOA apply_rlimits: headroom above current address space + CPU seconds.
    try:
        baseline = 0
        status = Path("/proc/self/status")
        if status.is_file():
            for line in status.read_text().splitlines():
                if line.startswith("VmSize:"):
                    baseline = int(line.split()[1]) * 1024
                    break
        limit = baseline + 2048 * 1024 * 1024
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (OSError, ValueError):
        # macOS may reject some RLIMIT_AS values; CPU still applies.
        pass
    try:
        secs = 300
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        if hard != resource.RLIM_INFINITY:
            secs = min(secs, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (secs, secs))
    except (OSError, ValueError):
        pass


def install_linux(reads, writes):
    """Applied by the trusted fresh child bootstrap before loading user code/state."""
    # rlimits are available on both Linux and macOS; filesystem backends differ.
    if sys.platform == "darwin":
        _apply_host_rlimits()
        _apply_nproc_rlimit()
        return
    if not sys.platform.startswith("linux"):
        return
    import ctypes
    import platform

    from _guard_config import LandlockRule
    from _guard_rules import apply_landlock, apply_rlimits

    rules = [
        LandlockRule(str(Path(p).resolve()), required=True)
        for p in [*runtime_reads(), *reads]
        if Path(p).exists()
    ]
    rules += [LandlockRule(str(Path(p).resolve()), write=True, required=True) for p in writes]
    rules += [
        LandlockRule("/dev/null", write=True, required=True),
        LandlockRule("/dev/urandom"),
        LandlockRule("/dev/random"),
    ]
    # Resolve the trusted process baseline before /proc becomes inaccessible.
    apply_rlimits(max_memory_mb=2048, max_cpu_seconds=300)
    _apply_nproc_rlimit()
    mode = os.environ.get("NOOA_CELL_BACKEND", "landlock")
    if mode == "process":
        # No filesystem confinement here: relies on the container (read-only root).
        _apply_process_rlimits()
        _drop_privileges(writes)
        os.chdir(Path(writes[0]).resolve())
    elif mode == "landlock":
        apply_landlock(tuple(rules))
    elif mode == "bubblewrap":
        marker = Path("/.nooa-cell-backend")
        if not marker.is_file() or marker.read_text() != "nooa-private-mount-root-v1":
            raise RuntimeError("Bubblewrap backend was requested outside its trusted mount root")
        import stat

        if not os.statvfs("/").f_flag & getattr(stat, "ST_RDONLY", 1):
            raise RuntimeError("Bubblewrap root must be read-only")
        # Defense in depth: stack Landlock inside the private mount namespace.
        apply_landlock(tuple(rules))
    else:
        raise RuntimeError("Unknown cell execution backend")
    # Seccomp blacklist: network/io_uring, signals, ptrace/process-vm, namespaces,
    # bpf/perf/userfaultfd, keyring, kexec/modules, mount graph, handle lookups,
    # swap/reboot/acct/quotactl/personality, and x86 I/O privilege. clone3 -> ENOSYS
    # so libc falls back to clone (NEW* flags filtered). fork/exec remain for science.
    arch = platform.machine()
    # EPERM denylist (arch-specific numbers).
    deny_eperm = {
        "x86_64": [
            41,  # socket
            53,  # socketpair
            101,  # ptrace
            310,  # process_vm_readv
            311,  # process_vm_writev
            62,  # kill
            200,  # tkill
            234,  # tgkill
            129,  # rt_sigqueueinfo
            297,  # rt_tgsigqueueinfo
            424,  # pidfd_send_signal
            434,  # pidfd_open
            438,  # pidfd_getfd
            425,  # io_uring_setup
            426,  # io_uring_enter
            427,  # io_uring_register
            272,  # unshare
            308,  # setns
            298,  # perf_event_open
            323,  # userfaultfd
            321,  # bpf
            250,  # keyctl
            248,  # add_key
            249,  # request_key
            246,  # kexec_load
            320,  # kexec_file_load
            175,  # init_module
            313,  # finit_module
            176,  # delete_module
            165,  # mount
            166,  # umount2
            155,  # pivot_root
            428,  # open_tree
            429,  # move_mount
            430,  # fsopen
            431,  # fsconfig
            432,  # fsmount
            433,  # fspick
            304,  # open_by_handle_at
            303,  # name_to_handle_at
            87,  # swapon
            168,  # swapoff
            169,  # reboot
            163,  # acct (51 is getsockname on x86_64)
            212,  # lookup_dcookie
            179,  # quotactl
            135,  # personality
            172,  # iopl
            173,  # ioperm
            154,  # modify_ldt
        ],
        "aarch64": [
            198,  # socket
            199,  # socketpair
            117,  # ptrace
            270,  # process_vm_readv
            271,  # process_vm_writev
            129,  # kill
            130,  # tkill
            131,  # tgkill
            138,  # rt_sigqueueinfo
            139,  # rt_tgsigqueueinfo
            424,  # pidfd_send_signal
            434,  # pidfd_open
            438,  # pidfd_getfd
            425,  # io_uring_setup
            426,  # io_uring_enter
            427,  # io_uring_register
            97,  # unshare
            268,  # setns
            241,  # perf_event_open
            282,  # userfaultfd
            280,  # bpf
            219,  # keyctl
            217,  # add_key
            218,  # request_key
            104,  # kexec_load
            294,  # kexec_file_load
            105,  # init_module
            273,  # finit_module
            106,  # delete_module
            40,  # mount
            39,  # umount2
            41,  # pivot_root
            428,  # open_tree
            429,  # move_mount
            430,  # fsopen
            431,  # fsconfig
            432,  # fsmount
            433,  # fspick
            265,  # open_by_handle_at
            264,  # name_to_handle_at
            224,  # swapon
            225,  # swapoff
            142,  # reboot
            89,  # acct
            18,  # lookup_dcookie
            60,  # quotactl
            92,  # personality
        ],
    }
    if allow_network():
        # Opt-in data-lake access: only socket creation is filtered (connect, send*,
        # recv*, sockopt are never denied); keep the rest of the denylist intact.
        networking = {"x86_64": {41, 53}, "aarch64": {198, 199}}
        deny_eperm = {
            key: [n for n in numbers if n not in networking[key]]
            for key, numbers in deny_eperm.items()
        }
    clone_nr = {"x86_64": 56, "aarch64": 220}
    clone3_nr = {"x86_64": 435, "aarch64": 435}
    # Namespace-related clone flags (NEWNS|NEWCGROUP|NEWUTS|NEWIPC|NEWUSER|NEWPID|NEWNET).
    clone_ns_mask = (
        0x00020000 | 0x02000000 | 0x04000000 | 0x08000000 | 0x10000000 | 0x20000000 | 0x40000000
    )
    if arch not in deny_eperm:
        if mode == "process":
            sys.stderr.write("warning: seccomp unsupported on this architecture; skipped\n")
            return
        raise RuntimeError("Unsupported seccomp architecture")

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    BPF_LD = 0x00
    BPF_JMP = 0x05
    BPF_RET = 0x06
    BPF_W = 0x00
    BPF_ABS = 0x20
    BPF_JEQ = 0x10
    BPF_JGE = 0x30
    BPF_JSET = 0x40
    BPF_K = 0x00
    SECCOMP_RET_KILL = 0x80000000
    SECCOMP_RET_ERRNO = 0x00050000
    SECCOMP_RET_ALLOW = 0x7FFF0000
    ENOSYS = 38
    EPERM = 1

    expected_arch = 0xC000003E if arch == "x86_64" else 0xC00000B7
    instructions: list[tuple[int, int, int, int]] = [
        # A = arch
        (BPF_LD | BPF_W | BPF_ABS, 0, 0, 4),
        (BPF_JMP | BPF_JEQ | BPF_K, 1, 0, expected_arch),
        (BPF_RET | BPF_K, 0, 0, SECCOMP_RET_KILL),
        # A = nr
        (BPF_LD | BPF_W | BPF_ABS, 0, 0, 0),
    ]
    if arch == "x86_64":
        # Reject the alternate x32 syscall table (NR | 0x40000000).
        instructions.extend(
            [
                (BPF_JMP | BPF_JGE | BPF_K, 0, 1, 0x40000000),
                (BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | EPERM),
            ]
        )
    # clone3 -> ENOSYS (libc retries with clone).
    instructions.extend(
        [
            (BPF_JMP | BPF_JEQ | BPF_K, 0, 1, clone3_nr[arch]),
            (BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | ENOSYS),
        ]
    )
    # clone with any NEW* namespace bit -> EPERM; plain fork/clone for threads OK.
    instructions.extend(
        [
            (BPF_JMP | BPF_JEQ | BPF_K, 0, 4, clone_nr[arch]),
            # A = args[0] low 32 bits (flags)
            (BPF_LD | BPF_W | BPF_ABS, 0, 0, 16),
            (BPF_JMP | BPF_JSET | BPF_K, 0, 1, clone_ns_mask),
            (BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | EPERM),
            # restore A = nr for subsequent compares
            (BPF_LD | BPF_W | BPF_ABS, 0, 0, 0),
        ]
    )
    for number in deny_eperm[arch]:
        instructions.extend(
            [
                (BPF_JMP | BPF_JEQ | BPF_K, 0, 1, number),
                (BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | EPERM),
            ]
        )
    instructions.append((BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ALLOW))
    array = (Filter * len(instructions))(*(Filter(*i) for i in instructions))
    program = Program(len(instructions), array)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        if mode == "process":
            sys.stderr.write(
                f"warning: seccomp unavailable (errno {ctypes.get_errno()}); skipped\n"
            )
            return
        raise OSError(ctypes.get_errno(), "Cannot install seccomp; execution refused")


def _proc_children(root_pid: int) -> set[int]:
    """All descendants of root_pid via /proc (covers setsid grandchildren)."""
    if not Path("/proc").is_dir():
        return set()
    parents: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            raw = (entry / "stat").read_text()
            close = raw.rfind(")")
            fields = raw[close + 2 :].split()
            ppid = int(fields[1])
            parents.setdefault(ppid, []).append(pid)
        except (OSError, ValueError, IndexError):
            continue
    found: set[int] = set()
    stack = [root_pid]
    while stack:
        current = stack.pop()
        for child in parents.get(current, []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def reap_cell_children(root_pid: int, infrastructure: set[int]) -> None:
    """Stop cell descendants while preserving the pre-source Linux kernel chain.

    The host observes /proc outside the sandbox. Orphans stay under namespace
    init/subreaper, including children that called setsid or rapidly forked.
    Fail closed if a live descendant survives the bounded sweep.
    """
    if not sys.platform.startswith("linux"):
        return
    import time

    host_namespace = os.readlink("/proc/self/ns/pid")
    namespaces = set()
    for pid in infrastructure:
        try:
            namespace = os.readlink(f"/proc/{pid}/ns/pid")
            if namespace != host_namespace:
                namespaces.add(namespace)
        except FileNotFoundError:
            pass

    protected_groups = {os.getpgid(pid) for pid in infrastructure | {root_pid}}

    def live_children():
        candidates = _proc_children(root_pid)
        if namespaces:
            # A fork-exit chain can momentarily disappear from a parent graph
            # assembled from separate /proc reads. Namespace membership survives
            # reparenting, and never includes unrelated host processes.
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    if os.readlink(entry / "ns/pid") in namespaces:
                        candidates.add(int(entry.name))
                except (FileNotFoundError, PermissionError):
                    # Unrelated users' /proc entries are not readable. Known
                    # descendants are still checked below, with errors fatal.
                    pass
        found = set()
        for pid in candidates - infrastructure:
            try:
                raw = Path(f"/proc/{pid}/stat").read_text()
                fields = raw[raw.rfind(")") + 2 :].split()
                group = int(fields[2])
                if group not in protected_groups:
                    # Even a zombie records the setsid group's ID. Killing that
                    # group catches the moving live tip of a rapid fork chain.
                    try:
                        os.killpg(group, signal.SIGSTOP)
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if fields[0] != "Z":
                    found.add(pid)
            except FileNotFoundError:
                pass
        return found

    deadline = time.monotonic() + 3
    quiet = 0
    while time.monotonic() < deadline:
        live = live_children()
        if not live:
            quiet += 1
            if quiet >= 3:
                return
            time.sleep(0.002)
            continue
        quiet = 0
        for pid in live:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        for pid in live | live_children():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    raise RuntimeError("Cell subprocess cleanup did not quiesce")


def _darwin_group_has_no_live_members(group: int) -> bool:
    """Verify Darwin's zombie-only EPERM without hiding live permission failures."""
    import subprocess

    result = subprocess.run(
        ["/bin/ps", "-axo", "pgid=,stat="],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    for line in result.stdout.splitlines():
        pgid, status = line.split()
        if int(pgid) == group and not status.startswith("Z"):
            return False
    return True


def _kill_tree(root_pid: int) -> None:
    """SIGKILL the process group and every live /proc descendant (setsid-safe).

    Freeze with SIGSTOP first so a fork-chain cannot outrun a single snapshot.
    """
    import time

    deadline = time.monotonic() + 3.0
    quiet = 0
    while time.monotonic() < deadline:
        descendants = _proc_children(root_pid)
        try:
            os.killpg(root_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Darwin returns EPERM for an exited, unreaped group leader too.
            # Root-PID absence alone is insufficient: live group members may remain.
            if sys.platform != "darwin" or not _darwin_group_has_no_live_members(root_pid):
                raise
        if not descendants:
            quiet += 1
            if quiet >= 3:
                break
            time.sleep(0.002)
            continue
        quiet = 0
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        for pid in _proc_children(root_pid) | descendants:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def ensure_child_subreaper() -> None:
    """Become subreaper so setsid orphans re-parent here instead of PID 1."""
    if not sys.platform.startswith("linux"):
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_CHILD_SUBREAPER = 36
    libc.prctl(36, 1, 0, 0, 0)


async def reap(process):
    if process is None:
        return
    ensure_child_subreaper()
    _kill_tree(process.pid)
    await process.wait()
    _kill_tree(process.pid)
