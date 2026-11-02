"""Capability-selected configuration for the unmodified NOOA sandbox executor."""

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.guards import probe_capabilities

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def native_capabilities():
    return probe_capabilities()


def sandbox_settings(*, capabilities=None):
    """Select explicit guards, never change the upstream executor/REPL protocol."""
    mode = os.environ.get("FACTOR_SANDBOX_FILESYSTEM", "auto").strip().lower()
    if mode not in {"auto", "required", "off"}:
        raise ValueError("FACTOR_SANDBOX_FILESYSTEM must be auto, required or off")
    network_value = os.environ.get("FACTOR_CELL_ALLOW_NETWORK", "0").strip()
    if network_value not in {"0", "1"}:
        raise ValueError("FACTOR_CELL_ALLOW_NETWORK must be 0 or 1")
    # Decorators are constructed on import, including documentation tools and
    # non-Linux clients. Only the deployed Linux process performs kernel probes.
    caps = capabilities
    if caps is None and sys.platform.startswith("linux"):
        caps = native_capabilities()
    filesystem = mode == "required" or (mode == "auto" and bool(caps and caps.filesystem))
    network = network_value == "1"
    if caps is not None:
        if filesystem and not caps.filesystem:
            raise RuntimeError("Required native sandbox filesystem guard needs Landlock ABI >= 1")
        if not network and not caps.network:
            raise RuntimeError("Network blocking requested but native seccomp is unavailable")
    workspace = os.environ.get("FACTOR_SANDBOX_WORKSPACE", "").strip()
    if workspace:
        target = Path(workspace).expanduser().resolve()
        if not target.is_dir():
            raise ValueError("FACTOR_SANDBOX_WORKSPACE must be an existing directory")
        workspace = str(target)
    settings = SandboxConfig(
        filesystem=filesystem,
        workspace=workspace or None,
        network=network,
        max_memory_mb=int(os.environ.get("FACTOR_SANDBOX_MEMORY_MB", "2048")),
        max_cpu_seconds=int(os.environ.get("FACTOR_SANDBOX_CPU_SECONDS", "300")),
        broker_timeout_s=float(os.environ.get("FACTOR_CELL_TOOL_TIMEOUT", "330")),
        require=True,
    )
    log.info(
        "NOOA native sandbox: filesystem=%s network_allowed=%s workspace=%s require=True",
        settings.filesystem,
        settings.network,
        settings.workspace,
    )
    if caps is not None and not filesystem:
        log.warning(
            "NOOA native sandbox has no filesystem confinement (mode=%s, Landlock ABI=%s)",
            mode,
            caps.landlock_abi,
        )
    return settings


def sandbox_status():
    caps = native_capabilities()
    settings = sandbox_settings(capabilities=caps)
    return {
        "executor": "nooa.runtime.sandbox.executor.SandboxedExecutor",
        "landlock_abi": caps.landlock_abi,
        "seccomp": caps.seccomp,
        "filesystem": settings.filesystem,
        "network_allowed": settings.network,
        "workspace": settings.workspace,
        "require": settings.require,
        "pid_namespace": False,
    }
