"""Linux policy startup imports only attributed guard primitives, not the agent SDK."""

import json
import subprocess
import sys

from factor_demo.sandboxing import child_environment, prepare_guard_modules


def test_guard_bootstrap_is_small_and_never_loads_agent_or_model_client(tmp_path):
    kit = tmp_path / "runtime"
    kit.mkdir()
    prepare_guard_modules(kit)
    code = f"import sys,json;sys.path.insert(0,{str(kit)!r});from _guard_config import LandlockRule;from _guard_rules import apply_landlock;print(json.dumps([x for x in sys.modules if x=='nooa' or x.startswith('nooa.') or x=='litellm']))"
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True, timeout=20
    )
    assert json.loads(result.stdout) == []
    assert "SPDX-License-Identifier: Apache-2.0" in (kit / "_guard_rules.py").read_text()
    assert child_environment(tmp_path)["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert child_environment(tmp_path)["PYTHON_DOTENV_DISABLED"] == "1"


def test_child_environment_forces_litellm_production_mode(tmp_path, monkeypatch):
    # Issue #4 B: DEV mode auto-loads the project .env, which Landlock denies.
    monkeypatch.delenv("LITELLM_MODE", raising=False)
    assert child_environment(tmp_path)["LITELLM_MODE"] == "PRODUCTION"


def test_dotenv_warnings_flags_project_env_without_reading_it(tmp_path):
    from factor_demo.sandboxing import dotenv_warnings

    assert dotenv_warnings([tmp_path]) == []
    (tmp_path / ".env").write_text("SECRET=do-not-print\n")
    warnings = dotenv_warnings([tmp_path, tmp_path])
    assert len(warnings) == 1 and ".env" in warnings[0]
    assert "do-not-print" not in warnings[0]
