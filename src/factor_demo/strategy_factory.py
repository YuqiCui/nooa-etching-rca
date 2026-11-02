"""Build native strategy instances from the mission's effective policy."""

from nooa.config import CodeActConfig, LoopGuardConfig
from nooa.config.strategy_config import ReflexionConfig
from nooa.strategies import CodeActV2

from .evidence import research_postcondition, research_precondition
from .local_reflection import TrackedReflexion
from .native_sandbox import sandbox_settings


def research_strategy(policy, local=False):
    return CodeActV2(
        config=CodeActConfig(
            max_iterations=6 if local else policy.max_research_iterations,
            max_retries=policy.validation_retries,
            cell_timeout=60 if local else 330,
            execution_backend="sandbox",
            sandbox=sandbox_settings(),
            loop_guard=LoopGuardConfig(
                repeat_threshold=policy.loop_repeat_threshold, window=policy.loop_window
            )
            if policy.loop_guard
            else None,
            preconditions=[research_precondition],
            postconditions=[research_postcondition],
        )
    )


def reflection_strategy(policy):
    return TrackedReflexion(
        base=research_strategy(policy, local=True),
        config=ReflexionConfig(max_iterations=policy.reflexion_iterations),
    )
