"""One NOOA CodeAct agent; no workflow or additional agent framework."""

from __future__ import annotations

from nooa import Agent, hidden, strategy

from .models import Conclusion

with hidden:
    from nooa.config.strategy_config import CodeActConfig
    from nooa.strategies import CodeActV2

    from ..native_sandbox import sandbox_settings


class FactorResearchAgent(Agent):
    """Research associations between sensor-step features and expert labels.

    You have access only to bounded research tools. Use those tools for every
    numerical claim. Do not access the filesystem, imports, private attributes,
    external APIs, holdout data, or generator internals. Do not fabricate results.
    Factor scores show associations, never confirmed mechanical root causes.
    """

    @hidden
    def __init__(self, research, **kwargs):
        super().__init__(**kwargs)
        self._research = research

    def describe_dataset(self) -> dict:
        """Inspect dataset, expert notes, budget, and current best experiment."""
        return self._research.describe()

    def screen_features(self, top_k: int = 12) -> list[dict]:
        """Rank raw features using training data only. No generalization guarantee."""
        return self._research.screen(top_k)

    def related_features(self, feature: str) -> list[dict]:
        """Find all available steps and statistics for this feature's sensor."""
        return self._research.related(feature)

    def inspect_features(self, columns: list[str]) -> dict:
        """Compare class distributions and correlations on train and development data."""
        return self._research.inspect(columns)

    def evaluate_candidate(self, factors: list[dict], rationale: str) -> dict:
        """Evaluate 1-5 explicit factors; returns real metrics and experiment ID.

        Each factor: dict(name='...', operation='raw'|'difference'|'ratio'|'product',
        columns=['catalogued_column'] or ['first_column', 'second_column']).
        Difference is first minus second; ratio is first divided by second.
        Fit preprocessing and a fixed classifier on training; score on development.
        Every new candidate, including failures, consumes one experiment.
        """
        return self._research.evaluate(factors, rationale)

    def list_experiments(self) -> list[dict]:
        """Read persisted experiments, including failures and actual metric values."""
        return self._research.records

    def record_finding(self, text: str) -> dict:
        """Save a short finding with experiment IDs and remaining uncertainties."""
        return self._research.finding(text)

    @strategy(
        CodeActV2(
            config=CodeActConfig(
                max_iterations=22,
                max_retries=1,
                execution_backend="sandbox",
                cell_timeout=60,
                sandbox=sandbox_settings(),
            )
        )
    )
    async def research(self, objective: str) -> Conclusion:
        """Autonomously research useful, simple factors for the stated objective.

        First inspect the dataset and existing experiments. Adapt the next action
        to actual observations: screen, inspect, construct, test, reject or refine.
        Use available experiment budget productively. Compare raw and composed
        factors when useful, and examine generalization rather than training score.
        You may use related_features to inspect cross-step candidates. No discovery
        path or winning formula is prescribed. Keep the final set small.
        Record evidence-supported findings and failed hypotheses. Finish by returning
        a real successful experiment ID, summary and limitations. Never invent an ID.
        """
        ...  # NOOA generation marker; this is executable framework syntax.
