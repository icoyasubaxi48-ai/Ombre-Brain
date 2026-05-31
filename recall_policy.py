from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory_relevance import (
    MemoryRelevanceOptions,
    memory_relevance_options_from_config,
    query_has_explicit_entity_marker,
    query_has_technical_recall_marker,
    recall_admission_decision,
)


CONTEXT_ONLY_SECTIONS = frozenset({"affect_anchor", "favorite_reason", "comment"})


@dataclass(frozen=True)
class RecallPolicyDecision:
    admit_direct: bool
    admit_diffused: bool
    seed_allowed: bool
    reason: str
    suppressed: bool
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def admit(self) -> bool:
        return self.admit_direct


class RecallPolicy:
    def __init__(
        self,
        options: MemoryRelevanceOptions | None = None,
        *,
        semantic_threshold: float = 0.72,
        rerank_threshold: float = 0.65,
    ) -> None:
        self.options = options or memory_relevance_options_from_config()
        self.semantic_threshold = _safe_float(semantic_threshold, 0.72)
        self.rerank_threshold = _safe_float(rerank_threshold, 0.65)

    def requires_topic_evidence(self, query: str) -> bool:
        return query_has_explicit_entity_marker(query) or query_has_technical_recall_marker(query)

    def has_strong_score(
        self,
        *,
        semantic_score: float | None = None,
        rerank_score: float | None = None,
    ) -> bool:
        return (
            _safe_float(semantic_score, 0.0) >= self.semantic_threshold
            or _safe_float(rerank_score, 0.0) >= self.rerank_threshold
        )

    def assess(
        self,
        query: str,
        node: dict,
        *,
        has_topic_evidence: bool = False,
        semantic_score: float | None = None,
        rerank_score: float | None = None,
        high_confidence_edge: bool = False,
        context_only: bool = False,
    ) -> RecallPolicyDecision:
        debug = {
            "requires_topic_evidence": self.requires_topic_evidence(query),
            "has_topic_evidence": bool(has_topic_evidence),
            "semantic_score": _maybe_float(semantic_score),
            "rerank_score": _maybe_float(rerank_score),
            "high_confidence_edge": bool(high_confidence_edge),
            "context_only": bool(context_only),
        }

        if context_only:
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="context_only_temperature_moment",
                suppressed=True,
                debug=debug,
            )

        base = recall_admission_decision(
            query,
            node,
            self.options,
            semantic_score=semantic_score,
            rerank_score=rerank_score,
            high_confidence_edge=high_confidence_edge,
            semantic_threshold=self.semantic_threshold,
            rerank_threshold=self.rerank_threshold,
        )
        debug["base_reason"] = base.reason

        if not base.admit:
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason=base.reason,
                suppressed=True,
                debug=debug,
            )

        if (
            debug["requires_topic_evidence"]
            and not has_topic_evidence
            and not self.has_strong_score(
                semantic_score=semantic_score,
                rerank_score=rerank_score,
            )
            and not high_confidence_edge
        ):
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="query_topic_evidence_missing",
                suppressed=True,
                debug=debug,
            )

        return RecallPolicyDecision(
            admit_direct=True,
            admit_diffused=True,
            seed_allowed=True,
            reason=base.reason,
            suppressed=False,
            debug=debug,
        )


def is_context_only_section(section: Any) -> bool:
    return str(section or "") in CONTEXT_ONLY_SECTIONS


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    number = _maybe_float(value)
    return default if number is None else number
