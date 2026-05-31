from recall_policy import RecallPolicy


def test_context_only_moment_cannot_be_direct_seed():
    policy = RecallPolicy()

    decision = policy.assess(
        "情书找门",
        {"text": "世界继续筑墙，小雨一叫，Haven转向那扇门"},
        has_topic_evidence=True,
        context_only=True,
    )

    assert decision.reason == "context_only_temperature_moment"
    assert not decision.admit_direct
    assert not decision.seed_allowed


def test_technical_query_requires_topic_evidence_without_strong_score():
    policy = RecallPolicy()

    decision = policy.assess(
        "handoff bridge 注入 原文",
        {"text": "一封情书，世界继续筑墙，我会继续寻找门"},
        has_topic_evidence=False,
        semantic_score=0.2,
    )

    assert decision.reason == "query_topic_evidence_missing"
    assert not decision.admit_direct


def test_broad_context_words_do_not_make_normal_chat_technical():
    policy = RecallPolicy()

    assert not policy.requires_topic_evidence("这张图片的上下文我想起来了")
    assert not policy.requires_topic_evidence("memory context makes me nostalgic")
    assert policy.requires_topic_evidence("读图 原文 怎么注入")
    assert policy.requires_topic_evidence("handoff 原文")


def test_technical_query_can_admit_strong_semantic_match_without_literal_topic_evidence():
    policy = RecallPolicy()

    decision = policy.assess(
        "handoff bridge 注入 原文",
        {"text": "一封情书，世界继续筑墙，我会继续寻找门"},
        has_topic_evidence=False,
        semantic_score=0.9,
    )

    assert decision.reason == "non_explicit_query"
    assert decision.admit_direct


def test_explicit_entity_query_keeps_existing_reliable_evidence_gate():
    policy = RecallPolicy()

    decision = policy.assess(
        "Titans",
        {"text": "临时雨夜和记忆写入偏好"},
        has_topic_evidence=False,
    )

    assert decision.reason == "explicit_query_without_reliable_evidence"
    assert not decision.admit_direct
