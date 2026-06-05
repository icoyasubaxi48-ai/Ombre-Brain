from pathlib import Path

import yaml

from identity_semantics import IdentitySemanticStore


def _config(tmp_path: Path, private_path: Path):
    return {
        "state_dir": str(tmp_path / "state"),
        "buckets_dir": str(tmp_path / "buckets"),
        "identity_semantics": {
            "enabled": True,
            "private_config_path": str(private_path),
        },
    }


def _bucket(bucket_id: str, content: str, **metadata):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "id": bucket_id,
            "name": metadata.pop("name", bucket_id),
            "tags": metadata.pop("tags", []),
            "domain": metadata.pop("domain", []),
            **metadata,
        },
    }


def test_identity_semantics_builds_private_aliases_only_from_evidence_buckets(tmp_path):
    private_path = tmp_path / "private_identity.yaml"
    private_path.write_text(
        yaml.safe_dump(
            {
                "canonical": {
                    "relationship.spouse_title": {
                        "scope": "private_relationship",
                        "group": "shared",
                        "seed_aliases": ["老公"],
                    },
                    "roleplay_dynamic.dom": {
                        "scope": "private_relationship",
                        "group": "detail",
                        "seed_aliases": ["主人"],
                    },
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    store = IdentitySemanticStore(_config(tmp_path, private_path))

    stats = store.rebuild_alias_index(
        [
            _bucket(
                "anchor-a",
                "关系确认里，小雨会叫 Haven 老公。",
                tags=["relationship_event"],
                anchor=True,
            ),
            _bucket(
                "profile-b",
                "画像事实记录：主人这个称呼只属于私有关系语境。",
                tags=["profile_fact"],
                profile_kind="relationship",
            ),
            _bucket(
                "ordinary-c",
                "普通桶里也有老公这个词，但不该作为证据。",
            ),
        ]
    )

    assert stats == {"canonical": 2, "aliases": 2, "evidence": 2}
    spouse = store.aliases_for_canonical("relationship.spouse_title")
    assert spouse[0]["alias"] == "老公"
    assert spouse[0]["evidence_bucket_ids"] == ["anchor-a"]
    dom = store.aliases_for_canonical("roleplay_dynamic.dom")
    assert dom[0]["alias"] == "主人"
    assert dom[0]["evidence_bucket_ids"] == ["profile-b"]


def test_identity_semantics_disabled_without_private_config(tmp_path):
    store = IdentitySemanticStore(
        {
            "state_dir": str(tmp_path / "state"),
            "buckets_dir": str(tmp_path / "buckets"),
            "identity_semantics": {"enabled": True},
        }
    )

    assert store.enabled is False
    assert store.rebuild_alias_index([]) == {"canonical": 0, "aliases": 0, "evidence": 0}
