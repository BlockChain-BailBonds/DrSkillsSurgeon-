"""Structural validation for preregistered skill testing contracts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised for incomplete, ambiguous, or inconsistent experiment contracts."""


def _nonblank(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{path} must be a nonempty string")
    return value.strip()


def validate_contract(data: Any, minimum_probes: int = 1) -> dict[str, int | str]:
    """Check data structure, reference integrity, and prompt-set disjointness."""
    if isinstance(minimum_probes, bool) or not isinstance(minimum_probes, int) or minimum_probes < 1:
        raise ContractError("minimum_probes must be a positive integer")
    if not isinstance(data, dict):
        raise ContractError("root must be a JSON object")
    skill = data.get("skill")
    metric = data.get("metric")
    if not isinstance(skill, dict) or not isinstance(metric, dict):
        raise ContractError("skill and metric must each be JSON objects")
    skill_id = _nonblank(skill.get("id"), "skill.id")
    _nonblank(skill.get("description"), "skill.description")
    _nonblank(skill.get("in_scope"), "skill.in_scope")
    _nonblank(skill.get("out_of_scope"), "skill.out_of_scope")
    _nonblank(metric.get("name"), "metric.name")
    if not isinstance(metric.get("higher_is_more_skill"), bool):
        raise ContractError("metric.higher_is_more_skill must be a boolean")
    floor = metric.get("floor")
    target = metric.get("target_success")
    for name, number in (("floor", floor), ("target_success", target)):
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 1:
            raise ContractError(f"metric.{name} must be a number between 0 and 1")
    if metric["higher_is_more_skill"] and target <= floor:
        raise ContractError("target_success must exceed floor when higher is better")
    if not metric["higher_is_more_skill"] and target >= floor:
        raise ContractError("target_success must be below floor when lower is better")

    all_ids: set[str] = set()
    ids_by_group: dict[str, set[str]] = {}
    groups = ("probe", "contrast", "retain", "adversarial")
    for group in groups:
        items = data.get(group)
        if not isinstance(items, list) or not items:
            raise ContractError(f"{group} must be a nonempty array")
        ids: set[str] = set()
        for index, item in enumerate(items):
            path = f"{group}[{index}]"
            if not isinstance(item, dict):
                raise ContractError(f"{path} must be an object")
            prompt_id = _nonblank(item.get("id"), f"{path}.id")
            _nonblank(item.get("prompt"), f"{path}.prompt")
            if prompt_id in all_ids:
                raise ContractError(f"duplicate prompt ID: {prompt_id}")
            all_ids.add(prompt_id)
            ids.add(prompt_id)
        ids_by_group[group] = ids

    if len(ids_by_group["probe"]) < minimum_probes:
        raise ContractError(f"probe needs at least {minimum_probes} examples")

    pairs = data.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ContractError("pairs must be a nonempty array")
    seen_pairs: set[tuple[str, str]] = set()
    paired_probes: set[str] = set()
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ContractError(f"pairs[{i}] must be an object")
        probe_id = _nonblank(pair.get("probe_id"), f"pairs[{i}].probe_id")
        contrast_id = _nonblank(pair.get("contrast_id"), f"pairs[{i}].contrast_id")
        if probe_id not in ids_by_group["probe"] or contrast_id not in ids_by_group["contrast"]:
            raise ContractError(f"pairs[{i}] references an ID outside its declared set")
        key = (probe_id, contrast_id)
        if key in seen_pairs:
            raise ContractError(f"duplicate pair: {probe_id}, {contrast_id}")
        seen_pairs.add(key)
        paired_probes.add(probe_id)
    if paired_probes != ids_by_group["probe"]:
        missing = sorted(ids_by_group["probe"] - paired_probes)
        raise ContractError(f"unpaired probe IDs: {', '.join(missing)}")

    return {"skill_id": skill_id, **{group: len(ids_by_group[group]) for group in groups}, "pairs": len(pairs)}


def load_contract(path: str | Path, minimum_probes: int = 1) -> dict[str, int | str]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    return validate_contract(document, minimum_probes=minimum_probes)
