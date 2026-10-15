"""Closure checks for Minimal Verifiable Evidence Units."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TEMPLATE_REQUIRED_SLOTS: dict[str, tuple[str, ...]] = {
    "RoundTransition": ("round_update", "input_var", "output_var", "round_scope"),
    "SBoxConstraint": ("sbox", "propagation_table", "input_var", "output_var", "bit_width"),
    "LinearLayer": ("input_var", "output_var", "linear_map"),
    "ObjectiveTerm": ("objective", "optimization_direction"),
    "BoundaryCondition": ("boundary_type", "variable", "value"),
    "PermutationLayer": ("permutation", "input_var", "output_var"),
    "PropagationTable": ("table_type", "input_var", "output_var"),
    "OperationModel": ("operation", "input_var", "output_var", "model_relation"),
    "PropagationRule": ("rule_type", "operation", "input_property", "output_property"),
    "MILPConstraint": ("target_operation", "variables", "constraints", "variable_domain"),
    "ComplexityClaim": ("resource", "bound", "setting", "claim"),
    "AttackProcedure": ("target", "attack_goal", "steps", "assumptions"),
    "ProblemDefinition": ("problem", "domain", "relation"),
    "ReductionClaim": ("source_problem", "target_problem", "reduction_map", "claim"),
    "AlgorithmClaim": ("algorithm", "task", "guarantee"),
}

INTEGER_SLOTS = {"bit_width"}
SLOT_FALLBACK_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "variables": (("variables",),),
    "round_update": (("round_function", "update"),),
    "round_scope": (("round_function", "round_scope"),),
    "linear_map": (("round_function", "linear_map"),),
    "permutation": (("round_function", "permutation"),),
    "propagation_table": (("sbox_table", "table"), ("sbox_table", "table_type")),
    "sbox": (("sbox_table", "sbox"),),
    "input_var": (("sbox_table", "input_var"),),
    "output_var": (("sbox_table", "output_var"),),
    "bit_width": (("sbox_table", "bit_width"),),
    "model_relation": (("round_function", "linear_map"), ("round_function", "update")),
    "operation": (("round_function", "permutation"), ("sbox_table", "sbox")),
    "objective": (("objective", "objective"),),
    "optimization_direction": (("objective", "optimization_direction"),),
    "solver": (("objective", "solver"),),
    "boundary_variable": (("objective", "boundary_variable"),),
    "boundary_type": (("objective", "boundary_type"),),
    "boundary_value": (("objective", "boundary_value"),),
    "variable": (("objective", "boundary_variable"),),
    "value": (("objective", "boundary_value"),),
    "table_type": (("sbox_table", "table_type"),),
    "claim": (("objective", "claim"),),
    "source_consistency": (("source_evidence",),),
    "verification_backend": (("template_slots", "verification_backend"),),
    "result_rows": (("template_slots", "result_rows"),),
    "primitive": (("scope", "primitive"),),
    "primitive_family": (("scope", "primitive"), ("state_layout", "layout")),
    "cipher_family": (("scope", "primitive"), ("state_layout", "layout")),
    "model_family": (("scope", "primitive"), ("state_layout", "layout")),
    "target": (("scope", "primitive"), ("objective", "boundary_value")),
    "target_family": (("scope", "primitive"), ("objective", "boundary_value")),
    "analysis_model": (("scope", "analysis_model"),),
    "analysis_type": (("scope", "analysis_model"), ("scope", "attack_model")),
    "attack_model": (("scope", "attack_model"), ("scope", "analysis_model")),
    "attack_type": (("scope", "attack_model"), ("scope", "analysis_model")),
    "rounds": (("scope", "round_range"), ("objective", "boundary_value")),
    "algorithm": (("template_slots", "algorithm"), ("objective", "claim")),
    "heuristic_formula": (("template_slots", "heuristic_formula"), ("objective", "claim")),
    "success_probability": (("template_slots", "success_probability"), ("objective", "claim")),
    "simulation_rows": (("template_slots", "simulation_rows"),),
}

SLOT_SEARCH_ALIASES: dict[str, tuple[str, ...]] = {
    "balanced_bits": ("balanced_bits",),
    "data": ("data", "data_exp"),
    "log_estimate": ("log_estimate", "heuristic_log_estimate", "log2_queries"),
    "number": ("number", "count"),
    "probability_exp": ("probability_exp", "probability", "probability_exponent"),
    "query_count": ("queries", "query_count", "number_of_queries"),
    "rounds": ("round", "rounds", "total_rounds"),
    "trail_count": ("trails", "trail_count", "trail_count"),
}


@dataclass(frozen=True)
class ClosureEvaluation:
    """Result of the three closure layers for one evidence unit."""

    syn: bool
    type: bool
    sem: bool
    reasons: list[str] = field(default_factory=list)
    sem_slots_required: list[str] = field(default_factory=list)
    sem_slots_missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "syn": self.syn,
            "type": self.type,
            "sem": self.sem,
            "reasons": list(self.reasons),
            "sem_slots_required": list(self.sem_slots_required),
            "sem_slots_missing": list(self.sem_slots_missing),
        }


def evaluate_mveu_closure(record: dict[str, Any] | None) -> ClosureEvaluation:
    """Compute SynClosure, TypeClosure, and SemClosure from an MVEU-like record."""

    if not isinstance(record, dict):
        return ClosureEvaluation(
            syn=False,
            type=False,
            sem=False,
            reasons=["record is not an object"],
        )

    evidence = _evidence_from_record(record)
    reasons: list[str] = []
    template = evidence.get("template")
    required_slots = required_slots_for_record(record)
    template_known = _template_known(record, template, required_slots)
    template_slots = evidence.get("template_slots")
    source_evidence = evidence.get("source_evidence")
    scope = evidence.get("scope")

    syn_reasons = _syntactic_reasons(
        evidence=evidence,
        template=template,
        template_known=template_known,
        template_slots=template_slots,
        required_slots=required_slots,
        source_evidence=source_evidence,
        scope=scope,
    )
    type_reasons = _type_reasons(evidence, template_slots, source_evidence, scope)
    sem_missing = _semantic_missing_slots(
        evidence=evidence,
        template_slots=template_slots,
        required_slots=required_slots,
    )

    reasons.extend(syn_reasons)
    reasons.extend(type_reasons)
    return ClosureEvaluation(
        syn=not syn_reasons,
        type=not type_reasons,
        sem=template_known and not sem_missing,
        reasons=reasons,
        sem_slots_required=list(required_slots),
        sem_slots_missing=sem_missing,
    )


def required_slots_for_record(record: dict[str, Any] | None) -> tuple[str, ...]:
    """Return semantic slots required by legacy or release-v1 MVEU records."""

    if not isinstance(record, dict):
        return ()
    evidence = _evidence_from_record(record)
    eval_closure = _eval_closure_from_record(record)
    release_slots = _string_list(eval_closure.get("sem_slots_required"))
    if release_slots:
        return tuple(release_slots)
    template = str(evidence.get("template") or "")
    return TEMPLATE_REQUIRED_SLOTS.get(template, ())


def _evidence_from_record(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record.get("evidence", record)
    return evidence if isinstance(evidence, dict) else {}


def _eval_closure_from_record(record: dict[str, Any]) -> dict[str, Any]:
    eval_block = record.get("eval")
    if not isinstance(eval_block, dict):
        return {}
    closure = eval_block.get("closure")
    return closure if isinstance(closure, dict) else {}


def _template_known(
    record: dict[str, Any],
    template: Any,
    required_slots: tuple[str, ...],
) -> bool:
    template_name = str(template or "")
    if not template_name:
        return False
    if template_name in TEMPLATE_REQUIRED_SLOTS:
        return True
    if required_slots and _eval_closure_from_record(record):
        return True
    return False


def _syntactic_reasons(
    *,
    evidence: dict[str, Any],
    template: Any,
    template_known: bool,
    template_slots: Any,
    required_slots: tuple[str, ...],
    source_evidence: Any,
    scope: Any,
) -> list[str]:
    reasons: list[str] = []
    if not _non_empty(evidence.get("mveu_id")):
        reasons.append("syn: missing mveu_id")
    if not _non_empty(template):
        reasons.append("syn: missing template")
    elif not template_known:
        reasons.append(f"syn: unknown template {template!r}")
    if not isinstance(scope, dict):
        reasons.append("syn: scope must be an object")
    if not isinstance(source_evidence, list) or not source_evidence:
        reasons.append("syn: source_evidence must be a non-empty list")
    if not isinstance(template_slots, dict):
        reasons.append("syn: template_slots must be an object")
    else:
        missing_keys = [
            slot
            for slot in required_slots
            if not _slot_has_syntactic_anchor(evidence, template_slots, slot)
        ]
        if missing_keys:
            reasons.append(f"syn: missing template slot keys {missing_keys}")
    return reasons


def _semantic_missing_slots(
    *,
    evidence: dict[str, Any],
    template_slots: Any,
    required_slots: tuple[str, ...],
) -> list[str]:
    if not isinstance(template_slots, dict):
        return list(required_slots)

    return [
        slot
        for slot in required_slots
        if not _slot_has_semantic_value(evidence, template_slots, slot)
    ]


def _type_reasons(
    evidence: dict[str, Any],
    template_slots: Any,
    source_evidence: Any,
    scope: Any,
) -> list[str]:
    reasons: list[str] = []
    if isinstance(scope, dict):
        reasons.extend(_scope_type_reasons(scope))
    reasons.extend(_source_evidence_type_reasons(source_evidence))
    reasons.extend(_variable_type_reasons(evidence.get("variables")))
    if isinstance(template_slots, dict):
        reasons.extend(_slot_type_reasons(template_slots))
    return reasons


def _scope_type_reasons(scope: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    round_range = scope.get("round_range")
    if round_range is not None and not _valid_round_range(round_range):
        reasons.append("type: scope.round_range must be a two-item increasing integer range")
    for key in ("primitive", "analysis_model", "attack_model"):
        if key in scope and scope[key] is not None and not isinstance(scope[key], str):
            reasons.append(f"type: scope.{key} must be a string when present")
    assumptions = scope.get("assumptions")
    if assumptions is not None and not isinstance(assumptions, list):
        reasons.append("type: scope.assumptions must be a list when present")
    return reasons


def _source_evidence_type_reasons(source_evidence: Any) -> list[str]:
    if source_evidence is None:
        return []
    if not isinstance(source_evidence, list):
        return ["type: source_evidence must be a list"]
    reasons: list[str] = []
    for index, item in enumerate(source_evidence):
        if not isinstance(item, dict):
            reasons.append(f"type: source_evidence[{index}] must be an object")
            continue
        page = item.get("page")
        if page is not None and not _parse_int(page, minimum=1):
            reasons.append(f"type: source_evidence[{index}].page must be a positive integer")
        fact_id = item.get("fact_id")
        if fact_id is not None and not _non_empty(fact_id):
            reasons.append(f"type: source_evidence[{index}].fact_id must be non-empty")
    return reasons


def _variable_type_reasons(variables: Any) -> list[str]:
    if variables is None:
        return []
    if not isinstance(variables, list):
        return ["type: variables must be a list"]
    reasons: list[str] = []
    for index, variable in enumerate(variables):
        if not isinstance(variable, dict):
            reasons.append(f"type: variables[{index}] must be an object")
            continue
        if not _non_empty(variable.get("canonical")) and not _non_empty(variable.get("symbol")):
            reasons.append(f"type: variables[{index}] must have canonical or symbol")
        bit_width = variable.get("bit_width")
        if bit_width is not None and not _parse_int(bit_width, minimum=0):
            reasons.append(f"type: variables[{index}].bit_width must be a non-negative integer")
        domain = variable.get("domain")
        if domain is not None and not isinstance(domain, str):
            reasons.append(f"type: variables[{index}].domain must be a string when present")
        for key in ("defined_by", "used_by"):
            value = variable.get(key)
            if value is not None and not isinstance(value, list):
                reasons.append(f"type: variables[{index}].{key} must be a list when present")
    return reasons


def _slot_type_reasons(template_slots: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for slot_name, slot_item in template_slots.items():
        value = _slot_value(slot_item)
        if slot_name in INTEGER_SLOTS and _non_empty(value) and not _parse_int(value, minimum=1):
            reasons.append(f"type: slot {slot_name!r} must be a positive integer")
        if isinstance(slot_item, dict):
            fact_ids = slot_item.get("fact_ids")
            if fact_ids is not None:
                if not isinstance(fact_ids, list):
                    reasons.append(f"type: slot {slot_name!r}.fact_ids must be a list")
                elif not all(isinstance(item, str) and item for item in fact_ids):
                    reasons.append(f"type: slot {slot_name!r}.fact_ids must contain strings")
        elif isinstance(slot_item, (list, str, int, float, bool)) or slot_item is None:
            continue
        else:
            reasons.append(f"type: slot {slot_name!r} has unsupported value container")
    return reasons


def _slot_has_semantic_value(
    evidence: dict[str, Any],
    template_slots: dict[str, Any],
    slot: str,
) -> bool:
    if slot in template_slots and _non_empty(_slot_value(template_slots[slot])):
        return True
    fallback_paths = SLOT_FALLBACK_PATHS.get(slot, ())
    if any(_non_empty(_nested_value(evidence, path)) for path in fallback_paths):
        return True
    return _contains_non_empty_key(evidence, _slot_search_aliases(slot))


def _slot_has_syntactic_anchor(
    evidence: dict[str, Any],
    template_slots: dict[str, Any],
    slot: str,
) -> bool:
    if slot in template_slots:
        return True
    if any(_nested_key_exists(evidence, path) for path in SLOT_FALLBACK_PATHS.get(slot, ())):
        return True
    return _contains_key(evidence, _slot_search_aliases(slot))


def _slot_value(slot_item: Any) -> Any:
    if isinstance(slot_item, dict) and "value" in slot_item:
        return slot_item.get("value")
    return slot_item


def _nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _nested_key_exists(data: dict[str, Any], path: tuple[str, ...]) -> bool:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def _slot_search_aliases(slot: str) -> tuple[str, ...]:
    normalized = _normalize_key(slot)
    aliases = {normalized}
    aliases.update(_normalize_key(alias) for alias in SLOT_SEARCH_ALIASES.get(slot, ()))
    if normalized.endswith("_exp"):
        aliases.add(normalized[:-4])
        aliases.add(normalized[:-4] + "_exponent")
    if normalized.endswith("_count"):
        aliases.add(normalized[:-6])
    if normalized.endswith("s"):
        aliases.add(normalized[:-1])
    else:
        aliases.add(normalized + "s")
    return tuple(sorted(alias for alias in aliases if alias))


def _contains_key(value: Any, aliases: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if _normalize_key(key) in aliases:
                return True
            if _contains_key(child, aliases):
                return True
    elif isinstance(value, list):
        return any(_contains_key(item, aliases) for item in value)
    return False


def _contains_non_empty_key(value: Any, aliases: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if _normalize_key(key) in aliases and _non_empty(_slot_value(child)):
                return True
            if _contains_non_empty_key(child, aliases):
                return True
    elif isinstance(value, list):
        return any(_contains_non_empty_key(item, aliases) for item in value)
    return False


def _valid_round_range(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    try:
        left, right = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return False
    return left <= right


def _parse_int(value: Any, *, minimum: int) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return False
    return parsed >= minimum


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()
