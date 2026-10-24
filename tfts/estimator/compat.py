"""Compose schema feature selection with backbone input capabilities."""

from typing import Any, Optional

from tfts.data.schema import (
    FeatureDType,
    FeaturePlan,
    FeatureRole,
    FeatureSelection,
    FeatureSpec,
    Schema,
    resolve_feature_plan as select_features,
)


def resolve_feature_plan(schema, selection=None, input_spec=None, unsupported="raise"):
    if unsupported not in {"raise", "drop"}:
        raise ValueError("unsupported must be either 'raise' or 'drop'")
    if input_spec is not None and not input_spec.supports_multivariate_target and len(schema.target_cols) > 1:
        raise ValueError("model does not support multivariate targets")
    plan = select_features(schema, selection)
    selected, excluded, problems = [], dict(plan.excluded), []
    for spec in plan.selected:
        reason = _input_incompatibility(spec, input_spec)
        if reason:
            if unsupported == "raise" or (selection and spec.name in selection.include_names):
                problems.append(f"feature {spec.name!r} is not supported: {reason}")
            excluded[spec.name] = reason
        else:
            selected.append(spec)
    if problems:
        raise ValueError("\n".join(problems))
    return FeaturePlan(tuple(selected), excluded)


def resolve_model_features(model_name, schema, selection=None, unsupported="raise"):
    from tfts.models.catalog import get_model_capabilities

    return resolve_feature_plan(schema, selection, get_model_capabilities(model_name).input_spec, unsupported)


def _input_incompatibility(spec: FeatureSpec, input_spec: Optional[Any]) -> Optional[str]:
    if input_spec is None:
        return None
    accepted_roles = {getattr(role, "value", role) for role in input_spec.accepted_roles}
    if spec.role.value not in accepted_roles:
        return f"model does not accept role {spec.role.value!r}"
    if spec.dtype == FeatureDType.CATEGORICAL and not input_spec.supports_categorical:
        return "model does not accept categorical features"
    if spec.role == FeatureRole.STATIC and not input_spec.supports_static:
        return "model does not accept static features"
    accepted_dtypes = input_spec.accepted_dtypes_by_role.get(spec.role.value)
    if accepted_dtypes is not None and spec.dtype.value not in accepted_dtypes:
        return f"model does not accept dtype {spec.dtype.value!r} for role {spec.role.value!r}"
    return None
