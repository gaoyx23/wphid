"""Lossless reporting of uncalibrated multiscale passive support evidence.

This is an output contract, not another estimator or confidence construction.
Whole supports stay distinct from their atom union. No truth label is consumed.
"""


def summarize_passive_supports(*, per_scale, candidate_names, allowed_supports):
    names = tuple(candidate_names)
    if not names or len(set(names)) != len(names) or not per_scale:
        raise ValueError("nonempty unique dictionary and scale evidence required")
    order = {name: index for index, name in enumerate(names)}

    def canonical(support):
        support = tuple(support)
        if len(set(support)) != len(support) or not set(support) <= set(names):
            raise ValueError("support contains duplicate or unknown atoms")
        return tuple(sorted(support, key=order.__getitem__))

    universe = tuple(dict.fromkeys(canonical(s) for s in allowed_supports))
    if not universe:
        raise ValueError("complete allowed support universe required")
    preferred, eligible, by_scale = [], [], {}
    for scale, evidence in per_scale.items():
        selected = evidence["selected"]
        point = canonical(selected["support"])
        sets = tuple(dict.fromkeys(canonical(s) for s in selected["one_standard_error_supports"]))
        if point not in sets or any(s not in universe for s in sets):
            raise ValueError("selected and eligible supports must belong to the declared universe")
        preferred.append(point)
        eligible.extend(sets)
        by_scale[scale] = dict(preferred_support=list(point),
            heuristic_eligible_supports=[list(s) for s in sets])
    preferred = tuple(dict.fromkeys(preferred))
    eligible = tuple(dict.fromkeys(eligible))
    preferred_atoms = {atom for support in preferred for atom in support}
    eligible_atoms = {atom for support in eligible for atom in support}
    return dict(
        per_scale=by_scale,
        preferred_supports=[list(s) for s in preferred],
        heuristic_eligible_supports=[list(s) for s in eligible],
        preferred_atom_summary=[n for n in names if n in preferred_atoms],
        heuristic_atom_summary=[n for n in names if n in eligible_atoms],
        compression_omitted_eligible_atoms=[n for n in names if n in eligible_atoms-preferred_atoms],
        allowed_support_universe=[list(s) for s in universe],
        statistically_excluded_supports=[],
        joint_coverage_calibrated=False,hard_pruning_authorized=False,
        full_dictionary_retention_is_recovery=False,
        parameter_confidence_intervals=None,
        status="uncalibrated_advisory_support_evidence",
        interpretation="Preferred supports and one-SE heuristic supports are summaries, not guaranteed physical exclusions. The atom union is not one fitted joint equation.",
    )
