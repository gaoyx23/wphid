from experiments.paper_r98_engineering_observability_audit import (
    MECHANISMS,
    _missions,
    _parameter_sets,
)


def test_r98_passive_missions_share_thermal_loads():
    missions = _missions()
    reference = missions["nominal"]
    for mission in missions.values():
        assert mission.duration == reference.duration
        assert mission.chamber_pressure == reference.chamber_pressure
        assert mission.heat_flux_throat == reference.heat_flux_throat


def test_r98_each_perturbation_changes_only_its_declared_group():
    baseline = _parameter_sets()
    for mechanism in MECHANISMS:
        changed = _parameter_sets(mechanism)
        difference_groups = [
            index for index, (left, right) in enumerate(zip(baseline, changed))
            if left != right
        ]
        assert len(difference_groups) == 1
