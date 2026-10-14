import experiments.s3c_e3_repeated_mission_cycles as experiment


def test_s3c_e3_tracks_every_cycle_without_reset():
    row = experiment.run_case("constant_duty", 34301)
    assert len(row["truth_cycle_end_damage_means"]) == experiment.PROTOCOL["cycle_count"]
    assert row["truth_damage_accumulates"]
    assert set(row["field_nrmse"]) == set(experiment.FIELDS)
