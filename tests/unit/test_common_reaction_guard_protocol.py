import numpy as np
import pytest
from experiments.audit_common_reaction_mismatch_guard import reaction_forcing,VARIANTS
from wphid.truth.reusable_nozzle import NozzleChemistry


@pytest.mark.parametrize("variant",VARIANTS)
def test_reaction_omission_preserves_cross_species_stoichiometry(variant):
    ro=np.array([1.,2.]);rw=np.array([3.,4.]);chem=NozzleChemistry()
    forcing=reaction_forcing(ro,rw,chem,variant)
    oxygen=forcing["pO2"]/-chem.oxygen_pressure_consumption
    steam=forcing["pH2O"]/-chem.steam_pressure_consumption
    np.testing.assert_allclose(forcing["pProducts"],chem.product_pressure_yield*(oxygen+steam))
    np.testing.assert_array_equal(oxygen,np.zeros_like(ro) if variant=="omit_oxygen_reaction" else ro)
    np.testing.assert_array_equal(steam,np.zeros_like(rw) if variant=="omit_steam_reaction" else rw)
    np.testing.assert_array_equal(ro,[1.,2.]);np.testing.assert_array_equal(rw,[3.,4.])


def test_unknown_equation_control_rejected():
    with pytest.raises(ValueError):reaction_forcing(np.ones(1),np.ones(1),NozzleChemistry(),"silent")


def test_saved_controls_change_common_source_not_structural_contrasts():
    import json
    from experiments.audit_common_reaction_mismatch_guard import OUTPUT,NAMES
    if not OUTPUT.exists():pytest.skip("completed saved controls unavailable")
    report=json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert report["complete"]
    for source in NAMES:
        baseline=report["cases"][f"{source}:declared_reactions"]
        assert baseline["joint_guard"]["candidate_set"]==[source]
        for variant in VARIANTS[1:]:
            control=report["cases"][f"{source}:{variant}"]
            shifts=[np.asarray(control["residuals"][n])-baseline["residuals"][n] for n in NAMES]
            for shift in shifts[1:]:np.testing.assert_allclose(shift,shifts[0],rtol=1.e-10,atol=1.e-14)
            assert control["paired_only"]["candidate_set"]==baseline["paired_only"]["candidate_set"]
    assert report["summary"]["omit_oxygen_reaction"]["guard_rejects_entire_library"]==4
    assert report["summary"]["omit_steam_reaction"]["guard_rejects_entire_library"]==2
