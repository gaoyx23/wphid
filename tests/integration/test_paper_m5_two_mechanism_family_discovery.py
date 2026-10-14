import numpy as np

from experiments import paper_m5_two_mechanism_family_discovery as experiment


def test_competing_family_atoms_are_finite_and_distinct():
    local, bases, targets = experiment.m1._assemble(36901, 1.0, 1.0)
    reaction = [experiment._builder(local, spec)(spec["center"])[0]
                for spec in experiment.REACTION_FAMILIES.values()]
    leakage = [experiment._builder(local, spec)(spec["center"])[0]
               for spec in experiment.LEAKAGE_FAMILIES.values()]
    assert all(np.all(np.isfinite(atom)) for atom in reaction + leakage)
    assert all(not np.allclose(reaction[0], atom) for atom in reaction[1:])
    assert all(not np.allclose(leakage[0], atom) for atom in leakage[1:])
    assert all(base.shape[0] == target.size for base, target in zip(bases, targets))
