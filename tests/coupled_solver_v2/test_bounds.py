"""Only new endpoint regression checks, no repeated scientific fits."""
import numpy as np
import pytest
from wphid.coupled_solver_v1 import candidate_space,parameter_box
from wphid.coupled_solver_v2 import decode_unit_box

@pytest.mark.parametrize("candidate",candidate_space(),ids=lambda c:c.identity)
def test_all_inclusive_parameter_endpoints_are_representable(candidate):
    box=parameter_box(candidate)
    for fraction in (0.,1.):
        point=decode_unit_box(candidate,np.full(len(box),fraction))
        for k,(low,high) in box.items():
            assert point[k]==(low if fraction==0. else high)

def test_roundoff_example_is_corrected_but_true_unit_domain_failure_is_rejected():
    c=candidate_space()[3];box=parameter_box(c)
    assert .6+(1.8-.6)>1.8
    p=decode_unit_box(c,np.ones(len(box)))
    assert p["kinetic.oxygen.energy_1e5"]==1.8
    x=np.ones(len(box));x[0]+=1e-12
    with pytest.raises(ValueError):decode_unit_box(c,x)

def test_interior_parameterization_unchanged():
    for c in candidate_space():
        box=parameter_box(c);x=np.linspace(.1,.9,len(box));p=decode_unit_box(c,x)
        assert list(p)==list(box)
        for (k,(lo,hi)),v in zip(box.items(),x):assert p[k]==lo+(hi-lo)*v
