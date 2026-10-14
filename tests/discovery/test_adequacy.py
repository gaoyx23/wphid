import numpy as np
import pytest

from wphid.discovery.adequacy import (
    absolute_adequacy_decision,
    maximum_reference_bound,
    one_sided_prediction_upper_bound,
)


def test_calibrated_absolute_adequacy_contract():
    reference = np.array([0.48, 0.50, 0.49, 0.51, 0.495])
    prediction = one_sided_prediction_upper_bound(reference, confidence=0.95)
    assert prediction["upper_bound"] > np.mean(reference)
    assert prediction["reference_count"] == 5
    assert absolute_adequacy_decision(0.49, prediction["upper_bound"])[
        "adequacy_pass"
    ]
    assert not absolute_adequacy_decision(0.80, prediction["upper_bound"])[
        "adequacy_pass"
    ]


def test_maximum_reference_bound_and_invalid_inputs():
    assert maximum_reference_bound([1.0, 2.0], relative_margin=0.1) == pytest.approx(2.2)
    with pytest.raises(ValueError):
        one_sided_prediction_upper_bound([1.0])
    with pytest.raises(ValueError):
        maximum_reference_bound([], relative_margin=0.1)

