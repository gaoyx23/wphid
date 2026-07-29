import numpy as np
import pytest
from wphid.types import Trajectory


def test_trajectory_rejects_mismatched_time_axis():
    with pytest.raises(ValueError, match="time axis"):
        Trajectory(np.arange(3.0), {"Do": np.zeros(2)}, "c0")
