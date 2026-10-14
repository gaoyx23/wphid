import numpy as np
import pytest

from wphid.transport_mechanisms import TransportCandidate
from wphid.truth.reusable_nozzle import NozzleTransport
from wphid.truth.reusable_cmc import CMCMechanics
from wphid.weak.transport_candidate import project_transport_candidate


def inputs():
    t, s, x = np.linspace(0, 1, 7), np.linspace(0, 1, 5), np.linspace(0, .01, 5)
    z = np.broadcast_to(s[:, None], (5, 5))
    r = np.broadcast_to(.1 + x[None, :], (5, 5))
    shape = (7, 5, 5)
    fields = {"T": np.full(shape, 1000.), "stress": np.full(shape, 38.e6),
              "carbon": np.full(shape, .9), "crack": np.full(shape, .1),
              "pO2": np.broadcast_to(1.e5 * (s[None, :, None] + x[None, None, :]), shape)}
    return dict(transport=NozzleTransport(), mechanics=CMCMechanics(), species="pO2",
        observed_fields=fields, field_provenance={k:"measured" for k in fields},
        t=t, s=s, x=x, z=z, r=r, forcing=np.zeros(shape),
        window_shape=(5, 5, 5), stride_shape=(2, 1, 1))


def test_actual_weak_operator_accepts_shared_candidate_and_returns_finite_equations():
    kwargs = inputs()
    for law, alpha in [("Darcy", 0.), ("Forchheimer", 2.)]:
        candidate = TransportCandidate("combined_opening", law, 3., alpha)
        result = project_transport_candidate(candidate=candidate, **kwargs)
        assert result["candidate_id"] == candidate.candidate_id
        assert result["matrix"].shape[1] == 2
        assert np.all(np.isfinite(result["residual"]))
        assert result["known_parameters_not_estimated"]


def test_hidden_truth_provenance_is_rejected():
    kwargs = inputs(); kwargs["field_provenance"]["carbon"] = "truth"
    with pytest.raises(ValueError, match="hidden truth"):
        project_transport_candidate(candidate=TransportCandidate("constant", "Darcy"), **kwargs)


def test_sparse_surface_data_is_not_silently_reconstructed():
    kwargs = inputs(); kwargs["observed_fields"]["pO2"] = kwargs["observed_fields"]["pO2"][:, :, 0]
    with pytest.raises(ValueError, match="volumetric"):
        project_transport_candidate(candidate=TransportCandidate("constant", "Darcy"), **kwargs)
