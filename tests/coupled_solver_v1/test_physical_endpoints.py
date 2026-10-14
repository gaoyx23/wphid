"""Artificial parameterization equivalence, not another scientific fit."""
import json
from reproducibility.coupled_solver_v1.generator import truth_point
from reproducibility.coupled_solver_v1.physical_endpoints import physical_coordinates,grade
from reproducibility.coupled_solver_v1.workflow import PROTOCOL


def test_coupled_and_embedded_free_coordinates_are_identical():
    p,a=truth_point("both_shared");r=p["reaction.relative_rate"]
    free={"carbon.oxygen_response":a["carbon_rate_scale"],"carbon.steam_response":r*a["carbon_rate_scale"],
        "pO2.oxygen_response":a["oxygen_response"],"pH2O.steam_response":a["steam_response"],
        "pProducts.oxygen_response":a["product_response"],"pProducts.steam_response":r*a["product_response"],
        "T.oxygen_response":a["heat_response"],"T.steam_response":r*a["heat_response"]}
    assert physical_coordinates("both_shared",p,a)==physical_coordinates("both_free",p,free)


def test_missing_path_does_not_pass_on_available_accurate_parameters():
    p,a=truth_point("both_shared");precision=json.loads(PROTOCOL.read_text(encoding="utf-8"))["precision"]
    result=grade("oxygen_only",{"fitted":True,"parameters":p,"coefficients":a},{"source":"both_shared","parameters":p,"coefficients":a},precision)
    assert not result["complete_physical_parameter_pass"]
    assert result["missing_coordinates"] or result["reaction_score"]>1


def test_independent_role_keeps_each_destination_parameter():
    p,a=truth_point("both_independent");precision=json.loads(PROTOCOL.read_text(encoding="utf-8"))["precision"]
    result=grade("both_independent",{"fitted":True,"parameters":p,"coefficients":a},{"source":"both_independent","parameters":p,"coefficients":a},precision)
    assert result["complete_physical_parameter_pass"] and result["reaction_score"]==0


def test_wrong_heat_graph_missing_real_steam_heat_is_not_precision():
    p,a=truth_point("both_shared");precision=json.loads(PROTOCOL.read_text(encoding="utf-8"))["precision"]
    result=grade("both_oxygen_heat",{"fitted":True,"parameters":p,"coefficients":a},{"source":"both_shared","parameters":p,"coefficients":a},precision)
    assert not result["complete_physical_parameter_pass"]
