import csv

import numpy as np
import pytest

from wphid.truth.reusable_nozzle import NozzleGeometry
from wphid.validation.external_fea import (
    OrthotropicMaterial,
    audit_calculix_deck,
    build_axisymmetric_strip_mesh,
    build_nozzle_axisymmetric_mesh,
    convert_calculix_frd_to_normalized_csv,
    interpolate_cell_field_to_nodes,
    interpolate_cell_profile_to_inner_faces,
    load_external_results,
    render_calculix_static_deck,
)


def _material():
    return OrthotropicMaterial.isotropic_equivalent(
        name="CMC_EQUIV",
        young_modulus=60.0e9,
        poisson_ratio=0.032,
        thermal_expansion_rows=((293.15, 0.5e-6), (1873.15, 2.5e-6)),
        reference_temperature=330.0,
        density=1900.0,
    )


def test_nozzle_mesh_is_positive_and_inner_wall_is_face_four():
    mesh = build_nozzle_axisymmetric_mesh(
        NozzleGeometry(), meridional_elements=6, thickness_elements=3
    )

    assert mesh.node_count == 28
    assert mesh.element_count == 18
    assert np.all(mesh.signed_areas > 0.0)
    assert mesh.connectivity[0].tolist() == [1, 2, 6, 5]
    assert mesh.inner_face_elements.tolist() == [1, 4, 7, 10, 13, 16]
    for element_id in mesh.inner_face_elements:
        node_four, node_one = mesh.connectivity[element_id - 1, [3, 0]]
        assert node_four in mesh.inner_nodes
        assert node_one in mesh.inner_nodes


def test_structured_interpolation_is_constant_preserving_and_resolution_independent():
    mesh = build_nozzle_axisymmetric_mesh(
        meridional_elements=11, thickness_elements=5
    )
    mapped = interpolate_cell_field_to_nodes(np.full((31, 8), 735.0), mesh)
    pressure = interpolate_cell_profile_to_inner_faces(
        np.full(31, 2.1e6), mesh
    )

    assert mapped.shape == (mesh.node_count,)
    assert pressure.shape == (mesh.meridional_elements,)
    assert np.allclose(mapped, 735.0, rtol=0.0, atol=1.0e-12)
    assert np.allclose(pressure, 2.1e6, rtol=0.0, atol=1.0e-9)


def test_calculix_deck_has_complete_temperature_and_inner_pressure_contract():
    mesh = build_nozzle_axisymmetric_mesh(
        meridional_elements=8, thickness_elements=2
    )
    temperature = np.linspace(330.0, 1500.0, mesh.node_count)
    pressure = np.linspace(3.0e6, 1.0e5, mesh.meridional_elements)
    deck = render_calculix_static_deck(
        mesh,
        nodal_temperature=temperature,
        inner_pressure=pressure,
        material=_material(),
        title="Independent nozzle snapshot",
    )
    audit = audit_calculix_deck(deck, mesh)

    assert audit["passed"] is True
    assert audit["temperature_count"] == mesh.node_count
    assert audit["pressure_face_count"] == mesh.meridional_elements
    assert "*ELASTIC,TYPE=ENGINEERING CONSTANTS" in deck
    assert "*EXPANSION,TYPE=ORTHO,ZERO=" in deck
    assert all(
        ",P4," in line
        for line in deck.split("*DLOAD,OP=NEW\n", maxsplit=1)[1].split("*NODE FILE", maxsplit=1)[0].splitlines()
        if line
    )


def test_calculix_deck_can_lump_a_distributed_axial_interface_foundation():
    mesh = build_nozzle_axisymmetric_mesh(
        meridional_elements=8, thickness_elements=3
    )
    stiffness_per_area = 1.2e12
    deck = render_calculix_static_deck(
        mesh,
        nodal_temperature=np.full(mesh.node_count, 700.0),
        inner_pressure=np.full(mesh.meridional_elements, 1.0e6),
        material=_material(),
        title="Compliant nozzle attachment",
        inlet_boundary="axial_foundation",
        inlet_axial_foundation_stiffness_per_area=stiffness_per_area,
    )
    audit = audit_calculix_deck(deck, mesh)

    assert audit["passed"] is True
    assert audit["support_representation"] == "distributed_spring"
    assert deck.count("*ELEMENT,TYPE=SPRING1") == mesh.inlet_nodes.size
    assert deck.count("*SPRING,ELSET=IAXSPR") == mesh.inlet_nodes.size
    assert "*BOUNDARY" not in deck


def test_axial_foundation_rejects_missing_or_misapplied_stiffness():
    mesh = build_nozzle_axisymmetric_mesh(
        meridional_elements=3, thickness_elements=1
    )
    common = {
        "mesh": mesh,
        "nodal_temperature": np.full(mesh.node_count, 700.0),
        "inner_pressure": np.full(mesh.meridional_elements, 1.0e6),
        "material": _material(),
        "title": "Compliant attachment validation",
    }
    with pytest.raises(ValueError, match="requires a positive"):
        render_calculix_static_deck(**common, inlet_boundary="axial_foundation")
    with pytest.raises(ValueError, match="only valid"):
        render_calculix_static_deck(
            **common,
            inlet_boundary="axial_roller",
            inlet_axial_foundation_stiffness_per_area=1.0e12,
        )


def _write_normalized_results(tmp_path, mesh, *, add_forbidden_column=False):
    nodal_path = tmp_path / "nodal_results.csv"
    element_path = tmp_path / "element_results.csv"
    nodal_header = [
        "node_id",
        "r_m",
        "z_m",
        "u_radial_m",
        "u_axial_m",
        "temperature_k",
    ]
    if add_forbidden_column:
        nodal_header.append("mechanism_truth")
    with nodal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(nodal_header)
        for node_id, coordinate in zip(mesh.node_ids, mesh.coordinates, strict=True):
            row = [node_id, coordinate[0], coordinate[1], 0.0, 0.0, 330.0]
            if add_forbidden_column:
                row.append(1.0)
            writer.writerow(row)
    with element_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "element_id",
                "sigma_radial_pa",
                "sigma_axial_pa",
                "sigma_hoop_pa",
                "tau_radial_axial_pa",
                "von_mises_pa",
            ]
        )
        for element_id in mesh.element_ids:
            radial, axial, hoop, shear = -2.0e6, 1.0e6, 4.0e6, 0.5e6
            von_mises = np.sqrt(
                0.5
                * ((radial - axial) ** 2 + (axial - hoop) ** 2 + (hoop - radial) ** 2)
                + 3.0 * shear**2
            )
            writer.writerow([element_id, radial, axial, hoop, shear, von_mises])
    return nodal_path, element_path


def test_external_result_reader_aligns_ids_and_checks_tensor_consistency(tmp_path):
    mesh = build_axisymmetric_strip_mesh(
        inner_radius=0.05,
        wall_thickness=0.01,
        axial_length=0.02,
        axial_elements=2,
        thickness_elements=2,
    )
    nodal_path, element_path = _write_normalized_results(tmp_path, mesh)
    results = load_external_results(nodal_path, element_path, mesh)

    assert results.radial_displacement.shape == (mesh.node_count,)
    assert results.von_mises_stress.shape == (mesh.element_count,)
    assert np.all(results.temperature == 330.0)


def test_external_result_reader_rejects_mechanism_label_leakage(tmp_path):
    mesh = build_axisymmetric_strip_mesh(
        inner_radius=0.05,
        wall_thickness=0.01,
        axial_length=0.02,
        axial_elements=1,
        thickness_elements=1,
    )
    nodal_path, element_path = _write_normalized_results(
        tmp_path, mesh, add_forbidden_column=True
    )

    with pytest.raises(ValueError, match="information-leakage"):
        load_external_results(nodal_path, element_path, mesh)


def test_unstable_orthotropic_constants_are_rejected():
    with pytest.raises(ValueError, match="stable compliance"):
        OrthotropicMaterial(
            name="BAD",
            young_radial=1.0,
            young_axial=100.0,
            young_hoop=1.0,
            poisson_radial_axial=0.49,
            poisson_radial_hoop=0.49,
            poisson_axial_hoop=0.49,
            shear_radial_axial=1.0,
            shear_radial_hoop=1.0,
            shear_axial_hoop=1.0,
            thermal_expansion_rows=((300.0, 1.0e-6, 1.0e-6, 1.0e-6),),
            reference_temperature=300.0,
        )


def test_stable_anisotropic_poisson_ratio_above_isotropic_limit_is_allowed():
    material = OrthotropicMaterial(
        name="DLR_CC_SIC",
        young_radial=17.6e9,
        young_axial=160.0e9,
        young_hoop=18.0e9,
        poisson_radial_axial=0.2,
        poisson_radial_hoop=0.2,
        poisson_axial_hoop=0.7,
        shear_radial_axial=9.8e9,
        shear_radial_hoop=9.8e9,
        shear_axial_hoop=20.0e9,
        thermal_expansion_rows=((293.15, 1.0e-6, 1.0e-6, 1.0e-6),),
        reference_temperature=293.15,
    )

    assert material.poisson_axial_hoop == pytest.approx(0.7)


def test_calculix_frd_conversion_round_trips_through_normalized_contract(tmp_path):
    mesh = build_axisymmetric_strip_mesh(
        inner_radius=0.05,
        wall_thickness=0.01,
        axial_length=0.02,
        axial_elements=1,
        thickness_elements=1,
    )

    def result_line(node_id, values):
        return " -1" + f"{int(node_id):10d}" + "".join(
            f"{float(value):12.5E}" for value in values
        )

    lines = ["    1UVERSION           Version 2.23", " -4  DISP        4    1"]
    lines.extend(result_line(node, (1.0e-6, 2.0e-6, 0.0)) for node in mesh.node_ids)
    lines.extend([" -3", " -4  NDTEMP      1    1"])
    lines.extend(result_line(node, (500.0,)) for node in mesh.node_ids)
    lines.extend([" -3", " -4  STRESS      6    1"])
    stress = (-2.0e6, 1.0e6, 4.0e6, 0.5e6, 0.0, 0.0)
    lines.extend(result_line(node, stress) for node in mesh.node_ids)
    lines.extend([" -3", " 9999"])
    frd = tmp_path / "result.frd"
    frd.write_text("\n".join(lines) + "\n", encoding="ascii")
    nodal = tmp_path / "nodal_results.csv"
    element = tmp_path / "element_results.csv"

    metadata = convert_calculix_frd_to_normalized_csv(frd, mesh, nodal, element)
    results = load_external_results(nodal, element, mesh)

    assert metadata["solver_version"] == "Version 2.23"
    assert np.allclose(results.radial_displacement, 1.0e-6)
    assert np.allclose(results.axial_displacement, 2.0e-6)
    assert np.allclose(results.temperature, 500.0)
    assert np.allclose(results.radial_stress, -2.0e6)


def test_frd_conversion_can_extract_a_registered_submesh_from_a_larger_model(tmp_path):
    mesh = build_axisymmetric_strip_mesh(
        inner_radius=0.05,
        wall_thickness=0.01,
        axial_length=0.02,
        axial_elements=1,
        thickness_elements=1,
    )

    def result_line(node_id, values):
        return " -1" + f"{int(node_id):10d}" + "".join(
            f"{float(value):12.5E}" for value in values
        )

    extra_node = 9001
    node_ids = [*mesh.node_ids.tolist(), extra_node]
    lines = ["    1UVERSION           Version 2.23", " -4  DISP        4    1"]
    lines.extend(result_line(node, (1.0e-6, 2.0e-6, 0.0)) for node in node_ids)
    lines.extend([" -3", " -4  NDTEMP      1    1"])
    lines.extend(result_line(node, (500.0,)) for node in node_ids)
    lines.extend([" -3", " -4  STRESS      6    1"])
    stress = (-2.0e6, 1.0e6, 4.0e6, 0.5e6, 0.0, 0.0)
    lines.extend(result_line(node, stress) for node in node_ids)
    lines.extend([" -3", " 9999"])
    frd = tmp_path / "larger_model.frd"
    frd.write_text("\n".join(lines) + "\n", encoding="ascii")
    nodal = tmp_path / "nodal_results.csv"
    element = tmp_path / "element_results.csv"

    with pytest.raises(ValueError, match="node IDs do not match"):
        convert_calculix_frd_to_normalized_csv(frd, mesh, nodal, element)

    metadata = convert_calculix_frd_to_normalized_csv(
        frd,
        mesh,
        nodal,
        element,
        allow_additional_nodes=True,
    )
    results = load_external_results(nodal, element, mesh)

    assert metadata["frd_node_count"] == mesh.node_count + 1
    assert metadata["normalized_node_count"] == mesh.node_count
    assert metadata["additional_frd_nodes_ignored"] == 1
    assert results.node_ids.tolist() == mesh.node_ids.tolist()
