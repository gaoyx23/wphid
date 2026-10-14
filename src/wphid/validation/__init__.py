"""Independent validation and external-solver exchange utilities."""

from .external_fea import (
    AxisymmetricQuadMesh,
    ExternalFEAResults,
    OrthotropicMaterial,
    audit_calculix_deck,
    build_axisymmetric_strip_mesh,
    build_nozzle_axisymmetric_mesh,
    convert_calculix_frd_to_normalized_csv,
    external_result_schema,
    interpolate_cell_field_to_nodes,
    interpolate_cell_profile_to_inner_faces,
    load_external_results,
    render_calculix_static_deck,
)

__all__ = [
    "AxisymmetricQuadMesh",
    "ExternalFEAResults",
    "OrthotropicMaterial",
    "audit_calculix_deck",
    "build_axisymmetric_strip_mesh",
    "build_nozzle_axisymmetric_mesh",
    "convert_calculix_frd_to_normalized_csv",
    "external_result_schema",
    "interpolate_cell_field_to_nodes",
    "interpolate_cell_profile_to_inner_faces",
    "load_external_results",
    "render_calculix_static_deck",
]
