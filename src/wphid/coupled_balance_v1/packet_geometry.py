"""Independent reconstruction of a declared Hermite shell's geometric measures.

Uses supplied geometry only. No simulation or chemistry/material truth classes
are imported. The caller must verify reconstructed coordinates against its input.
"""

import numpy as np

from .operators import ShellMesh


def declared_shell_mesh(geometry, meridional_cells, thickness_cells):
    keys = ("length", "exit_radius", "expansion_ratio", "wall_thickness",
            "throat_angle_deg", "exit_angle_deg")
    values = {key: float(geometry[key]) for key in keys}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("declared geometry must be finite")
    if min(values[key] for key in keys[:4]) <= 0 or min(meridional_cells, thickness_cells) < 1:
        raise ValueError("geometry lengths and cell counts must be positive")
    length, radius, thickness = values["length"], values["exit_radius"], values["wall_thickness"]
    throat = radius/np.sqrt(values["expansion_ratio"])
    inlet_slope = np.tan(np.deg2rad(values["throat_angle_deg"]))
    outlet_slope = np.tan(np.deg2rad(values["exit_angle_deg"]))

    def curve(z):
        q = z/length
        r = ((2*q**3-3*q**2+1)*throat+(q**3-2*q**2+q)*length*inlet_slope
             +(-2*q**3+3*q**2)*radius+(q**3-q**2)*length*outlet_slope)
        slope = ((6*q**2-6*q)*throat/length+(3*q**2-4*q+1)*inlet_slope
                 +(-6*q**2+6*q)*radius/length+(3*q**2-2*q)*outlet_slope)
        return r, slope

    z_faces = np.linspace(0., length, meridional_cells+1)
    r_faces, slope_faces = curve(z_faces)
    ds = np.sqrt(np.diff(z_faces)**2+np.diff(r_faces)**2)
    s_faces = np.r_[0., np.cumsum(ds)]
    z_inner = .5*(z_faces[:-1]+z_faces[1:])
    r_inner, slope_inner = curve(z_inner)
    normal_inner = 1/np.sqrt(1+slope_inner**2)
    normal_faces = 1/np.sqrt(1+slope_faces**2)
    x_faces = np.linspace(0., thickness, thickness_cells+1)
    x = .5*(x_faces[:-1]+x_faces[1:])
    dx = thickness/thickness_cells
    z = z_inner[:, None]-slope_inner[:, None]*normal_inner[:, None]*x
    r = r_inner[:, None]+normal_inner[:, None]*x
    radius_s = r_faces[:, None]+normal_faces[:, None]*x
    radius_x = r_inner[:, None]+normal_inner[:, None]*x_faces
    mesh = ShellMesh(
        r*ds[:, None]*dx, radius_s[1:-1]*dx, radius_x[:, 1:-1]*ds[:, None],
        np.broadcast_to(.5*(ds[1:]+ds[:-1])[:, None], (meridional_cells-1, thickness_cells)),
        np.full((meridional_cells, thickness_cells-1), dx),
        {"inner": radius_x[:, 0]*ds, "outer": radius_x[:, -1]*ds,
         "inlet": radius_s[0]*dx, "outlet": radius_s[-1]*dx},
    )
    return mesh, {"s": .5*(s_faces[:-1]+s_faces[1:]), "x": x, "z": z, "r": r}
