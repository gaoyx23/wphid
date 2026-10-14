"""Geometry conventions versus nuisance realizations: declarations, not proof."""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class GeometryConvention:
    reference_coordinate: str
    metric_mode: str
    volume_rule: str
    face_area_rule: str
    face_coefficient_rule: str = "arithmetic_cell_to_face"
    version: str = "geometry-convention-v1"

    def __post_init__(self):
        if any(not isinstance(v,str) or not v.strip() for v in asdict(self).values()):
            raise ValueError("explicit nonempty geometry conventions required")
        if self.metric_mode not in ("unit_reference_metric","declared_meridional_metric"):
            raise ValueError("unknown geometry metric mode")

    @property
    def convention_id(self):
        encoded=json.dumps(asdict(self),sort_keys=True,ensure_ascii=False).encode("utf-8")
        return "geometry-convention-v1:"+hashlib.sha256(encoded).hexdigest()


def geometry_realization_id(*, convention, geometry, s, x):
    """Hash a declared discrete geometry; does not certify its physical quality."""
    s,x=np.asarray(s,dtype=float),np.asarray(x,dtype=float)
    for coordinate in (s,x):
        if coordinate.ndim!=1 or len(coordinate)<3 or not np.all(np.isfinite(coordinate)) or np.any(np.diff(coordinate)<=0):
            raise ValueError("finite increasing geometry coordinates required")
    shapes=dict(volume=(len(s),len(x)),area_s=(len(s)-1,len(x)),
        area_x=(len(s),len(x)-1),distance_s=(len(s)-1,),distance_x=(len(x)-1,))
    arrays=dict(s=s,x=x)
    for name,shape in shapes.items():
        value=np.asarray(geometry.get(name),dtype=float)
        if value.shape!=shape or not np.all(np.isfinite(value)) or np.any(value<=0):
            raise ValueError("invalid explicit geometry: "+name)
        arrays[name]=value
    if not np.allclose(arrays["distance_s"],np.diff(s),rtol=1.e-10,atol=1.e-14) or not np.allclose(arrays["distance_x"],np.diff(x),rtol=1.e-10,atol=1.e-14):
        raise ValueError("geometry distances disagree with coordinates")
    if convention.metric_mode=="declared_meridional_metric" and "meridional_face_metric" not in geometry:
        raise ValueError("convention requires explicitly declared metric")
    metric=np.asarray(geometry.get("meridional_face_metric",np.ones(shapes["area_s"])),dtype=float)
    if metric.shape!=shapes["area_s"] or not np.all(np.isfinite(metric)) or np.any(metric<=0):
        raise ValueError("invalid geometry metric")
    if convention.metric_mode=="unit_reference_metric" and not np.all(metric==1.):
        raise ValueError("nonunit metric contradicts declared convention")
    arrays["meridional_face_metric"]=metric
    digest=hashlib.sha256(convention.convention_id.encode())
    for name,value in sorted(arrays.items()):
        digest.update(name.encode());digest.update(str(value.shape).encode())
        digest.update(value.astype("<f8").tobytes())
    return "geometry-realization-v1:"+digest.hexdigest()
