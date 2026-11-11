"""Raster-mask orthogonalization by fixed-axis least-squares adjustment.

See docs/orthogonalization.md for the equations, references and safeguards.
"""

import logging
import math
from numbers import Integral
from typing import Any

logger = logging.getLogger(__name__)


def _equal_coordinates(values, edges):
    """Minimize squared displacement with equality constraints along edges."""
    parents = list(range(len(values)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, equal in enumerate(edges):
        if equal:
            parents[find(i)] = find((i + 1) % len(values))
    groups = {}
    for i, value in enumerate(values):
        groups.setdefault(find(i), []).append(value)
    means = {i: math.fsum(group) / len(group) for i, group in groups.items()}
    return [means[find(i)] for i in range(len(values))]


def _fit_ring(ring, origin, cosine, sine):
    points = []
    for x, y in list(ring.coords)[:-1]:
        dx, dy = x - origin[0], y - origin[1]
        points.append((dx * cosine + dy * sine, -dx * sine + dy * cosine))
    if len(points) < 4:
        return None
    horizontal = [
        abs(points[(i + 1) % len(points)][0] - x)
        >= abs(points[(i + 1) % len(points)][1] - y)
        for i, (x, y) in enumerate(points)
    ]
    xs = _equal_coordinates([p[0] for p in points], [not h for h in horizontal])
    ys = _equal_coordinates([p[1] for p in points], horizontal)
    fitted = []
    for x, y in zip(xs, ys):
        point = (origin[0] + x * cosine - y * sine, origin[1] + x * sine + y * cosine)
        if not fitted or point != fitted[-1]:
            fitted.append(point)
    if len(fitted) > 1 and fitted[0] == fitted[-1]:
        fitted.pop()
    return fitted if len(fitted) >= 4 else None


def _fit_polygon(
    polygon, epsilon, min_area, min_segments, area_tolerance, detect_triangles
):
    from shapely.geometry import Polygon

    if polygon.is_empty or not polygon.is_valid or polygon.area < min_area:
        return polygon
    simplified = polygon.simplify(epsilon, preserve_topology=True)
    if detect_triangles and len(simplified.exterior.coords) == 4:
        return polygon
    if len(simplified.exterior.coords) - 1 < min_segments or len(
        simplified.interiors
    ) != len(polygon.interiors):
        simplified = polygon
    if len(simplified.exterior.coords) - 1 < min_segments:
        return polygon
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        return polygon
    a, b = list(rectangle.exterior.coords)[:2]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    cosine, sine = dx / length, dy / length
    origin = polygon.exterior.coords[0]
    rings = [
        _fit_ring(ring, origin, cosine, sine)
        for ring in [simplified.exterior, *simplified.interiors]
    ]
    if any(ring is None for ring in rings):
        return polygon
    candidate = Polygon(rings[0], rings[1:])
    if (
        candidate.is_empty
        or not candidate.is_valid
        or len(candidate.interiors) != len(polygon.interiors)
    ):
        return polygon
    ratio = candidate.area / polygon.area
    if not area_tolerance <= ratio <= 1 / area_tolerance:
        return polygon
    return candidate


def _avoid_overlaps(originals, candidates):
    """Reject a fit that overlaps another original or proposed feature.

    Checking both sets also keeps simultaneous rejections safe. This is
    conservative: a fit can be rejected even if its neighbor is later rejected.
    """
    import geopandas as gpd

    count = len(originals)
    if count < 2:
        return candidates
    geometries = originals + candidates
    index = gpd.GeoSeries(geometries).sindex
    result = list(candidates)
    for i, candidate in enumerate(candidates):
        if not candidate.is_valid or candidate.equals(originals[i]):
            continue
        for j in index.query(candidate):
            if j % count == i:
                continue
            if (
                not geometries[j].is_valid
                or candidate.intersection(geometries[j]).area > 0
            ):
                result[i] = originals[i]
                break
    return result


def orthogonalize(
    input_path,
    output_path=None,
    epsilon=0.2,
    min_area=10,
    min_segments=4,
    area_tolerance=0.7,
    detect_triangles=True,
) -> Any:
    """Fit positive raster-mask regions to perpendicular axes.

    Each region uses the axes of its minimum-area enclosing rectangle. In that
    frame, edges are assigned to their nearest axis and vertices are adjusted
    by constrained least squares. Holes use the same axes. Invalid, collapsed,
    excessively changed or overlapping fits retain the original geometry.
    This is a fixed-axis fitting method, not a bit-for-bit reproduction of
    the former implementation; see ``docs/orthogonalization.md``.

    Args:
        input_path: GeoTIFF mask path. Positive values identify regions;
            zero, negative values and NoData are excluded. Uses band 1.
        output_path: Optional vector output path, passed to GeoDataFrame.to_file.
        epsilon: Nonnegative topology-preserving simplification tolerance,
            in raster CRS units. Default 0.2.
        min_area: Nonnegative minimum region area to adjust, in squared CRS
            units. Smaller regions remain in the output. Default 10.
        min_segments: Minimum exterior segment count to retain. Must be an
            integer of at least 4. Default 4.
        area_tolerance: Lower bound on output/input area ratio; the reciprocal
            is the upper bound. Must be in (0, 1]. Default 0.7.
        detect_triangles: Preserve regions whose simplified exterior is a
            triangle. Default True. Degenerate fits are always rejected.

    Returns:
        geopandas.GeoDataFrame: Geometry, original positive ``value`` labels,
        and the input CRS, including a typed empty geometry column if no
        foreground regions exist.
    """
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    for name, value in (
        ("epsilon", epsilon),
        ("min_area", min_area),
        ("area_tolerance", area_tolerance),
    ):
        try:
            finite = math.isfinite(value)
        except TypeError:
            finite = False
        if not finite or isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number")
    if epsilon < 0 or min_area < 0:
        raise ValueError("epsilon and min_area must be nonnegative")
    if not 0 < area_tolerance <= 1:
        raise ValueError("area_tolerance must be in (0, 1]")
    if (
        not isinstance(min_segments, Integral)
        or isinstance(min_segments, bool)
        or min_segments < 4
    ):
        raise ValueError("min_segments must be an integer of at least 4")
    if not isinstance(detect_triangles, bool):
        raise ValueError("detect_triangles must be a boolean")

    with rasterio.open(input_path) as source:
        band = source.read(1, masked=True)
        foreground = (
            (band.data > 0) & np.isfinite(band.data) & ~np.ma.getmaskarray(band)
        )
        regions = shapes(band.data, mask=foreground, transform=source.transform)
        originals, values = [], []
        for geometry, value in regions:
            originals.append(shape(geometry))
            values.append(int(value) if value.is_integer() else value)
        crs = source.crs
    candidates = [
        _fit_polygon(
            p, epsilon, min_area, min_segments, area_tolerance, detect_triangles
        )
        for p in originals
    ]
    adjusted = _avoid_overlaps(originals, candidates)
    unchanged = sum(p.equals(q) for p, q in zip(originals, adjusted))
    logger.info(
        "Orthogonalization: %d regions adjusted; %d retained",
        len(originals) - unchanged,
        unchanged,
    )
    result = gpd.GeoDataFrame({"value": values}, geometry=adjusted, crs=crs)
    if output_path:
        result.to_file(output_path)
    return result
