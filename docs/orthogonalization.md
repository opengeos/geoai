# Raster-mask orthogonalization

`orthogonalize` uses an independently derived fixed-axis least-squares fit.
No implementation code was taken from another project's orthogonalization routine.
The former implementation has been removed in full. Fitted coordinates can differ.

The mathematical basis is constrained least-squares adjustment of building outlines,
as discussed by M. Sester (2000), [Generalization Based on Least Squares Adjustment,
ISPRS XXXIII/B4, pp.931–938](https://www.ikg.uni-hannover.de/fileadmin/ikg/Forschung/publications/Sester_ISPRS2000.pdf).
This implementation uses the general adjustment framework, not Sester's complete
facade-substitution and weighted-model algorithm. It does not copy that paper's code.

For each positive connected mask region:

1. Simplify with Shapely's topology-preserving simplification using `epsilon` in
   input CRS units. Retain the original outline if simplification loses a hole or
   violates `min_segments`. Optionally preserve simplified triangles.
2. Obtain fixed perpendicular axes from Shapely's
   [minimum rotated rectangle](https://shapely.readthedocs.io/en/stable/reference/shapely.minimum_rotated_rectangle.html).
   Translate near the outline before rotating to reduce coordinate cancellation.
3. Assign each edge to its nearer axis. For horizontal edges require `y[i]=y[i+1]`;
   for vertical edges require `x[i]=x[i+1]`. Minimize
   `sum((x[i]-x_original[i])**2 + (y[i]-y_original[i])**2)` under these equalities.
   Coordinates in each connected equality group share a value; differentiating
   this sum shows its unique minimizer is their arithmetic mean. This gives a
   linear-size union/find solution without a dense least-squares matrix.
4. Transform back. Accept the entire polygon, including all holes, only if it is
   valid, nonempty and its net area ratio is in `[area_tolerance,1/area_tolerance]`.
   Reject proposals overlapping another original or proposed mask region. All
   rejection paths retain the original feature and label.

The method adjusts independent building outlines. It does not enforce common walls,
maintain every boundary contact, merge buildings, infer semantic corners, optimize
the axes jointly with vertices, or guarantee a fit for every region. Overlap checks
are conservative; a fit near an invalid input outline is rejected without an
overlay against that outline. Invalid source outlines are retained, not repaired.
Triangles, tiny regions and unsuitable outlines can remain unchanged. Use a suitable projected CRS for meaningful linear/area thresholds.
Positive NoData values are excluded; holes, labels, CRS and empty masks are preserved.

The public arguments and `value`/geometry output columns remain. Parameter validation
now rejects nonfinite tolerances, `area_tolerance` values outside `(0, 1]` and
segment counts below 4.
OpenCV is no longer needed by this function. Review these behavior changes before
adopting the replacement; it is not a reproduction of the former algorithm.
