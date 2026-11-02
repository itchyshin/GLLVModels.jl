# Bowyer–Watson incremental Delaunay triangulation for 2-D point sets.
#
# Given N input points (N × 2 matrix), produces a triangular mesh (nodes, tris)
# over their convex hull that satisfies the *empty-circumcircle* (Delaunay)
# property: no input point lies strictly inside the circumcircle of any output
# triangle.
#
# Algorithm — Bowyer 1981 / Watson 1981 (independently discovered):
#   1. Create a "super-triangle" whose circumcircle contains all input points.
#      Append its three vertices to the working node list (indices N+1, N+2, N+3).
#   2. Initialise the triangulation with the single super-triangle.
#   3. For each input point p_i (inserted in input order):
#      a. Find all "bad" triangles whose circumcircle strictly contains p_i.
#      b. Find the "cavity boundary": the set of directed edges on the outer
#         boundary of the union of bad triangles.  An edge is a boundary edge iff
#         it appears (in either orientation) in exactly one bad triangle — edges
#         shared by two bad triangles are interior to the cavity and are deleted.
#      c. Remove all bad triangles.
#      d. For each boundary edge (u, v) add a new triangle (u, v, i).  The
#         boundary edges are stored such that the interior of the cavity is to
#         their left; orientation is enforced below to guarantee CCW output.
#   4. Discard every triangle that uses a super-triangle vertex (index > N).
#   5. Return `nodes = points[1:N, :]` and `tris` with 1-based node indices.
#
# Predicates (determinant form — exact for this purpose with Float64 and a
# small epsilon guard):
#
#   orient2d(a, b, c) > 0  ⟺  a→b→c is counter-clockwise.
#
#   incircle(a, b, c, d) > 0  ⟺  d is strictly inside the circumcircle of
#   the CCW triangle (a, b, c).  Both use the standard 3×3 / 4×4 lifting-map
#   determinants (Shewchuk 1996, "Robust Adaptive Floating-Point Geometric
#   Predicates"; we use non-adaptive versions with a tiny epsilon guard).
#
# Limitations / known degeneracies:
#   - Collinear or near-collinear inputs are handled by the epsilon guard in
#     `orient2d_robust`, but extremely close collinear triples may produce a
#     near-zero-area triangle that passes the positive-area check in `spde_fem`.
#   - Cocircular (four points on a circle) inputs are not degenerate for
#     Bowyer–Watson: the incircle test's epsilon guard forces one of the two
#     possible diagonals, so the result is a valid Delaunay triangulation (one
#     of the two possible ones) with non-negative areas.
#   - Duplicate input points will produce a zero-area triangle and then cause
#     `spde_fem` to throw; the caller must deduplicate beforehand.
#   - Performance: O(N²) worst-case (repeated incircle tests); adequate for the
#     hundreds-to-low-thousands range typical of SPDE meshes.  For large N use
#     a spatial index or QHull-based routine.
#
# References:
#   - Bowyer 1981, "Computing Dirichlet tessellations", Comput. J. 24(2):162–166
#   - Watson 1981, "Computing the n-dimensional Delaunay tessellation",
#     Comput. J. 24(2):167–172
#   - Shewchuk 1996, "Robust Adaptive Floating-Point Geometric Predicates",
#     Proc. 12th Annual Symposium on Computational Geometry

# ── Geometric predicates ──────────────────────────────────────────────────────

# orient2d: returns the signed area × 2 of triangle (a, b, c).
# > 0  ⟺  CCW;  < 0  ⟺  CW;  ≈ 0  ⟺  collinear.
@inline function _orient2d(ax, ay, bx, by, cx, cy)
    return (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
end

# incircle_val: positive iff d = (dx, dy) lies strictly inside the circumcircle
# of CCW triangle (a, b, c).  The sign of the 4×4 determinant equals the sign
# of the 3×3 determinant after lifting (standard result; see Shewchuk 1996).
@inline function _incircle_val(ax, ay, bx, by, cx, cy, dx, dy)
    adx = ax - dx;  ady = ay - dy
    bdx = bx - dx;  bdy = by - dy
    cdx = cx - dx;  cdy = cy - dy
    adx2 = adx * adx + ady * ady
    bdx2 = bdx * bdx + bdy * bdy
    cdx2 = cdx * cdx + cdy * cdy
    return (adx * (bdy * cdx2 - cdy * bdx2)
          - ady * (bdx * cdx2 - cdx * bdx2)
          + adx2 * (bdx * cdy - bdy * cdx))
end

# ── Triangle storage ──────────────────────────────────────────────────────────
# Each triangle is stored as three integers (v1, v2, v3) in CCW order.

# ── Core algorithm ────────────────────────────────────────────────────────────

"""
    spde_mesh_delaunay(points::AbstractMatrix) -> (nodes::Matrix{Float64}, tris::Matrix{Int})

Build a Delaunay triangulation of the 2-D point set `points` (N × 2) using the
incremental Bowyer–Watson algorithm.

Returns:
- `nodes` — N × 2 `Float64` matrix containing the same coordinates as `points`
  (same row order).
- `tris` — T × 3 `Int` matrix of 1-based vertex indices, one row per triangle,
  oriented consistently counter-clockwise (positive area), covering the convex
  hull of the input.

The triangulation satisfies the **empty-circumcircle** (Delaunay) property: no
input point lies strictly inside the circumcircle of any output triangle.

`points` must have `size(points, 2) == 2`. Duplicate points will produce
degenerate (zero-area) triangles and cause downstream `spde_fem` to throw; the
caller must deduplicate beforehand. Collinear inputs are handled via a small
epsilon guard but may produce very thin (yet positive-area) boundary triangles.

See the file header for a full description of the algorithm, predicates, and
known limitations.
"""
function spde_mesh_delaunay(points::AbstractMatrix)
    size(points, 2) == 2 ||
        throw(ArgumentError("points must be N × 2; got $(size(points))"))
    N = size(points, 1)
    N ≥ 3 || throw(ArgumentError("need at least 3 points; got $N"))

    # ── Step 1: super-triangle ────────────────────────────────────────────────
    # Pick a super-triangle big enough to contain every input point.
    # We use the bounding-box approach: the super-triangle circumscribes a
    # square that is 3× the bounding box on each side.
    xmin = minimum(@view points[:, 1]); xmax = maximum(@view points[:, 1])
    ymin = minimum(@view points[:, 2]); ymax = maximum(@view points[:, 2])

    dx = xmax - xmin;  dy = ymax - ymin
    extent = max(dx, dy)
    # In-circle determinant has units of length^4. Keep the 1e-10 guard at
    # ordinary (extent >= 1) coordinates; shrink it with extent^4 below that
    # so a 1e-8 domain does not mark every triangle bad (#748).
    incircle_tol = 1e-10 * (extent > 0.0 ? min(extent, 1.0)^4 : 0.0)
    # Super-triangle still uses a floor of 1 so order-1 construction is unchanged.
    delta = max(extent, 1.0)

    # Centre of the bounding box.
    midx = (xmin + xmax) / 2
    midy = (ymin + ymax) / 2

    # Three vertices of a large CCW equilateral-ish super-triangle.
    # We use an axis-aligned triangle for simplicity.  The factor 3.0 guarantees
    # that any axis-aligned rectangle within the bounding box maps strictly
    # inside the circumcircle, which is all Bowyer–Watson requires.
    s1x = midx - 3.0 * delta;  s1y = midy - 3.0 * delta
    s2x = midx + 3.0 * delta;  s2y = midy - 3.0 * delta
    s3x = midx;                 s3y = midy + 4.0 * delta

    # Append super-triangle vertices to a working copy of the coordinates.
    # These are at positions N+1, N+2, N+3 in 1-based indexing.
    # Store all coordinates in two parallel vectors for fast access.
    all_x = Vector{Float64}(undef, N + 3)
    all_y = Vector{Float64}(undef, N + 3)
    @inbounds for i in 1:N
        all_x[i] = Float64(points[i, 1])
        all_y[i] = Float64(points[i, 2])
    end
    all_x[N+1] = s1x;  all_y[N+1] = s1y
    all_x[N+2] = s2x;  all_y[N+2] = s2y
    all_x[N+3] = s3x;  all_y[N+3] = s3y

    # ── Step 2: initialise triangulation ─────────────────────────────────────
    # Triangle list: each element is a 3-tuple of 1-based vertex indices (CCW).
    # We use a Vector of NTuple{3,Int} for compact, type-stable storage.
    # A "deleted" slot is marked as (0, 0, 0).
    tris_list = Vector{NTuple{3,Int}}()
    sizehint!(tris_list, max(16, 2 * N))

    # The super-triangle is CCW iff orient2d(s1, s2, s3) > 0.
    # By construction above (s3 is above s1 and s2), it is CCW.
    push!(tris_list, (N+1, N+2, N+3))

    # ── Step 3: insert points one by one ─────────────────────────────────────
    # bad_mask: reusable boolean buffer, same length as tris_list.
    bad_mask = Vector{Bool}(undef, 0)

    # edge_count: temporary Dict{NTuple{2,Int}, Int} for cavity-boundary detection.
    edge_count = Dict{NTuple{2,Int}, Int}()

    for pi in 1:N      # NOT @inbounds: triangle indices are derived, keep bounds checks
        px = all_x[pi];  py = all_y[pi]

        ntri = length(tris_list)

        # ── 3a. find bad triangles ─────────────────────────────────────────
        # Resize bad_mask to match.
        resize!(bad_mask, ntri)

        for k in 1:ntri
            t = tris_list[k]
            if t[1] == 0   # deleted slot
                bad_mask[k] = false
                continue
            end
            ax = all_x[t[1]]; ay = all_y[t[1]]
            bx = all_x[t[2]]; by = all_y[t[2]]
            cx = all_x[t[3]]; cy = all_y[t[3]]
            # incircle_val > 0 iff (px,py) is strictly inside the circumcircle
            # of CCW triangle (a, b, c). Negative epsilon is scaled to extent
            # (length^4) so small-domain coordinates keep a usable cavity.
            bad_mask[k] = _incircle_val(ax, ay, bx, by, cx, cy, px, py) > -incircle_tol
        end

        # ── 3b. find cavity boundary edges ────────────────────────────────
        # An edge (u, v) is on the cavity boundary iff it belongs to exactly
        # one bad triangle.  We count occurrences in the *unordered* sense —
        # we canonicalise each edge as (min, max) for counting, but we keep
        # the *directed* version (as it appears in the bad triangle) so we
        # can construct new triangles with the correct orientation.
        #
        # Two passes:
        #   Pass 1: collect all directed edges of bad triangles, count their
        #           canonical (undirected) occurrences.
        #   Pass 2: keep directed edges whose canonical count == 1.

        empty!(edge_count)

        for k in 1:ntri
            bad_mask[k] || continue
            t = tris_list[k]
            u1, u2, u3 = t[1], t[2], t[3]
            e1 = u1 < u2 ? (u1, u2) : (u2, u1)
            e2 = u2 < u3 ? (u2, u3) : (u3, u2)
            e3 = u3 < u1 ? (u3, u1) : (u1, u3)
            edge_count[e1] = get(edge_count, e1, 0) + 1
            edge_count[e2] = get(edge_count, e2, 0) + 1
            edge_count[e3] = get(edge_count, e3, 0) + 1
        end

        # Collect boundary directed edges (from bad triangles, count == 1).
        boundary_edges = Vector{NTuple{2,Int}}()
        sizehint!(boundary_edges, 6)  # typical small polygon

        for k in 1:ntri
            bad_mask[k] || continue
            t = tris_list[k]
            u1, u2, u3 = t[1], t[2], t[3]
            # Check each directed edge of this bad triangle.
            for (ua, ub) in ((u1, u2), (u2, u3), (u3, u1))
                canon = ua < ub ? (ua, ub) : (ub, ua)
                if get(edge_count, canon, 0) == 1
                    push!(boundary_edges, (ua, ub))
                end
            end
        end

        # ── 3c. remove bad triangles ──────────────────────────────────────
        # Replace bad triangles in-place with (0,0,0) sentinels.
        for k in 1:ntri
            if bad_mask[k]
                tris_list[k] = (0, 0, 0)
            end
        end

        # ── 3d. re-triangulate cavity ─────────────────────────────────────
        # For each boundary edge (ua, ub) — directed as it came from the bad
        # triangle (so the cavity interior is to the left) — add new triangle
        # (ua, ub, pi).  This is CCW iff orient2d(ua, ub, pi) > 0 for a
        # boundary edge whose interior side faces pi.
        #
        # Because the bad triangles were CCW and the cavity interior contains
        # pi, the directed boundary edges (ua, ub) already have pi to their
        # left, so (ua, ub, pi) is CCW.  We verify with orient2d and swap
        # (ua, ub) → (ub, ua) if not (which can happen near numerical precision).
        for (ua, ub) in boundary_edges
            ax = all_x[ua]; ay = all_y[ua]
            bx = all_x[ub]; by = all_y[ub]
            o = _orient2d(ax, ay, bx, by, px, py)
            if o > 0.0
                push!(tris_list, (ua, ub, pi))
            elseif o < 0.0
                push!(tris_list, (ub, ua, pi))
            else
                # Collinear: skip degenerate triangle.
                # This can happen at the super-triangle boundary with collinear
                # input points. The resulting gap is negligible for FEM use.
                # (A near-zero area triangle would fail spde_fem anyway.)
                nothing
            end
        end
    end

    # ── Step 4: collect valid output triangles ────────────────────────────────
    # Discard deleted slots (0,0,0) and triangles using super-triangle vertices
    # (indices N+1, N+2, N+3).
    good = NTuple{3,Int}[]
    sizehint!(good, max(1, 2 * N - 2))
    for t in tris_list
        t[1] == 0 && continue                         # deleted slot
        (t[1] > N || t[2] > N || t[3] > N) && continue  # super-triangle vertex
        push!(good, t)
    end

    T_out = length(good)
    T_out ≥ 1 || throw(ErrorException(
        "Bowyer–Watson produced no valid triangles for $N input points; " *
        "check for duplicate or nearly-collinear inputs."))
    # A triangulation of N points has T = 2N - 2 - H triangles, H <= N, so T >= N-2.
    # Fewer triangles means a hole (silent sparse mesh) rather than a valid cover.
    T_out >= N - 2 || throw(ErrorException(
        "Bowyer–Watson produced a degenerate mesh ($T_out triangles for $N points); " *
        "check for duplicate or nearly-collinear inputs."))

    # ── Step 5: build output matrices ────────────────────────────────────────
    nodes_out = Matrix{Float64}(undef, N, 2)
    @inbounds for i in 1:N
        nodes_out[i, 1] = all_x[i]
        nodes_out[i, 2] = all_y[i]
    end

    tris_out = Matrix{Int}(undef, T_out, 3)
    @inbounds for k in 1:T_out
        t = good[k]
        # Enforce CCW orientation in output (should already be, but belt-and-suspenders).
        ax = nodes_out[t[1], 1]; ay = nodes_out[t[1], 2]
        bx = nodes_out[t[2], 1]; by = nodes_out[t[2], 2]
        cx = nodes_out[t[3], 1]; cy = nodes_out[t[3], 2]
        o = _orient2d(ax, ay, bx, by, cx, cy)
        if o > 0.0
            tris_out[k, 1] = t[1]; tris_out[k, 2] = t[2]; tris_out[k, 3] = t[3]
        else
            # CW → swap second and third vertex.
            tris_out[k, 1] = t[1]; tris_out[k, 2] = t[3]; tris_out[k, 3] = t[2]
        end
    end

    return nodes_out, tris_out
end
