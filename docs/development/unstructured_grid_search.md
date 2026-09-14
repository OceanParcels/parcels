# Unstructured Grid Search: Spatial Hashing with Morton Encoding

This page documents the algorithm used in Parcels to locate which grid cell a particle occupies on both curvilinear (`XGrid`) and unstructured (`UxGrid`) grids. The implementation lives in two files:

- `src/parcels/_core/spatialhash.py` — `SpatialHash` class and Morton encoding utilities
- `src/parcels/_core/index_search.py` — point-in-cell tests and the high-level search dispatch

For debugging information regarding the spatial hash grid that underlies a curvilinear or unstructured grid, call `SpatialHash.describe()`, which prints a summary of the hash table's statistics. This table includes the total number of occupied hash cells, the percentage of hash cells that are occupied, the minimum, mean, and maximum number of mesh faces in a hash cell, and more.

---

## Motivation

On a rectilinear grid, finding which cell contains a particle is a trivial O(1) operation. We can compute an index from the coordinate directly. On curvilinear and unstructured grids, no such shortcut exists; each cell has an arbitrary shape and position.

The naive approach is an O(N) linear scan: compute the distance from the particle to each face centroid and find the minimum. For large meshes this becomes a bottleneck, and distance minimisation does not guarantee the nearest centroid is actually the containing cell.

Parcels uses **spatial hashing** to reduce this to an O(1) average-case lookup. The key idea is to impose a cheap, regular "hash grid" on top of the irregular source grid, assign each source-grid face to the hash cells its bounding box overlaps, and then use the hash grid to narrow candidate faces for any query point down to a small set before running an exact point-in-cell test. The key data structure that relates the cells in the hash grid to the source grid is the **hash table**. In Parcels, the hash table is a compressed sparse row matrix whose row indices are morton encodings of the positions in the grid and the non-zero values in each row the of matrix are the face id's in a grid that likely contain the query point. Once a short list of candidate faces is obtained, Parcels then uses point-in-cell checks to locate which face a particle resides in.

The search methods that exist in the literature — KD-trees, BVH trees, quad-trees — give O(log N) query time. Spatial hashing achieves O(1) on average, which is particularly important when the same grid is queried millions of times per time step (one query per particle, per step).

---

## Algorithm Overview

The algorithm has two phases: **initialisation** (once, when the grid is first used) and **query** (once per particle per time step).

### Initialisation

1. Determine the bounding box of every face in the source grid.
2. Choose the hash-grid resolution (the quantization `bitwidth`) so that the total table size stays within a fixed memory budget (see **Hash-grid Resolution and the Entry Budget** below).
3. Map each face's bounding box to a set of integer hash cell coordinates using **Morton encoding** (see below).
4. Sort all (face, morton-code) pairs by morton code and store them in a compact CSR-like structure: arrays of unique codes `keys`, their starting positions `starts`, hit counts `counts`, and a single flat array `faces` of the (raveled) face indices.

### Query

1. Convert each particle position to a Morton code using the same encoding as step 2 above.
2. Binary-search `keys` for the particle's code (`np.searchsorted`).
3. For each candidate face returned by the hash table, run an **exact point-in-cell test**.
4. Return the first face that passes, along with its local (computational) coordinates.

---

## Coordinate Systems

The hash grid is three-dimensional regardless of the source grid:

| Source grid | Mesh type | Hash-grid space                          |
| ----------- | --------- | ---------------------------------------- |
| `XGrid`     | spherical | Cartesian bounding box (lon/lat → x,y,z) |
| `XGrid`     | flat      | 2-D lon/lat bounding box (z set to 0)    |
| `UxGrid`    | spherical | Cartesian bounding box (lon/lat → x,y,z) |
| `UxGrid`    | flat      | 2-D lon/lat bounding box                 |

For spherical grids, node coordinates are converted from degrees to radians and then to Cartesian (x, y, z) on the unit sphere using `parcels._core.index_search._latlon_rad_to_xyz`:

```
x = cos(lon) * cos(lat)
y = sin(lon) * cos(lat)
z = sin(lat)
```

The hash grid then spans the axis-aligned Cartesian bounding box of the face bounds. For a global grid this is (nearly) the unit cube `[-1, 1]³`; for a regional grid it is much tighter, so the full quantization resolution (1024 bins per axis) is spent on the region actually covered by the grid rather than the whole sphere. Working in Cartesian space avoids the longitude wrap-around discontinuity that would otherwise cause the bounding boxes of cells crossing the antimeridian to erroneously span the entire domain.

For flat meshes the hash grid simply spans `[lon_min, lon_max] × [lat_min, lat_max]`, with z fixed at 0.

---

## Morton Encoding

Morton encoding is the core of the hashing function. It converts a 3-D floating-point coordinate into a single unsigned 32-bit integer in a way that preserves spatial locality — nearby points in 3-D space get similar Morton codes. This makes it an ideal hash function for spatial search.

The process has three steps, all implemented in `spatialhash.py`.

### Step 1: Quantization

**Function:** `parcels._core.spatialhash.quantize_coordinates`

Each coordinate axis is mapped from its floating-point range `[min, max]` to an integer in `[0, bitwidth]` by:

```
xq = clip( floor( (x - xmin) / (xmax - xmin) * bitwidth ), 0, bitwidth )
```

With `bitwidth = 1023` (the maximum), each axis gets 1024 distinct integer levels. This is equivalent to overlaying a regular 1024×1024×1024 grid — the "hash grid" — on the domain. Each integer triple `(xq, yq, zq)` identifies one cell of this hash grid. The `bitwidth` actually used may be lower: it is capped at build time to keep the table within a memory budget.

**Choosing bitwidth:** The ceiling of 1023 was chosen because it fits in 10 bits per axis, and 3 × 10 = 30 bits, which fits exactly in a `uint32`. Increasing bitwidth beyond 1023 would require a `uint64` Morton code. In practice, 1023 provides more than enough resolution — for a global ocean model with ~1 million faces, the hash grid cells are already smaller than individual grid cells — so the build starts at 1023 and only coarsens if the entry budget requires it.

**Effect of domain extent:** If the quantization bounds `[xmin, xmax]` exactly match the source grid's coordinate range, then at `bitwidth = 1023` most hash cells contain at most one or two source faces. Expanding the bounds beyond the data range is equivalent to coarsening the hash grid, increasing the average number of candidates per query.

### Step 2: Bit Dilation

**Function:** `parcels._core.spatialhash._dilate_bits`

Before interleaving, each 10-bit integer is "dilated" so that its bits are spread out with two zero bits between each active bit:

```
Input:  b9 b8 b7 b6 b5 b4 b3 b2 b1 b0
Output: b9 0 0 b8 0 0 b7 0 0 ... b1 0 0 b0
```

This is done via a sequence of shift-and-mask operations operating on `uint32` values:

```python
n &= 0x000003FF          # keep only 10 bits
n = (n | (n << 16)) & 0xFF0000FF
n = (n | (n <<  8)) & 0x0300F00F
n = (n | (n <<  4)) & 0x030C30C3
n = (n | (n <<  2)) & 0x09249249
```

Each stage moves bits further apart. After five stages the 10 active bits are each separated by exactly two zeros, filling positions 0, 3, 6, 9, ..., 27 in the 30-bit result.

### Step 3: Bit Interleaving

**Function:** `parcels._core.spatialhash._encode_quantized_morton3d`

Once all three axes are dilated, the Morton code is formed by OR-ing the shifted dilated values:

```python
code = (dz << 2) | (dy << 1) | dx
```

The resulting bit layout (relative to the least significant bit) is:

```
x0, y0, z0, x1, y1, z1, ..., x9, y9, z9
```

The combined encode function `parcels._core.spatialhash._encode_morton3d` chains all three steps (quantize, dilate, interleave) and is the function used during queries.

---

## Hash-grid Resolution and the Entry Budget

The CSR table stores one entry per (face, overlapping-hash-cell) pair. A finer hash grid (higher `bitwidth`) means smaller hash cells and fewer candidate faces per query — but each face's bounding box then overlaps more hash cells, so the table holds more entries and costs more memory to build. To keep this bounded, the resolution is chosen at build time against a fixed entry budget rather than always using the maximum 10-bit resolution.

The relevant constants live at the top of `spatialhash.py`:

- `_HASH_MAX_BITWIDTH = 1023` — the maximum (and initial) quantization level, 10 bits per axis.
- `_HASH_ENTRIES_PER_FACE = 16` — target average number of hash-cell overlaps per face.
- `_HASH_ENTRY_BUDGET_MIN = 2**22` — a floor so that small grids are never forced onto an over-coarse hash grid.

The budget on the total number of (face, hash-cell) entries is:

```
budget = max(_HASH_ENTRIES_PER_FACE * nfaces, _HASH_ENTRY_BUDGET_MIN)
```

`SpatialHash._total_hash_entries(bitwidth)` computes what the table size _would_ be at a given resolution: it quantizes the low and high corner of every face bounding box and sums `(nx · ny · nz)` over all faces. If the table at `bitwidth = 1023` already fits the budget, that resolution is kept. Otherwise the constructor binary-searches for the largest bitwidth whose table fits the budget and uses that, coarsening the hash grid just enough to stay within memory.

---

## Hash Table Construction

**Method:** `parcels._core.spatialhash.SpatialHash._initialize_hash_table`
The hash table is built by iterating over every face in the source grid:

1. **Compute bounding boxes.** For each face, find the min and max reached along each axis anywhere on the face. For `XGrid`, the face is bounded by the four corner nodes at `[j,i]`, `[j,i+1]`, `[j+1,i+1]`, `[j+1,i]`. For `UxGrid`, it is bounded by the nodes listed in `face_node_connectivity`. On a flat mesh a face is a straight-sided polygon, so its extent is the min and max of the node coordinates. On a sphere it is not, and the bounds are computed with `_spherical_face_bounds` instead (see below).

2. **Quantize bounding box corners.** Both the lower-left and upper-right corners of each bounding box are quantized with `quantize_coordinates`. The difference `(xqhigh - xqlow + 1)` × `(yqhigh - yqlow + 1)` × `(zqhigh - zqlow + 1)` gives the number of hash cells the face overlaps.

3. **Generate one Morton code per (face, hash cell) pair.** For each face, every hash cell within its bounding box gets a Morton code. This is done vectorised: for each face `f` spanning a `nx × ny × nz` block of hash cells, `nx*ny*nz` Morton codes are generated by iterating over all `(xi, yi, zi)` offsets within the block. The result is a flat array `morton_codes` (one entry per face-hashcell overlap) and a parallel array `face_ids`.

4. **Sort by Morton code and compress.** Each `(morton_code, face_id)` pair is fused into a single `uint64` — the Morton code in the high 32 bits, the face id in the low 32 bits — and the fused array is sorted in place. Unsigned comparison then orders entries by Morton code, with ties broken by ascending face id, without allocating a separate `argsort` permutation or gather. The code and face id are split back out afterwards. Runs of equal codes are located by comparing adjacent entries (rather than `np.unique`), producing a CSR (Compressed Sparse Row) structure:
   - `keys` — sorted unique Morton codes
   - `starts` — index into `faces` where each unique code's entries begin
   - `counts` — number of face entries for each unique code
   - `faces` — the flat (raveled) face index for each entry, in sorted order

   Only the flat face id is stored (4 bytes per entry). The `(j, i)` pair is _not_ precomputed and stored; instead `query` unravels just the gathered candidate faces on demand (see below). This avoids holding two `int64` index arrays (16 bytes per entry) for the lifetime of the grid.

This CSR layout enables O(1) hash table lookup via binary search on `keys`.

### Exact face bounds on the sphere

**Function:** `parcels._core.spatialhash._spherical_face_bounds`

Whenever a mesh is "spherical", a cell's edges are arcs of great circles on the sphere that the mesh sits on. Therefore, the face itself is a curved patch that bulges away from the flat polygon through its nodes. The extremes of $x$, $y$ and $z$ reached anywhere on that patch must be taken as the bounds for the hash grid.

`_spherical_face_bounds` computes these minima and maxima for each face. On a convex spherical face, a coordinate's extreme can only sit in one of three places, so we take the most extreme candidate from each:

1. **At a node.** The `np.min`/`np.max` over the face's nodes.

2. **At one of the unit sphere's six axis points.** $(\pm1, 0, 0)$, $(0, \pm1, 0)$ and $(0, 0, \pm1)$ are where $x$, $y$ and $z$ reach their global extremes on the unit sphere. A face that contains one of them attains $\pm1$ exactly, so that bound is set directly rather than searched for. A point $p$ lies inside the spherical triangle spanned by nodes $v_0, v_1, v_2$ exactly when it is a non-negative combination of them, so we solve $Mw = p$ with the nodes as the columns of $M$ and ask whether every $w_i \ge 0$. That solve needs exactly three nodes, so for spherical curvilinear grids, a face is first split into two triangles about its first node; a point inside either triangle is considered to be in the face.

3. **Partway along an edge.** To compute minima and maxima along edges, we parameterise the edge as a cosine, and then analytically compute whether that cosine hits a minimum or maximum within the part of the arc the edge comprises. Let $\hat u$ be the unit vector perpendicular to node $a$ pointing towards node $b$. Every point of that edge is then $a\cos\theta + \hat u \sin\theta$ for $\theta \in [0, \delta]$, with $\delta$ the angular length of the arc. Each coordinate along the edge is therefore a sinusoid, $A\cos\theta + B\sin\theta = R\cos(\theta - \varphi)$ with amplitude $R = \sqrt{A^2 + B^2}$ and phase $\varphi = \mathrm{atan2}(B, A)$. Its maximum $+R$ falls at $\theta = \varphi$ and its minimum $-R$ at $\theta = \varphi + \pi$, as long as those min/max occur when $\theta$ is within $[0, \delta]$. If the min/max for $\theta$ do not fall within $[0, \delta]$, then the edge's min/max is at a node, which is already accounted for above.

---

## Querying the Hash Table

**Method:** `parcels._core.spatialhash.SpatialHash.query`

Given arrays of particle latitudes `y` and longitudes `x`:

1. **Convert to hash space.** For spherical grids, convert to Cartesian `(qx, qy, qz)`. For flat grids, use `(x, y, 0)` directly.

2. **Encode query Morton codes.** Call `_encode_morton3d` on the query coordinates.

3. **Binary search.** Use `np.searchsorted(keys, query_codes)` to find the position of each query code in the sorted unique-codes array. Queries whose code does not appear exactly in `keys` return no candidates (the hash cell the particle falls in has no registered faces).

4. **Gather candidates.** For each query with a valid hit, use `starts[pos]` and `counts[pos]` to gather the candidate face ids from the flat `faces` array. This is done fully vectorised using a CSR traversal with `np.repeat` and cumulative sums. The gathered flat face ids are then unravelled to `(j_all, i_all)` with `np.unravel_index` — only the gathered candidates are unravelled, not the whole table. At the end of this stage, there are potentially multiple faces to check with a point-in-cell test.

5. **Point-in-cell test.** Call `self._point_in_cell` (either `curvilinear_point_in_cell` or `uxgrid_point_in_cell`) on all candidates simultaneously. The provided point-in-cell tests for both unstructured and structured grids guarantee that at most one candidate cell is found containing each particle.

6. **Return.** Returns `(j_best, i_best, coords_best)`. For particles with no containing face found, `j_best` and `i_best` are `GRID_SEARCH_ERROR = -3`, and `coords_best` is `(-1, -1)`.

---

## Point-in-Cell Tests

### Curvilinear grids (`XGrid`)

**Function:** `parcels._core.index_search.curvilinear_point_in_cell`

Each quadrilateral cell is parameterised by bilinear mapping from the unit square `[0,1]²` (computational coordinates `xsi`, `eta`) to physical space. The test solves the resulting quadratic equation:

```
P = a₀ + a₁·xsi + a₂·eta + a₃·xsi·eta
```

The coefficients `a` and `b` are derived from the cell's four corner coordinates. The particle is inside the cell if `0 ≤ xsi ≤ 1` and `0 ≤ eta ≤ 1`.

This inverse assumes the cell has straight edges. That holds on a flat mesh, where `_bilinear_inverse_latlon` solves it directly in (lon, lat). On a sphere the edges are great-circle arcs, so `_bilinear_inverse_tangent_plane` first gnomonically projects (radially projects) the four corners and the particle onto a plane tangent to the sphere at the cell's centre, using `_spherical_project_cell_and_query`. The projection conserves which side of an edge a point lies on, while converting the curved edges to straight edges, allowing the bilinear inverse to measure the particle against the cell's true boundary.

### Unstructured grids (`UxGrid`)

**Function:** `parcels._core.index_search.uxgrid_point_in_cell`

For triangular UxGrid faces, the test uses **barycentric coordinates**, obtained differently on each mesh type.

On a **flat** mesh, barycentric coordinates are computed via **area ratios** in `parcels._core.index_search._barycentric_coordinates`:

```
λ₀ = area(P, v₁, v₂) / area(v₀, v₁, v₂)
λ₁ = area(P, v₂, v₀) / area(v₀, v₁, v₂)
λ₂ = area(P, v₀, v₁) / area(v₀, v₁, v₂)
```

The particle is inside the face if `λ₀ ≥ 0`, `λ₁ ≥ 0`, `λ₂ ≥ 0`, and `λ₀ + λ₁ + λ₂ ≈ 1`.

On a **spherical** mesh, the particle $p$ and the face's three vertices are converted to Cartesian coordinates and `np.linalg.solve` solves $Mw = p$, with the vertices as the columns of $M$. This is the same test used for the axis points in [Exact face bounds on the sphere](#exact-face-bounds-on-the-sphere). The particle is inside the face if every $w_i \ge 0$ and the weights sum to a positive number.

```{note}
The current implementation is limited to triangular faces (K=3). Generalising to
n-sided convex polygons would require Wachspress barycentric coordinates. This is
noted as a TODO in `_barycentric_coordinates`.
```

---

## Integration with the Index Search

**Function:** `parcels._core.index_search._search_indices_curvilinear_2d`

The `SpatialHash` sits inside the broader index search used every time fields are interpolated onto particle positions. The flow is:

1. If a previous cell guess (`yi`, `xi`) is available (i.e., from the previous time step), first try `curvilinear_point_in_cell` at those indices. Particles that haven't moved far enough to leave their cell are resolved immediately without touching the hash table.

2. For particles not resolved by the guess (or with no guess), call `grid.get_spatial_hash().query(y, x)`.

3. The `SpatialHash` object is lazily constructed on the first call and cached on the grid object.

This two-stage approach (cheap guess check, then hash lookup) means that once particles have been located once, subsequent steps are fast for the common case where particles remain in the same cell.

---

## Design Notes and Limitations

**Hash grid resolution vs. memory.** The CSR structure stores one entry per (face, overlapping-hash-cell) pair. For a mesh with highly variable face sizes, large faces overlap many hash cells and inflate the table. Rather than relying on the 10-bit quantisation alone to bound this, the build caps the total entry count against a fixed budget and lowers the `bitwidth` as needed (see **Hash-grid Resolution and the Entry Budget** above). The trade-off is moved from memory to query time: a coarser hash grid returns more candidate faces per query.

**Degenerate faces (XGrid).** On curvilinear spherical grids, a face whose corner nodes are undefined (e.g. land points with lon/lat masked to `0.0`) can span a huge portion of the domain and, on its own, dominate the table size. During construction the curvilinear spherical path runs `_find_degenerate_xgrid_faces`, which flags any cell whose longest edge — the maximum great-circle chord across its four edges and two diagonals — exceeds `threshold_factor` (default 10) times the 99th-percentile edge length, and emits a `FieldSetWarning` naming the first few offending `(j, i)` locations. This detection currently runs **only for the curvilinear (`XGrid`) spherical path**; there is no equivalent check on the `UxGrid` path.

**Periodic boundaries.** `SpatialHash` does not support queries on periodic elements — there is no longitudinal wrapping or remapping of query points. A particle that crosses a periodic seam is not folded back into the domain by the search. For spherical grids the antimeridian is handled only _implicitly_, by working in Cartesian space, not by treating the domain as periodic.

**Spherical geometry degeneracy.** Near the poles, lon/lat cells become highly elongated in lon/lat space. Working in Cartesian space mitigates this: all cells have similar extents in the unit cube, so the hash grid resolution is more uniform.

**Flat-grid z-coordinate.** For flat meshes, the z-coordinate is fixed at 0. The Morton code is still 3-D, but zq is always 0, so the z-bits of every code are 0. This wastes one third of the available code space but causes no correctness issues.
