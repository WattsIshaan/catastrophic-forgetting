"""Convex-hull Pareto-frontier utilities.

Two idioms appear across the old plotting code:

* `monotone_hull`: start at the min-x vertex, walk CCW, stop when x stops
  increasing. This traces the **lower hull** from min-x to max-x, i.e. the
  Pareto frontier when "down and to the left is better". Used by every
  small-scale Pareto plot.
* `open_hull(points, sort_key=...)`: walk between min-key and max-key vertices
  along the shorter side of the hull. Used by the 1B Pareto plots where
  endpoints are min-LR and max-LR runs on the convex hull.

Both take `points` as sequences of tuples whose first two elements are (x, y);
extra elements are available to `sort_key`.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.spatial import ConvexHull


def monotone_hull(points: Sequence[tuple]) -> Tuple[List[float], List[float]]:
    """Lower hull from min-x to max-x (x-monotone, CCW walk).

    Reproduces the Pareto-frontier traversal used across the small-scale
    figures: ConvexHull vertices in CCW order, rotate to start at the min-x
    vertex, then truncate as soon as x stops increasing.
    """
    pts = list(points)
    if len(pts) < 3:
        return [p[0] for p in pts], [p[1] for p in pts]

    xy = np.array([(p[0], p[1]) for p in pts])
    verts = ConvexHull(xy).vertices
    closed = np.append(verts, verts[0])
    path = xy[closed]
    start = int(np.argmin(path[:, 0]))
    path = np.roll(path, -start, axis=0)
    for i in range(len(path) - 1):
        if path[i, 0] > path[i + 1, 0]:
            path = path[: i + 1]
            break
    return path[:, 0].tolist(), path[:, 1].tolist()


def open_hull(
    points: Sequence[tuple],
    *,
    sort_key: Callable[[tuple], float],
) -> Tuple[List[float], List[float]]:
    """Hull polyline between min-key and max-key vertices, avoiding the direct edge.

    Matches ``pareto_1b._convex_hull_boundary`` (when lr_vals aligns with
    points): pick the two hull vertices with smallest/largest sort_key and walk
    the CCW cycle from min to max, going forward unless that is the single
    connecting edge (i.e. min and max are adjacent CCW) in which case reverse.
    The returned polyline therefore follows the "interesting" side of the hull,
    excluding the direct chord between the two endpoints.
    """
    pts = list(points)
    if len(pts) < 3:
        return [p[0] for p in pts], [p[1] for p in pts]

    xy = np.array([(p[0], p[1]) for p in pts])
    keys = np.array([sort_key(p) for p in pts])
    verts = ConvexHull(xy).vertices
    vert_keys = keys[verts]
    lo = int(np.argmin(vert_keys))
    hi = int(np.argmax(vert_keys))
    n = len(verts)

    step = 1 if (lo + 1) % n != hi else -1
    k = 0
    order = []
    while True:
        idx = verts[(lo + step * k) % n]
        order.append(idx)
        if (lo + step * k) % n == hi:
            break
        k += 1
    path = xy[order]
    return path[:, 0].tolist(), path[:, 1].tolist()


def highest_x_at(xs: Sequence[float], ys: Sequence[float], y_line: float):
    """Largest x where the polyline (xs, ys) crosses y = y_line.

    y_line is clamped to the y-range of the polyline. Returns None for <2
    points.
    """
    if len(xs) < 2:
        return None
    y_line = min(max(y_line, min(ys)), max(ys))
    best = None
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        lo, hi = sorted((y0, y1))
        if not (lo - 1e-12 <= y_line <= hi + 1e-12):
            continue
        t = 0.0 if abs(y1 - y0) < 1e-15 else (y_line - y0) / (y1 - y0)
        x = x0 + t * (x1 - x0)
        best = x if best is None else max(best, x)
    return best


def x_at_y_on_hull(xs: Sequence[float], ys: Sequence[float], y_line: float):
    """Interpolate the polyline (xs, ys) at y = y_line.

    Differs from `highest_x_at` by taking the **first** crossing (left-to-right
    in the polyline order), which matches the "intersect monotone hull with
    horizontal threshold" pattern in tradeoff.py.
    """
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        lo, hi = sorted((y0, y1))
        if not (lo - 1e-12 <= y_line <= hi + 1e-12):
            continue
        t = 0.0 if abs(y1 - y0) < 1e-15 else (y_line - y0) / (y1 - y0)
        return x0 + t * (x1 - x0)
    return None
