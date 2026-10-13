"""Crop each specimen photograph and build the 64 by 64 working array.

The iCollections photographs show a pinned butterfly in a unit tray beside its
labels, with a ruler. A portrait photograph has the ruler down the left and the
labels across the bottom. A landscape one has the labels on the right, and a
dark tray wall then the ruler along the bottom. The labels carry collectors'
handwriting and names, so the crop must remove them.

The crop rule, a function of the photograph alone with no learned part:

1. Cut away the ruler and the label section, by orientation. Each cut is read
   from a small copy of the photograph and kept as a proportion of it, so
   originals and 512-pixel renditions crop alike.
   - Portrait ruler: the white strip that is bright in the top tenth of the
     photograph, from its first bright column to the first three dark ones.
   - Portrait labels: the first row, from 45 per cent of the height down, where
     the right margin stops matching its own colour higher up (a dark wall, a
     lighter platform or a label).
   - Landscape ruler: from the foot of the photograph up, the rows at least 30
     per cent bright (the ruler with its numbers, ticks and grey top rows),
     then the rows darker than the card on both sides of the tray (the wall).
   - Landscape labels: the first column, from half the width across, where both
     the top and the bottom margins stop matching the card's colour.
   A cut that cannot be read falls back to a fixed proportion.
2. On what is left, model the card's colour as a plane fitted to its border
   pixels, so uneven lighting across the tray is not marked, and mark every
   pixel that differs from it by more than 22 levels in any channel. Shrink
   then grow the marks by a pixel, so specks drop out, and trim away any band
   along an edge of what is left that is at least 80 per cent marked, such as
   a tray wall or the top rows of a ruler.
3. Take the bounding box of the largest connected region. The marks are grown
   by three pixels only to join wing halves across an unmarked body; the box
   is taken from the marks themselves.
4. Pad it by 5 per cent, square it about its centre, and keep it inside what
   steps 1 and 2 left. Where the padded specimen is larger than that, the
   crop runs its full length on that side and is not square.

The fixed box, A4's alternative when the threshold rule fails too often, is
the tray's specimen area at the fixed proportions the cuts fall back to: right
of the ruler and above the labels in portrait, left of the labels and above
the wall and ruler in landscape. It reads nothing but the photograph's size.

The segment rule, a third candidate, finds the specimen as an object rather
than by cutting fixed sections away. On a 300-pixel copy, median filtered:

1. The ruler strips are the edge lines white along nearly their whole
   length, the strongest first, with the ruler's shaded inner edge.
2. The foreground is what differs from the tray: a quadratic surface fitted
   to the plain mid-grey pixels near the tray's commonest colour, read away
   from the label side. Next to a ruler, and across the label side, a line
   that is mostly marked by one plain grey (a wall, the label platform) is
   its own background. Long thin lines left over are dropped.
3. The regions are labelled by a run-length routine of this module's own.
   Label cards are told from white wings by shape, plainness and the dark
   texture of their writing, not by brightness alone, and only on the
   label side, across the tray from the ruler.
4. The specimen is the largest region left that is not in the outer part of
   the label side, with the regions close to it. Where it runs into a label,
   it is cut where the label's paper begins and chosen again.
5. Its box gets a margin of 3 per cent of its longer side, held short of the
   ruler, the label line and any white paper nearer than the margin, and is
   clipped to the photograph. It is not squared here.

Every rule's crop is padded to a square with the tray's colour, the median
of its border pixels, so the specimen keeps its shape, then resized to 64 by
64 with Pillow's bilinear filter.

The working array holds one row per photograph of the fixed manifest, in the
manifest's order, as uint8, except that crop failures are left out by
default. A crop failure is a row a verdicts file (file_name and verdict
columns) marks "fail". Every manifest row stays in the table written beside
the array, with its part (training or held out, by the seed-2026 split), its
failure flag and its row in the array, so nothing is removed silently.

    python scripts/build_working_set.py build --rule threshold \\
        --manifest data/manifest_fixed_20261013.csv --raw-dir data/raw \\
        --out data/nhm_butterflies_64.npy --crop-failures verdicts.csv \\
        --workers 8

Needs NumPy and Pillow. It never overwrites an existing array or table.
"""

import argparse
import csv
import hashlib
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 64
PROFILE_SIDE = 270
MASK_SIDE = 200
COLOUR_LINE = 22
BRIGHT = 150
BAND = 0.8
PORTRAIT_MARGIN_LINE = 22
LANDSCAPE_MARGIN_LINE = 28
JOIN_STEPS = 3
PAD = 0.05
PORTRAIT_LEFT = 0.25
PORTRAIT_BOTTOM = 0.64
LANDSCAPE_RIGHT = 0.665
LANDSCAPE_BOTTOM = 0.70
# The segment rule's settings. Distances are in the small copy's pixels or
# as fractions of its sides, so every size of photograph is read alike.
SEG_SIDE = 300
SEG_TOL = 22
SEG_HOME = 0.35
SEG_WHITE = 170
SEG_STRETCHES = 12
SEG_STRETCH_WHITE = 0.15
SEG_RULER_STRETCHES = 11
SEG_RULER_START = 0.08
SEG_RULER_GAP = 0.04
SEG_RULER_MOST = 0.4
SEG_RULER_EDGE = 0.03
SEG_WALL = 0.10
SEG_WALL_MARKED = 0.35
SEG_FULL = 0.9
SEG_NEAR_RULER = 0.12
SEG_PLAIN = 35
SEG_LINE_WIDTH = 16
SEG_LINE_LENGTH = 0.15
SEG_SMALLEST = 0.0015
SEG_JOIN_GAP = 0.02
SEG_JOIN_AREA = 0.05
SEG_REACH = 3
SEG_FRINGE = 5
SEG_MARGIN = 0.03
SEG_SLACK = 0.05
SEG_TEXT = 0.01
SEG_CARD_RECT = 0.85
SEG_CARD_WHITE = 0.4
SEG_CARD_SAT = 70
SEG_LABEL_PART = 0.58
SEG_LABEL_WHITE = 0.2
SEG_LABEL_SAT = 80
SEG_LABEL_RECT = 0.6
SEG_PAPER_START = 0.02
SEG_PAPER_EDGE = 0.15
SEG_PAPER_MIN = 0.02
# The iCollections layout: the labels lie across the tray from the ruler's
# corner, below a ruler on the left and to the right of one along the foot.
SEG_LABEL_SIDE = {"left": "bottom", "right": "top", "bottom": "right",
                  "top": "left"}
SEED = 2026
HELD_OUT_FRACTION = 0.10
TABLE_FIELDS = [
    "row", "file_name", "part", "crop_failed", "in_array", "array_row"
]


def _check_rgb(rgb, smallest=2):
    """Refuse anything but an RGB uint8 array of shape (height, width, 3)."""
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
        or min(rgb.shape[:2]) < smallest
    ):
        raise ValueError(
            "expected an RGB uint8 array of shape (height, width, 3)"
        )


def _shrunk(rgb, long_side):
    """A new int16 copy of rgb, box-filtered to long_side pixels long."""
    h, w = rgb.shape[:2]
    scale = long_side / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = Image.fromarray(rgb).resize(size, Image.BOX)
    return np.asarray(small, dtype=np.int16)


def _first_run(values, start, length, test):
    """The first index from start where length values in a row pass test.

    None when there is no such run.
    """
    for i in range(max(0, start), len(values) - length + 1):
        if all(test(values[j]) for j in range(i, i + length)):
            return i
    return None


def _portrait_left(grey):
    """The portrait cut past the ruler, as a fraction of the width."""
    sh, sw = grey.shape
    top = (grey[: max(1, sh // 10)] > BRIGHT).mean(axis=0)
    # Some photographs have a dark strip at the very edge, so I start at
    # the ruler's first bright column.
    starts = [x for x in range(max(1, int(0.15 * sw))) if top[x] >= 0.3]
    if not starts:
        return PORTRAIT_LEFT
    x = _first_run(top, starts[0], 3, lambda v: v <= 0.1)
    if x is None or x >= 0.4 * sw:
        return PORTRAIT_LEFT
    return x / sw + 0.01


def _portrait_bottom(small):
    """The portrait cut above the label section, as a fraction of the height.

    It is the first of two rows, from 45 per cent down, where the right
    margin's median colour differs from that margin's own colour higher up,
    from 5 to 45 per cent of the height, by more than PORTRAIT_MARGIN_LINE.
    The left margin is not read: the ruler shades it, and wings often reach
    it. A hindwing that reaches the right margin is cut there.
    """
    sh, sw = small.shape[:2]
    x1 = int(0.90 * sw)
    x1_end = max(x1 + 1, int(0.98 * sw))
    line = np.median(small[:, x1:x1_end], axis=1)
    top, stop = int(0.05 * sh), max(1, int(0.45 * sh))
    card = np.median(line[top:stop], axis=0)
    off = np.abs(line - card).max(axis=1) > PORTRAIT_MARGIN_LINE
    y = _first_run(off, int(0.45 * sh), 2, bool)
    return y / sh - 0.005 if y is not None else PORTRAIT_BOTTOM


def _landscape_bottom(grey):
    """The landscape cut above the wall and the ruler, as a height fraction.

    The ruler is found from the foot of the photograph: from its lowest
    bright row in the bottom tenth, up while rows stay at least 30 per cent
    bright, past its numbers, ticks and grey top rows. Then up over the
    tray wall, for at most 8 per cent of the height.
    """
    sh, sw = grey.shape
    bright = (grey > BRIGHT).mean(axis=1)
    low = [y for y in range(int(0.9 * sh), sh) if bright[y] >= 0.5]
    if not low:
        return LANDSCAPE_BOTTOM
    y = low[-1]
    while y - 1 > 0.55 * sh and bright[y - 1] >= 0.3:
        y -= 1
    # The wall spans the tray, under the labels too, where no specimen
    # reaches. It is read against this card's own grey, because some cards
    # are dark enough that any fixed line would take the card for the wall.
    card_grey = np.median(grey[: max(1, sh // 20), : max(1, sw // 2)])
    dark = card_grey - 20
    right = int(0.7 * sw)
    wall = (np.median(grey[:, : int(0.6 * sw)], axis=1) < dark) & (
        np.median(grey[:, right:], axis=1) < dark
    )
    stop = max(int(0.55 * sh), y - int(0.08 * sh))
    while y - 1 > stop and wall[y - 1]:
        y -= 1
    return y / sh - 0.005


def _landscape_right(small, bottom):
    """The landscape cut left of the label section, as a width fraction."""
    sh, sw = small.shape[:2]
    y1 = max(2, int(bottom * sh))
    card = np.median(
        small[: max(1, sh // 20), : max(1, sw // 2)].reshape(-1, 3), axis=0
    )
    height = max(1, int(0.08 * sh))
    y0 = max(0, y1 - height)
    strips = (small[:height], small[y0:y1])
    off = [
        np.abs(np.median(s, axis=0) - card).max(axis=1) > LANDSCAPE_MARGIN_LINE
        for s in strips
    ]
    x = _first_run(off[0] & off[1], int(0.5 * sw), 2, bool)
    return x / sw - 0.005 if x is not None else LANDSCAPE_RIGHT


def cut_fractions(rgb):
    """Where the ruler and the label section are cut away from rgb.

    Returns (left, top, right, bottom) as fractions of the width and the
    height. A photograph taller than it is wide is read as portrait.
    """
    _check_rgb(rgb)
    h, w = rgb.shape[:2]
    small = _shrunk(rgb, PROFILE_SIDE)
    grey = small.mean(axis=2)
    if h > w:
        left = _portrait_left(grey)
        return (left, 0.0, 1.0, _portrait_bottom(small))
    bottom = _landscape_bottom(grey)
    return (0.0, 0.0, _landscape_right(small, bottom), bottom)


def _grow(mask):
    """A new mask with every marked pixel's four neighbours marked too."""
    grown = mask.copy()
    grown[1:] |= mask[:-1]
    grown[:-1] |= mask[1:]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown


def _shrink(mask):
    """A new mask of the marked pixels whose four neighbours are marked."""
    kept = mask.copy()
    kept[1:] &= mask[:-1]
    kept[:-1] &= mask[1:]
    kept[:, 1:] &= mask[:, :-1]
    kept[:, :-1] &= mask[:, 1:]
    return kept


def _card_plane(small):
    """The card's colour at each pixel of small, a plane fitted to the border.

    Each channel is fitted by least squares to the border pixels, then
    refitted four times to the half of them that sit nearest the last fit,
    so a dark wall or a wing along one side does not pull the card towards
    it while the card is at least half of the border.
    """
    sh, sw = small.shape[:2]
    yy, xx = np.mgrid[0:sh, 0:sw]
    edge = np.zeros((sh, sw), dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    design = np.stack([np.ones(edge.sum()), xx[edge] / sw, yy[edge] / sh], 1)
    values = small[edge].astype(float)
    keep = np.ones(len(values), dtype=bool)
    for _ in range(5):
        coef = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        resid = np.abs(values - design @ coef).max(axis=1)
        keep = resid <= np.percentile(resid, 50)
    every = np.stack([np.ones(sh * sw), xx.ravel() / sw, yy.ravel() / sh], 1)
    return (every @ coef).reshape(sh, sw, 3)


def _edge_band(values):
    """How many steps in from the start of values a marked band reaches.

    The band may begin within the first 3 per cent and runs inward while
    each value is at least BAND, up to a quarter of the length.
    """
    n = len(values)
    start = _first_run(
        values[: max(1, int(0.03 * n)) + 1], 0, 1, lambda v: v >= BAND
    )
    if start is None:
        return 0
    k = start
    while k < int(0.25 * n) and values[k] >= BAND:
        k += 1
    return k


def _band_limits(mask):
    """The part of mask inside any marked band running in from its edges.

    A band is a run of rows or columns, each at least BAND marked, that
    starts at an edge of mask: a tray wall, the top rows of a ruler or a
    label platform left inside the cuts. A wing reaching an edge does not
    make one, because the card beside it is not marked. Returns
    (x0, y0, x1, y1), x1 and y1 exclusive.
    """
    sh, sw = mask.shape
    rows, cols = mask.mean(axis=1), mask.mean(axis=0)
    y0, x0 = _edge_band(rows), _edge_band(cols)
    y1, x1 = sh - _edge_band(rows[::-1]), sw - _edge_band(cols[::-1])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return (0, 0, sw, sh)
    return (x0, y0, x1, y1)


def _largest_region(joined, marked):
    """The bounding box of the largest four-connected region of joined.

    The box covers only the marked pixels inside that region, as
    (x0, y0, x1, y1) with x1 and y1 exclusive. Ties go to the region found
    first in reading order. None when nothing is joined.
    """
    h, w = joined.shape
    flat = joined.ravel().tolist()
    keep = marked.ravel().tolist()
    seen = bytearray(len(flat))
    best_size, best = 0, None
    for seed in np.flatnonzero(joined).tolist():
        if seen[seed]:
            continue
        seen[seed] = 1
        stack, members = [seed], []
        while stack:
            i = stack.pop()
            members.append(i)
            y, x = divmod(i, w)
            for j, ok in (
                (i - w, y > 0),
                (i + w, y < h - 1),
                (i - 1, x > 0),
                (i + 1, x < w - 1),
            ):
                if ok and flat[j] and not seen[j]:
                    seen[j] = 1
                    stack.append(j)
        if len(members) > best_size:
            best_size, best = len(members), members
    if best is None:
        return None
    inside = [i for i in best if keep[i]] or best
    ys, xs = np.divmod(np.array(inside), w)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _place(lo, hi, side, length, origin=0):
    """The span of side pixels centred on lo to hi, kept within origin to
    origin plus length."""
    span = min(side, length)
    start = min(max((lo + hi) / 2 - span / 2, origin), origin + length - span)
    a = max(origin, int(round(start)))
    b = min(origin + length, max(a + 1, int(round(start + span))))
    return a, b


def crop_box(rgb):
    """The crop around the specimen, as (left, top, right, bottom) pixels.

    Square unless the padded specimen is larger than what the cuts leave,
    in which case that side runs the full length the cuts allow.
    """
    _check_rgb(rgb)
    h, w = rgb.shape[:2]
    left, top, right, bottom = cut_fractions(rgb)
    rx0, ry0 = int(left * w), int(top * h)
    rx1 = min(w, max(rx0 + 2, int(right * w)))
    ry1 = min(h, max(ry0 + 2, int(bottom * h)))
    rw, rh = rx1 - rx0, ry1 - ry0
    small = _shrunk(rgb[ry0:ry1, rx0:rx1], MASK_SIDE)
    sh, sw = small.shape[:2]
    marked = np.abs(small - _card_plane(small)).max(axis=2) > COLOUR_LINE
    # Shrinking then growing drops specks a pixel or two across.
    opened = _grow(_shrink(marked))
    # Any band along an edge of what the cuts left is trimmed away too.
    cx0, cy0, cx1, cy1 = _band_limits(opened)
    sx, sy = rw / sw, rh / sh
    lo_x, lo_y = round(cx0 * sx), round(cy0 * sy)
    hi_x, hi_y = round(cx1 * sx), round(cy1 * sy)
    inner = opened[cy0:cy1, cx0:cx1]
    # Growing only joins wing halves across an unmarked body. The box is
    # still taken from the opened pixels themselves.
    joined = inner
    for _ in range(JOIN_STEPS):
        joined = _grow(joined)
    found = _largest_region(joined, inner)
    if found is None:
        side = min(hi_x - lo_x, hi_y - lo_y)
        x0, x1 = _place(lo_x, hi_x, side, hi_x - lo_x, lo_x)
        y0, y1 = _place(lo_y, hi_y, side, hi_y - lo_y, lo_y)
        return (rx0 + x0, ry0 + y0, rx0 + x1, ry0 + y1)
    bx0, bx1 = (found[0] + cx0) * sx, (found[2] + cx0) * sx
    by0, by1 = (found[1] + cy0) * sy, (found[3] + cy0) * sy
    pw, ph = PAD * (bx1 - bx0), PAD * (by1 - by0)
    bx0, bx1, by0, by1 = bx0 - pw, bx1 + pw, by0 - ph, by1 + ph
    side = max(bx1 - bx0, by1 - by0)
    x0, x1 = _place(bx0, bx1, side, hi_x - lo_x, lo_x)
    y0, y1 = _place(by0, by1, side, hi_y - lo_y, lo_y)
    return (rx0 + x0, ry0 + y0, rx0 + x1, ry0 + y1)


def fixed_box(rgb):
    """The fixed box for rgb's orientation, as (left, top, right, bottom).

    The unit tray is built so that every specimen is photographed with its
    labels in the same layout (Paterson et al., 2016, p. 4), so the box is
    that layout's specimen area, at the proportions crop_box's cuts fall back
    to. Its edges are whole pixels taken as crop_box takes them, rounded
    down, so the box never reaches past a cut line. A photograph taller than
    it is wide is read as portrait.
    """
    _check_rgb(rgb)
    h, w = rgb.shape[:2]
    if h > w:
        return (int(PORTRAIT_LEFT * w), 0, w, int(PORTRAIT_BOTTOM * h))
    return (0, 0, int(LANDSCAPE_RIGHT * w), int(LANDSCAPE_BOTTOM * h))


def label_regions(mask):
    """The four-connected regions of a boolean mask, as (labels, count).

    labels is a new int32 array the shape of mask, 0 where mask is False and
    1 to count elsewhere, numbered in the reading order of each region's
    first pixel. Diagonal neighbours are not joined. The mask is read a run
    of marked pixels at a time, and runs that overlap on neighbouring rows
    are joined with a union-find, so no pixel is visited twice.
    """
    h, w = mask.shape
    parent = [0]

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = np.zeros((h, w + 2), dtype=np.int8)
    edges[:, 1:-1] = mask
    steps = np.diff(edges, axis=1)
    runs, above = [], []
    for y in range(h):
        here = []
        j = 0
        starts = np.flatnonzero(steps[y] == 1).tolist()
        ends = np.flatnonzero(steps[y] == -1).tolist()
        for s, e in zip(starts, ends):
            run_id = 0
            while j < len(above) and above[j][1] <= s:
                j += 1
            k = j
            while k < len(above) and above[k][0] < e:
                other = root(above[k][2])
                if run_id == 0:
                    run_id = other
                elif other != run_id:
                    low, high = min(run_id, other), max(run_id, other)
                    parent[high] = low
                    run_id = low
                k += 1
            if run_id == 0:
                run_id = len(parent)
                parent.append(run_id)
            here.append((s, e, run_id))
            runs.append((y, s, e, run_id))
        above = here
    labels = np.zeros((h, w), dtype=np.int32)
    number = {}
    for y, s, e, run_id in runs:
        r = root(run_id)
        if r not in number:
            number[r] = len(number) + 1
        labels[y, s:e] = number[r]
    return labels, len(number)


def _median3(img):
    """A new int16 copy of img with each channel 3 by 3 median filtered."""
    h, w = img.shape[:2]
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    stack = np.stack([padded[dy:dy + h, dx:dx + w]
                      for dy in range(3) for dx in range(3)])
    return np.median(stack, axis=0).astype(np.int16)


def _white_line(grey):
    """The grey level above which a pixel counts as white paper or ruler.

    Normally SEG_WHITE, but lower on a dim photograph, where the ruler and
    the labels, the brightest things in the tray, sit under it. It never
    comes within 25 levels of the median, so a photograph with no white in
    it has none.
    """
    p50, p95 = np.percentile(grey, [50, 95])
    return max(min(SEG_WHITE, float(p95) - 25), float(p50) + 25)


def _stretches_white(white, axis):
    """For each line across axis, how many of SEG_STRETCHES stretches along
    it hold some white."""
    n = white.shape[axis]
    cuts = np.linspace(0, n, SEG_STRETCHES + 1).astype(int)
    count = np.zeros(white.shape[1 - axis], dtype=int)
    for a, b in zip(cuts[:-1], cuts[1:]):
        part = white[a:b] if axis == 0 else white[:, a:b]
        count += part.mean(axis=axis) >= SEG_STRETCH_WHITE
    return count


def _ruler_depth(counts):
    """How many lines in from the edge a ruler strip reaches, or 0.

    counts runs from the edge inward. A ruler line is white along nearly its
    whole length, which no wing or label is. The strip must start near the
    edge, and may dip for a few lines between a ruler's two scales.
    """
    n = len(counts)
    hits = [k for k in range(int(SEG_RULER_MOST * n))
            if counts[k] >= SEG_RULER_STRETCHES]
    if not hits or hits[0] >= max(2, int(SEG_RULER_START * n)):
        return 0
    last = hits[0]
    for k in hits[1:]:
        if k - last > max(2, SEG_RULER_GAP * n):
            break
        last = k
    return last + 1


def _ruler_depths(grey, white):
    """The ruler strip's depth in small-copy pixels from each side.

    The strip that is white along the most of its length is taken first and
    cut away before the other sides are read again, because a ruler along
    the foot puts white into every column and, with a column of labels
    above it, could pass for a second ruler down the side.
    """
    white_px = grey > white
    sh, sw = grey.shape
    depth = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    for _ in range(4):
        x0, y0 = depth["left"], depth["top"]
        x1, y1 = sw - depth["right"], sh - depth["bottom"]
        if x1 - x0 < 10 or y1 - y0 < 10:
            break
        rest = white_px[y0:y1, x0:x1]
        cols = _stretches_white(rest, 0)
        rows = _stretches_white(rest, 1)
        counts = {"left": cols, "right": cols[::-1], "top": rows,
                  "bottom": rows[::-1]}
        found = {}
        for s, c in counts.items():
            d = _ruler_depth(c)
            if d and not depth[s]:
                found[s] = (float(c[:d].mean()), d)
        if not found:
            break
        best = max(found, key=lambda s: found[s])
        # The ruler's inner edge is often shaded, paler than the tray but
        # under the white line, so the cut runs on over lines that are still
        # bright along most of their length.
        lines = _from_side(grey[y0:y1, x0:x1], best)
        d = found[best][1]
        stop = d + max(1, int(SEG_RULER_EDGE * lines.shape[0]))
        while d < min(stop, lines.shape[0]) and np.median(lines[d]) > white - 40:
            d += 1
        depth[best] = d
    return depth


def ruler_strips(rgb):
    """The ruler strips along the edges of rgb, as {side: depth fraction}.

    side is left, top, right or bottom, and the depth is a fraction of the
    width or the height. A side without a ruler is left out.
    """
    _check_rgb(rgb)
    grey = _median3(_shrunk(rgb, SEG_SIDE)).mean(axis=2)
    depths = _ruler_depths(grey, _white_line(grey))
    sh, sw = grey.shape
    return {
        side: d / (sw if side in ("left", "right") else sh)
        for side, d in depths.items() if d
    }


def _mode_colour(px):
    """The commonest colour of the pixels px: the commonest coarse colour,
    refined to the median of the pixels near it."""
    q = px // 16
    key = (q[:, 0] * 16 + q[:, 1]) * 16 + q[:, 2]
    centre = np.median(px[key == np.bincount(key).argmax()], axis=0)
    near = np.abs(px - centre).max(axis=1) <= SEG_TOL + 8
    return np.median(px[near], axis=0)


def _tray_distance(sm, pool, home):
    """How far each pixel of sm sits from the tray, as a new float array.

    The tray's colour is fitted as a quadratic surface over the pool pixels
    near its mode colour, so even lighting falls away gently across the
    tray is not marked. The mode is read only from the pool pixels in home,
    the part of the tray away from the label platform, which can be a
    plainer grey than the tray and so outvote it. Each refit takes every
    pool pixel near the last fit, so the fit can follow the light into a
    darker corner, and takes in a platform close to the tray's grey.
    """
    sh, sw = sm.shape[:2]
    yy, xx = np.mgrid[0:sh, 0:sw]
    x, y = xx / sw - 0.5, yy / sh - 0.5
    terms = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], -1)
    colour = _mode_colour(sm[pool & home] if (pool & home).sum() > 50
                          else sm[pool])
    near = np.abs(sm - colour).max(axis=2)
    chosen = pool & (near <= SEG_TOL + 8)
    dist = near.astype(float)
    for _ in range(4):
        if chosen.sum() < 30:
            break
        coef = np.linalg.lstsq(terms[chosen], sm[chosen].astype(float),
                               rcond=None)[0]
        dist = np.abs(sm - terms @ coef).max(axis=2)
        chosen = pool & (dist <= SEG_TOL)
    return dist


def _from_side(arr, side):
    """A view of arr whose first axis runs from side inward."""
    return {
        "left": arr.swapaxes(0, 1),
        "right": arr.swapaxes(0, 1)[::-1],
        "top": arr,
        "bottom": arr[::-1],
    }[side]


def _run_lengths(mask, axis):
    """For each marked pixel, the length of its run along axis; 0 elsewhere."""
    m = mask if axis == 1 else mask.T
    h, w = m.shape
    out = np.zeros((h, w), dtype=np.int32)
    edges = np.zeros((h, w + 2), dtype=np.int8)
    edges[:, 1:-1] = m
    steps = np.diff(edges, axis=1)
    for y in range(h):
        starts = np.flatnonzero(steps[y] == 1).tolist()
        ends = np.flatnonzero(steps[y] == -1).tolist()
        for s, e in zip(starts, ends):
            out[y, s:e] = e - s
    return out if axis == 1 else out.T


def _hull(points):
    """The convex hull of (x, y) points, by Andrew's monotone chain."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _smallest_rectangle(hull):
    """The area of the smallest rectangle, at any angle, around a hull."""
    if len(hull) < 3:
        return 1.0
    pts = np.array(hull, dtype=float)
    best = None
    for k in range(len(pts)):
        edge = pts[(k + 1) % len(pts)] - pts[k]
        length = np.hypot(*edge)
        if length == 0:
            continue
        u = edge / length
        a, b = pts @ u, pts @ np.array([-u[1], u[0]])
        area = (a.max() - a.min()) * (b.max() - b.min())
        best = area if best is None else min(best, area)
    return best or 1.0


def _regions(core, grey, raw_grey, sat, white):
    """The regions of core worth keeping, largest first, with what they look
    like: their box, how rectangular, how white and how plain they are, and
    how much dark text texture they hold."""
    labels, n = label_regions(core)
    sh, sw = core.shape
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=n + 1)
    order = np.argsort(flat, kind="stable")
    bounds = np.concatenate(([0], np.cumsum(areas)))
    found = []
    for i in range(1, n + 1):
        if areas[i] < SEG_SMALLEST * sh * sw:
            continue
        idx = order[bounds[i]:bounds[i + 1]]
        ys, xs = np.divmod(idx, sw)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        shape = np.zeros((y1 - y0 + 2, x1 - x0 + 2), dtype=bool)
        shape[ys - y0 + 1, xs - x0 + 1] = True
        # The region with its holes filled: whatever the outside cannot reach.
        outside, _ = label_regions(~shape)
        filled = (outside != outside[0, 0])[1:-1, 1:-1]
        p = np.pad(filled, 1)
        rim = filled & ~(p[2:, 1:-1] & p[:-2, 1:-1] & p[1:-1, 2:] & p[1:-1, :-2])
        ry, rx = np.nonzero(rim)
        corners = [(cx + dx, cy + dy)
                   for cx, cy in zip(rx.tolist(), ry.tolist())
                   for dx in (0, 1) for dy in (0, 1)]
        inside = raw_grey[y0:y1, x0:x1][filled]
        paper = np.median(grey.ravel()[idx])
        found.append({
            "id": i,
            "area": int(areas[i]),
            "box": (int(x0), int(y0), int(x1), int(y1)),
            "paper": _paper_box(xs, ys, grey.ravel()[idx] > white,
                                sat.ravel()[idx] <= 60),
            "rect": float(filled.sum() / _smallest_rectangle(_hull(corners))),
            "white": float((grey.ravel()[idx] > white).mean()),
            "sat": float(np.median(sat.ravel()[idx])),
            "text": float((inside < paper - 50).mean()),
        })
    found.sort(key=lambda r: -r["area"])
    return labels, found


def _paper_box(xs, ys, bright, plain):
    """The box of a region's white paper, for where a label begins.

    A label region can carry a scrap of the platform's edge, the tray's wall
    or a wing's dark border that reaches past the card towards the specimen;
    the card itself begins where its paper does. The box leaves out the 3
    per cent of paper pixels furthest out on each side. A region with too
    little paper falls back to the box of all its pixels, read the same way.
    """
    paper = bright & plain
    if paper.sum() >= 20:
        xs, ys = xs[paper], ys[paper]
    return (int(np.percentile(xs, 3)), int(np.percentile(ys, 3)),
            int(np.percentile(xs, 97)) + 1, int(np.percentile(ys, 97)) + 1)


def _near_edge(box, side):
    """The edge of box that faces away from side."""
    return {"bottom": box[1], "top": box[3], "right": box[0], "left": box[2]}[side]


def _beyond(box, line, side, slack):
    """Whether box starts at or beyond line, seen from the specimen, less
    slack."""
    edge = _near_edge(box, side)
    if side in ("bottom", "right"):
        return edge >= line - slack
    return edge <= line + slack


def _gap(a, b):
    """The gap between two boxes, the larger of the two axes'."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)


def _boxes_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _looks_like_paper(r):
    """A label card by its look alone: a white, plain, rectangular region
    with text texture, wherever it lies."""
    return (r["rect"] >= SEG_CARD_RECT and r["white"] >= SEG_CARD_WHITE
            and r["sat"] <= SEG_CARD_SAT and r["text"] >= SEG_TEXT)


def _choose(core, clean, grey, raw_grey, sat, white, work, side, line):
    """Pick the specimen's regions from core, as a dict.

    A region is taken for a label if it looks like a card, or if it lies in
    the outer part of the tray on the label side and is compact, partly
    white, plain and textured: never by its whiteness alone. The label
    line is where the nearest such label begins, unless line is given, and
    any region from there outward is left out too. The specimen is the
    largest region left whose middle is not in that outer part, where the
    labels lie; regions close to it and not small join it.
    """
    x0, y0, x1, y1 = work
    sh, sw = core.shape
    labels, regions = _regions(core, grey, raw_grey, sat, white)

    def outward(r):
        """How far across the tray towards the labels r's middle lies, 0 to 1."""
        b = r["box"]
        if side in ("top", "bottom"):
            f = ((b[1] + b[3]) / 2 - y0) / max(1, y1 - y0)
        else:
            f = ((b[0] + b[2]) / 2 - x0) / max(1, x1 - x0)
        return 1 - f if side in ("top", "left") else f

    def label_like(r):
        if _looks_like_paper(r):
            return True
        if side is None:
            return False
        return (outward(r) >= SEG_LABEL_PART and r["white"] >= SEG_LABEL_WHITE
                and r["sat"] <= SEG_LABEL_SAT and r["rect"] >= SEG_LABEL_RECT
                and r["text"] >= SEG_TEXT)

    found = [r for r in regions if label_like(r)]
    # Two lines on the label side: the label line, from the labels' whole
    # regions, past which nothing joins the specimen, and the paper line,
    # where their cards begin, which the margin stops at. A scrap of wall
    # joined to a card can bring the first nearer the specimen than the
    # card is, which is safe for leaving regions out but not for the margin.
    paper_line = line
    if side and line is None:
        middle = (y0 + y1) / 2 if side in ("top", "bottom") else (x0 + x1) / 2
        nearest = min if side in ("bottom", "right") else max
        edges = [_near_edge(r["box"], side) for r in found
                 if _beyond(r["box"], middle, side, 0)]
        if edges:
            line = nearest(edges)
        papers = [_near_edge(r["paper"], side) for r in found
                  if _beyond(r["paper"], middle, side, 0)]
        if papers:
            paper_line = nearest(papers)
    slack = SEG_SLACK * max(sh, sw)
    left = [
        r for r in regions
        if not label_like(r)
        and not (line is not None and _beyond(r["box"], line, side, slack))
    ]
    out = {"labels": found, "line": line, "paper_line": paper_line,
           "chosen": []}
    if not left:
        return out
    if side is not None:
        home = [r for r in left if outward(r) < SEG_LABEL_PART]
        seed = home[0] if home else left[0]
    else:
        seed = left[0]
    chosen, box = [seed], list(seed["box"])
    reach = SEG_JOIN_GAP * max(sh, sw)
    joined = True
    while joined:
        joined = False
        for r in left:
            if r in chosen:
                continue
            if (r["area"] >= SEG_JOIN_AREA * seed["area"]
                    and _gap(r["box"], box) <= reach):
                chosen.append(r)
                b = r["box"]
                box = [min(box[0], b[0]), min(box[1], b[1]),
                       max(box[2], b[2]), max(box[3], b[3])]
                joined = True
    # The opening and the line removal shave a pixel or two off the wing
    # edges, so the regions grow back a little into what was marked, but
    # never into or beside another region, such as a label's top.
    keep = np.isin(labels, [r["id"] for r in chosen])
    others = _grow((labels > 0) & ~keep)
    for _ in range(SEG_REACH):
        keep = _grow(keep) & clean & ~others
    ys, xs = np.nonzero(keep)
    out.update(chosen=chosen, keep=keep, box=(
        int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return out


def _paper_line(keep, grey, sat, white, work, side):
    """Where label paper inside the chosen region begins, or None.

    When a wing touches a label, or both touch the platform's edge, the
    specimen and the labels are one region. Read from the label side
    inward, white paper then starts at the region's outer end, or near the
    tray's edge where remnants of the platform reach past the labels, and
    stops at a gap before the specimen; the line is that gap. A gap past
    the middle of the tray is a white specimen's own body, not a label's
    edge, so it gives no line.
    """
    x0, y0, x1, y1 = work
    region = (slice(y0, y1), slice(x0, x1))
    paper = (grey > white) & (sat <= 60)
    lines = _from_side(keep[region], side)
    held = lines.sum(axis=1)
    counts = (lines & _from_side(paper[region], side)).sum(axis=1)
    n = len(counts)
    outer = int(np.argmax(held >= 2)) if (held >= 2).any() else n
    reach = max(outer + SEG_PAPER_START * n, SEG_PAPER_EDGE * n)
    k = outer
    while k < n and counts[k] < 2 and k <= reach:
        k += 1
    if k >= n or counts[k] < 2:
        return None
    total = 0
    while k < n and counts[k:k + 2].max() >= 2:
        total += counts[k]
        k += 1
    if k > 0.5 * n or total < SEG_PAPER_MIN * keep.sum():
        return None
    return {"bottom": y1 - k, "right": x1 - k, "top": y0 + k, "left": x0 + k}[side]


def _clear_of(paper, box, own, sx, sy, sides):
    """box, in pixels, with the named sides' margins pulled in before paper.

    own is the specimen's own extent, which the margin surrounds and which
    never shrinks. paper is a mask on the small copy, sx and sy its scale.
    On each side named the margin's lines are read outward from the
    specimen, and the side stops a line short of the first line holding two
    or more paper pixels, because paper can begin within the line before.
    """
    left, top, right, bottom = box
    ox0, oy0, ox1, oy1 = own
    sh, sw = paper.shape
    cols = slice(max(0, int(left / sx)), min(sw, int(np.ceil(right / sx))))
    rows = slice(max(0, int(top / sy)), min(sh, int(np.ceil(bottom / sy))))
    if "bottom" in sides:
        for y in range(int(np.ceil(oy1 / sy)), min(sh, int(np.ceil(bottom / sy)))):
            if paper[y, cols].sum() >= 2:
                bottom = max(int(np.ceil(oy1)), int((y - 1) * sy))
                break
    if "top" in sides:
        for y in range(int(oy0 / sy) - 1, max(-1, int(top / sy) - 1), -1):
            if paper[y, cols].sum() >= 2:
                top = min(int(oy0), int(np.ceil((y + 2) * sy)))
                break
    if "right" in sides:
        for x in range(int(np.ceil(ox1 / sx)), min(sw, int(np.ceil(right / sx)))):
            if paper[rows, x].sum() >= 2:
                right = max(int(np.ceil(ox1)), int((x - 1) * sx))
                break
    if "left" in sides:
        for x in range(int(ox0 / sx) - 1, max(-1, int(left / sx) - 1), -1):
            if paper[rows, x].sum() >= 2:
                left = min(int(ox0), int(np.ceil((x + 2) * sx)))
                break
    return left, top, right, bottom


def _cut_beyond(mask, line, side):
    """A new copy of mask with everything from line outward unmarked."""
    out = mask.copy()
    if side == "bottom":
        out[line:] = False
    elif side == "top":
        out[:line] = False
    elif side == "right":
        out[:, line:] = False
    else:
        out[:, :line] = False
    return out


def segment_box(rgb):
    """The segment rule's crop around the specimen, as (left, top, right,
    bottom) pixels.

    On a small copy of the photograph: find the ruler strips by their extent,
    mark what differs from the tray's colour, unmark the tray's walls and
    the label platform, drop long thin lines, label the regions, leave out
    the labels and take the specimen's regions. Their box gets a margin of
    SEG_MARGIN of its longer side, kept out of the ruler strip and short of
    the label line, and is clipped to the photograph. It is not squared:
    pad_to_square does that.
    """
    _check_rgb(rgb)
    h, w = rgb.shape[:2]
    small = _shrunk(rgb, SEG_SIDE)
    sm = _median3(small)
    sh, sw = sm.shape[:2]
    grey = sm.mean(axis=2)
    raw_grey = small.mean(axis=2)
    sat = sm.max(axis=2) - sm.min(axis=2)
    white = _white_line(grey)
    depth = _ruler_depths(grey, white)
    x0, y0 = depth["left"], depth["top"]
    x1, y1 = sw - depth["right"], sh - depth["bottom"]
    if x1 - x0 < 10 or y1 - y0 < 10:
        x0, y0, x1, y1 = 0, 0, sw, sh
        depth = dict.fromkeys(depth, 0)
    work = (x0, y0, x1, y1)
    region = (slice(y0, y1), slice(x0, x1))
    inside = np.zeros((sh, sw), dtype=bool)
    inside[region] = True
    rulers = [s for s in depth if depth[s]]
    side = (SEG_LABEL_SIDE[max(rulers, key=lambda s: depth[s])]
            if rulers else None)
    # The tray is a plain mid grey, so only such pixels may set its colour.
    pool = inside & (grey > 25) & (grey < white - 5) & (sat <= 45)
    home = inside.copy()
    if side is not None:
        _from_side(home[region], side)[:int(SEG_HOME * _from_side(
            home[region], side).shape[0])] = False
    if pool.sum() > 50:
        dist = _tray_distance(sm, pool, home)
    else:
        dist = np.abs(sm - np.median(sm[inside], axis=0)).max(axis=2)
    marked = (dist > SEG_TOL) & inside

    # Next to a ruler lie its own grey edge and the tray's dark wall. Each
    # such line that is mostly marked is its own background: what matches
    # the commonest colour of its marked pixels is unmarked, and a wing
    # crossing it, being another colour, is not. The commonest colour, not
    # each channel's median, because the medians of a line part wing, part
    # wall and part label can make a colour that none of them is.
    for s in rulers:
        lines_sm = _from_side(sm[region], s)
        lines = _from_side(marked[region], s)
        for k in range(int(SEG_WALL * lines.shape[0])):
            if lines[k].mean() < SEG_WALL_MARKED:
                break
            common = _mode_colour(lines_sm[k][lines[k]])
            lines[k] &= np.abs(lines_sm[k] - common).max(axis=1) > SEG_TOL

    # The same for whole lines on the label side or close to a ruler: the
    # raised label platform and its side wall, where they differ from the
    # tray, when the line's commonest colour is a plain grey. Where label
    # paper is a line's commonest colour, the plain greys of the other lines
    # stand in for it, so the platform still goes and the labels stay.
    full, greys = [], []
    for axis in (0, 1):
        lines = marked[region] if axis == 1 else marked[region].T
        lines_sm = sm[region] if axis == 1 else sm[region].swapaxes(0, 1)
        n = lines.shape[0]
        for k in range(n):
            if axis == 1:
                half = "top" if k < n / 2 else "bottom"
            else:
                half = "left" if k < n / 2 else "right"
            near_ruler = depth[half] and min(k, n - 1 - k) <= SEG_NEAR_RULER * n
            if not (half == side or near_ruler) or lines[k].mean() < SEG_FULL:
                continue
            common = _mode_colour(lines_sm[k][lines[k]])
            plain = (common.max() - common.min() <= SEG_PLAIN
                     and common.mean() < white - 10)
            full.append((lines, lines_sm, k, common if plain else None))
            if plain:
                greys.append(common)
    stand_in = np.median(greys, axis=0) if greys else None
    for lines, lines_sm, k, common in full:
        common = common if common is not None else stand_in
        if common is not None:
            lines[k] &= np.abs(lines_sm[k] - common).max(axis=1) > SEG_TOL

    # Long thin lines left over, such as a platform edge or a ruler's edge,
    # would join a wing to the labels, so they are dropped. A wing tip is
    # thin too, but never for long.
    across = _run_lengths(marked, 1)
    down = _run_lengths(marked, 0)
    lines = (
        (_run_lengths(marked & (across <= SEG_LINE_WIDTH), 0)
         >= SEG_LINE_LENGTH * sh)
        | (_run_lengths(marked & (down <= SEG_LINE_WIDTH), 1)
           >= SEG_LINE_LENGTH * sw)
    )
    clean = marked & ~lines
    core = _grow(_shrink(clean))

    picked = _choose(core, clean, grey, raw_grey, sat, white, work, side, None)
    labels = list(picked["labels"])
    if side is not None and picked["chosen"]:
        # A label the specimen's box runs into is cut away where its paper
        # begins, so that a scrap of wall joined to it cannot pull the cut
        # into a wing.
        hits = [_near_edge(r["paper"], side) for r in picked["labels"]
                if _boxes_overlap(picked["box"], r["paper"])]
        cuts = [min(hits) if side in ("bottom", "right") else max(hits)] if hits else []
        paper = _paper_line(picked["keep"], grey, sat, white, work, side)
        if paper is not None:
            cuts.append(paper)
        if cuts:
            line = min(cuts) if side in ("bottom", "right") else max(cuts)
            # The first row of paper on the small copy can begin anywhere
            # within the row before it, so the cut is one row nearer the
            # specimen.
            line += -1 if side in ("bottom", "right") else 1
            again = _choose(_cut_beyond(core, line, side),
                            _cut_beyond(clean, line, side),
                            grey, raw_grey, sat, white, work, side, line)
            if again["chosen"]:
                picked = again
                labels += again["labels"]

    sx, sy = w / sw, h / sh
    if not picked["chosen"]:
        return (int(x0 * sx), int(y0 * sy), min(w, int(np.ceil(x1 * sx))),
                min(h, int(np.ceil(y1 * sy))))
    bx0, by0, bx1, by1 = picked["box"]
    bx0, bx1, by0, by1 = bx0 * sx, bx1 * sx, by0 * sy, by1 * sy
    m = SEG_MARGIN * max(bx1 - bx0, by1 - by0)
    left, top = int(np.floor(bx0 - m)), int(np.floor(by0 - m))
    right, bottom = int(np.ceil(bx1 + m)), int(np.ceil(by1 + m))
    # The margin stops at the ruler strip and at the label line, though the
    # specimen's own box never shrinks for them.
    left = max(left, min(int(bx0), int(np.ceil(x0 * sx))))
    top = max(top, min(int(by0), int(np.ceil(y0 * sy))))
    right = min(right, max(int(np.ceil(bx1)), int(x1 * sx)))
    bottom = min(bottom, max(int(np.ceil(by1)), int(y1 * sy)))
    line = picked["paper_line"]
    if line is not None:
        # One row back from the first row of paper, as for the cut.
        if side == "bottom":
            bottom = max(int(np.ceil(by1)), min(bottom, int((line - 1) * sy)))
        elif side == "right":
            right = max(int(np.ceil(bx1)), min(right, int((line - 1) * sx)))
        elif side == "top":
            top = min(int(by0), max(top, int(np.ceil((line + 1) * sy))))
        else:
            left = min(int(bx0), max(left, int(np.ceil((line + 1) * sx))))
    # On the label side and the ruler sides the margin stops before any
    # white paper that is not the specimen's, such as the top of a label or
    # the ruler's edge, that lies closer to the specimen than the margin
    # reaches. White within a few pixels of the specimen is its own pale
    # fringe, not paper.
    near = picked["keep"]
    for _ in range(SEG_FRINGE):
        near = _grow(near)
    paper = (grey > white) & (sat <= 60) & ~near
    sides = {s for s in rulers} | ({side} if side else set())
    left, top, right, bottom = _clear_of(
        paper, (left, top, right, bottom), (bx0, by0, bx1, by1), sx, sy,
        sides)
    # A label beside the specimen, rather than beyond the line, can still
    # catch the margin at a corner: the margin gives way on whichever side
    # costs it least.
    own = (bx0, by0, bx1, by1)
    for r in labels:
        lx0, ly0, lx1, ly1 = (r["paper"][0] * sx, r["paper"][1] * sy,
                              r["paper"][2] * sx, r["paper"][3] * sy)
        if _boxes_overlap(own, (lx0, ly0, lx1, ly1)):
            continue
        if not _boxes_overlap((left, top, right, bottom), (lx0, ly0, lx1, ly1)):
            continue
        options = []
        if lx0 >= bx1:
            options.append((right - lx0, "right", max(int(np.ceil(bx1)), int(lx0))))
        if lx1 <= bx0:
            options.append((lx1 - left, "left", min(int(bx0), int(np.ceil(lx1)))))
        if ly0 >= by1:
            options.append((bottom - ly0, "bottom", max(int(np.ceil(by1)), int(ly0))))
        if ly1 <= by0:
            options.append((ly1 - top, "top", min(int(by0), int(np.ceil(ly1)))))
        if options:
            _, edge, value = min(options)
            if edge == "right":
                right = value
            elif edge == "left":
                left = value
            elif edge == "bottom":
                bottom = value
            else:
                top = value
    return (max(0, left), max(0, top), min(w, right), min(h, bottom))


RULES = {"threshold": crop_box, "fixed": fixed_box, "segment": segment_box}


def pad_to_square(rgb):
    """A new square copy of rgb, padded with the tray's colour.

    The colour is each channel's median over rgb's border pixels. The padding
    is split evenly, the odd pixel after, so the subject stays centred and
    keeps its shape when the square is resized.
    """
    _check_rgb(rgb, smallest=1)
    h, w = rgb.shape[:2]
    side = max(h, w)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    out = np.empty((side, side, 3), dtype=np.uint8)
    out[:] = np.round(np.median(border, axis=0)).astype(np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = rgb
    return out


def square_and_resize(crop, size=SIZE):
    """A new size by size uint8 image: crop padded to a square and resized."""
    square = Image.fromarray(pad_to_square(crop))
    resized = square.resize((size, size), Image.BILINEAR)
    return np.array(resized, dtype=np.uint8)


def crop_and_resize(rgb, size=SIZE, rule="threshold"):
    """A new size by size uint8 image of the specimen, by the named rule."""
    if rule not in RULES:
        raise ValueError(
            f"unknown crop rule {rule!r}, expected one of {sorted(RULES)}"
        )
    left, top, right, bottom = RULES[rule](rgb)
    return square_and_resize(rgb[top:bottom, left:right], size)


def load_photograph(path):
    """The photograph at path as a new RGB uint8 array."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def _crop_file(job):
    """Load and crop one photograph; job is (path, size, rule), for a pool."""
    path, size, rule = job
    return crop_and_resize(load_photograph(path), size, rule)


def build_array(paths, size=SIZE, workers=8, rule="threshold"):
    """The working array: one cropped size by size image per path, in order.

    Returns a new uint8 array of shape (len(paths), size, size, 3). With
    more than one worker the photographs are cropped in spawned processes,
    which gives the same bytes as one process.
    """
    if rule not in RULES:
        raise ValueError(
            f"unknown crop rule {rule!r}, expected one of {sorted(RULES)}"
        )
    jobs = [(Path(p), size, rule) for p in paths]
    out = np.empty((len(jobs), size, size, 3), dtype=np.uint8)
    if workers <= 1 or len(jobs) < 2:
        for k, job in enumerate(jobs):
            out[k] = _crop_file(job)
        return out
    # Spawned workers behave the same under a script, under Jupyter and on
    # Windows or macOS, where forking a threaded process is unsafe or absent.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for k, image in enumerate(pool.map(_crop_file, jobs, chunksize=16)):
            out[k] = image
    return out


def held_out_count(n):
    """The number of held-out images: a tenth of n, rounded, halves up."""
    return int(math.floor(HELD_OUT_FRACTION * n + 0.5))


def held_out_names(names, seed=SEED):
    """The held-out file names, as a set.

    They are the first held_out_count of NumPy's default_rng(seed)
    permutation over the names sorted, which is the fixed manifest's own
    order, so the order the names arrive in does not move the split.
    """
    ordered = sorted(names)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    return {ordered[i] for i in perm[:held_out_count(len(ordered))]}


def read_crop_failures(path):
    """The file names a verdicts CSV marks "fail", as a set.

    The CSV needs file_name and verdict columns. A doubtful crop, or any
    other verdict, is not a failure.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not {"file_name", "verdict"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path} needs file_name and verdict columns")
        return {
            row["file_name"]
            for row in reader
            if row["verdict"].strip().lower() == "fail"
        }


def row_table(names, failures, held_out, exclude=True):
    """One row per manifest name, in order, saying where it went.

    Each row has the manifest row, the file name, the part (training or
    held_out), crop_failed (1 or 0), in_array (1 or 0) and array_row, which
    is blank for a row left out. With exclude, a crop failure is left out
    of the array; without it, it is kept and only flagged.
    """
    table, next_row = [], 0
    for k, name in enumerate(names):
        failed = name in failures
        kept = not (failed and exclude)
        table.append({
            "row": k,
            "file_name": name,
            "part": "held_out" if name in held_out else "training",
            "crop_failed": int(failed),
            "in_array": int(kept),
            "array_row": next_row if kept else "",
        })
        next_row += kept
    return table


def totals(table):
    """Each part's count with and without the crop failures, and the array's
    row count."""
    counts = {}
    for part in ("training", "held_out", "all"):
        rows = [r for r in table if part in ("all", r["part"])]
        failed = sum(r["crop_failed"] for r in rows)
        counts[part] = {
            "with_failures": len(rows),
            "failures": failed,
            "without_failures": len(rows) - failed,
        }
    counts["in_array"] = sum(r["in_array"] for r in table)
    return counts


def file_sha256(path):
    """The SHA-256 of the file at path, as hexadecimal."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_array(path, array):
    """Save array through a .part file, so a stopped build leaves none."""
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as fh:
        np.save(fh, array)
    os.replace(part, path)


def _save_table(path, table):
    """Write the row table as CSV, through a .part file as the array is."""
    part = path.with_name(path.name + ".part")
    with open(part, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(table)
    os.replace(part, path)


def main(argv=None):
    """Build the working array and its table from the fixed manifest.

    Returns 0 on success and 1 on a refusal, when nothing is written.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser(
        "build", help="crop every photograph and save the array"
    )
    build.add_argument(
        "--manifest", default="data/manifest_fixed_20261013.csv"
    )
    build.add_argument("--raw-dir", default="data/raw")
    build.add_argument("--out", default="data/nhm_butterflies_64.npy")
    build.add_argument("--size", type=int, default=SIZE)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument(
        "--rule", required=True, choices=sorted(RULES),
        help="the threshold rule, the fixed box or the segment rule",
    )
    build.add_argument(
        "--crop-failures",
        help="a verdicts CSV; its rows marked fail are the crop failures",
    )
    build.add_argument(
        "--failures", choices=("exclude", "include"), default="exclude",
        help="leave crop failures out of the array (default) or keep them",
    )
    build.add_argument(
        "--table",
        help="the row table, by default beside the array as _rows.csv",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    table_path = (
        Path(args.table)
        if args.table
        else out.with_name(out.stem + "_rows.csv")
    )
    for path in (out, table_path):
        if path.exists():
            print(
                f"{path} exists, and I never overwrite the working array "
                "or its table.",
                file=sys.stderr,
            )
            return 1
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        names = [row["file_name"] for row in csv.DictReader(fh)]
    raw = Path(args.raw_dir)
    missing = [raw / n for n in names if not (raw / n).is_file()]
    if missing:
        print(
            f"{len(missing)} photographs in the manifest are not on disk, "
            f"the first {missing[0]}. Nothing was written.",
            file=sys.stderr,
        )
        return 1
    failures = (
        read_crop_failures(args.crop_failures) if args.crop_failures else set()
    )
    # A failure that names no manifest row would flag nothing, so a list for
    # another manifest is refused rather than quietly ignored.
    stray = sorted(failures - set(names))
    if stray:
        print(
            f"{len(stray)} crop failures are not in the manifest, the first "
            f"{stray[0]}. Nothing was written.",
            file=sys.stderr,
        )
        return 1

    exclude = args.failures == "exclude"
    table = row_table(names, failures, held_out_names(names), exclude)
    kept = [raw / r["file_name"] for r in table if r["in_array"]]
    start = time.perf_counter()
    array = build_array(
        kept, size=args.size, workers=args.workers, rule=args.rule
    )
    seconds = time.perf_counter() - start
    _save_array(out, array)
    _save_table(table_path, table)
    counts = totals(table)
    print(
        f"Built {len(kept)} images into {out} by the {args.rule} rule: "
        f"shape {array.shape}, {out.stat().st_size} bytes, SHA-256 "
        f"{file_sha256(out)}, {seconds:.1f} s with {args.workers} worker "
        f"processes."
    )
    n = counts["all"]["failures"]
    if args.crop_failures:
        where = "left out of" if exclude else "kept in"
        print(
            f"{n} crop failure{'s' if n != 1 else ''} flagged in "
            f"{table_path}, {where} the array."
        )
    else:
        print(f"No crop-failure list was given, so {table_path} flags no row.")
    for part, label in (("training", "Training"), ("held_out", "Held-out")):
        c = counts[part]
        print(
            f"{label} images: {c['with_failures']} with the crop failures, "
            f"{c['without_failures']} without."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
