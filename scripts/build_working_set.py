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

Either way the crop is padded to a square with the tray's colour, the median
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


RULES = {"threshold": crop_box, "fixed": fixed_box}


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
        help="the threshold rule or the fixed box, as A4 rules",
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
