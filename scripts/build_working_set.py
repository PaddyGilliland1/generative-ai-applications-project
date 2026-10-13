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

Then the crop is resized to 64 by 64 with Pillow's bilinear filter, and the
working array holds one row per photograph of the fixed manifest, in the
manifest's order, as uint8.

    python scripts/build_working_set.py build \\
        --manifest data/manifest_fixed_20261013.csv --raw-dir data/raw \\
        --out data/nhm_butterflies_64.npy --workers 8

Needs NumPy and Pillow. It never overwrites an existing array.
"""

import argparse
import csv
import hashlib
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


def _check_rgb(rgb):
    """Refuse anything but an RGB uint8 array of shape (height, width, 3)."""
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
        or min(rgb.shape[:2]) < 2
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


def crop_and_resize(rgb, size=SIZE):
    """A new size by size uint8 image of the specimen, by the crop rule."""
    box = crop_box(rgb)
    crop = Image.fromarray(rgb).crop(box)
    return np.array(crop.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def load_photograph(path):
    """The photograph at path as a new RGB uint8 array."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def _crop_file(job):
    """Load and crop one photograph; job is (path, size), for the pool."""
    path, size = job
    return crop_and_resize(load_photograph(path), size)


def build_array(paths, size=SIZE, workers=8):
    """The working array: one cropped size by size image per path, in order.

    Returns a new uint8 array of shape (len(paths), size, size, 3). With
    more than one worker the photographs are cropped in spawned processes,
    which gives the same bytes as one process.
    """
    jobs = [(Path(p), size) for p in paths]
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


def file_sha256(path):
    """The SHA-256 of the file at path, as hexadecimal."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    """Build the working array from the fixed manifest.

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
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        print(
            f"{out} exists, and I never overwrite the working array.",
            file=sys.stderr,
        )
        return 1
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        names = [row["file_name"] for row in csv.DictReader(fh)]
    paths = [Path(args.raw_dir) / name for name in names]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(
            f"{len(missing)} photographs in the manifest are not on disk, "
            f"the first {missing[0]}. Nothing was written.",
            file=sys.stderr,
        )
        return 1

    start = time.perf_counter()
    array = build_array(paths, size=args.size, workers=args.workers)
    seconds = time.perf_counter() - start
    part = out.with_name(out.name + ".part")
    with open(part, "wb") as fh:
        np.save(fh, array)
    os.replace(part, out)
    print(
        f"Built {len(paths)} images into {out}: shape {array.shape}, "
        f"{out.stat().st_size} bytes, SHA-256 {file_sha256(out)}, "
        f"{seconds:.1f} s with {args.workers} worker processes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
