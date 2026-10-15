"""Download the Natural History Museum butterfly photographs the notebook uses.

Source: the Natural History Museum (NHM) Data Portal, dataset "Collection
specimens" (https://data.nhm.ac.uk/dataset/collection-specimens), resource
05ff2255-c38a-40c9-b657-4ccb55ab2feb, read through the portal's public CKAN API
(`datastore_search`, no key). The records are CC0. Each image carries its own
`license` field, and the NHM terms say images are CC BY "by default", so this
script keeps only images whose own field is the CC BY 4.0 address. Attribution,
in the form the NHM citation page asks for: "Natural History Museum (2026).
Collection specimens [Photographs]. Natural History Museum. Licensed under
CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0)." Each image's own page
is https://data.nhm.ac.uk/object/<occurrence_id>. The images are changed:
cropped and scaled for training, and a few resized first by the NHM's image
service (below). The Museum cannot warrant the quality or accuracy of the data,
and nothing here is endorsed by it.

Each image is fetched from its original address on the portal, the record's own
`identifier` (https://data.nhm.ac.uk/media/<asset id>), which serves the
portal's default JPEG. The portal also offers smaller sizes through its IIIF
Image API 3.0 service (the same address plus, for example,
`/full/!512,512/0/default.jpg`), and `--rendition` selects one. Either route
can fail for an image the other serves: the 512-pixel size can answer HTTP 500,
and the original address HTTP 503 "Failed to retrieve the requested image
data". So when one route cannot serve an image, the script tries the other
once, unless `--no-fallback` is given. Each manifest row's `rendition` column
says which address served it, and in the fixed manifest 165 rows are the
512-pixel rendition.

Steps:
1. `list`: page through the records that match a selection, keep each record
   whose one specimen photograph (category "Specimen", not the separate label
   photograph) is CC BY 4.0 by its own `license` field, and write the
   candidates CSV in a seeded uniform random order (seed 2026), so taking the
   first N is a uniform random sample. The default selection is the
   iCollections British and Irish butterflies (Paterson et al., 2016): project
   "icollections", one query for each of the six butterfly families. Records
   left out go to a second CSV with the reason.
2. `download`: fetch candidates in order until `--target` good images are on
   disk or the stop time passes, whichever comes first. At most 3 workers and a
   pause after every image. A rate limit (HTTP 429, or any answer with
   Retry-After) holds every worker, for the Retry-After where given and never
   less than an exponential backoff; a 403, 500 or other 5xx backs off only the
   worker that got it. A throughput line goes to the log every 5 minutes. Each
   file is written to `<name>.part` and renamed when complete, and the manifest
   row is written only after that, so a crash costs at most the files in
   flight. Rerunning the same command resumes: a manifest row whose file is on
   disk at the recorded size is not fetched again.
3. `verify`: re-hash every file against the manifest, check every licence, list
   orphans and part files, and count families and countries.
4. `freeze`: write the fixed manifest, sorted by file name, from the rows that
   verify, and print its SHA-256.

The manifest holds the occurrence ID, catalogue number, family, genus, species
(the specific epithet), country, the original and rendition addresses, the
licence, the file name, SHA-256, bytes, pixel size and download time (UTC). It
leaves out every field that can name a person: collectors, identifiers, donors,
photographers, the scientific-name authorship, and free-text fields such as
locality and the image title.

    python scripts/download_nhm_images.py list --out data/candidates.csv
    python scripts/download_nhm_images.py download \\
        --candidates data/candidates.csv --target 22000 --max-hours 4 \\
        2> data/download.log
    python scripts/download_nhm_images.py verify
    python scripts/download_nhm_images.py freeze \\
        --out data/manifest_fixed_20261013.csv

To rebuild the submitted set exactly, pass the fixed manifest as `--candidates`
with `--target` equal to its row count: it carries every column `download`
reads. Needs `requests` and Pillow.
"""

import argparse
import csv
import datetime as dt
import email.utils
import hashlib
import io
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image

API_URL = "https://data.nhm.ac.uk/api/3/action/datastore_search"
RESOURCE_ID = "05ff2255-c38a-40c9-b657-4ccb55ab2feb"
RENDITION = "original"
IIIF_512 = "full/!512,512/0/default.jpg"
USER_AGENT = (
    "msc-capstone-generative-ai-applications/1.0 "
    "(one-off download of NHM Lepidoptera specimen images "
    "for a student project)"
)
MAX_WORKERS = 3
# HTTP 500 answers allowed before giving up on an address: one for the
# rendition, since the original address is there to fall back on, and two for
# the original itself.
RENDITION_MAX_500 = 1
ORIGINAL_MAX_500 = 2
# The image server's answer when it cannot get one image's data at the address
# asked for. It comes with HTTP 503 on some original addresses whose IIIF 512
# rendition works.
UNAVAILABLE_MARK = b"Failed to retrieve the requested image data"
SHUFFLE_SEED = 2026

# The census choice of 13 Oct 2026 (selection.json): iCollections British and
# Irish butterflies, one query per family, because a list of families in one
# filter is ANDed and matches nothing. The licence and category filters match
# at record level, so pick_media checks each image again.
BUTTERFLY_FAMILIES = [
    "lycaenidae",
    "nymphalidae",
    "pieridae",
    "hesperiidae",
    "riodinidae",
    "papilionidae",
]
DEFAULT_SELECTION = {
    "resource_id": RESOURCE_ID,
    "queries": [
        {
            "filters": {
                "order": "Lepidoptera",
                "project": "icollections",
                "family": family,
                "associatedMedia.license": (
                    "http://creativecommons.org/licenses/by/4.0/"
                ),
                "associatedMedia.category": "Specimen",
            }
        }
        for family in BUTTERFLY_FAMILIES
    ],
}

# Only these fields are asked for, so person names never reach this machine
# from the listing.
LIST_FIELDS = [
    "occurrenceID",
    "catalogNumber",
    "family",
    "genus",
    "specificEpithet",
    "country",
    "associatedMedia",
]

CANDIDATE_FIELDS = [
    "occurrence_id",
    "catalog_number",
    "family",
    "genus",
    "species",
    "country",
    "image_url",
    "rendition",
    "download_url",
    "licence",
    "file_name",
]
MANIFEST_FIELDS = CANDIDATE_FIELDS + [
    "sha256",
    "bytes",
    "width",
    "height",
    "download_utc",
]

log = logging.getLogger("download_nhm_images")


class DownloadError(Exception):
    """A download that failed for a reason the run records and moves past."""


class ServerError(DownloadError):
    """HTTP 500 again and again for one address: the server cannot make that
    image, so stop asking."""


def is_cc_by_4(value) -> bool:
    """True when value is the CC BY 4.0 licence address, with or without its
    scheme or a trailing slash."""
    if not isinstance(value, str):
        return False
    bare = re.sub(r"^https?://", "", value.strip().lower()).rstrip("/")
    return bare == "creativecommons.org/licenses/by/4.0"


def rendition_url(identifier: str, rendition: str = RENDITION) -> str:
    """The address that serves identifier's image at rendition: the identifier
    itself for the original, or it with the IIIF size path added."""
    if rendition == "original":
        return identifier
    return f"{identifier.rstrip('/')}/{rendition}"


def apply_rendition(rows, rendition: str) -> list:
    """The same rows set to download the original address or a IIIF size path.
    `as-listed` keeps each row's own address, which is how a rerun from the
    fixed manifest gets the same files."""
    if rendition == "as-listed":
        return list(rows)
    return [
        dict(
            r,
            rendition=rendition,
            download_url=rendition_url(r["image_url"], rendition),
        )
        for r in rows
    ]


def _text(value) -> str:
    """value as one line of text with its white space collapsed, or an empty
    string for None."""
    return " ".join(str(value).split()) if value is not None else ""


def pick_media(record: dict):
    """(media, None) for the record's one CC BY 4.0 specimen photograph, else
    (None, the reason)."""
    if not _text(record.get("occurrenceID")):
        return None, "no occurrence ID"
    specimen = [
        m
        for m in record.get("associatedMedia") or []
        if str(m.get("category") or "").lower() == "specimen"
    ]
    # iCollections records pair a specimen photograph with a label photograph.
    # More than one specimen photograph would break one image per specimen, so
    # those records are left out.
    if len(specimen) != 1:
        return None, f"{len(specimen)} specimen images"
    media = specimen[0]
    if not is_cc_by_4(media.get("license")):
        return None, f"licence is {media.get('license')!r}"
    if not media.get("identifier"):
        return None, "no image address"
    if media.get("format") not in (None, "", "image/jpeg"):
        return None, f"format is {media.get('format')!r}"
    return media, None


def record_to_row(record: dict, rendition: str = RENDITION):
    """One candidate row from an API record, or None when it has no single CC
    BY 4.0 specimen photograph."""
    media, _ = pick_media(record)
    if media is None:
        return None
    occurrence = _text(record.get("occurrenceID"))
    catalogue = _text(record.get("catalogNumber"))
    stem = re.sub(r"[^A-Za-z0-9-]", "-", catalogue or "nocatalogue")
    return {
        "occurrence_id": occurrence,
        "catalog_number": catalogue,
        "family": _text(record.get("family")),
        "genus": _text(record.get("genus")),
        "species": _text(record.get("specificEpithet")),
        "country": _text(record.get("country")),
        "image_url": media["identifier"],
        "rendition": rendition,
        "download_url": rendition_url(media["identifier"], rendition),
        "licence": media["license"],
        "file_name": f"{stem}_{re.sub(r'[^A-Za-z0-9]', '', occurrence)}.jpg",
    }


def select_rows(records, rendition: str = RENDITION, skipped=None) -> list:
    """Candidate rows in input order: CC BY 4.0 only, one image per specimen,
    no repeated image.

    Records left out are appended to `skipped`, when given, with the reason."""
    rows, seen = [], set()
    for record in records:
        media, reason = pick_media(record)
        row = record_to_row(record, rendition) if media is not None else None
        if row is not None:
            keys = {
                ("occ", row["occurrence_id"]),
                ("url", row["download_url"]),
                ("file", row["file_name"]),
            }
            if keys & seen:
                row, reason = None, "repeat of an earlier record"
            else:
                seen |= keys
                rows.append(row)
        if row is None and skipped is not None:
            skipped.append(
                {
                    "occurrence_id": _text(record.get("occurrenceID")),
                    "catalog_number": _text(record.get("catalogNumber")),
                    "reason": reason,
                }
            )
    return rows


def shuffle_rows(rows, seed: int) -> list:
    """A seeded uniform shuffle that does not depend on the order the API
    returned the rows in."""
    ordered = sorted(rows, key=lambda r: r["occurrence_id"])
    random.Random(seed).shuffle(ordered)
    return ordered


def parse_retry_after(value, now=None):
    """Seconds to wait from a Retry-After header, in either of its two forms,
    or None."""
    if value is None:
        return None
    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


def backoff_delay(
    attempt: int, base: float = 2.0, cap: float = 300.0
) -> float:
    """Seconds to wait after try attempt fails: base doubled for each attempt,
    never over cap."""
    return min(cap, base * (2**attempt))


class Cooldown:
    """A pause shared by every worker: one refusal holds them all, not just the
    one refused."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        """A pause that has already ended, with the clock and the sleep passed
        in, so a test can supply its own."""
        self.clock, self.sleep = clock, sleep
        self.lock = threading.Lock()
        self.until = clock()
        self.history = []

    def push(self, seconds: float):
        """Hold every worker for at least seconds from now, and record the
        pause."""
        with self.lock:
            self.history.append(seconds)
            self.until = max(self.until, self.clock() + seconds)

    def wait(self):
        """Return once the shared pause has passed, sleeping until then."""
        while True:
            with self.lock:
                left = self.until - self.clock()
            if left <= 0:
                return
            self.sleep(left)


class Stats:
    """Thread-safe counts of the answers a run receives, by HTTP status or kind
    of error."""

    def __init__(self):
        """Empty counts."""
        self.lock = threading.Lock()
        self.status = Counter()

    def count(self, status):
        """Count one answer with this status."""
        with self.lock:
            self.status[status] += 1

    @property
    def requests(self) -> int:
        """Every answer counted, refusals and errors included."""
        return sum(self.status.values())

    @property
    def refused(self) -> int:
        """The answers that were refusals: 403, 429 or any 5xx."""
        return sum(n for s, n in self.status.items() if is_refusal(s))


def is_refusal(status) -> bool:
    """True when status is a refusal the run backs off from: 403, 429 or any
    5xx."""
    # Cloudflare sits in front of the portal and can refuse with 403, so I
    # treat 403 like 429 and 5xx.
    return isinstance(status, int) and (
        status in (403, 429) or 500 <= status < 600
    )


def get_with_retry(
    session,
    url,
    cooldown,
    stats,
    params=None,
    max_tries=6,
    base=2.0,
    cap=300.0,
    timeout=60,
    max_500=2,
):
    """The 200 response for url, as a requests Response, after waiting out
    refusals.

    A rate limit holds every worker through cooldown, and any other refusal
    only this one. A 500 seen max_500 times, or the server's answer that it
    cannot get the image's data, raises ServerError. Anything else that
    fails raises DownloadError after max_tries."""
    last, count_500 = None, 0
    for attempt in range(max_tries):
        cooldown.wait()
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except OSError as exc:
            # requests' own exceptions are OSErrors too, so this covers resets
            # and timeouts.
            stats.count("error")
            last = f"{type(exc).__name__}: {exc}"
            cooldown.sleep(backoff_delay(attempt, base, cap))
            continue
        stats.count(resp.status_code)
        if resp.status_code == 200:
            return resp
        if not is_refusal(resp.status_code):
            raise DownloadError(f"HTTP {resp.status_code}")
        last = f"HTTP {resp.status_code}"
        if UNAVAILABLE_MARK in (resp.content or b"")[:500]:
            stats.count("unavailable")
            raise ServerError(
                f"HTTP {resp.status_code}: image data unavailable at this "
                "address"
            )
        retry_after = parse_retry_after(resp.headers.get("Retry-After"))
        wait = max(retry_after or 0.0, backoff_delay(attempt, base, cap))
        # Only a rate limit holds every worker: a 429, or any answer carrying
        # Retry-After. A 500, 403 or other 5xx is about this request, so only
        # the worker that got it waits.
        if resp.status_code == 429 or retry_after is not None:
            log.warning(
                "HTTP %s, all workers wait %.1f s (try %d of %d)",
                resp.status_code,
                wait,
                attempt + 1,
                max_tries,
            )
            cooldown.push(wait)
            continue
        if resp.status_code == 500:
            count_500 += 1
            if count_500 >= max_500:
                raise ServerError(f"HTTP 500 {count_500} times")
        log.warning(
            "HTTP %s, this worker waits %.1f s (try %d of %d)",
            resp.status_code,
            wait,
            attempt + 1,
            max_tries,
        )
        cooldown.sleep(wait)
    raise DownloadError(f"gave up after {max_tries} tries, last {last}")


def fetch(
    session,
    url,
    cooldown,
    stats,
    max_tries=6,
    base=2.0,
    cap=300.0,
    timeout=60,
    max_500=2,
) -> bytes:
    """The image bytes at url, by get_with_retry, or DownloadError when the
    answer is not an image."""
    resp = get_with_retry(
        session,
        url,
        cooldown,
        stats,
        max_tries=max_tries,
        base=base,
        cap=cap,
        timeout=timeout,
        max_500=max_500,
    )
    kind = resp.headers.get("Content-Type", "")
    if not kind.startswith("image/"):
        raise DownloadError(f"not an image: Content-Type {kind!r}")
    return resp.content


def make_session() -> requests.Session:
    """A new requests Session that names this project in its User-Agent."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


class ThreadLocalSession:
    """One requests.Session per worker thread, since a Session is not promised
    to be thread safe."""

    def __init__(self, factory=make_session):
        """Sessions made by factory, one for each thread, on first use."""
        self.factory = factory
        self.local = threading.local()

    def get(self, *args, **kwargs):
        """The answer to session.get on this thread's own session."""
        session = getattr(self.local, "session", None)
        if session is None:
            session = self.local.session = self.factory()
        return session.get(*args, **kwargs)


def list_records(
    session,
    selection,
    cooldown,
    stats,
    page_size=1000,
    max_records=None,
    pause=2.0,
    sleep=time.sleep,
    max_tries=6,
):
    """Yield API records for the selection, paging with the API's `after`
    token."""
    params = {
        "resource_id": selection.get("resource_id", RESOURCE_ID),
        "limit": page_size,
        "filters": json.dumps(selection.get("filters", {})),
        "fields": ",".join(LIST_FIELDS),
    }
    if selection.get("q"):
        params["q"] = selection["q"]
    after, yielded = None, 0
    while True:
        page = dict(params)
        if after is not None:
            page["after"] = json.dumps(after)
        payload = get_with_retry(
            session, API_URL, cooldown, stats, params=page, max_tries=max_tries
        ).json()
        if not payload.get("success", True):
            raise DownloadError(f"API error: {payload.get('error')}")
        result = payload["result"]
        records = result.get("records") or []
        for rec in records:
            yield rec
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return
        after = result.get("after")
        if not records or after is None:
            return
        sleep(pause)


def iter_selection(session, selection, cooldown, stats, **kwargs):
    """Records for every query in a selection, one query after another."""
    queries = selection.get("queries") or [
        {"filters": selection.get("filters", {}), "q": selection.get("q")}
    ]
    for query in queries:
        one = {
            "resource_id": selection.get("resource_id", RESOURCE_ID),
            "filters": query.get("filters", {}),
            "q": query.get("q"),
        }
        log.info("listing %s", json.dumps(one["filters"]))
        yield from list_records(session, one, cooldown, stats, **kwargs)


class Manifest:
    """Append-only CSV ledger. A row is written only once its file is complete
    on disk."""

    def __init__(self, path, fields=MANIFEST_FIELDS):
        """A ledger at path with these CSV fields."""
        self.path = Path(path)
        self.fields = fields
        self.lock = threading.Lock()

    def append(self, row: dict):
        """Append row, flushed and synced to disk, with the header first when
        the file is new."""
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists() or self.path.stat().st_size == 0
            with open(self.path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=self.fields, extrasaction="ignore"
                )
                if new:
                    writer.writeheader()
                writer.writerow(row)
                fh.flush()
                os.fsync(fh.fileno())

    def load(self) -> dict:
        """file_name -> the last complete row for it. A row cut short by a
        crash is skipped."""
        if not self.path.exists():
            return {}
        rows = {}
        with open(self.path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256") or "")
                    or not (row.get("bytes") or "").isdigit()
                ):
                    continue
                if not row.get("download_utc"):
                    continue
                rows[row["file_name"]] = row
        return rows


def read_csv(path) -> list:
    """Every row of the CSV file at path, as a list of dicts."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows, fields):
    """Write rows to path as CSV with these fields, making its folder if it is
    missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def utc_now() -> str:
    """The time now in UTC, as an ISO 8601 string to the second."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def download_one(
    row,
    session,
    raw_dir: Path,
    cooldown,
    stats,
    removed: list,
    fallback=True,
    **retry,
) -> dict:
    """Fetch, check, and save one image. If the server cannot make it at the
    address asked for, the other route is tried once: the original address for
    a IIIF size, the IIIF 512 size for an original. The row then records the
    address and rendition it actually came from."""
    on_original = row["download_url"] == row["image_url"]
    try:
        body = fetch(
            session,
            row["download_url"],
            cooldown,
            stats,
            **(
                dict(retry, max_500=ORIGINAL_MAX_500) if on_original else retry
            ),
        )
    except ServerError as exc:
        if not fallback:
            raise
        other = IIIF_512 if on_original else "original"
        other_url = rendition_url(row["image_url"], other)
        log.info("%s: %s, trying %s", row["file_name"], exc, other_url)
        body = fetch(
            session,
            other_url,
            cooldown,
            stats,
            **(
                retry if on_original else dict(retry, max_500=ORIGINAL_MAX_500)
            ),
        )
        row = dict(row, download_url=other_url, rendition=other)
    try:
        image = Image.open(io.BytesIO(body))
        image.load()
    except Exception as exc:
        raise DownloadError(f"not a readable image: {exc}") from exc
    if image.format != "JPEG":
        raise DownloadError(f"not a JPEG: {image.format}")
    final = raw_dir / row["file_name"]
    part = final.with_name(final.name + ".part")
    try:
        with open(part, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(part, final)
    except OSError as exc:
        # The part file is this run's own, written just above, so removing it
        # is safe.
        if part.exists():
            part.unlink()
            removed.append(str(part))
        raise DownloadError(f"write failed: {exc}") from exc
    width, height = image.size
    return dict(
        row,
        sha256=hashlib.sha256(body).hexdigest(),
        bytes=len(body),
        width=width,
        height=height,
        download_utc=utc_now(),
    )


def run(
    rows,
    target,
    raw_dir,
    manifest_path,
    session,
    workers=MAX_WORKERS,
    pause=1.0,
    deadline=None,
    clock=time.monotonic,
    sleep=time.sleep,
    max_tries=6,
    base=2.0,
    cap=300.0,
    max_500=RENDITION_MAX_500,
    fallback=True,
    max_consecutive_failures=50,
    stop_event=None,
    progress_every=1000,
    report_every=300.0,
) -> dict:
    """Download candidates in order until `target` good images exist, the
    deadline passes, or a stop."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(manifest_path)
    on_disk = {
        name
        for name, r in manifest.load().items()
        if (raw_dir / name).is_file()
        and (raw_dir / name).stat().st_size == int(r["bytes"])
    }
    todo = [r for r in rows if r["file_name"] not in on_disk]
    cooldown = Cooldown(clock=clock, sleep=sleep)
    stats = Stats()
    retry = {
        "max_tries": max_tries,
        "base": base,
        "cap": cap,
        "max_500": max_500,
    }
    lock = threading.Lock()
    pending = iter(todo)
    failures, removed = [], []
    state = {
        "done": len(on_disk),
        "new": 0,
        "in_flight": 0,
        "consecutive": 0,
        "fallbacks": 0,
        "stop_reason": None,
    }
    started = clock()
    window = {"at": started, "new": 0, "requests": 0, "refused": 0}
    log.info(
        "start: %d already on disk, %d candidates to try, target %d, "
        "%d workers, pause %.2f s",
        len(on_disk),
        len(todo),
        target,
        workers,
        pause,
    )

    def report():
        """Log one throughput line when report_every seconds have passed since
        the last. The caller holds the lock."""
        # Called under the lock: one throughput line per report_every seconds.
        now = clock()
        if now - window["at"] < report_every:
            return
        span, total = now - window["at"], max(now - started, 1e-9)
        images, requests_ = (
            state["new"] - window["new"],
            stats.requests - window["requests"],
        )
        refused = stats.refused - window["refused"]
        log.info(
            "throughput: %.2f images/s over the last %.0f s (%d images, "
            "%d requests, %d refused, %d rate-limited in all), %.2f images/s "
            "since start, %d done of %d, %d failed",
            images / span,
            span,
            images,
            requests_,
            refused,
            stats.status[429],
            state["new"] / total,
            state["done"],
            target,
            len(failures),
        )
        window.update(
            at=now,
            new=state["new"],
            requests=stats.requests,
            refused=stats.refused,
        )

    def claim():
        """The next candidate to download, counted as in flight, or None when
        the run should stop."""
        with lock:
            report()
            if state["stop_reason"]:
                return None
            if stop_event is not None and stop_event.is_set():
                state["stop_reason"] = "stop requested"
                return None
            # A worker that finds every remaining slot in flight leaves; if one
            # of those fails, the worker that held it comes back here and takes
            # the next candidate.
            if state["done"] + state["in_flight"] >= target:
                return None
            if deadline is not None and clock() >= deadline:
                state["stop_reason"] = "time limit"
                return None
            row = next(pending, None)
            if row is not None:
                state["in_flight"] += 1
            return row

    def worker():
        """Download claimed candidates one at a time until claim says stop,
        recording each result or failure."""
        while True:
            row = claim()
            if row is None:
                return
            try:
                result = download_one(
                    row,
                    session,
                    raw_dir,
                    cooldown,
                    stats,
                    removed,
                    fallback=fallback,
                    **retry,
                )
            except Exception as exc:
                if not isinstance(exc, DownloadError):
                    log.exception("unexpected error on %s", row["file_name"])
                with lock:
                    state["in_flight"] -= 1
                    state["consecutive"] += 1
                    failures.append(
                        {
                            "file_name": row["file_name"],
                            "download_url": row["download_url"],
                            "error": str(exc),
                        }
                    )
                    if (
                        state["consecutive"] >= max_consecutive_failures
                        and not state["stop_reason"]
                    ):
                        state["stop_reason"] = "persistent failures"
                log.warning("failed %s: %s", row["file_name"], exc)
            else:
                manifest.append(result)
                with lock:
                    state["in_flight"] -= 1
                    state["done"] += 1
                    state["new"] += 1
                    state["consecutive"] = 0
                    state["fallbacks"] += (
                        result["download_url"] != row["download_url"]
                    )
                    new, done = state["new"], state["done"]
                if new % progress_every == 0:
                    elapsed = max(clock() - started, 1e-9)
                    log.info(
                        "progress: %d done (%d this run), %.2f images/s, "
                        "%d requests, %d refused, %d failed, "
                        "%d from the original address",
                        done,
                        new,
                        new / elapsed,
                        stats.requests,
                        stats.refused,
                        len(failures),
                        state["fallbacks"],
                    )
            sleep(pause)

    threads = [
        threading.Thread(target=worker, name=f"worker-{k}", daemon=True)
        for k in range(workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        # Short joins keep the main thread free to take a signal.
        while t.is_alive():
            t.join(0.5)

    stop_reason = state["stop_reason"] or (
        "target reached" if state["done"] >= target else "candidates exhausted"
    )
    elapsed = clock() - started
    summary = {
        "stop_reason": stop_reason,
        "done": state["done"],
        "downloaded_this_run": state["new"],
        "skipped_existing": len(rows) - len(todo),
        "failures": failures,
        "fallbacks": state["fallbacks"],
        "removed_partials": removed,
        "requests": stats.requests,
        "refused": stats.refused,
        "status_counts": {
            str(k): v
            for k, v in sorted(stats.status.items(), key=lambda kv: str(kv[0]))
        },
        "elapsed_s": round(elapsed, 1),
        "images_per_s": (
            round(state["new"] / elapsed, 3) if elapsed > 0 else None
        ),
    }
    log.info(
        "stop: %s; %d done, %d this run, %d failed, %d requests, "
        "%d refused, %.0f s",
        stop_reason,
        state["done"],
        state["new"],
        len(failures),
        stats.requests,
        stats.refused,
        elapsed,
    )
    return summary


def sha256_file(path) -> str:
    """The SHA-256 of the file at path, read a mebibyte at a time, as a hex
    string."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_row(row, raw_dir: Path):
    """What is wrong with row's file in raw_dir against the manifest, as a
    string, or None when it verifies."""
    path = raw_dir / row["file_name"]
    if not path.is_file():
        return "file missing"
    if path.stat().st_size != int(row["bytes"]):
        return "size differs from the manifest"
    if sha256_file(path) != row["sha256"]:
        return "hash differs from the manifest"
    if not is_cc_by_4(row["licence"]):
        return f"licence is {row['licence']!r}"
    return None


def verify(manifest_path, raw_dir) -> dict:
    """A report on the manifest's rows against the files in raw_dir, as a dict:
    hash and licence counts, families, countries and every problem found."""
    raw_dir = Path(raw_dir)
    rows = Manifest(manifest_path).load()
    problems, good = [], []
    for name, row in sorted(rows.items()):
        issue = _check_row(row, raw_dir)
        if issue:
            problems.append(f"{name}: {issue}")
        else:
            good.append(row)
    on_disk = {p.name for p in raw_dir.glob("*.jpg")}
    problems += [
        f"orphan file not in the manifest: {n}"
        for n in sorted(on_disk - set(rows))
    ]
    problems += [
        f"part file left: {p.name}" for p in sorted(raw_dir.glob("*.part"))
    ]
    return {
        "rows": len(rows),
        "hash_ok": len(good),
        "licence_cc_by_4": sum(
            is_cc_by_4(r["licence"]) for r in rows.values()
        ),
        "bytes_total": sum(int(r["bytes"]) for r in good),
        "family": dict(
            Counter(r["family"] or "(blank)" for r in good).most_common()
        ),
        "country": dict(
            Counter(r["country"] or "(blank)" for r in good).most_common()
        ),
        "problems": problems,
    }


def freeze(manifest_path, raw_dir, out_path) -> dict:
    """Write the fixed manifest from the rows that verify, sorted by file name.
    Never overwrites."""
    out_path, raw_dir = Path(out_path), Path(raw_dir)
    if out_path.exists():
        raise DownloadError(
            f"{out_path} already exists; the fixed set is written once"
        )
    kept, excluded = [], []
    for name, row in sorted(Manifest(manifest_path).load().items()):
        issue = _check_row(row, raw_dir)
        if issue:
            excluded.append({"file_name": name, "reason": issue})
        else:
            kept.append(row)
    write_csv(out_path, kept, MANIFEST_FIELDS)
    return {
        "path": str(out_path),
        "rows": len(kept),
        "sha256": sha256_file(out_path),
        "excluded": excluded,
    }


def _stop_at(args):
    """The run's deadline on the monotonic clock and the UTC stop time, from
    --stop-at-utc or --max-hours, as (deadline, when), or (None, None) with
    no limit."""
    if args.stop_at_utc:
        when = dt.datetime.fromisoformat(
            args.stop_at_utc.replace("Z", "+00:00")
        )
        seconds = (when - dt.datetime.now(dt.timezone.utc)).total_seconds()
        return time.monotonic() + seconds, when
    if args.max_hours:
        return time.monotonic() + args.max_hours * 3600, None
    return None, None


def main(argv=None) -> int:
    """Run the command argv names, list, download, verify or freeze, and return
    the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="write the candidates CSV for a selection")
    p.add_argument(
        "--selection",
        help="JSON file with `filters` or `queries`, and optional "
        "`resource_id`; defaults to the iCollections British and Irish "
        "butterflies",
    )
    p.add_argument("--out", default="data/candidates.csv")
    p.add_argument(
        "--skipped",
        default="data/candidates_skipped.csv",
        help="records left out, with the reason",
    )
    p.add_argument(
        "--max-records",
        type=int,
        help="stop each query after this many API records",
    )
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument(
        "--pause", type=float, default=1.0, help="seconds between API pages"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SHUFFLE_SEED,
        help="seed for the uniform random order",
    )
    p.add_argument(
        "--api-order",
        action="store_true",
        help="keep the API's order, do not shuffle",
    )

    p = sub.add_parser(
        "download",
        help="download candidates until the target or the stop time",
    )
    p.add_argument("--candidates", default="data/candidates.csv")
    p.add_argument("--target", type=int, default=22000)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument(
        "--rendition",
        default="as-listed",
        help="`as-listed` (the default: each row's own address), "
        f"`original`, or a IIIF size path such as {IIIF_512}",
    )
    p.add_argument(
        "--no-fallback",
        action="store_true",
        help="when one route cannot serve an image, list it as failed "
        "instead of trying the other",
    )
    p.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="seconds each worker waits after each image",
    )
    p.add_argument(
        "--max-hours", type=float, help="stop this many hours from now"
    )
    p.add_argument(
        "--stop-at-utc",
        help="stop at this UTC time, e.g. 2026-10-13T19:00:00Z (kept across "
        "restarts)",
    )
    p.add_argument("--summary", default="data/download_summary.json")

    p = sub.add_parser(
        "verify", help="re-hash the files and count families and countries"
    )
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--json", help="also write the report here")

    p = sub.add_parser(
        "freeze", help="write the fixed manifest from the rows that verify"
    )
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        stream=sys.stderr,
    )
    logging.Formatter.converter = time.gmtime

    if args.command == "list":
        selection = (
            json.loads(Path(args.selection).read_text(encoding="utf-8"))
            if args.selection
            else DEFAULT_SELECTION
        )
        session = make_session()
        cooldown, stats = Cooldown(), Stats()
        records = iter_selection(
            session,
            selection,
            cooldown,
            stats,
            page_size=args.page_size,
            max_records=args.max_records,
            pause=args.pause,
        )
        skipped = []
        rows = select_rows(records, skipped=skipped)
        if not args.api_order:
            rows = shuffle_rows(rows, args.seed)
        write_csv(args.out, rows, CANDIDATE_FIELDS)
        write_csv(
            args.skipped,
            skipped,
            ["occurrence_id", "catalog_number", "reason"],
        )
        log.info(
            "listed %d candidates (one CC BY 4.0 specimen image each) to %s; "
            "%d records left out (%s); %d API requests, %d refused",
            len(rows),
            args.out,
            len(skipped),
            dict(Counter(s["reason"] for s in skipped)),
            stats.requests,
            stats.refused,
        )
        return 0

    if args.command == "download":
        if not 1 <= args.workers <= MAX_WORKERS:
            parser.error(
                f"--workers must be 1 to {MAX_WORKERS}: the media server "
                "throttles, so stay polite"
            )
        rows = apply_rendition(read_csv(args.candidates), args.rendition)
        deadline, when = _stop_at(args)
        log.info(
            "candidates %s (%d rows); stop at %s",
            args.candidates,
            len(rows),
            (
                when.isoformat()
                if when
                else (
                    f"{args.max_hours} h from now"
                    if args.max_hours
                    else "target only"
                )
            ),
        )
        stop = threading.Event()

        def on_signal(signum, frame):
            """Ask the workers to finish the files in flight, then stop."""
            log.warning(
                "signal %s: finishing the files in flight, then stopping",
                signum,
            )
            stop.set()

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        summary = run(
            rows,
            args.target,
            args.raw_dir,
            args.manifest,
            ThreadLocalSession(),
            workers=args.workers,
            pause=args.pause,
            deadline=deadline,
            stop_event=stop,
            fallback=not args.no_fallback,
        )
        summary["finished_utc"] = utc_now()
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(summary, indent=1), encoding="utf-8"
        )
        return 0

    if args.command == "verify":
        report = verify(args.manifest, args.raw_dir)
        text = json.dumps(report, indent=1)
        if args.json:
            Path(args.json).write_text(text, encoding="utf-8")
        print(text)
        return 0 if not report["problems"] else 1

    if args.command == "freeze":
        result = freeze(args.manifest, args.raw_dir, args.out)
        print(json.dumps(result, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
