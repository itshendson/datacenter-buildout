"""Download Epoch AI's data center dataset to disk, re-downloading only when it changes."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

DATA_URL = "https://epoch.ai/data/data_centers/data_centers.zip"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ZIP_PATH = DATA_DIR / "data_centers.zip"
META_PATH = DATA_DIR / "meta.json"

logger = logging.getLogger(__name__)


def refresh() -> bool:
    """Download the dataset if Epoch has a newer version.

    Returns True if the local zip was replaced.
    """
    meta = _read_meta()
    now = datetime.now(timezone.utc)

    # Send back the ETag from our last download; Epoch replies 304 if the file is unchanged.
    headers = {}
    if ZIP_PATH.exists() and meta.get("etag"):
        headers["If-None-Match"] = meta["etag"]

    try:
        response = httpx.get(DATA_URL, headers=headers, follow_redirects=True, timeout=30)
        if response.status_code != 304:
            response.raise_for_status()
    except httpx.HTTPError:
        if ZIP_PATH.exists():
            logger.warning("Could not check for dataset updates; using local copy", exc_info=True)
            return False
        raise

    meta["last_checked"] = now.isoformat()

    if response.status_code == 304:
        logger.info("Dataset unchanged")
        _write_meta(meta)
        return False

    # Write to a temp file, then rename over the real one, so a crash mid-write
    # never leaves a half-written zip behind.
    DATA_DIR.mkdir(exist_ok=True)
    tmp_path = ZIP_PATH.with_name(ZIP_PATH.name + ".tmp")
    tmp_path.write_bytes(response.content)
    os.replace(tmp_path, ZIP_PATH)

    meta["etag"] = response.headers.get("etag")
    meta["downloaded_at"] = now.isoformat()
    _write_meta(meta)
    logger.info("Downloaded new dataset version (%d bytes)", len(response.content))
    return True


def _read_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text())
    except json.JSONDecodeError:
        # Without a usable ETag we just download a fresh copy, which rewrites meta.
        logger.warning("meta.json is unreadable; ignoring it")
        return {}


def _write_meta(meta: dict) -> None:
    META_PATH.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("changed:", refresh())
