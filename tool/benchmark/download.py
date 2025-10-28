from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve


MANIFEST_PATH = Path(__file__).with_name("checkpoints_manifest.json")
DEFAULT_CHECKPOINTS_DIR = Path("output/benchmark/checkpoints")


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def verify_file(path: Path, expected_sha256: Optional[str], expected_size: Optional[int]) -> Tuple[bool, Optional[str], Optional[int]]:
    if not path.exists():
        return False, None, None
    actual_size = path.stat().st_size
    actual_sha = compute_sha256(path)
    ok_hash = expected_sha256 is None or (actual_sha == expected_sha256)
    ok_size = expected_size is None or (actual_size == expected_size)
    return (ok_hash and ok_size), actual_sha, actual_size


def _is_gdrive_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc
    except Exception:
        return False
    return "drive.google.com" in netloc or "dropbox.com" in netloc


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    urlretrieve(url, dest)


def ensure_checkpoint(model_key: str, snapshot_filename: str, *, dest_root: Path = DEFAULT_CHECKPOINTS_DIR) -> Path:
    """Ensure a checkpoint exists locally, attempting download if a direct URL is provided.

    Manifest structure expected:
    {
      "<model_key>": {
        "snapshots": {
          "<snapshot_filename>": {"url": ..., "sha256": ..., "size_bytes": ...}
        }
      }
    }
    """
    manifest = load_manifest()
    if model_key not in manifest:
        raise KeyError(f"Model key '{model_key}' not found in manifest")
    snapshots = manifest[model_key].get("snapshots")
    if not snapshots or snapshot_filename not in snapshots:
        raise KeyError(f"Snapshot '{snapshot_filename}' not found for model '{model_key}' in manifest")

    entry = snapshots[snapshot_filename]
    url = entry.get("url")
    exp_sha = entry.get("sha256")
    exp_size = entry.get("size_bytes")

    dest_dir = dest_root / model_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / snapshot_filename

    ok, actual_sha, actual_size = verify_file(dest_path, exp_sha, exp_size)
    if ok:
        print(f"[verify] OK: {dest_path} (sha256={actual_sha}, size={actual_size})")
        return dest_path

    if url is None:
        raise FileNotFoundError(
            f"No URL provided for {model_key}/{snapshot_filename}. Please place the file at {dest_path} manually."
        )

    if _is_gdrive_url(url):
        # Avoid adding external deps; advise user to fetch with gdown/wget and place file.
        print(
            f"[note] URL points to Google Drive/Dropbox, which may require cookies/confirmation. "
            f"Please download manually (or use gdown/wget with proper flags) and place the file at {dest_path}."
        )
        raise FileNotFoundError(
            f"Manual download required for {model_key}/{snapshot_filename} from {url}."
        )

    # Direct URL attempt
    try:
        download_url(url, dest_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e

    ok, actual_sha, actual_size = verify_file(dest_path, exp_sha, exp_size)
    if not ok:
        msg = (
            f"Downloaded but verification failed for {dest_path}. "
            f"actual_sha={actual_sha}, actual_size={actual_size}, expected_sha={exp_sha}, expected_size={exp_size}."
        )
        print("[warn] " + msg)
    else:
        print(f"[verify] OK after download: {dest_path} (sha256={actual_sha}, size={actual_size})")
    return dest_path
