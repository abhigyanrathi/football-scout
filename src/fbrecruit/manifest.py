import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from fbrecruit.paths import MANIFEST


def file_sizes(paths, root):
    return {Path(p).relative_to(root).as_posix(): Path(p).stat().st_size for p in sorted(paths)}


def record(source, sources, files):
    """Write one source's provenance into data/manifest.json, keeping the other sources."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[source] = {
        "sources": sources,
        "retrieved_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_sizes": files,
        "versions": {p: version(p) for p in ("socceraction", "statsbombpy", "pandas")},
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
