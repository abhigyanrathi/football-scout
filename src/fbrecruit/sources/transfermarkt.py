import sys

import pandas as pd
import requests

from fbrecruit import manifest
from fbrecruit.paths import INTERIM, RAW

BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DATE_COLUMNS = {
    "competitions": [],
    "clubs": [],
    "players": ["date_of_birth", "contract_expiration_date"],
    "games": ["date"],
    "appearances": ["date"],
    "player_valuations": ["date"],
    "transfers": ["transfer_date"],
}
RAW_DIR = RAW / "transfermarkt"
INTERIM_DIR = INTERIM / "transfermarkt"


def download(table):
    url = f"{BASE}/{table}.csv.gz"
    path = RAW_DIR / f"{table}.csv.gz"
    head = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=60)
    head.raise_for_status()
    expected = int(head.headers["Content-Length"])
    if path.exists() and path.stat().st_size == expected:
        return path
    part = path.with_suffix(".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if part.stat().st_size != expected:
        raise OSError(f"{table}: got {part.stat().st_size} bytes, host reports {expected}")
    part.replace(path)
    return path


def to_parquet(table):
    df = pd.read_csv(RAW_DIR / f"{table}.csv.gz", keep_default_na=False, na_values=[""])
    for col in DATE_COLUMNS[table]:
        df[col] = pd.to_datetime(df[col])
    df.to_parquet(INTERIM_DIR / f"{table}.parquet", index=False)
    return df.shape


def ingest():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for table in DATE_COLUMNS:
        path = download(table)
        shapes[table] = (*to_parquet(table), path.stat().st_size)
    manifest.record(
        "transfermarkt",
        [f"{BASE}/{t}.csv.gz" for t in DATE_COLUMNS],
        manifest.file_sizes(RAW_DIR.glob("*.csv.gz"), RAW_DIR),
    )
    return shapes


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("table rows columns bytes_downloaded")
    for table, (rows, cols, size) in ingest().items():
        print(table, rows, cols, size)
