"""Daily discharge per gauging station from Rijkswaterstaat's Waterwebservices (DDAPI 2.0).

The discharge files in ``data/02_context/discharge`` end in 2024; the SAM surveys run to
2026. This fetches the 10-minute discharge (Grootheid Q) month by month and stores the
daily mean per station in ``data/02_context/discharge_daily/<code>.parquet``
(columns ``date``, ``discharge_m3s``).

Run: ``uv run python scripts/data_prep/fetch_discharge.py --start 2022-01-01
        [--stations lobith.bovenrijn.tolkamer olst megen.maas]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://ddapi20-waterwebservices.rijkswaterstaat.nl"
    "/ONLINEWAARNEMINGENSERVICES/OphalenWaarnemingen"
)
OUT = Path("data/02_context/discharge_daily")

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2022-01-01")
ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
ap.add_argument(
    "--stations",
    nargs="*",
    default=[
        "lobith.bovenrijn.tolkamer",
        "pannerden.pannerdenschkanaal",
        "olst",
        "megen.maas",
        "venlo",
    ],
)
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)


def month(code: str, a: pd.Timestamp, b: pd.Timestamp) -> list[tuple[str, float]]:
    body = {
        "Locatie": {"Code": code},
        "AquoPlusWaarnemingMetadata": {
            "AquoMetadata": {"Compartiment": {"Code": "OW"}, "Grootheid": {"Code": "Q"}}
        },
        "Periode": {
            "Begindatumtijd": a.strftime("%Y-%m-%dT%H:%M:%S.000+01:00"),
            "Einddatumtijd": b.strftime("%Y-%m-%dT%H:%M:%S.000+01:00"),
        },
    }
    r = requests.post(URL, json=body, timeout=120)
    if r.status_code != 200:
        return []
    out = []
    for series in r.json().get("WaarnemingenLijst", []):
        for m in series.get("MetingenLijst", []):
            v = m.get("Meetwaarde", {}).get("Waarde_Numeriek")
            if v is not None and abs(v) < 1e5:  # 999999999 marks a missing value
                out.append((m["Tijdstip"], v))
    return out


for code in args.stations:
    rows = []
    for a in pd.date_range(args.start, args.end, freq="MS"):
        rows += month(
            code, a, min(a + pd.offsets.MonthBegin(1), pd.Timestamp(args.end))
        )
    if not rows:
        print(f"{code}: no data")
        continue
    d = pd.DataFrame(rows, columns=["t", "discharge_m3s"])
    d["date"] = (
        pd.to_datetime(d["t"], utc=True).dt.tz_convert("Europe/Amsterdam").dt.date
    )
    daily = d.groupby("date")["discharge_m3s"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily.to_parquet(OUT / f"{code}.parquet", index=False)
    print(
        f"{code}: {daily['date'].min().date()} – {daily['date'].max().date()}, {len(daily)} days"
    )
