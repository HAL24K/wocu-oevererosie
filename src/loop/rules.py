"""Cleaning rules for the loop-engineering experiment.

Every rule is a pure function ``(samples, ctx, **params) -> (samples, stats)``
that removes rows from the sample table and reports what it did. Rules see the
effect of the rules applied before them (line/survey statistics are recomputed
on the surviving samples), so order matters and is part of a variant's
definition.

Levels of attack, from least to most invasive:
  sample  — drop individual sampled points (structure mask)
  line    — drop one delivered line, keep the survey (tortuosity, near-bank,
            far-line selection)
  survey  — drop one (region, date) observation, keep the region (fragments,
            temporal outliers, maze surveys)
  region  — exclude the region entirely (legacy |v| filter — lives in the
            harness, applied on dist_per_year, and is the move the survey- and
            line-level rules exist to make unnecessary)

The sample table is the one cached by ``scripts/loop_build_caches.py``:
columns line_idx, location_id, date, year, model, station, dist, x, y.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shapely

from src.sources.geometry import LOCATION_ID

N_STATION_BINS = 20
SURVEY = [LOCATION_ID, "date"]


@dataclass
class RuleContext:
    """Static inputs rules may need besides the (mutable) sample table."""

    line_metrics: pd.DataFrame  # indexed by line_idx: length, chord, tortuosity
    cl_len: pd.Series  # centreline length per location_id
    structures: object | None = None  # prepared shapely geometry (buffered union)
    stats: list[dict] = field(default_factory=list)


def make_structure_geom(gdf, buffer_m: float):
    """Buffered union of structure polygons, prepared for fast contains()."""
    geom = shapely.union_all(gdf.geometry.buffer(buffer_m).values)
    shapely.prepare(geom)
    return geom


# ── helpers ───────────────────────────────────────────────────────────────────


def _line_p50(s: pd.DataFrame) -> pd.Series:
    return s.groupby("line_idx")["dist"].median()


def _survey_p50(s: pd.DataFrame) -> pd.Series:
    """Per-survey robust position: median sample distance."""
    return s.groupby(SURVEY)["dist"].median()


def _stat(name: str, params: dict, before: pd.DataFrame, after: pd.DataFrame) -> dict:
    return {
        "rule": name,
        "params": {k: v for k, v in params.items() if not hasattr(v, "geom_type")},
        "samples_dropped": len(before) - len(after),
        "lines_dropped": before["line_idx"].nunique() - after["line_idx"].nunique(),
        "surveys_dropped": (
            before.groupby(SURVEY).ngroups - after.groupby(SURVEY).ngroups
        ),
        "regions_dropped": (
            before[LOCATION_ID].nunique() - after[LOCATION_ID].nunique()
        ),
    }


# ── sample level ──────────────────────────────────────────────────────────────


def structure_mask(s: pd.DataFrame, ctx: RuleContext) -> pd.DataFrame:
    """Drop samples inside the prepared structure buffer (kribben et al.)."""
    if ctx.structures is None:
        return s
    pts = shapely.points(s["x"].values, s["y"].values)
    inside = shapely.contains(ctx.structures, pts)
    out = s[~inside]
    ctx.stats.append(_stat("structure_mask", {}, s, out))
    return out


# ── line level ────────────────────────────────────────────────────────────────


def max_tortuosity_line(
    s: pd.DataFrame, ctx: RuleContext, max_tort: float = 3.0, min_len: float = 30.0
) -> pd.DataFrame:
    """Drop lines that wander: length over endpoint chord above ``max_tort``.

    Short scraps are spared via ``min_len`` — a 10 m hook is noise either way,
    and closed loops (chord ≈ 0) would otherwise dominate.
    """
    lm = ctx.line_metrics
    bad = lm[(lm["tortuosity"] > max_tort) & (lm["length"] > min_len)].index
    out = s[~s["line_idx"].isin(bad)]
    ctx.stats.append(
        _stat("max_tortuosity_line", {"max_tort": max_tort, "min_len": min_len}, s, out)
    )
    return out


def near_bank_line(
    s: pd.DataFrame, ctx: RuleContext, frac: float = 0.25, min_ref: float = 30.0
) -> pd.DataFrame:
    """Drop lines far inside the channel relative to the region's own history.

    A line whose median distance is below ``frac`` × the region's median survey
    position traces a sandbar, structure shadow or mid-channel feature, not the
    bank. Only fires when the reference itself is at least ``min_ref`` m out,
    so genuinely narrow regions are untouched.
    """
    p50 = _line_p50(s)
    ref = _survey_p50(s).groupby(LOCATION_ID).median()
    lines = s.groupby("line_idx")[LOCATION_ID].first()
    ref_per_line = lines.map(ref)
    bad = p50.index[(p50 < frac * ref_per_line) & (ref_per_line > min_ref)]
    out = s[~s["line_idx"].isin(bad)]
    ctx.stats.append(
        _stat("near_bank_line", {"frac": frac, "min_ref": min_ref}, s, out)
    )
    return out


def multiline_far_line(
    s: pd.DataFrame, ctx: RuleContext, gap: float = 25.0, ratio: float = 1.5
) -> pd.DataFrame:
    """In a two-banks survey, keep the line group consistent with history.

    Fires on surveys whose lines split by more than ``gap`` metres and
    ``ratio``× between nearest and furthest line median. The survey's lines
    are split at the midpoint of that gap; the group whose position is closer
    to the region's median across *other* surveys is kept. With no other
    surveys to consult, the nearer group is kept — every confirmed artefact so
    far has been the far bank.
    """
    p50 = _line_p50(s)
    lines = s.groupby("line_idx")[SURVEY].first()
    lines["p50"] = p50

    g = lines.groupby(SURVEY)["p50"]
    span = g.max() - g.min()
    rat = g.max() / g.min().clip(lower=0.1)
    hit = span.index[(span > gap) & (rat > ratio)]
    if not len(hit):
        ctx.stats.append(
            _stat("multiline_far_line", {"gap": gap, "ratio": ratio}, s, s)
        )
        return s

    survey_pos = _survey_p50(s)
    drop: list = []
    hit_set = set(hit)
    for key, grp in lines.groupby(SURVEY):
        if key not in hit_set:
            continue
        mid = (grp["p50"].min() + grp["p50"].max()) / 2
        near, far = grp[grp["p50"] <= mid], grp[grp["p50"] > mid]
        others = survey_pos.loc[key[0]].drop(index=key[1], errors="ignore")
        if len(others):
            ref = others.median()
            keep_near = abs(near["p50"].median() - ref) <= abs(
                far["p50"].median() - ref
            )
        else:
            keep_near = True
        drop.extend((far if keep_near else near).index)

    out = s[~s["line_idx"].isin(drop)]
    ctx.stats.append(_stat("multiline_far_line", {"gap": gap, "ratio": ratio}, s, out))
    return out


# ── survey level ──────────────────────────────────────────────────────────────


def min_samples_survey(
    s: pd.DataFrame, ctx: RuleContext, min_n: int = 12
) -> pd.DataFrame:
    """Drop surveys with too few surviving samples to trust the furthest-3."""
    n = s.groupby(SURVEY)["dist"].transform("size")
    out = s[n >= min_n]
    ctx.stats.append(_stat("min_samples_survey", {"min_n": min_n}, s, out))
    return out


def fragment_survey(
    s: pd.DataFrame, ctx: RuleContext, min_cov: float = 0.3
) -> pd.DataFrame:
    """Drop surveys covering less than ``min_cov`` of the region's length."""
    bins = np.minimum((s["station"] * N_STATION_BINS).astype(int), N_STATION_BINS - 1)
    cov = bins.groupby([s[LOCATION_ID], s["date"]]).nunique() / N_STATION_BINS
    bad = cov.index[cov < min_cov]
    keep = ~pd.MultiIndex.from_frame(s[SURVEY]).isin(bad)
    out = s[keep]
    ctx.stats.append(_stat("fragment_survey", {"min_cov": min_cov}, s, out))
    return out


def maze_survey(
    s: pd.DataFrame, ctx: RuleContext, max_ratio: float = 1.8
) -> pd.DataFrame:
    """Drop surveys whose delivered line length exceeds the region length.

    A bank can be at most about as long as its region's centreline; a survey
    delivering ``max_ratio``× that traces a land/water mosaic (harbours,
    floodplain pools), not a bank.
    """
    lm = ctx.line_metrics
    line_len = s.groupby("line_idx").size().index.to_series().map(lm["length"])
    lines = s.groupby("line_idx")[SURVEY].first()
    lines["length"] = line_len
    total = lines.groupby(SURVEY)["length"].sum()
    ratio = total / total.index.get_level_values(0).map(ctx.cl_len).values
    bad = ratio.index[ratio > max_ratio]
    keep = ~pd.MultiIndex.from_frame(s[SURVEY]).isin(bad)
    out = s[keep]
    ctx.stats.append(_stat("maze_survey", {"max_ratio": max_ratio}, s, out))
    return out


def temporal_outlier_survey(
    s: pd.DataFrame,
    ctx: RuleContext,
    max_dev: float = 30.0,
    min_surveys: int = 4,
) -> pd.DataFrame:
    """Drop surveys that disagree with the region's own history.

    Repair-over-removal counterpart of the legacy region-level |v| filter: a
    survey whose median position deviates more than ``max_dev`` metres from
    the median of the region's *other* surveys is an artefact (wrong bank,
    side channel); the region keeps its remaining surveys. Only regions with
    at least ``min_surveys`` surveys are touched, so a genuine rapid change
    seen once is never silently erased.

    ``max_dev`` should sit far above real erosion (a few m/yr) and below the
    far-bank jump (typically 50–200 m).
    """
    pos = _survey_p50(s)
    n = pos.groupby(LOCATION_ID).transform("size")
    med = pos.groupby(LOCATION_ID).transform("median")
    # median of the others: recompute without self only where it matters
    dev = (pos - med).abs()
    bad = pos.index[(dev > max_dev) & (n >= min_surveys)]
    keep = ~pd.MultiIndex.from_frame(s[SURVEY]).isin(bad)
    out = s[keep]
    ctx.stats.append(
        _stat(
            "temporal_outlier_survey",
            {"max_dev": max_dev, "min_surveys": min_surveys},
            s,
            out,
        )
    )
    return out


#: Registry for variant definitions: name → callable.
RULES = {
    "structure_mask": structure_mask,
    "max_tortuosity_line": max_tortuosity_line,
    "near_bank_line": near_bank_line,
    "multiline_far_line": multiline_far_line,
    "min_samples_survey": min_samples_survey,
    "fragment_survey": fragment_survey,
    "maze_survey": maze_survey,
    "temporal_outlier_survey": temporal_outlier_survey,
}


def apply_rules(
    samples: pd.DataFrame, ctx: RuleContext, rules: list[tuple[str, dict]]
) -> pd.DataFrame:
    """Apply ``rules`` (name, params) in order; stats accumulate on ``ctx``."""
    s = samples
    for name, params in rules:
        s = RULES[name](s, ctx, **params)
    return s
