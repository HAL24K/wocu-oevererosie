"""Composable cleaning rules for bank-line observations.

Graduated from the loop-engineering experiment (experiments/loop/
TRACK1_REPORT.md): every rule is a pure function over the sample table,
applied in order, clipping samples/lines/surveys — never whole regions.
"""
