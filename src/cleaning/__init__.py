"""Composable cleaning rules for bank-line observations.

Graduated from the loop-engineering experiment (docs/loop_experiment/
TRACK1_REPORT.md): every rule is a pure function over the sample table,
applied in order, clipping samples/lines/surveys — never whole regions.
"""
