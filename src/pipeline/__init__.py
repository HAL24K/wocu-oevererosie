"""Model stage: hybrid line set → cleaned observations → features → model → predictions.

config              ExperimentConfig: inputs, parameters, output locations
run                 the orchestration, step by step (``python -m src.pipeline``)
hybrid_prep         step 1: sample and clean the hybrid lines, per-year distances
region_split        step 2: quality gate, train/test split
feature_engineering step 3: context and discharge features
trajectory          step 3b: history descriptors per region
train               step 4: baselines and LightGBM
predict             step 5: iterative prediction (with feature_shifter)
centerline, export  step 6–7: bank positions, VVR crossing years, GeoPackage
segments            step 8: the segment-horizon artifact
provenance          the run record (run.json, docs/runs/)
report              the HTML report
"""
