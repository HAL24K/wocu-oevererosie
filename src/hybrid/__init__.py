"""Hybrid stage: raw SAM lines + height model → a hybrid line set in 03_hybrid/<name>/.

config        HybridConfig: inputs, SAM cleaning options, output locations
run           the stage, step by step (``python -m src.hybrid``)
luke_v2       port of Luke's hybrid model v2: nearest-crossing cut, points count
sam_cleaning  our cleaning of the SAM pieces between those two steps
"""
