"""Entry point: ``uv run python -m src.pipeline --experiment <name>``."""

from src.pipeline.run import main

raise SystemExit(main())
