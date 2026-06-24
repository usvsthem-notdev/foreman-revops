"""
Foreman scheduler entry point.

Usage:
    python scheduler.py                  # run with defaults
    FOREMAN_POLL_INTERVAL_HOURS=1 python scheduler.py

Docker Compose:
    The 'scheduler' service in docker-compose.yml runs this automatically.

All configuration is via environment variables — see src/polling/scheduler.py
for the full list.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault(
    "FOREMAN_DB_PATH",
    str(Path(__file__).parent / "data" / "foreman.db"),
)

from src.db import init_db  # noqa: E402
from src.polling.scheduler import load_config, run  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("foreman.scheduler")


def main() -> None:
    log.info("Foreman scheduler starting.")
    init_db()

    try:
        config = load_config()
    except ValueError as exc:
        log.error("Cannot start: %s", exc)
        sys.exit(1)

    run(config)


if __name__ == "__main__":
    main()
