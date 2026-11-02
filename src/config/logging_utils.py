"""Application logging configuration shared by UI and command-line entry points."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.config.settings import BASE_DIR, settings


def configure_logging() -> None:
    """Configure timestamped console logs and an optional rotating project log."""
    level = getattr(logging, settings.log_level, logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Make repeated Streamlit reruns and imports safe without duplicating handlers.
    if not any(getattr(handler, "_onboarding_handler", False) for handler in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler._onboarding_handler = True  # type: ignore[attr-defined]
        root.addHandler(console_handler)

        if settings.log_file:
            log_path = BASE_DIR / settings.log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler._onboarding_handler = True  # type: ignore[attr-defined]
            root.addHandler(file_handler)

    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
