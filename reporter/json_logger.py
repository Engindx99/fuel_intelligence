import json
import logging
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "validation.jsonl"


class JsonFormatter(logging.Formatter):

    def format(self, record):

        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "validation"):
            log_record["validation"] = record.validation

        return json.dumps(
            log_record,
            ensure_ascii=False,
        )


def _get_logger():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        "cement.validation"
    )

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        )

        handler.setFormatter(
            JsonFormatter()
        )

        logger.addHandler(handler)

        logger.propagate = False

    return logger


def log_validation(validation):

    logger = _get_logger()

    logger.info(
        "Validation completed",
        extra={
            "validation": validation
        },
    )