import logging
import os
import sys


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Initialize and return a logger with the specified name and level. If the logger already has handlers, it will not add new ones.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.propagate = True

    is_airflow = os.environ.get("AIRFLOW_CTX_DAG_ID") is not None

    if not is_airflow and not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
