from logging import Logger, getLogger

import coloredlogs


def get_logger(name: str) -> Logger:
    logger = getLogger(name)
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"

    coloredlogs.DEFAULT_FIELD_STYLES = {
        "asctime": {"color": "blue"},
        "name": {"color": "green"},
        "levelname": {"color": "yellow"},
    }
    coloredlogs.DEFAULT_LEVEL_STYLES["warning"] = {"color": 208}
    coloredlogs.install(level="INFO", logger=logger, fmt=fmt)

    return logger
