import json
import logging


def setup_logging():
    with open("src/pyrescue/logger/logger_config.json", "r") as f:
        config = json.load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(config["log_path"]), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


logger = setup_logging()
