import logging


def setup_logging(log_config: dict):
    logging.basicConfig(
        level=log_config["level"],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_config["log_file"]), logging.StreamHandler()],
    )

    return logging.getLogger("pyrescue")
