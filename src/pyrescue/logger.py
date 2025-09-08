import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler("training_monitor.log"), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


logger = setup_logging()
