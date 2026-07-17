import logging

from workers.main_worker import main_worker

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    main_worker()
