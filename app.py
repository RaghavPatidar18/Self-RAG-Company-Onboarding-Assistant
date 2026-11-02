from src.config.logging_utils import configure_logging

configure_logging()

from src.ui.app import main


if __name__ == "__main__":
    main()
