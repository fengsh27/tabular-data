import os
import logging

def initialize_logger(
    log_file: str,
    app_log_name: str,
    app_log_level: int,
    log_entries: dict[str, int],
):
    # prepare logger
    # logging.basicConfig(level=logging.INFO)
    logs_folder = os.environ.get("LOGS_FOLDER", "./logs")
    logs_file = os.path.join(logs_folder, log_file)
    
    # Root logger configuration (optional)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Silence noisy libraries

    file_handler = logging.handlers.RotatingFileHandler(logs_file)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # datefmt="%Y-%m-%d %H:%M:%S,uuu"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    for log_entry in log_entries:
        level = log_entries[log_entry]
        logger = logging.getLogger(log_entry)
        logger.setLevel(level)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.propagate = False

    app_logger = logging.getLogger(app_log_name)
    app_logger.setLevel(app_log_level)
    app_logger.handlers.clear()
    app_logger.addHandler(file_handler)
    app_logger.addHandler(stream_handler)
    app_logger.propagate = False

    return app_logger

