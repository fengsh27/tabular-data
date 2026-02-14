import os
import logging
from datetime import datetime


class DateStampedFileHandler(logging.Handler):
    def __init__(self, base_path: str, encoding: str = "utf-8"):
        super().__init__()
        self.base_path = base_path
        self.encoding = encoding
        self.stream = None
        self.current_date = None
        self._ensure_open()

    def _ensure_open(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        if self.current_date == date_str and self.stream:
            return
        if self.stream:
            try:
                self.stream.flush()
                self.stream.close()
            except Exception:
                pass
        base, ext = os.path.splitext(self.base_path)
        if not ext:
            ext = ".log"
        dated_path = f"{base}-{date_str}{ext}"
        self.baseFilename = os.path.abspath(dated_path)
        self.stream = open(self.baseFilename, "a", encoding=self.encoding)
        self.current_date = date_str

    def emit(self, record):
        try:
            self._ensure_open()
            msg = self.format(record)
            self.stream.write(msg + "\n")
            self.flush()
        except Exception:
            self.handleError(record)

    def flush(self):
        if self.stream and hasattr(self.stream, "flush"):
            self.stream.flush()

    def close(self):
        if self.stream:
            try:
                self.stream.flush()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        super().close()


def initialize_logger(
    log_file: str,
    app_log_name: str,
    app_log_level: int,
    log_entries: dict[str, int],
):
    # prepare logger
    # logging.basicConfig(level=logging.INFO)
    logs_folder = os.environ.get("LOGS_FOLDER", "./logs")
    if len(logs_folder.strip()) == 0:
        logs_folder = "./logs"
    os.makedirs(logs_folder, exist_ok=True)
    logs_file = os.path.join(logs_folder, log_file)

    # Root logger configuration (optional)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Silence noisy libraries

    file_handler = DateStampedFileHandler(logs_file)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
