# modules/utils/logs.py

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import platform
import psutil
import time
import json

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = setup_file_handler(log_file, level)
        console_handler = setup_console_handler(level)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.propagate = False  # не передаём в root

    return logger

def setup_file_handler(log_file: str, level=logging.INFO) -> RotatingFileHandler:
    file_path = LOG_DIR / log_file
    handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler

def setup_console_handler(level=logging.INFO) -> logging.StreamHandler:
    console = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    console.setLevel(level)
    return console

def get_logger(name: str) -> logging.Logger:
    return setup_logger(name, f"{name.lower()}.log")

# ----------------------------- SYSTEM METRICS -----------------------------

def log_system_info(logger: logging.Logger):
    info = {
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_mb": round(psutil.virtual_memory().total / (1024 * 1024)),
        "boot_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(psutil.boot_time())),
    }
    logger.info(f"System Info: {json.dumps(info)}")

def log_processing_stats(topics_processed: int, errors: int, success_rate: float, logger: logging.Logger):
    logger.info(f"Processing Stats | Topics: {topics_processed}, Errors: {errors}, Success Rate: {success_rate:.2f}")

def log_rag_performance(retrieval_time: float, context_length: int, logger: logging.Logger):
    logger.debug(f"RAG | Time: {retrieval_time:.2f}s, Context length: {context_length} chars")

def log_telegram_status(message_sent: bool, logger: logging.Logger, error_details: str = None):
    if message_sent:
        logger.info("Telegram message sent successfully.")
    else:
        logger.error(f"Telegram message failed. Reason: {error_details}")
