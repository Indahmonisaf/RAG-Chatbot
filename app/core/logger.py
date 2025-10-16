import logging

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)
