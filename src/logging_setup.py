import logging

logging.basicConfig(
    level=logging.DEBUG,  # DEBUG for detailed trace info
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Logging is configured.")
