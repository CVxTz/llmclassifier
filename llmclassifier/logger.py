import logging
import sys

# Configure the logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Define the log message format
    handlers=[
        logging.StreamHandler(sys.stdout)  # Send logs to stdout
    ],
)

# Example usage
logger = logging.getLogger(__name__)
