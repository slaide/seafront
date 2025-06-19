from loguru import logger

from .config.basics import GlobalConfigHandler

log_dir = GlobalConfigHandler.home() / "logs"
if not log_dir.exists():
    log_dir.mkdir()

# no timestamp required
logfile_path = log_dir / "log.txt"
logger.add(
    logfile_path,
    rotation="10 MB",
    # use gz compression for the very redundant text messages
    compression="gz",
    format="{time} | {level} | {file.path}:{line} | {message}",
)

logger.info(f"logger - opened logfile at {logfile_path!s}")

# combine all logs into a single file
# $ ( zcat log.txt*.gz; cat log.txt ) | gzip > combined.log.gz
