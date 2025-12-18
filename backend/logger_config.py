"""
نظام logging احترافي للتطبيق
CHANGES:
- Added a metrics logger and prevented duplicate handler addition more robustly.
- Added utility to create structured logs for metrics.
"""
import logging
import logging.handlers
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers (more strict check)
    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

video_logger = setup_logger('video_utils', log_file='logs/video_utils.log')
whisper_logger = setup_logger('whisper_engine', log_file='logs/whisper_engine.log')
nlp_logger = setup_logger('nlp_utils', log_file='logs/nlp_utils.log')
summarizer_logger = setup_logger('summarizer', log_file='logs/summarizer.log')
main_logger = setup_logger('main', log_file='logs/main.log')
api_logger = setup_logger('api', log_file='logs/api.log')
metrics_logger = setup_logger('metrics', log_file='logs/metrics.log')


def log_metric(name: str, value, **kwargs):
    """
    Structured metric log for later scraping by log collectors.
    """
    extra_parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
    metrics_logger.info(f"METRIC {name} value={value} {extra_parts}")