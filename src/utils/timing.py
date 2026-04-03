import time
from functools import wraps
from loguru import logger


def time_it(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        duration = (end_time - start_time) * 1000
        logger.info(f"{func.__name__} took {duration:.4f} ms")

        return result

    return wrapper
