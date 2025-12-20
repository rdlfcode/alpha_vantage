import time
from collections import deque
import logging
import threading

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, requests_per_minute, requests_per_day):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.day_timestamps = deque()
        self.lock = threading.Lock()
        
        # Calculate minimum interval between requests to space them evenly
        if self.requests_per_minute > 0:
            self.min_interval = 60.0 / self.requests_per_minute
        else:
            self.min_interval = 0
            
        self.last_request_time = 0

    def wait_if_needed(self):
        with self.lock:
            current_time = time.time()

            # Prune old daily timestamps
            while (
                self.day_timestamps
                and current_time - self.day_timestamps[0] > 24 * 60 * 60
            ):
                self.day_timestamps.popleft()

            # Check day limit
            if len(self.day_timestamps) >= self.requests_per_day:
                logger.info(
                    "Daily request limit reached. Cannot make more requests today."
                )
                raise Exception("Daily request limit reached.")

            # Enforce even spacing (interval)
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                logger.debug(f"Spacing out request. Waiting for {wait_time:.2f} seconds.")
                time.sleep(wait_time)
                # Update current time after sleep
                current_time = time.time()

            # Record the new request time
            self.last_request_time = current_time
            self.day_timestamps.append(current_time)
            
            logger.debug(
                "Request allowed. Interval: %.2fs, RPD: %d/%d",
                self.min_interval,
                len(self.day_timestamps),
                self.requests_per_day,
            )

    def set_daily_limit_reached(self):
        """
        Manually set the rate limiter to the daily limit reached state.
        Fills the day timestamps with current time to block further requests.
        """
        with self.lock:
            current_time = time.time()
            while len(self.day_timestamps) < self.requests_per_day:
                self.day_timestamps.append(current_time)
            logger.warning(
                f"Daily limit manually set. Day timestamps count: {len(self.day_timestamps)}"
            )

