import time
from collections import deque
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
   def __init__(self, requests_per_minute, requests_per_day):
      self.requests_per_minute = requests_per_minute
      self.requests_per_day = requests_per_day
      self.minute_timestamps = deque()
      self.day_timestamps = deque()

   def wait_if_needed(self):
      current_time = time.time()

      # Prune old timestamps
      while (
         self.minute_timestamps
         and current_time - self.minute_timestamps[0] > 60
      ):
         self.minute_timestamps.popleft()
      while (
         self.day_timestamps and current_time - self.day_timestamps[0] > 24 * 60 * 60
      ):
         self.day_timestamps.popleft()

      # Check day limit
      if len(self.day_timestamps) >= self.requests_per_day:
         logger.info("Daily request limit reached. Cannot make more requests today.")
         raise Exception("Daily request limit reached.")

      # Check minute limit and wait if necessary
      if len(self.minute_timestamps) >= self.requests_per_minute:
         wait_time = 60 - (current_time - self.minute_timestamps[0])
         logger.info(f"Rate limit reached. Waiting for {wait_time:.2f} seconds.")
         time.sleep(wait_time)

      # Record the new request time
      self.minute_timestamps.append(time.time())
      self.day_timestamps.append(time.time())
      logger.debug("Request made. RPM: %d/%d, RPD: %d/%d",
                len(self.minute_timestamps), self.requests_per_minute,
                len(self.day_timestamps), self.requests_per_day)

   def set_daily_limit_reached(self):
      """
      Manually set the rate limiter to the daily limit reached state.
      Fills the day timestamps with current time to block further requests.
      """
      current_time = time.time()
      while len(self.day_timestamps) < self.requests_per_day:
         self.day_timestamps.append(current_time)
      logger.warning(f"Daily limit manually set. Day timestamps count: {len(self.day_timestamps)}")
