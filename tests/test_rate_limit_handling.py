
import unittest
from unittest.mock import MagicMock, patch
import logging
from alpha_vantage import AlphaVantageClient

class TestRateLimitHandling(unittest.TestCase):
    @patch('requests.get')
    def test_daily_limit_handling(self, mock_get):
        # Mock response with the rate limit note
        mock_response = MagicMock()
        mock_response.text = '{"Note": "We have detected your API key as R5AYPFDI3RC4ZCSD and our standard API rate limit is 25 requests per day. Please visit https://www.alphavantage.co/premium/ if you would like to target a..."}'
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = AlphaVantageClient(api_key="TEST_KEY")
        
        # Ensure we start with no daily timestamps
        client.rate_limiter.day_timestamps.clear()
        
        with self.assertRaises(Exception) as cm:
            client._fetch_data("TIME_SERIES_DAILY", {"symbol": "IBM"})
            
        self.assertIn("Daily request limit reached", str(cm.exception))
        
        # Verify rate limiter was updated
        # It should be full (25 requests)
        self.assertEqual(len(client.rate_limiter.day_timestamps), client.rate_limiter.requests_per_day)
        print("Test passed: Exception raised and RateLimiter updated.")

if __name__ == '__main__':
    unittest.main()
