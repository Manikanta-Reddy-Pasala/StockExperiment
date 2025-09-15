import json
import os
import httpx
import pandas as pd
from datetime import datetime
import urllib.parse
import time
from .utils import get_api_response, get_logger
from .transform import get_br_symbol

logger = get_logger(__name__)

class BrokerData:
    def __init__(self, auth_token):
        """Initialize Fyers data handler with authentication token"""
        self.auth_token = auth_token
        self.timeframe_map = {
            '5s': '5S', '10s': '10S', '15s': '15S', '30s': '30S', '45s': '45S',
            '1m': '1', '2m': '2', '3m': '3', '5m': '5', '10m': '10', '15m': '15',
            '20m': '20', '30m': '30', '1h': '60', '2h': '120', '4h': '240', 'D': '1D'
        }

    def get_quotes(self, symbol: str, exchange: str) -> dict:
        try:
            br_symbol = get_br_symbol(symbol, exchange)
            encoded_symbol = urllib.parse.quote(br_symbol)
            response = get_api_response(f"/data/quotes?symbols={encoded_symbol}", self.auth_token)
            logger.debug(f"Fyers quotes API response: {response}")

            if response.get('s') != 'ok':
                raise Exception(f"Error from Fyers API: {response.get('message', 'Unknown error')}")

            quote_data = response.get('d', [{}])[0]
            v = quote_data.get('v', {})
            return {
                'bid': v.get('bid', 0), 'ask': v.get('ask', 0), 'open': v.get('open_price', 0),
                'high': v.get('high_price', 0), 'low': v.get('low_price', 0), 'ltp': v.get('lp', 0),
                'prev_close': v.get('prev_close_price', 0), 'volume': v.get('volume', 0)
            }
        except Exception as e:
            logger.exception(f"Error fetching quotes for {exchange}:{symbol}")
            raise

    def get_history(self, symbol: str, exchange: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            br_symbol = get_br_symbol(symbol, exchange)
            resolution = self.timeframe_map.get(interval)
            if not resolution:
                raise ValueError("Unsupported timeframe")

            start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
            dfs = []
            chunk_days = 300 if resolution == '1D' else 60

            current_start = start_dt
            while current_start <= end_dt:
                current_end = min(current_start + pd.Timedelta(days=chunk_days - 1), end_dt)
                chunk_start, chunk_end = current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d')

                encoded_symbol = urllib.parse.quote(br_symbol)
                endpoint = (f"/data/history?symbol={encoded_symbol}&resolution={resolution}&date_format=1"
                            f"&range_from={chunk_start}&range_to={chunk_end}&cont_flag=1")

                response = get_api_response(endpoint, self.auth_token)
                if response.get('s') == 'ok' and response.get('candles'):
                    df = pd.DataFrame(response['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    dfs.append(df)

                time.sleep(0.5)
                current_start = current_end + pd.Timedelta(days=1)

            if not dfs:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            final_df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            logger.info(f"Successfully collected data: {len(final_df)} total candles")
            return final_df
        except Exception as e:
            logger.exception(f"Error fetching historical data for {exchange}:{symbol}")
            raise

    def get_depth(self, symbol: str, exchange: str) -> dict:
        try:
            br_symbol = get_br_symbol(symbol, exchange)
            encoded_symbol = urllib.parse.quote(br_symbol)
            response = get_api_response(f"/data/depth?symbol={encoded_symbol}&ohlcv_flag=1", self.auth_token)

            if response.get('s') != 'ok':
                raise Exception(f"Error from Fyers API: {response.get('message', 'Unknown error')}")

            depth_data = response.get('d', {}).get(br_symbol, {})
            bids = [{'price': b['price'], 'quantity': b['volume']} for b in depth_data.get('bids', [])[:5]]
            asks = [{'price': a['price'], 'quantity': a['volume']} for a in depth_data.get('asks', [])[:5]]

            return {
                'bids': bids, 'asks': asks, 'totalbuyqty': depth_data.get('totalbuyqty', 0),
                'totalsellqty': depth_data.get('totalsellqty', 0), 'high': depth_data.get('h', 0),
                'low': depth_data.get('l', 0), 'ltp': depth_data.get('ltp', 0),
                'ltq': depth_data.get('ltq', 0), 'open': depth_data.get('o', 0),
                'prev_close': depth_data.get('c', 0), 'volume': depth_data.get('v', 0),
                'oi': int(depth_data.get('oi', 0))
            }
        except Exception as e:
            logger.exception(f"Error fetching market depth for {exchange}:{symbol}")
            raise
