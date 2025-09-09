"""
Fyers API Connector
"""
from typing import Dict, Any, List, Optional
from fyers_apiv3 import fyersModel
from .base_connector import BrokerConnector


class FyersConnector(BrokerConnector):
    """Interface with Fyers API for market data and order execution."""

    def __init__(self, client_id: str, access_token: str, log_path: str = ""):
        """
        Initialize the Fyers connector.

        Args:
            client_id (str): Fyers API client_id (or app_id)
            access_token (str): Access token for authentication
            log_path (str): Path to store logs.
        """
        self.fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, log_path=log_path, is_async=False)

    def get_profile(self) -> Dict[str, Any]:
        """
        Get user profile.

        Returns:
            Dict[str, Any]: User profile information
        """
        return self.fyers.get_profile()

    def get_margins(self) -> Dict[str, Any]:
        """
        Get user margins.

        Returns:
            Dict[str, Any]: Margin information
        """
        return self.fyers.funds()

    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders.

        Returns:
            List[Dict[str, Any]]: List of orders
        """
        response = self.fyers.orderbook()
        return response.get('orderbook', [])

    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions.

        Returns:
            Dict[str, Any]: Position information
        """
        return self.fyers.positions()

    def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get current holdings.

        Returns:
            List[Dict[str, Any]]: List of holdings
        """
        response = self.fyers.holdings()
        return response.get('holdings', [])

    def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place an order.

        Args:
            order_params (Dict[str, Any]): Order parameters

        Returns:
            Dict[str, Any]: Order response
        """
        return self.fyers.place_order(data=order_params)

    def modify_order(self, order_id: str, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify an existing order.

        Args:
            order_id (str): Order ID to modify
            order_params (Dict[str, Any]): Modified order parameters

        Returns:
            Dict[str, Any]: Order response
        """
        data = {"id": order_id, **order_params}
        return self.fyers.modify_order(data=data)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.

        Args:
            order_id (str): Order ID to cancel

        Returns:
            Dict[str, Any]: Order response
        """
        data = {"id": order_id}
        return self.fyers.cancel_order(data=data)

    def get_instruments(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get instrument master data from Fyers symbol master files.

        Args:
            exchange (str, optional): Exchange name (e.g., "NSE", "BSE")

        Returns:
            List[Dict[str, Any]]: List of instruments
        """
        import pandas as pd
        import requests
        from io import StringIO

        urls = {
            "NSE_CM": "https://public.fyers.in/sym_details/NSE_CM.csv",
            "BSE_CM": "https://public.fyers.in/sym_details/BSE_CM.csv",
            "NSE_FO": "https://public.fyers.in/sym_details/NSE_FO.csv",
            "BSE_FO": "https://public.fyers.in/sym_details/BSE_FO.csv",
        }

        exchanges_to_download = []
        if exchange:
            if f"{exchange}_CM" in urls:
                exchanges_to_download.append(urls[f"{exchange}_CM"])
            if f"{exchange}_FO" in urls:
                exchanges_to_download.append(urls[f"{exchange}_FO"])
        else:
            exchanges_to_download = list(urls.values())

        all_instruments = []

        headers = [
            'fytoken', 'symbol_details', 'exchange_instrument_type', 'minimum_lot_size', 'tick_size',
            'isin', 'trading_session', 'last_updated_date', 'expiry_date', 'symbol',
            'exchange', 'segment', 'scrip_code', 'underlying_scrip_code', 'strike_price',
            'option_type', 'underlying_fytoken'
        ]

        for url in exchanges_to_download:
            try:
                response = requests.get(url)
                response.raise_for_status()

                # Use StringIO to treat the string content as a file
                csv_data = StringIO(response.text)

                df = pd.read_csv(csv_data, header=None, names=headers)
                all_instruments.extend(df.to_dict('records'))
            except requests.exceptions.RequestException as e:
                print(f"Error downloading instruments from {url}: {e}")
            except Exception as e:
                print(f"Error processing instruments from {url}: {e}")

        return all_instruments

    def get_ltp(self, instrument_tokens: List[str]) -> Dict[str, Any]:
        """
        Get last traded price for instruments.

        Args:
            instrument_tokens (List[str]): List of instrument tokens (e.g., ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ"])

        Returns:
            Dict[str, Any]: LTP data
        """
        if not instrument_tokens:
            return {}
        data = {"symbols": ",".join(instrument_tokens)}
        response = self.fyers.quotes(data=data)

        ltp_data = {}
        if response.get('s') == 'ok':
            for quote in response.get('d', []):
                if 'n' in quote and 'v' in quote and 'lp' in quote['v']:
                    ltp_data[quote['n']] = {'last_price': quote['v']['lp']}
        return ltp_data

    def get_quote(self, instrument_tokens: List[str]) -> Dict[str, Any]:
        """
        Get quote data for instruments.

        Args:
            instrument_tokens (List[str]): List of instrument tokens (e.g., ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ"])

        Returns:
            Dict[str, Any]: Quote data
        """
        if not instrument_tokens:
            return {}
        data = {"symbols": ",".join(instrument_tokens)}
        return self.fyers.quotes(data=data)
