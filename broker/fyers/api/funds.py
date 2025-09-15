import os
import json
import httpx
from typing import Dict
from .utils import get_httpx_client, get_logger

logger = get_logger(__name__)

def get_margin_data(auth_token: str) -> Dict[str, str]:
    """
    Fetch and process margin/funds data from Fyers' API.
    """
    default_response = {
        "availablecash": "0.00", "collateral": "0.00",
        "m2munrealized": "0.00", "m2mrealized": "0.00",
        "utiliseddebits": "0.00"
    }

    api_key = os.getenv('BROKER_API_KEY')
    if not api_key:
        logger.error("BROKER_API_KEY environment variable not set")
        return default_response

    client = get_httpx_client()
    headers = {'Authorization': f'{api_key}:{auth_token}', 'Content-Type': 'application/json'}

    try:
        response = client.get('https://api-t1.fyers.in/api/v3/funds', headers=headers)
        response.raise_for_status()
        funds_data = response.json()
        logger.debug(f"Fyers funds API response: {json.dumps(funds_data, indent=2)}")

        if funds_data.get('code') != 200:
            logger.error(f"Error in Fyers funds API: {funds_data.get('message', 'Unknown error')}")
            return default_response

        processed_funds = {
            fund['title'].lower().replace(' ', '_'): {
                "equity_amount": float(fund.get('equityAmount', 0)),
                "commodity_amount": float(fund.get('commodityAmount', 0))
            }
            for fund in funds_data.get('fund_limit', [])
        }

        balance = processed_funds.get('available_balance', {})
        total_balance = float(balance.get('equity_amount', 0)) + float(balance.get('commodity_amount', 0))

        collateral = processed_funds.get('collaterals', {})
        total_collateral = float(collateral.get('equity_amount', 0)) + float(collateral.get('commodity_amount', 0))

        pnl = processed_funds.get('realized_profit_and_loss', {})
        total_real_pnl = float(pnl.get('equity_amount', 0)) + float(pnl.get('commodity_amount', 0))

        utilized = processed_funds.get('utilized_amount', {})
        total_utilized = float(utilized.get('equity_amount', 0)) + float(utilized.get('commodity_amount', 0))

        return {
            "availablecash": f"{total_balance:.2f}",
            "collateral": f"{total_collateral:.2f}",
            "m2munrealized": f"{total_collateral:.2f}",  # Assuming collateral as unrealized M2M
            "m2mrealized": f"{total_real_pnl:.2f}",
            "utiliseddebits": f"{total_utilized:.2f}"
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching Fyers funds: {e.response.text}")
    except (httpx.RequestError, json.JSONDecodeError, KeyError, ValueError) as e:
        logger.exception("Error processing funds data")

    return default_response
