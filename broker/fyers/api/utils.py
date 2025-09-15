import os
import json
import httpx
import logging

# --- HTTP Client ---
_httpx_client = None

def get_httpx_client() -> httpx.Client:
    """
    Returns a shared httpx.Client instance with connection pooling.
    """
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.Client(timeout=30.0)
    return _httpx_client

# --- Logger ---
def get_logger(name: str) -> logging.Logger:
    """
    Returns a basic logger.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)

logger = get_logger(__name__)

# --- API Response Handler ---
def get_api_response(endpoint, auth, method="GET", payload=''):
    """
    Make API requests to Fyers API using shared connection pooling.

    Args:
        endpoint: API endpoint (e.g., /api/v3/orders)
        auth: Authentication token
        method: HTTP method (GET, POST, etc.)
        payload: Request payload as a string or dict

    Returns:
        dict: Parsed JSON response from the API
    """
    try:
        client = get_httpx_client()
        api_key = os.getenv('BROKER_API_KEY')
        url = f"https://api-t1.fyers.in{endpoint}"
        headers = {
            'Authorization': f'{api_key}:{auth}',
            'Content-Type': 'application/json'
        }

        logger.debug(f"Making {method} request to Fyers API: {url}")

        if method == "GET":
            response = client.get(url, headers=headers)
        elif method == "POST":
            response = client.post(url, headers=headers, json=payload if isinstance(payload, dict) else json.loads(payload))
        else:
            response = client.request(method, url, headers=headers, json=payload if isinstance(payload, dict) else json.loads(payload))

        response.raise_for_status()
        response_data = response.json()
        logger.debug(f"API response: {json.dumps(response_data, indent=2)}")
        return response_data

    except httpx.HTTPError as e:
        logger.error(f"HTTP error during API request: {e}")
        return {"s": "error", "message": f"HTTP error: {e}"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return {"s": "error", "message": f"Invalid JSON response: {e}"}
    except Exception as e:
        logger.exception("Error during API request")
        return {"s": "error", "message": f"General error: {e}"}
