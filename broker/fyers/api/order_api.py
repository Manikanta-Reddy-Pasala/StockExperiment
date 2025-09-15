import json
import os
import httpx
from .transform import get_br_symbol, map_product_type, transform_data, transform_modify_order_data
from .utils import get_api_response, get_logger

logger = get_logger(__name__)

def get_order_book(auth):
    return get_api_response("/api/v3/orders", auth)

def get_trade_book(auth):
    return get_api_response("/api/v3/tradebook", auth)

def get_positions(auth):
    return get_api_response("/api/v3/positions", auth)

def get_holdings(auth):
    return get_api_response("/api/v3/holdings", auth)

def get_open_position(tradingsymbol, exchange, product, auth):
    tradingsymbol = get_br_symbol(tradingsymbol, exchange)
    positions_data = get_positions(auth)
    net_qty = '0'

    if positions_data and positions_data.get('s') == 'ok' and positions_data.get('netPositions'):
        for position in positions_data['netPositions']:
            if position.get('symbol') == tradingsymbol and position.get("productType") == product:
                net_qty = position.get('netQty', '0')
                logger.debug(f"Net Quantity {net_qty}")
                break
    return net_qty

def place_order_api(data, auth):
    try:
        payload = transform_data(data)
        response_data = get_api_response("/api/v3/orders/sync", auth, "POST", payload)

        orderid = None
        if response_data.get('s') == 'ok':
            orderid = response_data.get('id')
            logger.info(f"Order placed successfully. Order ID: {orderid}")
        else:
            error_msg = response_data.get('message', 'Unknown error')
            logger.warning(f"Order placement failed: {error_msg}")

        # To maintain compatibility with original return signature, we create a mock response object
        response = httpx.Response(200, json=response_data)
        return response, response_data, orderid

    except Exception as e:
        logger.exception("Error during order placement")
        response = httpx.Response(500, json={"s": "error", "message": f"General error: {e}"})
        return response, {"s": "error", "message": f"General error: {e}"}, None

def place_smartorder_api(data, auth):
    symbol = data.get("symbol")
    exchange = data.get("exchange")
    product = data.get("product")
    position_size = int(data.get("position_size", "0"))

    current_position = int(get_open_position(symbol, exchange, map_product_type(product), auth))
    logger.debug(f"position_size : {position_size}, Open Position : {current_position}")

    action = None
    quantity = 0

    if position_size == current_position:
        if int(data.get('quantity', 0)) == 0:
            response = {"status": "success", "message": "No OpenPosition Found. Not placing Exit order."}
        else:
            response = {"status": "success", "message": "No action needed. Position size matches current position"}
        return None, response, None

    if position_size == 0:
        action = "SELL" if current_position > 0 else "BUY"
        quantity = abs(current_position)
    elif current_position == 0:
        action = "BUY" if position_size > 0 else "SELL"
        quantity = abs(position_size)
    else:
        if position_size > current_position:
            action = "BUY"
            quantity = position_size - current_position
        else: # position_size < current_position
            action = "SELL"
            quantity = current_position - position_size

    if action:
        order_data = data.copy()
        order_data["action"] = action
        order_data["quantity"] = str(quantity)
        return place_order_api(order_data, auth)

    return None, {"status": "success", "message": "No action taken."}, None

def close_all_positions(auth):
    payload = {"exit_all": 1}
    response_data = get_api_response("/api/v3/positions", auth, "DELETE", payload)

    if response_data.get("s") == "ok":
        return {"status": "success", "message": "All positions closed successfully"}, 200
    else:
        error_msg = response_data.get("message", "Failed to close positions")
        return {"status": "error", "message": error_msg}, 400

def cancel_order(orderid, auth):
    payload = {"id": orderid}
    response_data = get_api_response("/api/v3/orders/sync", auth, "DELETE", payload)

    if response_data.get("s") == "ok":
        return {"status": "success", "orderid": response_data.get('id')}, 200
    else:
        error_msg = response_data.get("message", "Failed to cancel order")
        return {"status": "error", "message": error_msg}, 400

def modify_order(data, auth):
    payload = transform_modify_order_data(data)
    response_data = get_api_response("/api/v3/orders/sync", auth, "PATCH", payload)

    if response_data.get("s") in ["ok", "OK"]:
        return {"status": "success", "orderid": response_data.get("id")}, 200
    else:
        error_msg = response_data.get("message", "Failed to modify order")
        return {"status": "error", "message": error_msg}, 400

def cancel_all_orders_api(auth):
    order_book_response = get_order_book(auth)
    if order_book_response.get('s') != 'ok':
        error_msg = order_book_response.get('message', 'Failed to retrieve order book')
        logger.error(f"Could not fetch order book to cancel all orders: {error_msg}")
        return [], []

    orders_to_cancel = [
        order for order in order_book_response.get('orderBook', [])
        if order.get('status') in [4, 6]  # 4: Trigger-pending, 6: Open
    ]

    if not orders_to_cancel:
        logger.info("No open orders to cancel.")
        return [], []

    canceled_orders, failed_cancellations = [], []
    for order in orders_to_cancel:
        orderid = order.get('id')
        if not orderid:
            logger.warning(f"Skipping order with no ID: {order}")
            continue

        cancel_response, status_code = cancel_order(orderid, auth)
        if status_code == 200:
            canceled_orders.append(orderid)
        else:
            failed_cancellations.append(orderid)

    return canceled_orders, failed_cancellations
