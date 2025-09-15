# Fyers API v2 - Order Management

This document provides details on the order management APIs for the Fyers API v2.

---

## API: Get Order Book

- **Function:** `get_order_book(auth)`
- **HTTP Method:** `GET`
- **Endpoint:** `https://api-t1.fyers.in/api/v3/orders`

### Description
Fetches the user's order book.

### Request
- **Arguments:** `auth` (str): The authentication token.

### Sample Response
```json
{
  "s": "ok",
  "code": 200,
  "message": "Order book retrieved successfully",
  "orderBook": [
    {
      "id": "123456789",
      "symbol": "NSE:SBIN-EQ",
      "side": 1,
      "type": 2,
      "status": 6,
      "qty": 10,
      "remainingQuantity": 10,
      "filledQty": 0,
      "limitPrice": 500.0,
      "stopPrice": 0.0,
      "disclosedQty": 0,
      "orderDateTime": "2023-01-01 10:00:00",
      "tradedPrice": 0.0
    }
  ]
}
```

---

## API: Get Trade Book

- **Function:** `get_trade_book(auth)`
- **HTTP Method:** `GET`
- **Endpoint:** `https://api-t1.fyers.in/api/v3/tradebook`

### Description
Fetches the user's trade book.

### Request
- **Arguments:** `auth` (str): The authentication token.

### Sample Response
```json
{
  "s": "ok",
  "code": 200,
  "message": "Trade book retrieved successfully",
  "tradeBook": [
    {
      "id": "987654321",
      "symbol": "NSE:SBIN-EQ",
      "side": 1,
      "type": 2,
      "qty": 10,
      "price": 501.0,
      "orderId": "123456789",
      "tradeTime": "2023-01-01 10:05:00"
    }
  ]
}
```

---

## API: Get Positions

- **Function:** `get_positions(auth)`
- **HTTP Method:** `GET`
- **Endpoint:** `https://api-t1.fyers.in/api/v3/positions`

### Description
Fetches the user's current positions.

### Request
- **Arguments:** `auth` (str): The authentication token.

### Sample Response
```json
{
  "s": "ok",
  "code": 200,
  "message": "Positions retrieved successfully",
  "netPositions": [
    {
      "symbol": "NSE:SBIN-EQ",
      "netQty": 10,
      "avgPrice": 501.0,
      "ltp": 505.0,
      "pl": 40.0
    }
  ]
}
```

---

## API: Get Holdings

- **Function:** `get_holdings(auth)`
- **HTTP Method:** `GET`
- **Endpoint:** `https://api-t1.fyers.in/api/v3/holdings`

### Description
Fetches the user's holdings.

### Request
- **Arguments:** `auth` (str): The authentication token.

### Sample Response
```json
{
  "s": "ok",
  "code": 200,
  "message": "Holdings retrieved successfully",
  "holdings": [
    {
      "symbol": "NSE:RELIANCE-EQ",
      "quantity": 5,
      "avgPrice": 2500.0,
      "ltp": 2600.0,
      "pl": 500.0
    }
  ]
}
```

---

## API: Place Order

- **Function:** `place_order_api(data, auth)`
- **HTTP Method:** `POST`
- **Endpoint:** `https://api-t1.fyers.in/api/v3/orders/sync`

### Description
Places a new order. Note that this function uses a placeholder `transform_data` function. You may need to implement the data transformation logic in `transform.py` to match your specific needs.

### Request
- **Arguments:**
  - `data` (dict): A dictionary containing the order details.
  - `auth` (str): The authentication token.
- **Sample Payload:**
  ```json
  {
    "symbol": "NSE:SBIN-EQ",
    "qty": 10,
    "type": 2,
    "side": 1,
    "productType": "INTRADAY",
    "limitPrice": 500,
    "stopPrice": 0,
    "validity": "DAY",
    "disclosedQty": 0,
    "offlineOrder": false
  }
  ```

### Sample Response
```json
{
  "s": "ok",
  "code": 200,
  "message": "Order placed successfully",
  "id": "1234567890"
}
```

---

## API: Modify Order

- **Function:** `modify_order(data, auth)`
- **HTTP Method:** `PATCH`
- **Endpoint:** `https://api-t1.fyers.in/api/v3/orders/sync`

### Description
Modifies an existing order. Note that this function uses a placeholder `transform_modify_order_data` function. You may need to implement the data transformation logic in `transform.py` to match your specific needs.

### Request
- **Arguments:**
  - `data` (dict): A dictionary containing the order modification details.
  - `auth` (str): The authentication token.
- **Sample Payload:**
  ```json
  {
    "id": "1234567890",
    "limitPrice": 501
  }
  ```

### Sample Response
```json
{
  "s": "ok",
  "code": 200,
  "message": "Order modified successfully",
  "id": "1234567890"
}
```

---

## API: Cancel Order

- **Function:** `cancel_order(orderid, auth)`
- **HTTP Method:** `DELETE`
- **Endpoint:** `https://api-t1.fyers.in/api/v3/orders/sync`

### Description
Cancels a single order.

### Request
- **Arguments:**
  - `orderid` (str): The ID of the order to be cancelled.
  - `auth` (str): The authentication token.
- **Sample Payload:**
  ```json
  {
    "id": "1234567890"
  }
  ```

### Sample Response
```json
{
  "s": "ok",
  "code": 200,
  "message": "Order cancelled successfully",
  "id": "1234567890"
}
```

---

## API: Close All Positions

- **Function:** `close_all_positions(auth)`
- **HTTP Method:** `DELETE`
- **Endpoint:** `https://api-t1.fyers.in/api/v3/positions`

### Description
Closes all open positions.

### Request
- **Arguments:**
  - `auth` (str): The authentication token.
- **Sample Payload:**
  ```json
  {
    "exit_all": 1
  }
  ```

### Sample Response
```json
{
  "s": "ok",
  "code": 200,
  "message": "All positions closed successfully"
}
```

---

## API: Cancel All Orders

- **Function:** `cancel_all_orders_api(auth)`

### Description
Cancels all open orders. This function first fetches the order book and then cancels each open order individually.

### Request
- **Arguments:**
  - `auth` (str): The authentication token.

### Sample Response
The function returns a tuple: `(canceled_orders, failed_cancellations)`.
- `canceled_orders`: A list of successfully cancelled order IDs.
- `failed_cancellations`: A list of order IDs that failed to be cancelled.
```python
(['12345', '67890'], [])
```
