# Fyers API v2 - Market Data

This document provides details on the market data APIs for the Fyers API v2.

---

## Class: `BrokerData`

This class handles all the market data-related API calls.

### Initialization

```python
broker_data = BrokerData(auth_token)
```

- **Arguments:** `auth_token` (str): The authentication token.

---

## API: Get Quotes

- **Method:** `broker_data.get_quotes(symbol, exchange)`
- **HTTP Method:** `GET`
- **Endpoint:** `https://api-t1.fyers.in/data/quotes`

### Description
Fetches real-time quotes for a given symbol.

### Request
- **Arguments:**
  - `symbol` (str): The trading symbol (e.g., `SBIN-EQ`).
  - `exchange` (str): The exchange (e.g., `NSE`).

### Sample Response
```json
{
  "bid": 500.0,
  "ask": 500.1,
  "open": 498.0,
  "high": 502.0,
  "low": 497.0,
  "ltp": 500.05,
  "prev_close": 497.5,
  "volume": 1234567
}
```

---

## API: Get Historical Data

- **Method:** `broker_data.get_history(symbol, exchange, interval, start_date, end_date)`
- **HTTP Method:** `GET`
- **Endpoint:** `https://api-t1.fyers.in/data/history`

### Description
Fetches historical data for a given symbol.

### Request
- **Arguments:**
  - `symbol` (str): The trading symbol.
  - `exchange` (str): The exchange.
  - `interval` (str): The candle interval. Supported values: `5s`, `10s`, `15s`, `30s`, `45s`, `1m`, `2m`, `3m`, `5m`, `10m`, `15m`, `20m`, `30m`, `1h`, `2h`, `4h`, `D`.
  - `start_date` (str): The start date in `YYYY-MM-DD` format.
  - `end_date` (str): The end date in `YYYY-MM-DD` format.

### Sample Response
The method returns a Pandas DataFrame with the following columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `oi`.

```
       timestamp   open   high    low  close   volume  oi
0     1672531200  500.0  502.0  498.0  501.0  1000000   0
1     1672534800  501.0  503.0  499.0  502.0  1200000   0
...
```

---

## API: Get Market Depth

- **Method:** `broker_data.get_depth(symbol, exchange)`
- **HTTP Method:** `GET`
- **Endpoint:** `https://api-t1.fyers.in/data/depth`

### Description
Fetches the market depth for a given symbol.

### Request
- **Arguments:**
  - `symbol` (str): The trading symbol.
  - `exchange` (str): The exchange.

### Sample Response
```json
{
  "bids": [
    {"price": 500.0, "quantity": 100},
    {"price": 499.95, "quantity": 200},
    ...
  ],
  "asks": [
    {"price": 500.05, "quantity": 150},
    {"price": 500.1, "quantity": 250},
    ...
  ],
  "totalbuyqty": 12345,
  "totalsellqty": 54321,
  "high": 502.0,
  "low": 497.0,
  "ltp": 500.05,
  "ltq": 10,
  "open": 498.0,
  "prev_close": 497.5,
  "volume": 1234567,
  "oi": 0
}
```
