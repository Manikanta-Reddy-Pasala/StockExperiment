# Fyers API v2 - Funds Management

This document provides details on the funds management API for the Fyers API v2.

---

## API: Get Margin Data

- **Function:** `get_margin_data(auth_token)`
- **HTTP Method:** `GET`
- **Endpoint:** `https://api-t1.fyers.in/api/v3/funds`

### Description
Fetches and processes the user's margin and funds data from the Fyers API.

### Request
- **Arguments:**
  - `auth_token` (str): The authentication token for the Fyers API.

### Sample Response
```json
{
  "availablecash": "100000.00",
  "collateral": "50000.00",
  "m2munrealized": "50000.00",
  "m2mrealized": "1000.00",
  "utiliseddebits": "25000.00"
}
```

### Response Fields
- `availablecash`: Total available balance.
- `collateral`: Collateral value.
- `m2munrealized`: Unrealized M2M (Mark-to-Market). **Note:** This implementation assumes the unrealized M2M is the same as the collateral value. This should be verified against the Fyers API documentation for your specific use case.
- `m2mrealized`: Realized M2M.
- `utiliseddebits`: Utilized amount.
