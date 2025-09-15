# Fyers API v2 - Authentication

This document provides details on the authentication process for the Fyers API v2.

## API: Authenticate Broker

This API is used to authenticate the broker and get an access token.

- **Function:** `authenticate_broker(request_token: str)`
- **HTTP Method:** `POST`
- **Endpoint:** `https://api-t1.fyers.in/api/v3/validate-authcode`

### Description

This function takes a `request_token` (authorization code) and exchanges it for an `access_token`. It requires the `BROKER_API_KEY` and `BROKER_API_SECRET` to be set as environment variables.

### Request

#### Arguments

- `request_token` (str): The authorization code received from Fyers after a successful login.

#### Sample Request Payload

```json
{
  "grant_type": "authorization_code",
  "appIdHash": "your_app_id_hash",
  "code": "your_request_token"
}
```

**Note:** The `appIdHash` is a SHA-256 hash of `BROKER_API_KEY:BROKER_API_SECRET`.

### Response

#### Success Response

- **Status Code:** `200`
- **Content:**

```json
{
  "status": "success",
  "message": "Authentication successful",
  "data": {
    "access_token": "your_access_token",
    "refresh_token": "your_refresh_token",
    "expires_in": 86400
  }
}
```

#### Error Response

- **Status Code:** Varies (e.g., `400`, `500`)
- **Content:**

```json
{
  "status": "error",
  "message": "Authentication failed: [error_details]",
  "data": null
}
```
