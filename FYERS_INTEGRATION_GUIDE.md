# FYERS API Integration - Complete Setup Guide

## Overview
This document explains the complete FYERS API integration flow with proper OAuth2 authentication and token management.

## Architecture

### Key Components

1. **FyersService** (`src/services/brokers/fyers_service.py`)
   - Main service class handling all FYERS operations
   - Manages configuration storage in database
   - Handles OAuth2 flow and token management
   - Provides API wrapper methods

2. **FyersOAuth2Flow** 
   - OAuth2 authentication flow handler
   - Generates authorization URLs
   - Exchanges auth codes for access tokens

3. **FyersAPIConnector**
   - Direct API connection testing
   - Wraps FYERS API calls
   - Fallback to direct HTTP requests if library unavailable

4. **Routes** (`src/web/routes/brokers/fyers_routes.py`)
   - `/brokers/fyers/` - Main configuration page
   - `/brokers/fyers/api/config` - Save configuration
   - `/brokers/fyers/oauth/callback` - OAuth callback handler
   - `/brokers/fyers/api/oauth/complete` - Complete OAuth flow
   - `/brokers/fyers/api/test` - Test connection
   - Various API endpoints for funds, holdings, positions, etc.

5. **Frontend** (`src/web/templates/brokers/fyers.html`)
   - Configuration UI for entering credentials
   - Connection status display
   - Token management interface

## Setup Process

### Step 1: Create FYERS App
1. Go to [FYERS API Dashboard](https://myapi.fyers.in/dashboard/)
2. Create a new App
3. Note down:
   - App ID (Client ID) - e.g., "XC9XXX5XM-100"
   - App Secret - e.g., "MH*****TJ5"
   - Set Redirect URI - e.g., "http://localhost:5001/brokers/fyers/oauth/callback"

### Step 2: Configure in Application
1. Navigate to FYERS configuration page
2. Enter:
   - App ID (Client ID)
   - App Secret
   - Redirect URI (must match what you set in FYERS dashboard)
3. Click "Save Configuration"

### Step 3: Authorization Flow
1. After saving config, the system automatically opens FYERS login page
2. Login with your FYERS credentials
3. Approve the app permissions
4. FYERS redirects to callback URL with auth_code
5. Our callback handler automatically:
   - Captures the auth_code
   - Exchanges it for access_token
   - Saves token to database
   - Updates connection status to "Connected"

## Token Management

### Token Expiration
- FYERS access tokens expire every 2 hours
- System automatically detects expired tokens
- Shows warning message when token is expired
- "Refresh Token" button to re-authenticate

### Token Storage
Tokens are stored in the `BrokerConfiguration` table:
- `client_id` - App ID
- `api_secret` - App Secret (encrypted recommended)
- `redirect_url` - OAuth redirect URI
- `access_token` - Current access token
- `is_connected` - Connection status
- `connection_status` - Detailed status ('connected', 'disconnected', 'expired')

## API Flow Diagram

```
User → Configure Credentials → Generate Auth URL
                                      ↓
                            Open FYERS Login Page
                                      ↓
                              User Authenticates
                                      ↓
                        FYERS Redirects to Callback
                                      ↓
                      Callback Page Captures auth_code
                                      ↓
                    Exchange auth_code for access_token
                                      ↓
                        Save Token to Database
                                      ↓
                     Update Status to "Connected"
```

## Key Features Implemented

1. **No Hardcoded URLs**: Redirect URI is dynamically read from configuration
2. **Automatic Token Capture**: Callback page automatically processes auth_code
3. **Token Persistence**: Tokens saved to database for reuse
4. **Status Tracking**: Real-time connection status updates
5. **Error Handling**: Comprehensive error messages and logging
6. **Token Expiration Detection**: Automatic detection of expired tokens
7. **UI Notifications**: Clean notification system instead of alert popups

## Troubleshooting

### Common Issues

1. **"Token Expired" Status**
   - Click "Refresh Token" to re-authenticate
   - Complete the login flow again

2. **"Connection Failed"**
   - Verify App ID and Secret are correct
   - Check if redirect URI matches FYERS app settings
   - Ensure FYERS API is accessible

3. **OAuth Callback Not Working**
   - Ensure redirect URI in FYERS app matches exactly
   - Check browser console for JavaScript errors
   - Verify backend server is running on correct port

### Debug Steps

1. Check logs in:
   - Browser console (F12)
   - Python application logs
   - Database BrokerConfiguration table

2. Test connection manually:
   - Click "Test Connection" button
   - Check response time and status

3. Verify database entries:
   ```sql
   SELECT * FROM broker_configuration WHERE broker_name = 'fyers';
   ```

## Security Considerations

1. **Never commit credentials** to version control
2. **Use environment variables** for sensitive data in production
3. **Implement token encryption** before storing in database
4. **Use HTTPS** in production for redirect URIs
5. **Implement rate limiting** for API calls
6. **Add user authentication** before allowing broker configuration

## Future Enhancements

1. Automatic token refresh before expiration
2. WebSocket support for real-time data
3. Order placement functionality
4. Historical data fetching
5. Multi-user support with proper isolation
6. Audit logging for all broker operations

## Support

For FYERS API documentation: https://myapi.fyers.in/docsv3
For issues with this integration: Check application logs and database status
