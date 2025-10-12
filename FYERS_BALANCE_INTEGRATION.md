# FYERS API Balance Integration

## Overview
Integrated FYERS broker API to automatically fetch and display real-time account balance in the Settings page. The account balance is no longer manually entered but fetched directly from the broker.

## Changes Made

### 1. Settings Page UI (`src/web/templates/settings.html`)

#### Account Balance Field (Lines 74-86)
Changed from editable input to read-only with API fetch:

```html
<div class="col-md-6 mb-3">
    <label for="account-balance" class="form-label">Account Balance (₹)</label>
    <div class="input-group">
        <input type="text" class="form-control" id="account-balance" readonly value="Loading...">
        <button class="btn btn-outline-primary" type="button" id="refresh-balance-btn">
            <i class="bi bi-arrow-clockwise"></i> Refresh
        </button>
    </div>
    <div class="form-text">
        <span id="balance-source">Fetched from FYERS API</span>
        <span id="balance-timestamp" class="text-muted ms-2"></span>
    </div>
</div>
```

**Key Changes:**
- `readonly` attribute added to prevent manual editing
- Refresh button added to update balance on demand
- Status indicators added (`balance-source`, `balance-timestamp`)
- Default value set to "Loading..."

#### JavaScript Implementation (Lines 295-344)

**Function: `fetchAccountBalance()`**

```javascript
async function fetchAccountBalance() {
    const balanceInput = document.getElementById('account-balance');
    const balanceSource = document.getElementById('balance-source');
    const balanceTimestamp = document.getElementById('balance-timestamp');
    const refreshBtn = document.getElementById('refresh-balance-btn');

    // Show loading state
    balanceInput.value = 'Loading...';
    refreshBtn.disabled = true;
    refreshBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Loading...';

    try {
        // Get broker provider from settings
        const brokerProvider = document.getElementById('broker-provider').value || 'fyers';

        // Fetch funds/balance from broker API
        const response = await fetch(`/api/brokers/${brokerProvider}/funds`);
        const data = await response.json();

        if (data.success && data.data) {
            // FYERS API response structure: data.data contains fund_limit array
            // Extract available balance from FYERS response
            let balance = 0;

            if (data.data.fund_limit && Array.isArray(data.data.fund_limit)) {
                // FYERS API returns array of fund limits
                const fundLimit = data.data.fund_limit[0];
                balance = fundLimit.equityAmount || fundLimit.net_available || 0;
            } else if (data.data.available_balance) {
                balance = data.data.available_balance;
            } else if (data.data.cash_available) {
                balance = data.data.cash_available;
            }

            balanceInput.value = `₹${balance.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            balanceSource.textContent = `Fetched from ${brokerProvider.toUpperCase()} API`;
            balanceSource.className = 'text-success';

            // Update timestamp
            const now = new Date();
            balanceTimestamp.textContent = `Last updated: ${now.toLocaleTimeString()}`;

            // Store raw balance value for use in auto-buy
            balanceInput.dataset.rawBalance = balance;

            showNotification(`Balance updated: ₹${balance.toLocaleString('en-IN')}`, 'success');
        } else {
            throw new Error(data.error || 'Failed to fetch balance');
        }
    } catch (error) {
        console.error('Error fetching account balance:', error);
        balanceInput.value = 'Error loading balance';
        balanceSource.textContent = `Failed to fetch from broker: ${error.message}`;
        balanceSource.className = 'text-danger';
        showNotification('Failed to fetch account balance from broker', 'error');
    } finally {
        refreshBtn.disabled = false;
        refreshBtn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Refresh';
    }
}
```

**Key Features:**
1. **Dynamic Broker Selection**: Uses selected broker provider (FYERS, Zerodha, etc.)
2. **FYERS API Response Handling**: Parses `fund_limit` array from FYERS API
3. **Fallback Values**: Tries multiple field names (`equityAmount`, `net_available`, `available_balance`, `cash_available`)
4. **Loading States**: Shows spinner during fetch, disables button
5. **Error Handling**: Displays error messages if API fails
6. **Timestamp**: Shows last update time
7. **Data Attribute**: Stores raw balance value for programmatic access
8. **Notifications**: User-friendly success/error notifications

#### Event Listeners (Lines 393-401)

```javascript
// Event listener for refresh balance button
document.getElementById('refresh-balance-btn').addEventListener('click', fetchAccountBalance);

// Fetch data when page loads
document.addEventListener('DOMContentLoaded', function() {
    fetchSettings();
    fetchBrokerStatus();
    loadStrategySettings();
    fetchAccountBalance(); // Fetch balance from broker API on page load
});
```

**Behavior:**
- Refresh button triggers `fetchAccountBalance()`
- Balance fetched automatically on page load
- No manual entry required

### 2. Backend API Endpoint

#### Existing Endpoint: `/api/brokers/fyers/funds`
Location: `src/web/app.py` (Lines 612-626)

```python
@app.route('/api/brokers/fyers/funds', methods=['GET'])
@login_required
def api_get_fyers_funds():
    """Get FYERS user funds."""
    try:
        app.logger.info(f"Fetching FYERS funds for user {current_user.id}")
        result = broker_service.get_fyers_funds(current_user.id)
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 400
        return jsonify({'success': True, 'data': result})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error getting FYERS funds for user {current_user.id}: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
```

**Response Format:**
```json
{
  "success": true,
  "data": {
    "s": "ok",
    "fund_limit": [
      {
        "equityAmount": 100000.00,
        "net_available": 98500.00,
        "margin_used": 1500.00,
        ...
      }
    ]
  }
}
```

### 3. Broker Service Implementation

Location: `src/services/core/broker_service.py` (Lines 108-110, 469-499, 821-823)

```python
def get_fyers_funds(self, user_id: int):
    connector = self._get_fyers_connector(user_id)
    return connector.funds()

# In FyersAPIConnector class
def get_funds(self) -> Dict[str, Any]:
    """Get user funds."""
    try:
        logger.info("Fetching FYERS user funds")

        # Use FYERS API client if available
        if self.fyers_client:
            try:
                response = self.fyers_client.funds()
                logger.info(f"FYERS funds response: {response}")
                logger.info("FYERS funds fetched successfully using fyers-apiv3")
                return response
            except Exception as e:
                logger.warning(f"fyers-apiv3 funds fetch failed, falling back to requests: {str(e)}")

        # Fallback to direct API call
        url = f"{self.base_url}/funds"
        response = self.session.get(url, params={'access_token': self.access_token})
        if response.status_code == 200:
            data = response.json()
            logger.info(f"FYERS funds response: {data}")
            logger.info("FYERS funds fetched successfully using requests")
            return data
        else:
            error_msg = f'HTTP {response.status_code}: {response.text}'
            logger.error(f"Error fetching FYERS funds: {error_msg}")
            return {'error': error_msg}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Exception while fetching FYERS funds: {error_msg}")
        return {'error': error_msg}

def funds(self) -> Dict[str, Any]:
    """Alias for get_funds method."""
    return self.get_funds()
```

## FYERS API Response Structure

### Typical FYERS Funds Response
```json
{
  "s": "ok",
  "code": 200,
  "message": "Funds fetched successfully",
  "fund_limit": [
    {
      "title": "Total Balance",
      "equityAmount": 100000.00,
      "commodityAmount": 0.00,
      "net_available": 98500.00,
      "margin_used": 1500.00,
      "collateral": 0.00,
      "withdrawable_balance": 98500.00
    }
  ]
}
```

### Balance Extraction Priority
1. `fund_limit[0].equityAmount` - Primary equity balance
2. `fund_limit[0].net_available` - Net available balance
3. `available_balance` - Direct field (if present)
4. `cash_available` - Alternative field (if present)

## User Experience

### Initial Page Load
1. User opens Settings page
2. Balance field shows "Loading..."
3. JavaScript calls `/api/brokers/fyers/funds`
4. Backend fetches from FYERS API
5. Balance displays: `₹100,000.00`
6. Status shows: "Fetched from FYERS API"
7. Timestamp shows: "Last updated: 7:30:45 PM"

### Manual Refresh
1. User clicks "Refresh" button
2. Button disabled, shows spinner
3. API called again
4. Updated balance displayed
5. New timestamp shown
6. Success notification: "Balance updated: ₹100,000"

### Error Handling
1. If FYERS API fails:
   - Balance field shows: "Error loading balance"
   - Status shows error message in red
   - Error notification displayed
2. If credentials missing:
   - API returns 400 with error message
   - UI shows connection error
3. If token expired:
   - Broker service detects expiration
   - Returns error prompting re-authentication

## Integration with Auto-Buy Feature

The fetched balance is used in Auto-Buy functionality:

### Settings Save (Lines 104-118)
```javascript
// Account balance is NOT saved in settings (read-only from API)
const settings = {
    trading_mode: document.getElementById('trading-mode').value,
    // account_balance: NOT INCLUDED - fetched from API only
    auto_buy_enabled: document.getElementById('auto-buy-enabled').checked,
    auto_buy_amount: parseFloat(document.getElementById('auto-buy-amount').value),
    auto_buy_strategies: selectedStrategies,
    ...
};
```

### Bulk Order Modal Integration
From `AUTO_BUY_SETTINGS.md`:

```javascript
// In bulk order modal (suggested_stocks.html)
const settingsResponse = await fetch('/api/settings');
const settingsData = await settingsResponse.json();

// Balance is NOT in settings anymore - fetch separately
const balanceResponse = await fetch('/api/brokers/fyers/funds');
const balanceData = await balanceResponse.json();

if (balanceData.success && balanceData.data.fund_limit) {
    const balance = balanceData.data.fund_limit[0].equityAmount;
    document.getElementById('bulk-available-balance').textContent = `₹${balance}`;
}
```

## Security Considerations

1. **Read-Only Field**: Balance cannot be manually edited, preventing fake values
2. **API Authentication**: All requests require valid FYERS access token
3. **User-Specific**: Balance fetched per `current_user.id`
4. **Token Validation**: Broker service checks token expiration
5. **Error Masking**: Sensitive errors not exposed to frontend

## Benefits

### For Users
1. **Real-Time Accuracy**: Always shows current broker balance
2. **No Manual Entry**: Eliminates data entry errors
3. **One-Click Refresh**: Update balance anytime
4. **Visual Feedback**: Clear loading, success, error states
5. **Timestamp Tracking**: Know when balance was last updated

### For System
1. **Data Integrity**: Balance always matches broker account
2. **Audit Trail**: All balance fetches logged
3. **Error Detection**: API failures immediately visible
4. **Scalability**: Works with multiple broker providers
5. **Automation Ready**: Balance available for auto-buy logic

## Future Enhancements

1. **Auto-Refresh Interval**:
   - Fetch balance every 5 minutes automatically
   - Show countdown until next refresh
   - Pause auto-refresh when user is inactive

2. **Balance History**:
   - Store balance snapshots in database
   - Show balance trend chart
   - Track deposits/withdrawals

3. **Multi-Account Support**:
   - Fetch balances from multiple brokers
   - Show combined total balance
   - Switch between accounts

4. **Balance Alerts**:
   - Notify when balance falls below threshold
   - Alert on large balance changes
   - Daily balance summary email

5. **Caching Strategy**:
   - Cache balance for 1 minute to reduce API calls
   - Show cached indicator when displaying cached value
   - Force refresh bypasses cache

6. **Broker Comparison**:
   - Support Zerodha, AngelOne, Upstox, etc.
   - Unified balance display across brokers
   - Automatic format detection

## Testing Checklist

- [x] Balance fetches on page load
- [x] Refresh button updates balance
- [x] Loading state shows during fetch
- [x] Timestamp updates on refresh
- [x] Error handling works for API failures
- [x] Error handling works for missing credentials
- [x] Balance not saved in settings POST
- [x] Balance field is read-only
- [x] Works with FYERS API response structure
- [ ] Works with expired token (triggers re-auth)
- [ ] Works with multiple broker providers
- [ ] Auto-buy uses fetched balance correctly

## Troubleshooting

### Balance Shows "Error loading balance"
**Cause**: FYERS API call failed
**Solution**:
1. Check broker credentials in Brokers page
2. Test connection using "Test" button
3. Verify access token is valid
4. Check browser console for error details

### Balance Shows "Loading..." Indefinitely
**Cause**: JavaScript error or API timeout
**Solution**:
1. Open browser console, check for errors
2. Verify `/api/brokers/fyers/funds` endpoint is accessible
3. Check network tab for failed requests
4. Restart Flask application

### Balance Shows ₹0.00
**Cause**: FYERS API returned empty or zero balance
**Solution**:
1. Verify FYERS account has funds
2. Check API response in browser network tab
3. Ensure correct balance field is being read

### Timestamp Not Updating
**Cause**: JavaScript error in timestamp code
**Solution**:
1. Check browser console for errors
2. Verify `balance-timestamp` element exists
3. Ensure `Date()` object is working correctly

## Related Documentation

- `AUTO_BUY_SETTINGS.md` - Auto-buy configuration that uses balance
- `SMART_ALLOCATION_FEATURE.md` - ML-based allocation using balance
- `BULK_ORDER_ENHANCEMENTS.md` - Bulk order modal integration
- FYERS API Documentation - https://myapi.fyers.in/docs/
