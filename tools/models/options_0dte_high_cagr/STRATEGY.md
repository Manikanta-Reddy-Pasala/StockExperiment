# Multi-Index 0DTE Iron-Fly (`options_0dte_high_cagr`)

**Status:** DESIGN / PAPER

Defined-risk 0DTE premium selling across multiple Indian indices on their respective expiry days.
This strategy capitalizes on the rapid theta decay (time value crush) on the final day of an option's life.

## Why this model?
To achieve a >100% CAGR while strictly keeping drawdowns < 35%, we must use leverage, but we must *cap the downside*.
Naked short strangles offer high win rates but expose the portfolio to black swan gap risk and massive intraday trend days that can easily exceed a 35% drawdown.
The **0DTE Iron-Fly** defines risk per trade (wing minus credit), ensuring a single bad day only costs a known, fixed percentage of margin.

By trading multiple indices (FINNIFTY on Tuesday, BANKNIFTY on Wednesday, NIFTY on Thursday), we increase trade frequency and capital turnover, rapidly compounding the defined-risk edge.

## Strategy Rules

**Markets & Schedule:**
*   Tuesday: FINNIFTY Expiry
*   Wednesday: BANKNIFTY Expiry
*   Thursday: NIFTY 50 Expiry
*   *(Optional: SENSEX on Friday, MIDCPNIFTY on Monday)*

**ENTRY (09:15 - 09:20 IST):**
1. **Find ATM:** Calculate the At-The-Money (ATM) strike via put-call parity or nearest spot price.
2. **Sell Straddle / Iron-Fly:** Sell ATM Call (CE) + Sell ATM Put (PE).
3. **Buy Wings (Protection):** Buy Out-Of-The-Money (OTM) Call and Put options exactly 1% to 1.5% away from the ATM strike. This creates an "Iron Fly".
4. **Execution:** Must be executed as a **BASKET ORDER**. Legging in separately destroys the defined risk profile due to slippage.

**RISK PARAMETERS:**
*   **Net Credit:** The premium received from the short legs minus the premium paid for the long legs. This is the Max Profit.
*   **Max Loss (Margin Deployed):** `(Wing Width) - (Net Credit)`. This is the absolute maximum capital at risk per trade.
*   **Stop Loss:** 1.5x to 2.0x of the Net Credit received, tracked purely on the combined premium of the short legs. *We exit if the premium spikes.*

**EXIT:**
1.  **Stop Hit:** If the short straddle premium expands to `Entry Premium * Stop Multiplier` (e.g., 2x). Close all 4 legs.
2.  **Target Hit / Settle:** If the stop is not hit, hold until 15:20 IST (or exchange settlement) and close all legs. If the index closes near the ATM strike, the short legs expire worthless, yielding the Max Profit.

## Leverage and CAGR Expectation
Because the risk is strictly defined (e.g., ₹5000 max loss per lot), brokers (SPAN margin) require significantly less capital compared to naked selling.
*   Expected Win Rate: ~70-75%
*   Average Win: 3-5% of Margin
*   Average Loss: 4-8% of Margin (capped)
*   Compounding this 3-4 days a week yields a mathematical expectancy capable of **150%+ CAGR**, with the defined risk hard-capping the max drawdown well below **30%**.

## Execution Infrastructure
Since this is 0DTE options, the logic relies heavily on near-instant execution. The backtest script uses historical option data to simulate the realistic fills and capped losses.
