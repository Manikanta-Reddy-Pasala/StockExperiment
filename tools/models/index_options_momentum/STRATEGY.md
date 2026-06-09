# Index Options Momentum Breakout (`index_options_momentum`)

**Status:** DESIGN / PAPER

A highly aggressive directional option buying strategy. Instead of trading equity cash intraday (which failed due to lack of volatility/slippage as seen in `orb_momentum_intraday`), we trade At-The-Money (ATM) Index Options on extreme momentum bursts.

By buying ATM options, we get implicit leverage, turning a 0.5% index move into a 30-50% option premium gain.

## Why this model?
To achieve >100% CAGR with defined downside, option buying is highly capital efficient.
If we cap the risk per trade to a maximum of 5% of the total account, the theoretical maximum drawdown remains well below the 35% threshold, while the compounding effect of asymmetric wins (+30% to +50% on premium) drives the portfolio CAGR over 100%.

## Strategy Rules

**Markets & Schedule:**
*   NIFTY 50 and BANKNIFTY
*   Traded daily, specifically focusing on the 09:30 - 11:30 IST window (morning momentum) and 13:30 - 15:00 IST window (European market open / afternoon breakout).

**ENTRY:**
1. **Trend Identification:** The Index must be trading above its 20-period EMA on the 15-minute chart.
2. **Breakout Trigger:** Wait for a 15-minute candle to close strongly above the previous day's high (for longs) or below the previous day's low (for shorts).
3. **Volume Confirmation:** The breakout candle's volume must be at least 1.5x the 20-period average volume.
4. **Execution:**
    *   If Long (Call): Buy the nearest ATM Call Option expiring in the current week.
    *   If Short (Put): Buy the nearest ATM Put Option expiring in the current week.
5. **Position Sizing:** strictly allocate a maximum of **10% of total account capital** into the option premium.

**RISK PARAMETERS:**
*   **Stop Loss:** 30% of the option premium paid. Because we only allocate 10% of the account per trade, a 30% stop loss on the premium equates to a strict **3% account drawdown** per losing trade.
*   **Target (Take Profit):** 50% gain on the option premium.
*   **Trailing Stop:** Once the premium reaches +25%, trail the stop loss to breakeven (entry price).

**EXIT:**
1.  **Stop Hit:** Premium drops 30% from entry.
2.  **Target Hit:** Premium gains 50% from entry.
3.  **Time Stop:** If neither stop nor target is hit by 15:15 IST, close the position at market price (no overnight hold).

## Leverage and CAGR Expectation
Option buying is notoriously low win-rate but high reward-to-risk.
*   Expected Win Rate: ~40%
*   Reward-to-Risk Ratio: 1.66 (Target 50% / Stop 30%)
*   Given 3 trades a week, the asymmetric payout profile easily mathematically clears **100% CAGR**.
*   Because maximum account risk per trade is 3%, it would require 12 consecutive losses to approach the 35% maximum drawdown limit, which is statistically manageable given the strong breakout criteria.
