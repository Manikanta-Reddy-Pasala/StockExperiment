# High Win Rate MACD Trading Strategy

Based on the trading strategy outlined in the YouTube video (https://www.youtube.com/watch?v=rf_EQvubKlk).

## 1. Core Indicator: MACD (Moving Average Convergence Divergence)
The MACD indicator is used to find trends and momentum in the market. It consists of 4 main components:
*   **MACD Line (Blue Line):** Usually a 12-day moving average.
*   **Signal Line (Orange Line):** Usually a 26-day moving average.
*   **Histogram:** Represents the space/difference between the MACD line and the Signal line.
    *   An expanding histogram indicates increasing momentum.
    *   A shrinking histogram indicates decreasing momentum.
    *   Turns green when MACD crosses above the Signal line; turns red when MACD crosses below the Signal line.
*   **Zero Line:** The center line of the MACD indicator.

### Base MACD Rules (Flawed on their own)
*   **Long:** MACD line crosses *upward* above the Signal line, but **only if the cross happens below the Zero Line**.
*   **Short:** MACD line crosses *downward* below the Signal line, but **only if the cross happens above the Zero Line**.
*   *Flaw:* Using this base strategy alone fails when the market is in a downtrend (for longs) or moving sideways, leading to false signals.

---

## 2. Trend Confirmation: 200-Day Moving Average (200 DMA)
To fix the issue of trading against the trend, the strategy adds a 200-Day Moving Average (200 DMA) to the chart.

*   **Uptrend:** Price is trading *above* the 200 DMA. (Only look for Long setups).
*   **Downtrend:** Price is trading *below* the 200 DMA. (Only look for Short setups).

### Strategy Combination 1 (MACD + 200 DMA)

#### Long Entry Rules:
1.  Current price must be **above the 200 DMA** (confirming an uptrend).
2.  Wait for the MACD lines to cross **upward**.
3.  The MACD cross must happen **below the Zero Line**.

#### Short Entry Rules:
1.  Current price must be **below the 200 DMA** (confirming a downtrend).
2.  Wait for the MACD lines to cross **downward**.
3.  The MACD cross must happen **above the Zero Line**.

#### Exit Rules (Stop Loss & Take Profit):
*   **Stop Loss:** Placed strictly **below the 200 DMA**. The 200 DMA acts as a "wall" that the price must break through to hit the stop loss.
*   **Take Profit:** Target a **1.5 Risk-to-Reward ratio** based on your stop loss distance.

---

## 3. Fine-Tuning: Price Action (Support & Resistance)
Even with the 200 DMA, the strategy can still give false signals if the market loses momentum and starts moving sideways in a tight range.

To fine-tune the strategy and maximize the win rate, you must combine the MACD + 200 DMA with **Price Action**.

#### Ultimate Fine-Tuned Entry Rules (Long Example):
1.  **Trend Filter:** Verify the current price is **above the 200 DMA**.
2.  **Price Action (Support):** Identify a key support level on the chart where the price previously hit and bounced upwards.
3.  **Wait for the Retest:** Wait for the price to drop and hit that exact same key support level again.
4.  **MACD Trigger:** Once the price hits support, do not enter blindly (it could break lower). Instead, wait for the MACD lines to **cross upward** while being **below the Zero Line**.
5.  **Execution:** Enter the trade immediately upon the MACD cross.

By combining the 200 DMA (Trend), Support/Resistance (Price Action), and the MACD (Momentum/Trigger), you drastically reduce false signals and enter trades with an extremely high probability of success.