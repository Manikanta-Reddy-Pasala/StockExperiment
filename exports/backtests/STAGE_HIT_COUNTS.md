# EMA 200/400 1H — Stage Hit Distribution (Nifty 500 Fyers)

_Source: Fyers production data, 504 NSE Nifty 500 symbols, 720 days, 1H bars._
_Backtest run on 77.42.45.12 inside trading_system_app container._

## Total Signal Counts

| Stage | Count | Description |
|-------|-------|-------------|
| CROSSOVER | 3200 | Trend identification (EMA200 crosses EMA400) |
| ALERT1 | 2750 | First alert (price breaks crossover candle high/low + close) |
| ALERT2 | 2625 | Second alert (Retest 1: price touches EMA200 from opposite side) |
| ENTRY1 | 2034 | First Entry — break of retest1 candle |
| ALERT3 | 1508 | Third alert (Retest 2: price touches EMA400) |
| ENTRY2 | 707 | Second Entry — break of retest2 candle |
| EXIT | 1976 | 1H close on wrong side of EMA400 |

## BUY vs SELL Split

| Stage | BUY | SELL | Total |
|-------|-----|------|-------|
| CROSSOVER | 1513 | 1687 | 3200 |
| ALERT1 | 1251 | 1499 | 2750 |
| ALERT2 | 1133 | 1492 | 2625 |
| ALERT3 | 656 | 852 | 1508 |
| ENTRY1 | 869 | 1165 | 2034 |
| ENTRY2 | 286 | 421 | 707 |
| EXIT | 850 | 1126 | 1976 |

## Conversion Ratios (funnel)

| Transition | Ratio | Interpretation |
|------------|-------|----------------|
| Crossover → Alert1 | 85.9% | How often trend confirms with breakout |
| Alert1 → Alert2 | 95.5% | How often price pulls back to EMA200 |
| Alert2 → Entry1 | 77.5% | How often retest1 break triggers entry |
| Entry1 → Alert3 | 74.1% | How often price tests EMA400 after entry |
| Alert3 → Entry2 | 46.9% | How often pyramid 2nd entry triggers |
| Entry → Exit | 72.1% | How often EMA400 cross stops trade |

## Notes
- ENTRY1 and ENTRY2 counts are individual entries; some cycles produce 2.
- ALERT3/ENTRY2 can fire multiple times per cycle (pyramid loop on EMA400 retests).
- EXIT fires once per cycle, closing all open entries together.
- CROSSOVER count = number of trend flips × 504 stocks.
