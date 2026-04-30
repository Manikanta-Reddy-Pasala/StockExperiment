# ABB India Ltd. (ABB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 7230.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 9 |
| ENTRY1 | 7 |
| ENTRY2 | 5 |
| EXIT | 6 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 1
- **Winners / losers:** 2 / 9
- **Total realized P&L (per unit):** 807.05
- **Avg P&L per closed trade:** 73.37

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-10-01 09:15:00 | ALERT1 | BUY | 8218.80 | 7856.26 | 7828.32 | Break + close above crossover candle high |
| 2024-10-07 10:15:00 | ALERT2 | BUY | 7850.00 | 7902.05 | 7855.76 | EMA200 retest candle locked |
| 2024-10-08 10:15:00 | ENTRY1 | BUY | 7989.50 | 7896.31 | 7854.42 | Buy entry 1 (retest1 break) |
| 2024-10-23 09:15:00 | ALERT3 | BUY | 8089.95 | 8230.38 | 8066.34 | EMA400 retest candle locked |
| 2024-10-23 10:15:00 | EXIT | BUY | 8062.05 | 8228.70 | 8066.32 | Close below EMA400 |
| 2024-11-04 10:15:00 | CROSSOVER | SELL | 7289.60 | 7940.49 | 7941.87 | EMA200 below EMA400 |
| 2024-11-05 09:15:00 | ALERT1 | SELL | 7006.35 | 7902.48 | 7922.56 | Break + close below crossover candle low |
| 2024-11-26 10:15:00 | ALERT2 | SELL | 7347.70 | 7325.69 | 7563.31 | EMA200 retest candle locked |
| 2024-12-20 12:15:00 | ENTRY1 | SELL | 7105.10 | 7519.72 | 7573.70 | Sell entry 1 (retest1 break) |
| 2025-04-17 10:15:00 | EXIT | SELL | 5617.00 | 5319.03 | 5574.88 | Close above EMA400 |
| 2025-05-26 10:15:00 | CROSSOVER | BUY | 5997.50 | 5621.80 | 5621.00 | EMA200 above EMA400 |
| 2025-05-29 12:15:00 | ALERT1 | BUY | 6047.00 | 5702.93 | 5663.93 | Break + close above crossover candle high |
| 2025-06-19 12:15:00 | ALERT2 | BUY | 5885.00 | 5922.37 | 5817.98 | EMA200 retest candle locked |
| 2025-06-20 10:15:00 | ENTRY1 | BUY | 5962.50 | 5921.33 | 5820.02 | Buy entry 1 (retest1 break) |
| 2025-07-02 13:15:00 | ALERT3 | BUY | 5897.50 | 5956.65 | 5866.08 | EMA400 retest candle locked |
| 2025-07-02 14:15:00 | ENTRY2 | BUY | 5907.50 | 5956.16 | 5866.29 | Buy entry 2 (retest2 break) |
| 2025-07-03 14:15:00 | ALERT3 | BUY | 5875.00 | 5951.89 | 5867.21 | EMA400 retest candle locked |
| 2025-07-03 15:15:00 | EXIT | BUY | 5866.50 | 5951.04 | 5867.21 | Close below EMA400 |
| 2025-07-23 10:15:00 | CROSSOVER | SELL | 5691.50 | 5820.86 | 5821.17 | EMA200 below EMA400 |
| 2025-07-24 09:15:00 | ALERT1 | SELL | 5663.50 | 5813.58 | 5817.47 | Break + close below crossover candle low |
| 2025-09-11 10:15:00 | ALERT2 | SELL | 5233.30 | 5204.64 | 5363.89 | EMA200 retest candle locked |
| 2025-09-26 09:15:00 | ENTRY1 | SELL | 5148.70 | 5269.12 | 5352.27 | Sell entry 1 (retest1 break) |
| 2025-10-07 09:15:00 | ALERT3 | SELL | 5316.50 | 5244.00 | 5322.51 | EMA400 retest candle locked |
| 2025-10-07 14:15:00 | ENTRY2 | SELL | 5215.00 | 5244.49 | 5320.83 | Sell entry 2 (retest2 break) |
| 2025-10-29 10:15:00 | ALERT3 | SELL | 5269.50 | 5211.68 | 5270.33 | EMA400 retest candle locked |
| 2025-10-29 11:15:00 | EXIT | SELL | 5294.00 | 5212.50 | 5270.45 | Close above EMA400 |
| 2026-02-04 12:15:00 | CROSSOVER | BUY | 5778.00 | 5143.68 | 5141.10 | EMA200 above EMA400 |
| 2026-02-05 11:15:00 | ALERT1 | BUY | 5811.50 | 5179.14 | 5159.17 | Break + close above crossover candle high |
| 2026-03-23 12:15:00 | ALERT2 | BUY | 5997.00 | 6014.84 | 5767.05 | EMA200 retest candle locked |
| 2026-03-24 09:15:00 | ENTRY1 | BUY | 6114.50 | 6016.82 | 5772.95 | Buy entry 1 (retest1 break) |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| SELL | 2023-10-18 11:15:00 | 4139.25 | 2023-11-06 14:15:00 | 4205.95 | -66.70 |
| SELL | 2023-11-06 11:15:00 | 4156.75 | 2023-11-06 14:15:00 | 4205.95 | -49.20 |
| BUY | 2024-01-02 12:15:00 | 4667.60 | 2024-02-01 10:15:00 | 4620.70 | -46.90 |
| BUY | 2024-01-24 10:15:00 | 4798.00 | 2024-02-01 10:15:00 | 4620.70 | -177.30 |
| BUY | 2024-01-31 13:15:00 | 4672.90 | 2024-02-01 10:15:00 | 4620.70 | -52.20 |
| BUY | 2024-10-08 10:15:00 | 7989.50 | 2024-10-23 10:15:00 | 8062.05 | 72.55 |
| SELL | 2024-12-20 12:15:00 | 7105.10 | 2025-04-17 10:15:00 | 5617.00 | 1488.10 |
| BUY | 2025-06-20 10:15:00 | 5962.50 | 2025-07-03 15:15:00 | 5866.50 | -96.00 |
| BUY | 2025-07-02 14:15:00 | 5907.50 | 2025-07-03 15:15:00 | 5866.50 | -41.00 |
| SELL | 2025-09-26 09:15:00 | 5148.70 | 2025-10-29 11:15:00 | 5294.00 | -145.30 |
| SELL | 2025-10-07 14:15:00 | 5215.00 | 2025-10-29 11:15:00 | 5294.00 | -79.00 |
