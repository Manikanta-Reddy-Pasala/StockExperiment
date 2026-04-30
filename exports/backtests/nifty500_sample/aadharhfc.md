# Aadhar Housing Finance Ltd. (AADHARHFC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-15 09:15:00 → 2026-04-30 15:30:00 (3373 bars)
- **Last close:** 488.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Total realized P&L (per unit):** -15.05
- **Avg P&L per closed trade:** -3.01

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-11-26 10:15:00 | CROSSOVER | SELL | 405.50 | 435.15 | 435.24 | EMA200 below EMA400 |
| 2025-01-15 09:15:00 | ALERT1 | SELL | 399.95 | 424.31 | 427.86 | Break + close below crossover candle low |
| 2025-02-07 09:15:00 | ALERT2 | SELL | 409.90 | 401.58 | 411.70 | EMA200 retest candle locked |
| 2025-02-10 09:15:00 | ENTRY1 | SELL | 392.55 | 401.71 | 411.41 | Sell entry 1 (retest1 break) |
| 2025-03-06 10:15:00 | ALERT3 | SELL | 398.05 | 387.97 | 398.30 | EMA400 retest candle locked |
| 2025-03-06 14:15:00 | EXIT | SELL | 399.60 | 388.36 | 398.29 | Close above EMA400 |
| 2025-03-20 12:15:00 | CROSSOVER | BUY | 424.15 | 405.53 | 405.47 | EMA200 above EMA400 |
| 2025-03-20 13:15:00 | ALERT1 | BUY | 428.30 | 405.75 | 405.59 | Break + close above crossover candle high |
| 2025-03-25 09:15:00 | ALERT2 | BUY | 407.10 | 408.22 | 406.91 | EMA200 retest candle locked |
| 2025-03-27 10:15:00 | ENTRY1 | BUY | 418.40 | 407.80 | 406.78 | Buy entry 1 (retest1 break) |
| 2025-05-09 09:15:00 | ALERT3 | BUY | 442.05 | 454.20 | 439.45 | EMA400 retest candle locked |
| 2025-05-09 11:15:00 | ENTRY2 | BUY | 444.60 | 453.97 | 439.48 | Buy entry 2 (retest2 break) |
| 2025-05-12 13:15:00 | ALERT3 | BUY | 440.00 | 453.08 | 439.66 | EMA400 retest candle locked |
| 2025-05-13 09:15:00 | ENTRY2 | BUY | 453.90 | 452.86 | 439.75 | Buy entry 2 (retest2 break) |
| 2025-05-20 11:15:00 | ALERT3 | BUY | 442.95 | 452.45 | 441.76 | EMA400 retest candle locked |
| 2025-05-20 12:15:00 | EXIT | BUY | 441.60 | 452.34 | 441.76 | Close below EMA400 |
| 2025-11-17 12:15:00 | CROSSOVER | SELL | 497.00 | 505.99 | 506.03 | EMA200 below EMA400 |
| 2025-11-19 09:15:00 | ALERT1 | SELL | 491.75 | 504.84 | 505.44 | Break + close below crossover candle low |
| 2025-12-15 09:15:00 | ALERT2 | SELL | 492.95 | 490.61 | 495.87 | EMA200 retest candle locked |
| 2025-12-17 09:15:00 | ENTRY1 | SELL | 483.70 | 490.85 | 495.64 | Sell entry 1 (retest1 break) |
| 2026-01-02 09:15:00 | EXIT | SELL | 499.60 | 486.58 | 491.56 | Close above EMA400 |
| 2026-04-22 12:15:00 | CROSSOVER | BUY | 487.55 | 471.25 | 471.21 | EMA200 above EMA400 |
| 2026-04-22 13:15:00 | ALERT1 | BUY | 492.65 | 471.46 | 471.31 | Break + close above crossover candle high |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| SELL | 2025-02-10 09:15:00 | 392.55 | 2025-03-06 14:15:00 | 399.60 | -7.05 |
| BUY | 2025-03-27 10:15:00 | 418.40 | 2025-05-20 12:15:00 | 441.60 | 23.20 |
| BUY | 2025-05-09 11:15:00 | 444.60 | 2025-05-20 12:15:00 | 441.60 | -3.00 |
| BUY | 2025-05-13 09:15:00 | 453.90 | 2025-05-20 12:15:00 | 441.60 | -12.30 |
| SELL | 2025-12-17 09:15:00 | 483.70 | 2026-01-02 09:15:00 | 499.60 | -15.90 |
