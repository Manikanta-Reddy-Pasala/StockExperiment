# AWL Agri Business Ltd. (AWL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 196.37
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 7 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Total realized P&L (per unit):** -82.65
- **Avg P&L per closed trade:** -11.81

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-10-24 14:15:00 | ALERT2 | SELL | 339.50 | 338.47 | 345.13 | EMA200 retest candle locked |
| 2024-11-21 09:15:00 | ENTRY1 | SELL | 294.90 | 334.52 | 339.92 | Sell entry 1 (retest1 break) |
| 2024-12-26 13:15:00 | ALERT3 | SELL | 320.05 | 311.75 | 320.71 | EMA400 retest candle locked |
| 2024-12-26 14:15:00 | EXIT | SELL | 321.00 | 311.84 | 320.71 | Close above EMA400 |
| 2025-04-24 14:15:00 | CROSSOVER | BUY | 283.15 | 270.05 | 270.02 | EMA200 above EMA400 |
| 2025-05-07 10:15:00 | CROSSOVER | SELL | 263.45 | 270.11 | 270.12 | EMA200 below EMA400 |
| 2025-05-07 15:15:00 | ALERT1 | SELL | 261.45 | 269.75 | 269.94 | Break + close below crossover candle low |
| 2025-05-14 09:15:00 | ALERT2 | SELL | 267.40 | 267.32 | 268.59 | EMA200 retest candle locked |
| 2025-05-20 13:15:00 | ENTRY1 | SELL | 264.55 | 267.75 | 268.64 | Sell entry 1 (retest1 break) |
| 2025-05-27 11:15:00 | EXIT | SELL | 272.90 | 265.64 | 267.36 | Close above EMA400 |
| 2025-07-21 14:15:00 | CROSSOVER | BUY | 278.10 | 265.49 | 265.43 | EMA200 above EMA400 |
| 2025-07-21 15:15:00 | ALERT1 | BUY | 280.00 | 265.64 | 265.51 | Break + close above crossover candle high |
| 2025-07-28 12:15:00 | ALERT2 | BUY | 267.50 | 268.35 | 267.02 | EMA200 retest candle locked |
| 2025-08-05 12:15:00 | CROSSOVER | SELL | 253.45 | 265.98 | 266.00 | EMA200 below EMA400 |
| 2025-08-05 13:15:00 | ALERT1 | SELL | 252.95 | 265.85 | 265.94 | Break + close below crossover candle low |
| 2025-08-19 15:15:00 | ALERT2 | SELL | 260.20 | 259.73 | 262.34 | EMA200 retest candle locked |
| 2025-08-22 09:15:00 | ENTRY1 | SELL | 258.75 | 259.87 | 262.23 | Sell entry 1 (retest1 break) |
| 2025-09-03 10:15:00 | ALERT3 | SELL | 260.10 | 257.39 | 260.33 | EMA400 retest candle locked |
| 2025-09-03 11:15:00 | EXIT | SELL | 263.85 | 257.46 | 260.34 | Close above EMA400 |
| 2025-10-09 09:15:00 | CROSSOVER | BUY | 263.05 | 261.08 | 261.07 | EMA200 above EMA400 |
| 2025-10-09 11:15:00 | ALERT1 | BUY | 265.10 | 261.14 | 261.10 | Break + close above crossover candle high |
| 2025-10-17 14:15:00 | ALERT2 | BUY | 261.75 | 263.21 | 262.27 | EMA200 retest candle locked |
| 2025-10-27 14:15:00 | ENTRY1 | BUY | 265.85 | 262.80 | 262.17 | Buy entry 1 (retest1 break) |
| 2025-11-21 09:15:00 | ALERT3 | BUY | 269.50 | 269.79 | 266.94 | EMA400 retest candle locked |
| 2025-11-25 09:15:00 | EXIT | BUY | 264.10 | 270.15 | 267.32 | Close below EMA400 |
| 2025-12-04 10:15:00 | CROSSOVER | SELL | 246.40 | 265.17 | 265.27 | EMA200 below EMA400 |
| 2025-12-10 11:15:00 | ALERT1 | SELL | 245.35 | 261.10 | 263.07 | Break + close below crossover candle low |
| 2026-03-11 11:15:00 | ALERT2 | SELL | 196.40 | 196.30 | 210.74 | EMA200 retest candle locked |
| 2026-03-12 09:15:00 | ENTRY1 | SELL | 176.50 | 195.55 | 210.01 | Sell entry 1 (retest1 break) |
| 2026-04-22 09:15:00 | EXIT | SELL | 196.35 | 184.19 | 193.40 | Close above EMA400 |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| SELL | 2024-04-15 10:15:00 | 338.50 | 2024-04-30 09:15:00 | 355.35 | -16.85 |
| BUY | 2024-08-16 12:15:00 | 362.15 | 2024-09-09 12:15:00 | 357.50 | -4.65 |
| SELL | 2024-11-21 09:15:00 | 294.90 | 2024-12-26 14:15:00 | 321.00 | -26.10 |
| SELL | 2025-05-20 13:15:00 | 264.55 | 2025-05-27 11:15:00 | 272.90 | -8.35 |
| SELL | 2025-08-22 09:15:00 | 258.75 | 2025-09-03 11:15:00 | 263.85 | -5.10 |
| BUY | 2025-10-27 14:15:00 | 265.85 | 2025-11-25 09:15:00 | 264.10 | -1.75 |
| SELL | 2026-03-12 09:15:00 | 176.50 | 2026-04-22 09:15:00 | 196.35 | -19.85 |
