# Aditya Birla Capital Ltd. (ABCAPITAL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 345.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Total realized P&L (per unit):** 45.51
- **Avg P&L per closed trade:** 5.06

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2024-08-14 14:15:00 | ALERT1 | SELL | 203.28 | 218.30 | 220.03 | Break + close below crossover candle low |
| 2024-08-20 10:15:00 | ALERT2 | SELL | 218.55 | 217.55 | 219.49 | EMA200 retest candle locked |
| 2024-09-09 09:15:00 | ENTRY1 | SELL | 214.77 | 220.28 | 220.44 | Sell entry 1 (retest1 break) |
| 2024-09-10 14:15:00 | ALERT3 | SELL | 217.61 | 219.92 | 220.25 | EMA400 retest candle locked |
| 2024-09-11 12:15:00 | ENTRY2 | SELL | 216.65 | 219.81 | 220.18 | Sell entry 2 (retest2 break) |
| 2024-09-12 13:15:00 | ALERT3 | SELL | 219.85 | 219.56 | 220.04 | EMA400 retest candle locked |
| 2024-09-12 14:15:00 | EXIT | SELL | 220.77 | 219.57 | 220.04 | Close above EMA400 |
| 2024-09-17 11:15:00 | CROSSOVER | BUY | 223.26 | 220.49 | 220.48 | EMA200 above EMA400 |
| 2024-09-17 12:15:00 | ALERT1 | BUY | 225.14 | 220.54 | 220.51 | Break + close above crossover candle high |
| 2024-10-04 14:15:00 | ALERT2 | BUY | 227.79 | 228.06 | 224.95 | EMA200 retest candle locked |
| 2024-10-23 14:15:00 | CROSSOVER | SELL | 210.64 | 223.47 | 223.49 | EMA200 below EMA400 |
| 2024-10-25 09:15:00 | ALERT1 | SELL | 207.28 | 222.49 | 222.99 | Break + close below crossover candle low |
| 2024-12-04 09:15:00 | ALERT2 | SELL | 199.72 | 199.27 | 206.78 | EMA200 retest candle locked |
| 2024-12-04 11:15:00 | ENTRY1 | SELL | 198.42 | 199.25 | 206.70 | Sell entry 1 (retest1 break) |
| 2025-03-19 09:15:00 | ALERT3 | SELL | 168.41 | 161.83 | 168.77 | EMA400 retest candle locked |
| 2025-03-19 10:15:00 | EXIT | SELL | 168.84 | 161.90 | 168.78 | Close above EMA400 |
| 2025-04-03 13:15:00 | CROSSOVER | BUY | 192.18 | 173.29 | 173.28 | EMA200 above EMA400 |
| 2025-04-03 14:15:00 | ALERT1 | BUY | 193.44 | 173.49 | 173.38 | Break + close above crossover candle high |
| 2025-07-25 09:15:00 | ALERT2 | BUY | 260.60 | 263.68 | 247.87 | EMA200 retest candle locked |
| 2025-08-04 14:15:00 | ENTRY1 | BUY | 278.40 | 260.53 | 249.40 | Buy entry 1 (retest1 break) |
| 2026-01-21 11:15:00 | ALERT3 | BUY | 348.65 | 353.60 | 343.18 | EMA400 retest candle locked |
| 2026-01-21 12:15:00 | ENTRY2 | BUY | 351.40 | 353.57 | 343.22 | Buy entry 2 (retest2 break) |
| 2026-01-27 13:15:00 | ALERT3 | BUY | 344.00 | 352.92 | 343.96 | EMA400 retest candle locked |
| 2026-01-27 14:15:00 | ENTRY2 | BUY | 349.10 | 352.89 | 343.98 | Buy entry 2 (retest2 break) |
| 2026-01-29 10:15:00 | EXIT | BUY | 344.00 | 352.36 | 344.15 | Close below EMA400 |
| 2026-03-09 10:15:00 | CROSSOVER | SELL | 316.45 | 342.52 | 342.61 | EMA200 below EMA400 |
| 2026-03-13 12:15:00 | ALERT1 | SELL | 311.35 | 337.63 | 339.99 | Break + close below crossover candle low |
| 2026-04-08 09:15:00 | ALERT2 | SELL | 328.65 | 317.79 | 326.88 | EMA200 retest candle locked |
| 2026-04-28 12:15:00 | CROSSOVER | BUY | 339.35 | 332.50 | 332.46 | EMA200 above EMA400 |
| 2026-04-29 09:15:00 | ALERT1 | BUY | 352.80 | 332.88 | 332.66 | Break + close above crossover candle high |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| SELL | 2023-11-08 14:15:00 | 174.45 | 2023-11-15 09:15:00 | 178.90 | -4.45 |
| SELL | 2024-02-01 09:15:00 | 166.10 | 2024-02-02 09:15:00 | 181.55 | -15.45 |
| BUY | 2024-02-29 14:15:00 | 184.05 | 2024-03-06 10:15:00 | 176.90 | -7.15 |
| SELL | 2024-09-09 09:15:00 | 214.77 | 2024-09-12 14:15:00 | 220.77 | -6.00 |
| SELL | 2024-09-11 12:15:00 | 216.65 | 2024-09-12 14:15:00 | 220.77 | -4.12 |
| SELL | 2024-12-04 11:15:00 | 198.42 | 2025-03-19 10:15:00 | 168.84 | 29.58 |
| BUY | 2025-08-04 14:15:00 | 278.40 | 2026-01-29 10:15:00 | 344.00 | 65.60 |
| BUY | 2026-01-21 12:15:00 | 351.40 | 2026-01-29 10:15:00 | 344.00 | -7.40 |
| BUY | 2026-01-27 14:15:00 | 349.10 | 2026-01-29 10:15:00 | 344.00 | -5.10 |
