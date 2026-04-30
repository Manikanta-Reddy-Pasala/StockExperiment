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
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** 62.15
- **Avg P&L per closed trade:** 6.91

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 12:15:00 | 175.55 | 183.12 | 183.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 172.50 | 180.50 | 181.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 176.30 | 176.28 | 178.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-08 14:15:00 | 174.45 | 176.23 | 178.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 13:15:00 | 177.95 | 176.06 | 178.16 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-15 09:15:00 | 178.90 | 176.12 | 178.16 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 14:15:00 | 174.60 | 172.39 | 172.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 15:15:00 | 175.60 | 172.42 | 172.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 170.15 | 172.40 | 172.39 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 10:15:00 | 169.35 | 172.37 | 172.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 166.85 | 172.32 | 172.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 13:15:00 | 171.50 | 170.83 | 171.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-01 09:15:00 | 166.10 | 170.79 | 171.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-02-02 09:15:00 | 181.55 | 170.64 | 171.38 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 13:15:00 | 178.55 | 172.11 | 172.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 14:15:00 | 180.00 | 172.19 | 172.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-29 09:15:00 | 180.50 | 180.98 | 177.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-29 14:15:00 | 184.05 | 181.03 | 177.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 180.35 | 182.76 | 179.08 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-06 10:15:00 | 176.90 | 182.70 | 179.07 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 10:15:00 | 205.44 | 222.01 | 222.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 14:15:00 | 203.28 | 218.30 | 220.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 10:15:00 | 218.55 | 217.55 | 219.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-09 09:15:00 | 214.77 | 220.28 | 220.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 217.61 | 219.92 | 220.25 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-11 12:15:00 | 216.65 | 219.81 | 220.18 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 219.85 | 219.56 | 220.04 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-12 14:15:00 | 220.77 | 219.57 | 220.04 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 11:15:00 | 223.26 | 220.49 | 220.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 12:15:00 | 225.14 | 220.54 | 220.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 227.79 | 228.06 | 224.95 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 14:15:00 | 210.64 | 223.47 | 223.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 207.28 | 222.49 | 222.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 199.72 | 199.27 | 206.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-04 11:15:00 | 198.42 | 199.25 | 206.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 168.41 | 161.83 | 168.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 10:15:00 | 168.84 | 161.90 | 168.78 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 192.18 | 173.29 | 173.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 193.44 | 173.49 | 173.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 260.60 | 263.68 | 247.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-04 14:15:00 | 278.40 | 260.53 | 249.40 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 348.65 | 353.60 | 343.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-21 12:15:00 | 351.40 | 353.57 | 343.22 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 344.00 | 352.92 | 343.96 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-27 14:15:00 | 349.10 | 352.89 | 343.98 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-29 10:15:00 | 344.00 | 352.36 | 344.15 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 316.45 | 342.52 | 342.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 311.35 | 337.63 | 339.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 328.65 | 317.79 | 326.88 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 12:15:00 | 339.35 | 332.50 | 332.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 352.80 | 332.88 | 332.66 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-08 14:15:00 | 174.45 | 2023-11-15 09:15:00 | 178.90 | EXIT_EMA400 | -4.45 |
| SELL | 2024-02-01 09:15:00 | 166.10 | 2024-02-02 09:15:00 | 181.55 | EXIT_EMA400 | -15.45 |
| BUY | 2024-02-29 14:15:00 | 184.05 | 2024-03-06 10:15:00 | 176.90 | EXIT_EMA400 | -7.15 |
| SELL | 2024-09-09 09:15:00 | 214.77 | 2024-09-12 14:15:00 | 220.77 | EXIT_EMA400 | -6.00 |
| SELL | 2024-09-11 12:15:00 | 216.65 | 2024-09-12 14:15:00 | 220.77 | EXIT_EMA400 | -4.12 |
| SELL | 2024-12-04 11:15:00 | 198.42 | 2025-01-09 14:15:00 | 173.59 | TARGET | 24.83 |
| BUY | 2025-08-04 14:15:00 | 278.40 | 2025-12-09 11:15:00 | 365.39 | TARGET | 86.99 |
| BUY | 2026-01-21 12:15:00 | 351.40 | 2026-01-29 10:15:00 | 344.00 | EXIT_EMA400 | -7.40 |
| BUY | 2026-01-27 14:15:00 | 349.10 | 2026-01-29 10:15:00 | 344.00 | EXIT_EMA400 | -5.10 |
