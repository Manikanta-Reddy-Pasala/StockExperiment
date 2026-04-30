# Engineers India Ltd. (ENGINERSIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 251.94
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 26.83
- **Avg P&L per closed trade:** 3.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 15:15:00 | 127.05 | 142.78 | 142.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 10:15:00 | 126.50 | 142.47 | 142.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 135.35 | 134.03 | 137.39 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 15:15:00 | 143.85 | 139.68 | 139.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 144.90 | 139.73 | 139.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 152.00 | 153.37 | 148.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-22 09:15:00 | 157.80 | 153.48 | 148.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 205.25 | 218.72 | 198.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-14 09:15:00 | 213.60 | 217.93 | 198.56 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 205.00 | 218.51 | 204.89 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-29 12:15:00 | 210.10 | 217.71 | 204.95 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 209.90 | 219.26 | 208.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-12 10:15:00 | 207.50 | 219.14 | 208.41 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 221.80 | 249.73 | 249.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 219.70 | 249.44 | 249.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 12:15:00 | 225.75 | 224.47 | 232.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-13 12:15:00 | 222.80 | 224.55 | 232.59 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-27 14:15:00 | 201.13 | 188.95 | 198.12 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 182.80 | 168.79 | 168.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 12:15:00 | 184.06 | 170.72 | 169.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 171.75 | 172.90 | 171.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 11:15:00 | 175.15 | 172.93 | 171.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 172.45 | 173.22 | 171.37 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-09 09:15:00 | 170.30 | 173.19 | 171.36 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 191.77 | 217.07 | 217.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 190.70 | 216.81 | 217.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 210.84 | 205.00 | 209.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-25 10:15:00 | 201.78 | 206.65 | 208.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-30 09:15:00 | 205.49 | 200.97 | 203.68 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 15:15:00 | 205.52 | 200.11 | 200.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 206.60 | 200.17 | 200.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 11:15:00 | 200.26 | 200.51 | 200.29 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 191.01 | 200.00 | 200.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 14:15:00 | 187.54 | 198.59 | 199.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 10:15:00 | 182.60 | 182.39 | 188.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 15:15:00 | 180.55 | 182.39 | 188.24 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-13 09:15:00 | 204.70 | 182.61 | 188.33 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 214.30 | 193.22 | 193.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 14:15:00 | 215.11 | 193.44 | 193.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 15:15:00 | 201.10 | 202.17 | 198.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 09:15:00 | 203.45 | 202.18 | 198.26 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 195.82 | 202.45 | 198.66 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 11:15:00 | 185.37 | 196.47 | 196.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 182.44 | 195.10 | 195.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 199.95 | 194.97 | 195.69 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 206.96 | 196.29 | 196.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 10:15:00 | 210.22 | 196.84 | 196.56 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-22 09:15:00 | 157.80 | 2024-01-02 11:15:00 | 186.25 | TARGET | 28.45 |
| BUY | 2024-02-29 12:15:00 | 210.10 | 2024-03-04 14:15:00 | 225.55 | TARGET | 15.45 |
| BUY | 2024-02-14 09:15:00 | 213.60 | 2024-03-12 10:15:00 | 207.50 | EXIT_EMA400 | -6.10 |
| SELL | 2024-09-13 12:15:00 | 222.80 | 2024-10-07 10:15:00 | 193.43 | TARGET | 29.37 |
| BUY | 2025-05-07 11:15:00 | 175.15 | 2025-05-09 09:15:00 | 170.30 | EXIT_EMA400 | -4.85 |
| SELL | 2025-09-25 10:15:00 | 201.78 | 2025-10-30 09:15:00 | 205.49 | EXIT_EMA400 | -3.71 |
| SELL | 2026-02-12 15:15:00 | 180.55 | 2026-02-13 09:15:00 | 204.70 | EXIT_EMA400 | -24.15 |
| BUY | 2026-03-05 09:15:00 | 203.45 | 2026-03-09 09:15:00 | 195.82 | EXIT_EMA400 | -7.63 |
