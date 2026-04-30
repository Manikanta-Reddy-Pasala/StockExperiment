# Indraprastha Gas Ltd. (IGL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 166.02
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -17.00
- **Avg P&L per closed trade:** -1.89

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 11:15:00 | 243.32 | 231.83 | 231.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 14:15:00 | 243.55 | 232.17 | 232.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 09:15:00 | 230.82 | 232.85 | 232.36 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 11:15:00 | 205.65 | 231.86 | 231.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 204.43 | 231.59 | 231.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 10:15:00 | 201.88 | 200.71 | 209.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-12 12:15:00 | 198.52 | 201.01 | 207.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-12-20 10:15:00 | 206.75 | 200.99 | 206.49 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 15:15:00 | 216.57 | 208.54 | 208.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-17 13:15:00 | 217.32 | 208.93 | 208.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 10:15:00 | 211.50 | 211.60 | 210.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-31 09:15:00 | 215.55 | 211.34 | 210.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 213.77 | 215.71 | 213.03 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-13 09:15:00 | 216.27 | 215.72 | 213.04 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 215.40 | 217.50 | 214.89 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-27 11:15:00 | 214.23 | 217.45 | 214.89 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 15:15:00 | 204.45 | 213.74 | 213.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 202.10 | 213.63 | 213.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 10:15:00 | 213.12 | 211.10 | 212.30 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 14:15:00 | 220.77 | 213.28 | 213.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 10:15:00 | 225.60 | 213.89 | 213.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 12:15:00 | 221.15 | 222.10 | 218.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-23 09:15:00 | 225.23 | 221.63 | 218.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 14:15:00 | 221.55 | 224.95 | 221.14 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-07 09:15:00 | 223.90 | 224.91 | 221.16 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-05-07 10:15:00 | 220.95 | 224.87 | 221.16 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 223.55 | 264.30 | 264.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 222.12 | 263.89 | 264.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 09:15:00 | 192.80 | 192.02 | 213.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 09:15:00 | 189.10 | 192.04 | 212.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 205.52 | 194.69 | 205.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-31 11:15:00 | 206.15 | 194.80 | 205.82 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 10:15:00 | 204.24 | 193.63 | 193.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 206.60 | 194.22 | 193.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 205.17 | 205.98 | 201.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 10:15:00 | 207.34 | 205.32 | 201.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 202.06 | 205.73 | 202.39 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 204.96 | 208.04 | 208.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 204.53 | 208.01 | 208.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 207.36 | 207.31 | 207.66 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 217.03 | 207.92 | 207.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 217.52 | 208.10 | 208.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 209.80 | 210.14 | 209.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-09 12:15:00 | 211.58 | 210.15 | 209.15 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-09 14:15:00 | 208.93 | 210.14 | 209.16 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 203.70 | 210.91 | 210.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 202.21 | 210.82 | 210.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 196.90 | 196.25 | 201.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-19 09:15:00 | 192.54 | 196.22 | 201.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 164.51 | 157.04 | 164.62 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 10:15:00 | 164.74 | 157.12 | 164.62 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-12-12 12:15:00 | 198.52 | 2023-12-20 10:15:00 | 206.75 | EXIT_EMA400 | -8.23 |
| BUY | 2024-01-31 09:15:00 | 215.55 | 2024-02-27 11:15:00 | 214.23 | EXIT_EMA400 | -1.32 |
| BUY | 2024-02-13 09:15:00 | 216.27 | 2024-02-27 11:15:00 | 214.23 | EXIT_EMA400 | -2.05 |
| BUY | 2024-04-23 09:15:00 | 225.23 | 2024-05-07 10:15:00 | 220.95 | EXIT_EMA400 | -4.28 |
| BUY | 2024-05-07 09:15:00 | 223.90 | 2024-05-07 10:15:00 | 220.95 | EXIT_EMA400 | -2.95 |
| SELL | 2024-12-09 09:15:00 | 189.10 | 2024-12-31 11:15:00 | 206.15 | EXIT_EMA400 | -17.05 |
| BUY | 2025-06-16 10:15:00 | 207.34 | 2025-06-19 12:15:00 | 202.06 | EXIT_EMA400 | -5.28 |
| BUY | 2025-09-09 12:15:00 | 211.58 | 2025-09-09 14:15:00 | 208.93 | EXIT_EMA400 | -2.65 |
| SELL | 2025-12-19 09:15:00 | 192.54 | 2026-02-13 09:15:00 | 165.73 | TARGET | 26.81 |
