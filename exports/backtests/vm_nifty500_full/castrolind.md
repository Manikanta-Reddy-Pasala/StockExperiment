# Castrol India Ltd. (CASTROLIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 184.82
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 3 |
| EXIT | 7 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / EMA400 exits:** 4 / 6
- **Total realized P&L (per unit):** 69.13
- **Avg P&L per closed trade:** 6.91

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 09:15:00 | 134.65 | 140.09 | 140.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 15:15:00 | 133.90 | 139.74 | 139.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 138.15 | 136.97 | 138.19 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 141.50 | 138.93 | 138.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 142.10 | 138.96 | 138.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 10:15:00 | 168.80 | 169.28 | 158.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-18 10:15:00 | 175.95 | 169.30 | 159.35 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 191.00 | 200.67 | 189.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-14 09:15:00 | 195.85 | 200.53 | 189.72 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 191.50 | 198.69 | 191.03 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-26 09:15:00 | 188.25 | 198.58 | 191.01 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 11:15:00 | 189.85 | 197.92 | 197.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 189.75 | 197.68 | 197.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 196.90 | 195.51 | 196.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-03 12:15:00 | 193.50 | 195.48 | 196.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 195.20 | 193.59 | 195.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-07 11:15:00 | 196.40 | 193.62 | 195.44 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 13:15:00 | 202.78 | 196.89 | 196.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 205.51 | 197.09 | 196.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 12:15:00 | 201.15 | 201.44 | 199.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-01 09:15:00 | 209.00 | 201.49 | 199.56 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-19 09:15:00 | 250.20 | 259.53 | 251.52 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 09:15:00 | 230.82 | 247.57 | 247.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 10:15:00 | 227.50 | 246.15 | 246.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 10:15:00 | 208.32 | 205.86 | 217.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-18 09:15:00 | 203.61 | 210.17 | 216.09 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 09:15:00 | 199.70 | 185.97 | 195.21 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 11:15:00 | 216.39 | 200.32 | 200.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 12:15:00 | 217.55 | 200.49 | 200.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 218.50 | 218.59 | 211.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-20 09:15:00 | 219.65 | 218.60 | 211.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-03-25 10:15:00 | 212.25 | 218.45 | 212.27 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 15:15:00 | 199.39 | 208.58 | 208.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 198.79 | 206.86 | 207.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 15:15:00 | 204.70 | 204.43 | 206.04 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 218.70 | 207.02 | 207.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 222.25 | 208.65 | 207.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 212.55 | 213.12 | 210.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-26 09:15:00 | 213.91 | 211.37 | 210.28 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 218.66 | 220.33 | 217.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-01 09:15:00 | 222.28 | 220.27 | 217.26 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 219.81 | 220.23 | 217.34 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-05 09:15:00 | 221.78 | 220.17 | 217.41 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 217.95 | 220.26 | 217.56 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-06 11:15:00 | 215.50 | 220.20 | 217.55 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 206.00 | 215.64 | 215.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 205.60 | 213.36 | 214.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-18 09:15:00 | 204.00 | 204.21 | 208.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-15 15:15:00 | 204.99 | 201.57 | 204.52 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-18 10:15:00 | 175.95 | 2024-03-26 09:15:00 | 188.25 | EXIT_EMA400 | 12.30 |
| BUY | 2024-03-14 09:15:00 | 195.85 | 2024-03-26 09:15:00 | 188.25 | EXIT_EMA400 | -7.60 |
| SELL | 2024-06-03 12:15:00 | 193.50 | 2024-06-04 10:15:00 | 184.26 | TARGET | 9.24 |
| BUY | 2024-07-01 09:15:00 | 209.00 | 2024-07-03 13:15:00 | 237.33 | TARGET | 28.33 |
| SELL | 2024-12-18 09:15:00 | 203.61 | 2025-01-28 09:15:00 | 166.18 | TARGET | 37.43 |
| BUY | 2025-03-20 09:15:00 | 219.65 | 2025-03-25 10:15:00 | 212.25 | EXIT_EMA400 | -7.40 |
| BUY | 2025-06-26 09:15:00 | 213.91 | 2025-07-01 09:15:00 | 224.80 | TARGET | 10.89 |
| BUY | 2025-08-01 09:15:00 | 222.28 | 2025-08-06 11:15:00 | 215.50 | EXIT_EMA400 | -6.78 |
| BUY | 2025-08-05 09:15:00 | 221.78 | 2025-08-06 11:15:00 | 215.50 | EXIT_EMA400 | -6.28 |
| SELL | 2025-09-18 09:15:00 | 204.00 | 2025-10-15 15:15:00 | 204.99 | EXIT_EMA400 | -0.99 |
