# National Aluminium Co. Ltd. (NATIONALUM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 399.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 17.20
- **Avg P&L per closed trade:** 2.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 91.20 | 93.73 | 93.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 12:15:00 | 91.00 | 93.70 | 93.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 10:15:00 | 93.05 | 93.05 | 93.35 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 12:15:00 | 96.65 | 93.63 | 93.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 13:15:00 | 99.90 | 93.69 | 93.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-13 11:15:00 | 140.00 | 140.64 | 128.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-14 09:15:00 | 152.90 | 140.79 | 128.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 145.55 | 154.59 | 143.58 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-13 10:15:00 | 142.85 | 154.47 | 143.57 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 172.27 | 185.05 | 185.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 169.98 | 184.78 | 184.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 12:15:00 | 178.78 | 178.73 | 181.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-04 09:15:00 | 175.38 | 179.78 | 181.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-13 09:15:00 | 183.70 | 177.76 | 179.97 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 14:15:00 | 195.08 | 181.58 | 181.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 11:15:00 | 198.47 | 182.14 | 181.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 221.49 | 222.90 | 211.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-18 09:15:00 | 237.80 | 222.90 | 212.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-13 09:15:00 | 227.00 | 239.71 | 228.24 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 10:15:00 | 205.73 | 222.69 | 222.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 11:15:00 | 204.75 | 222.51 | 222.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 09:15:00 | 196.15 | 195.33 | 203.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-24 13:15:00 | 189.82 | 195.50 | 202.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 171.85 | 163.17 | 171.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-14 12:15:00 | 172.22 | 163.40 | 171.89 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 185.45 | 176.39 | 176.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 188.00 | 176.51 | 176.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 181.50 | 181.64 | 179.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 183.26 | 181.66 | 179.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 188.93 | 190.12 | 186.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-29 14:15:00 | 189.73 | 190.07 | 186.74 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 185.83 | 189.86 | 186.78 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-14 09:15:00 | 152.90 | 2024-03-13 10:15:00 | 142.85 | EXIT_EMA400 | -10.05 |
| SELL | 2024-09-04 09:15:00 | 175.38 | 2024-09-13 09:15:00 | 183.70 | EXIT_EMA400 | -8.32 |
| BUY | 2024-11-18 09:15:00 | 237.80 | 2024-12-13 09:15:00 | 227.00 | EXIT_EMA400 | -10.80 |
| SELL | 2025-02-24 13:15:00 | 189.82 | 2025-04-07 09:15:00 | 150.75 | TARGET | 39.07 |
| BUY | 2025-06-20 09:15:00 | 183.26 | 2025-06-26 14:15:00 | 194.47 | TARGET | 11.21 |
| BUY | 2025-07-29 14:15:00 | 189.73 | 2025-07-31 09:15:00 | 185.83 | EXIT_EMA400 | -3.90 |
