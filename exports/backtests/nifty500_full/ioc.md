# Indian Oil Corporation Ltd. (IOC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 142.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 1
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -0.43
- **Avg P&L per closed trade:** -0.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 13:15:00 | 92.15 | 93.68 | 93.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 15:15:00 | 91.75 | 93.54 | 93.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 13:15:00 | 94.85 | 92.31 | 92.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-09-12 14:15:00 | 91.55 | 92.50 | 92.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-09-13 12:15:00 | 93.20 | 92.50 | 92.91 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 13:15:00 | 103.35 | 91.80 | 91.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 14:15:00 | 104.00 | 91.92 | 91.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 12:15:00 | 167.75 | 168.21 | 152.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-28 13:15:00 | 170.00 | 168.23 | 152.11 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-15 11:15:00 | 155.30 | 169.76 | 158.10 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 11:15:00 | 165.27 | 171.39 | 171.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 164.44 | 170.80 | 171.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 142.72 | 142.41 | 150.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-12 14:15:00 | 141.45 | 142.52 | 148.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 125.30 | 122.45 | 127.09 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-12 11:15:00 | 124.70 | 122.50 | 127.07 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-19 10:15:00 | 126.95 | 123.11 | 126.82 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 09:15:00 | 133.14 | 128.70 | 128.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 15:15:00 | 134.80 | 129.01 | 128.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 141.13 | 141.13 | 137.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-09 09:15:00 | 142.21 | 140.96 | 138.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 140.80 | 141.57 | 138.75 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 10:15:00 | 141.10 | 141.57 | 138.77 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 138.81 | 141.39 | 139.07 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 139.61 | 143.86 | 143.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 139.38 | 143.74 | 143.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 142.94 | 141.63 | 142.47 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 148.75 | 143.12 | 143.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 148.82 | 143.66 | 143.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 144.01 | 144.26 | 143.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-26 12:15:00 | 145.01 | 144.27 | 143.73 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 160.98 | 163.99 | 160.76 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-19 14:15:00 | 162.62 | 163.88 | 160.78 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 161.17 | 163.70 | 161.01 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-26 09:15:00 | 160.06 | 163.66 | 161.00 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 15:15:00 | 146.55 | 166.56 | 166.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 144.47 | 165.11 | 165.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 147.78 | 147.51 | 153.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-22 09:15:00 | 147.01 | 147.50 | 153.54 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-09-12 14:15:00 | 91.55 | 2023-09-13 12:15:00 | 93.20 | EXIT_EMA400 | -1.65 |
| BUY | 2024-02-28 13:15:00 | 170.00 | 2024-03-15 11:15:00 | 155.30 | EXIT_EMA400 | -14.70 |
| SELL | 2024-12-12 14:15:00 | 141.45 | 2025-02-12 09:15:00 | 118.87 | TARGET | 22.58 |
| SELL | 2025-03-12 11:15:00 | 124.70 | 2025-03-19 10:15:00 | 126.95 | EXIT_EMA400 | -2.25 |
| BUY | 2025-06-09 09:15:00 | 142.21 | 2025-06-19 12:15:00 | 138.81 | EXIT_EMA400 | -3.40 |
| BUY | 2025-06-13 10:15:00 | 141.10 | 2025-06-19 12:15:00 | 138.81 | EXIT_EMA400 | -2.29 |
| BUY | 2025-09-26 12:15:00 | 145.01 | 2025-09-29 09:15:00 | 148.84 | TARGET | 3.83 |
| BUY | 2025-12-19 14:15:00 | 162.62 | 2025-12-26 09:15:00 | 160.06 | EXIT_EMA400 | -2.56 |
