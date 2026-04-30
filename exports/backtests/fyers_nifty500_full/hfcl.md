# HFCL Ltd. (HFCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 115.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -10.01
- **Avg P&L per closed trade:** -1.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 12:15:00 | 119.15 | 137.81 | 137.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 14:15:00 | 117.11 | 137.42 | 137.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 12:15:00 | 129.93 | 129.93 | 133.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-06 14:15:00 | 128.45 | 129.91 | 133.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 129.63 | 129.50 | 132.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-12 14:15:00 | 126.27 | 129.44 | 132.52 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 129.36 | 128.69 | 131.71 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-19 15:15:00 | 127.92 | 128.69 | 131.69 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 130.39 | 128.45 | 131.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-27 10:15:00 | 131.48 | 128.48 | 131.16 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 15:15:00 | 92.20 | 86.37 | 86.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 92.96 | 86.43 | 86.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 86.80 | 87.30 | 86.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 10:15:00 | 88.40 | 87.31 | 86.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 88.40 | 87.31 | 86.87 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 15:15:00 | 86.43 | 87.31 | 86.88 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 81.22 | 86.47 | 86.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 80.56 | 86.41 | 86.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 86.24 | 85.46 | 85.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-07 11:15:00 | 83.44 | 85.60 | 85.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 75.90 | 73.14 | 75.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-16 14:15:00 | 76.50 | 73.20 | 75.98 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 73.21 | 68.77 | 68.76 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 13:15:00 | 65.10 | 68.81 | 68.82 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 70.63 | 68.83 | 68.83 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 66.35 | 68.81 | 68.82 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 71.13 | 68.82 | 68.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 71.67 | 68.91 | 68.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 69.75 | 70.07 | 69.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-20 09:15:00 | 71.70 | 70.09 | 69.54 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 68.03 | 70.15 | 69.59 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-06 14:15:00 | 128.45 | 2024-11-27 10:15:00 | 131.48 | EXIT_EMA400 | -3.03 |
| SELL | 2024-11-12 14:15:00 | 126.27 | 2024-11-27 10:15:00 | 131.48 | EXIT_EMA400 | -5.21 |
| SELL | 2024-11-19 15:15:00 | 127.92 | 2024-11-27 10:15:00 | 131.48 | EXIT_EMA400 | -3.56 |
| BUY | 2025-06-13 10:15:00 | 88.40 | 2025-06-13 15:15:00 | 86.43 | EXIT_EMA400 | -1.97 |
| SELL | 2025-07-07 11:15:00 | 83.44 | 2025-07-25 12:15:00 | 76.01 | TARGET | 7.43 |
| BUY | 2026-03-20 09:15:00 | 71.70 | 2026-03-23 09:15:00 | 68.03 | EXIT_EMA400 | -3.67 |
