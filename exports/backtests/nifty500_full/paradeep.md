# Paradeep Phosphates Ltd. (PARADEEP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 128.96
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / EMA400 exits:** 5 / 3
- **Total realized P&L (per unit):** 52.83
- **Avg P&L per closed trade:** 6.60

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 11:15:00 | 60.35 | 66.80 | 66.81 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 68.05 | 65.70 | 65.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 13:15:00 | 68.75 | 65.76 | 65.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 67.35 | 67.78 | 66.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-26 09:15:00 | 69.95 | 67.85 | 66.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 74.30 | 76.36 | 73.44 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-07 09:15:00 | 75.75 | 76.22 | 73.47 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 75.55 | 77.52 | 75.51 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-01 09:15:00 | 77.20 | 77.45 | 75.54 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-03-04 15:15:00 | 75.50 | 77.33 | 75.60 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 15:15:00 | 67.20 | 74.49 | 74.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 66.70 | 74.41 | 74.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 10:15:00 | 71.90 | 71.84 | 72.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-12 14:15:00 | 71.20 | 72.21 | 72.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 71.45 | 71.28 | 72.29 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-26 14:15:00 | 70.35 | 71.30 | 72.21 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-05-15 14:15:00 | 72.20 | 69.05 | 70.60 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 12:15:00 | 74.21 | 70.71 | 70.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 13:15:00 | 74.91 | 70.75 | 70.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 83.45 | 83.99 | 79.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-24 11:15:00 | 85.21 | 84.01 | 79.86 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-06 14:15:00 | 81.77 | 85.48 | 81.88 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 98.13 | 108.38 | 108.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 11:15:00 | 97.23 | 108.17 | 108.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 95.65 | 95.39 | 99.49 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 13:15:00 | 113.92 | 102.12 | 102.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-07 15:15:00 | 114.65 | 102.36 | 102.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 15:15:00 | 163.60 | 163.61 | 151.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 166.70 | 163.64 | 151.20 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-08 09:15:00 | 194.66 | 209.85 | 196.45 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 178.60 | 189.02 | 189.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 176.42 | 188.70 | 188.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 09:15:00 | 163.16 | 160.21 | 167.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-18 09:15:00 | 153.14 | 159.86 | 166.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-30 09:15:00 | 168.28 | 159.74 | 164.97 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-26 09:15:00 | 69.95 | 2024-01-03 09:15:00 | 78.88 | TARGET | 8.93 |
| BUY | 2024-02-07 09:15:00 | 75.75 | 2024-02-08 12:15:00 | 82.60 | TARGET | 6.85 |
| BUY | 2024-03-01 09:15:00 | 77.20 | 2024-03-04 15:15:00 | 75.50 | EXIT_EMA400 | -1.70 |
| SELL | 2024-04-12 14:15:00 | 71.20 | 2024-05-09 12:15:00 | 65.94 | TARGET | 5.26 |
| SELL | 2024-04-26 14:15:00 | 70.35 | 2024-05-10 09:15:00 | 64.77 | TARGET | 5.58 |
| BUY | 2024-07-24 11:15:00 | 85.21 | 2024-08-06 14:15:00 | 81.77 | EXIT_EMA400 | -3.44 |
| BUY | 2025-06-24 09:15:00 | 166.70 | 2025-07-29 09:15:00 | 213.20 | TARGET | 46.50 |
| SELL | 2025-12-18 09:15:00 | 153.14 | 2025-12-30 09:15:00 | 168.28 | EXIT_EMA400 | -15.14 |
