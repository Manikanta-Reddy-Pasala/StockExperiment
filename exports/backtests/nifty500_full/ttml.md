# Tata Teleservices (Maharashtra) Ltd. (TTML.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 42.93
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -4.56
- **Avg P&L per closed trade:** -1.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 15:15:00 | 88.45 | 91.69 | 91.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 87.70 | 91.62 | 91.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 14:15:00 | 89.95 | 89.70 | 90.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-11 09:15:00 | 87.25 | 89.67 | 90.57 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-23 09:15:00 | 89.45 | 80.26 | 83.36 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 11:15:00 | 108.28 | 79.11 | 78.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 09:15:00 | 108.55 | 80.37 | 79.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-06 14:15:00 | 90.07 | 90.34 | 86.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-07 10:15:00 | 91.73 | 90.35 | 86.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-09 09:15:00 | 89.95 | 93.38 | 90.33 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 83.30 | 89.20 | 89.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 82.90 | 89.13 | 89.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 74.96 | 72.94 | 77.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-10 14:15:00 | 69.45 | 76.76 | 77.88 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-20 09:15:00 | 80.80 | 74.75 | 76.57 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 75.33 | 62.42 | 62.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 80.09 | 67.97 | 65.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 67.81 | 69.37 | 66.68 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 63.10 | 65.95 | 65.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 62.48 | 65.87 | 65.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 59.90 | 58.36 | 60.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 14:15:00 | 57.71 | 58.34 | 59.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-07 12:15:00 | 59.10 | 57.22 | 58.77 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-11 09:15:00 | 87.25 | 2024-03-13 12:15:00 | 77.29 | TARGET | 9.96 |
| BUY | 2024-08-07 10:15:00 | 91.73 | 2024-09-09 09:15:00 | 89.95 | EXIT_EMA400 | -1.78 |
| SELL | 2025-01-10 14:15:00 | 69.45 | 2025-01-20 09:15:00 | 80.80 | EXIT_EMA400 | -11.35 |
| SELL | 2025-09-19 14:15:00 | 57.71 | 2025-10-07 12:15:00 | 59.10 | EXIT_EMA400 | -1.39 |
