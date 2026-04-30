# IDBI Bank Ltd. (IDBI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 76.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 0.44
- **Avg P&L per closed trade:** 0.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 11:15:00 | 88.88 | 91.51 | 91.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 12:15:00 | 88.50 | 91.48 | 91.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 09:15:00 | 84.34 | 84.16 | 86.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-04 09:15:00 | 81.84 | 84.09 | 86.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 85.85 | 83.91 | 86.12 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-07 09:15:00 | 86.65 | 83.94 | 86.12 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 80.76 | 76.33 | 76.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 81.85 | 76.52 | 76.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 79.30 | 79.55 | 78.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-02 09:15:00 | 80.48 | 79.56 | 78.25 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-06 14:15:00 | 77.79 | 79.62 | 78.40 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 88.54 | 92.87 | 92.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 88.10 | 92.83 | 92.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 92.80 | 92.18 | 92.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 12:15:00 | 90.01 | 92.58 | 92.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-08 09:15:00 | 92.02 | 91.27 | 91.94 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 93.30 | 92.40 | 92.39 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 89.77 | 92.38 | 92.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 88.88 | 92.15 | 92.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 92.30 | 91.97 | 92.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-08 10:15:00 | 91.60 | 92.12 | 92.22 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 09:15:00 | 93.90 | 92.05 | 92.18 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 94.16 | 92.30 | 92.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 95.28 | 92.33 | 92.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 98.30 | 99.56 | 97.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-22 09:15:00 | 101.49 | 98.27 | 97.41 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 100.02 | 102.99 | 100.81 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 10:15:00 | 75.08 | 103.64 | 103.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 14:15:00 | 73.90 | 102.50 | 103.18 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-04 09:15:00 | 81.84 | 2024-11-07 09:15:00 | 86.65 | EXIT_EMA400 | -4.81 |
| BUY | 2025-05-02 09:15:00 | 80.48 | 2025-05-06 14:15:00 | 77.79 | EXIT_EMA400 | -2.69 |
| SELL | 2025-08-28 12:15:00 | 90.01 | 2025-09-08 09:15:00 | 92.02 | EXIT_EMA400 | -2.01 |
| SELL | 2025-10-08 10:15:00 | 91.60 | 2025-10-10 09:15:00 | 93.90 | EXIT_EMA400 | -2.30 |
| BUY | 2025-12-22 09:15:00 | 101.49 | 2026-01-02 13:15:00 | 113.74 | TARGET | 12.25 |
