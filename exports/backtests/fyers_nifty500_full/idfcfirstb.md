# IDFC First Bank Ltd. (IDFCFIRSTB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 69.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -5.48
- **Avg P&L per closed trade:** -1.37

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 68.40 | 60.12 | 60.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 69.00 | 63.74 | 62.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 72.89 | 73.41 | 70.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-16 11:15:00 | 73.75 | 73.40 | 70.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 10:15:00 | 70.96 | 73.25 | 71.14 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 09:15:00 | 68.63 | 70.22 | 70.23 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 72.10 | 70.23 | 70.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 72.60 | 70.25 | 70.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 71.08 | 71.13 | 70.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-16 13:15:00 | 71.40 | 71.13 | 70.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-22 14:15:00 | 70.89 | 71.30 | 70.89 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 12:15:00 | 69.07 | 70.59 | 70.60 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 71.80 | 70.60 | 70.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 72.00 | 70.62 | 70.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 78.01 | 78.26 | 76.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-25 09:15:00 | 78.99 | 78.27 | 76.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-21 09:15:00 | 81.24 | 83.47 | 81.64 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 69.27 | 81.77 | 81.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 66.63 | 77.43 | 79.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 15:15:00 | 66.13 | 66.07 | 70.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 09:15:00 | 65.69 | 66.06 | 70.90 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-27 13:15:00 | 70.12 | 66.88 | 69.74 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-07-16 11:15:00 | 73.75 | 2025-07-25 10:15:00 | 70.96 | EXIT_EMA400 | -2.79 |
| BUY | 2025-09-16 13:15:00 | 71.40 | 2025-09-22 14:15:00 | 70.89 | EXIT_EMA400 | -0.51 |
| BUY | 2025-11-25 09:15:00 | 78.99 | 2026-01-21 09:15:00 | 81.24 | EXIT_EMA400 | 2.25 |
| SELL | 2026-04-09 09:15:00 | 65.69 | 2026-04-27 13:15:00 | 70.12 | EXIT_EMA400 | -4.43 |
