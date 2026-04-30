# Redington Ltd. (REDINGTON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 215.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -11.81
- **Avg P&L per closed trade:** -2.95

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 12:15:00 | 197.25 | 190.17 | 190.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 202.17 | 190.50 | 190.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 202.70 | 203.06 | 198.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-07 15:15:00 | 205.72 | 201.66 | 198.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 204.83 | 202.25 | 199.52 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-16 14:15:00 | 216.73 | 203.01 | 200.16 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 207.22 | 210.19 | 204.81 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-28 10:15:00 | 204.73 | 210.13 | 204.81 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 14:15:00 | 242.90 | 281.46 | 281.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 237.70 | 280.64 | 281.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 266.00 | 248.62 | 258.07 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 287.84 | 265.40 | 265.33 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 253.10 | 267.49 | 267.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 13:15:00 | 252.40 | 266.70 | 267.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 277.45 | 265.51 | 266.51 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 289.65 | 267.60 | 267.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 298.80 | 271.34 | 269.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 281.20 | 282.61 | 277.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-01 09:15:00 | 285.85 | 282.49 | 277.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 277.80 | 282.40 | 277.51 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-03 10:15:00 | 277.05 | 282.34 | 277.51 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 272.00 | 275.38 | 275.39 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 280.10 | 275.42 | 275.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 289.60 | 275.65 | 275.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 09:15:00 | 275.25 | 277.58 | 276.55 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 267.45 | 275.66 | 275.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 263.50 | 275.08 | 275.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 269.40 | 268.36 | 271.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 10:15:00 | 262.50 | 268.28 | 271.30 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 274.20 | 268.23 | 271.19 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-01-07 15:15:00 | 205.72 | 2025-01-21 09:15:00 | 226.41 | TARGET | 20.69 |
| BUY | 2025-01-16 14:15:00 | 216.73 | 2025-01-28 10:15:00 | 204.73 | EXIT_EMA400 | -12.00 |
| BUY | 2025-12-01 09:15:00 | 285.85 | 2025-12-03 10:15:00 | 277.05 | EXIT_EMA400 | -8.80 |
| SELL | 2026-02-02 10:15:00 | 262.50 | 2026-02-03 09:15:00 | 274.20 | EXIT_EMA400 | -11.70 |
