# Adani Power Ltd. (ADANIPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 222.69
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** -1.78
- **Avg P&L per closed trade:** -0.59

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 131.06 | 138.80 | 138.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 11:15:00 | 130.30 | 138.72 | 138.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 09:15:00 | 134.27 | 132.63 | 135.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-19 09:15:00 | 128.76 | 132.56 | 134.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 133.77 | 132.32 | 134.52 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-23 15:15:00 | 134.68 | 132.38 | 134.49 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 14:15:00 | 109.25 | 102.94 | 102.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 10:15:00 | 109.96 | 103.13 | 103.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 106.10 | 107.15 | 105.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 10:15:00 | 112.08 | 107.12 | 105.47 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-07 09:15:00 | 105.22 | 107.41 | 105.73 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 140.23 | 145.13 | 145.14 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 149.48 | 145.16 | 145.14 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 140.26 | 145.14 | 145.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 140.18 | 144.26 | 144.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 142.20 | 139.91 | 142.06 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 12:15:00 | 149.50 | 143.83 | 143.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 150.95 | 143.96 | 143.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 143.38 | 144.39 | 144.09 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 141.02 | 143.82 | 143.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 13:15:00 | 140.39 | 143.71 | 143.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 09:15:00 | 143.27 | 141.58 | 142.51 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 155.64 | 143.37 | 143.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 155.76 | 143.49 | 143.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 11:15:00 | 145.25 | 145.60 | 144.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-23 13:15:00 | 148.25 | 145.63 | 144.58 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-19 09:15:00 | 128.76 | 2024-09-23 15:15:00 | 134.68 | EXIT_EMA400 | -5.92 |
| BUY | 2025-05-05 10:15:00 | 112.08 | 2025-05-07 09:15:00 | 105.22 | EXIT_EMA400 | -6.86 |
| BUY | 2026-03-23 13:15:00 | 148.25 | 2026-04-02 14:15:00 | 159.25 | TARGET | 11.00 |
