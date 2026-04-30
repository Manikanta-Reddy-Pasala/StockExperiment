# GE Vernova T&D India Ltd. (GVT&D.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-11-05 09:15:00 → 2026-04-30 15:30:00 (2547 bars)
- **Last close:** 4466.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 199.08
- **Avg P&L per closed trade:** 49.77

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 1591.45 | 1885.98 | 1887.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-29 12:15:00 | 1590.05 | 1871.82 | 1880.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 1518.40 | 1495.76 | 1606.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-03 09:15:00 | 1451.20 | 1520.81 | 1586.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1501.70 | 1457.32 | 1525.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-25 09:15:00 | 1468.30 | 1464.42 | 1524.80 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 1521.50 | 1467.74 | 1523.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-28 14:15:00 | 1535.50 | 1468.42 | 1523.33 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 15:15:00 | 1770.00 | 1556.38 | 1555.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 1787.60 | 1558.68 | 1556.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 2277.50 | 2293.41 | 2132.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 09:15:00 | 2328.40 | 2293.12 | 2137.51 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-23 14:15:00 | 2804.20 | 2953.88 | 2835.54 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 2634.00 | 2951.03 | 2951.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 2594.80 | 2932.70 | 2941.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 2867.10 | 2852.66 | 2895.99 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 3525.00 | 2937.93 | 2935.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 3627.70 | 2960.46 | 2946.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 3606.30 | 3616.00 | 3413.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-17 10:15:00 | 3663.50 | 3607.14 | 3421.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 3519.00 | 3637.66 | 3461.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-23 10:15:00 | 3441.40 | 3635.71 | 3460.92 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-03 09:15:00 | 1451.20 | 2025-04-28 14:15:00 | 1535.50 | EXIT_EMA400 | -84.30 |
| SELL | 2025-04-25 09:15:00 | 1468.30 | 2025-04-28 14:15:00 | 1535.50 | EXIT_EMA400 | -67.20 |
| BUY | 2025-07-15 09:15:00 | 2328.40 | 2025-08-04 09:15:00 | 2901.08 | TARGET | 572.68 |
| BUY | 2026-03-17 10:15:00 | 3663.50 | 2026-03-23 10:15:00 | 3441.40 | EXIT_EMA400 | -222.10 |
