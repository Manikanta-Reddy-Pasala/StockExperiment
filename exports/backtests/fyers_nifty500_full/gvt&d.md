# GE Vernova T&D India Ltd. (GVT&D.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3402 bars)
- **Last close:** 4475.00
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
- **Total realized P&L (per unit):** 195.42
- **Avg P&L per closed trade:** 48.85

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 1590.05 | 1871.39 | 1872.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 1528.30 | 1772.29 | 1814.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 1520.30 | 1494.85 | 1602.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-03 09:15:00 | 1451.20 | 1520.38 | 1583.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1501.70 | 1457.16 | 1523.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-25 09:15:00 | 1468.30 | 1464.30 | 1523.13 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 1521.50 | 1467.62 | 1521.68 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-28 14:15:00 | 1535.50 | 1468.30 | 1521.75 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 1780.30 | 1554.23 | 1553.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 1787.30 | 1558.60 | 1555.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 2277.00 | 2293.19 | 2132.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 09:15:00 | 2328.40 | 2292.91 | 2137.13 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-23 14:15:00 | 2804.20 | 2953.72 | 2835.48 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 2634.00 | 2950.62 | 2950.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 2594.80 | 2932.34 | 2941.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 2867.10 | 2852.42 | 2895.74 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 3277.60 | 2933.20 | 2932.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 3417.00 | 2941.45 | 2936.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 3606.30 | 3618.98 | 3417.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-17 10:15:00 | 3668.30 | 3610.12 | 3424.94 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 3519.00 | 3640.02 | 3464.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-23 10:15:00 | 3441.40 | 3638.04 | 3464.25 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-03 09:15:00 | 1451.20 | 2025-04-28 14:15:00 | 1535.50 | EXIT_EMA400 | -84.30 |
| SELL | 2025-04-25 09:15:00 | 1468.30 | 2025-04-28 14:15:00 | 1535.50 | EXIT_EMA400 | -67.20 |
| BUY | 2025-07-15 09:15:00 | 2328.40 | 2025-08-04 09:15:00 | 2902.22 | TARGET | 573.82 |
| BUY | 2026-03-17 10:15:00 | 3668.30 | 2026-03-23 10:15:00 | 3441.40 | EXIT_EMA400 | -226.90 |
