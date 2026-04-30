# Bosch Ltd. (BOSCHLTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 36145.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -7848.70
- **Avg P&L per closed trade:** -981.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 12:15:00 | 31960.00 | 32758.45 | 32759.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 13:15:00 | 31936.55 | 32750.27 | 32755.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 32615.95 | 32560.72 | 32648.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-06 14:15:00 | 32361.90 | 32609.86 | 32669.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 32623.35 | 32603.76 | 32665.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-09 13:15:00 | 33050.00 | 32608.25 | 32666.54 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 14:15:00 | 33096.35 | 32722.52 | 32721.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 15:15:00 | 33323.00 | 32728.50 | 32724.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 14:15:00 | 36489.95 | 36504.99 | 35225.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-21 10:15:00 | 37149.95 | 36531.18 | 35301.57 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 35647.60 | 36491.34 | 35442.42 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-25 14:15:00 | 35974.30 | 36461.70 | 35448.21 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-29 09:15:00 | 35460.55 | 36421.74 | 35472.44 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 15:15:00 | 34856.00 | 35108.11 | 35108.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 15:15:00 | 34780.00 | 35094.74 | 35102.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 35096.70 | 35094.41 | 35101.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-03 11:15:00 | 34778.75 | 35091.27 | 35100.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 11:15:00 | 35149.00 | 35077.31 | 35092.82 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 12:15:00 | 36123.15 | 35110.69 | 35109.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 15:15:00 | 36300.00 | 35142.89 | 35125.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 12:15:00 | 35571.75 | 35599.79 | 35399.40 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 33676.20 | 35243.12 | 35243.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 33483.70 | 34863.40 | 35034.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 12:15:00 | 27565.10 | 27504.04 | 29029.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-21 14:15:00 | 27402.80 | 27523.65 | 28922.14 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 27960.00 | 27432.01 | 28221.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-23 14:15:00 | 28465.00 | 27474.13 | 28219.88 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 30835.00 | 28660.62 | 28655.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 31195.00 | 28794.86 | 28723.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 39620.00 | 39808.50 | 38107.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-17 12:15:00 | 40130.00 | 39803.14 | 38243.13 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-25 14:15:00 | 38425.00 | 39657.57 | 38475.44 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 36670.00 | 38258.93 | 38259.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 36235.00 | 38000.11 | 38123.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 37035.00 | 37020.23 | 37462.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-08 10:15:00 | 36565.00 | 36998.80 | 37419.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 09:15:00 | 37215.00 | 36345.10 | 36816.43 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 37685.00 | 37207.96 | 37205.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 37895.00 | 37218.66 | 37211.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 37210.00 | 37263.14 | 37234.43 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 36335.00 | 37200.60 | 37204.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 35935.00 | 37188.01 | 37197.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 14:15:00 | 36570.00 | 36536.67 | 36821.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-01 12:15:00 | 36090.00 | 36526.40 | 36809.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 37260.00 | 36507.66 | 36784.41 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 38200.00 | 34571.80 | 34557.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 38290.00 | 34644.49 | 34593.92 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-06 14:15:00 | 32361.90 | 2024-09-09 13:15:00 | 33050.00 | EXIT_EMA400 | -688.10 |
| BUY | 2024-10-21 10:15:00 | 37149.95 | 2024-10-29 09:15:00 | 35460.55 | EXIT_EMA400 | -1689.40 |
| BUY | 2024-10-25 14:15:00 | 35974.30 | 2024-10-29 09:15:00 | 35460.55 | EXIT_EMA400 | -513.75 |
| SELL | 2024-12-03 11:15:00 | 34778.75 | 2024-12-04 11:15:00 | 35149.00 | EXIT_EMA400 | -370.25 |
| SELL | 2025-03-21 14:15:00 | 27402.80 | 2025-04-23 14:15:00 | 28465.00 | EXIT_EMA400 | -1062.20 |
| BUY | 2025-09-17 12:15:00 | 40130.00 | 2025-09-25 14:15:00 | 38425.00 | EXIT_EMA400 | -1705.00 |
| SELL | 2025-12-08 10:15:00 | 36565.00 | 2026-01-02 09:15:00 | 37215.00 | EXIT_EMA400 | -650.00 |
| SELL | 2026-02-01 12:15:00 | 36090.00 | 2026-02-03 09:15:00 | 37260.00 | EXIT_EMA400 | -1170.00 |
