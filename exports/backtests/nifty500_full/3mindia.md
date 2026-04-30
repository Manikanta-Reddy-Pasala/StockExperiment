# 3M India Ltd. (3MINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 33300.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / EMA400 exits:** 0 / 9
- **Total realized P&L (per unit):** -6476.05
- **Avg P&L per closed trade:** -719.56

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 31009.25 | 32531.77 | 32531.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 30669.35 | 32513.23 | 32522.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 32278.45 | 31954.62 | 32212.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-04 11:15:00 | 30911.05 | 31916.32 | 32178.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-03-27 14:15:00 | 31597.00 | 30769.71 | 31344.47 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 12:15:00 | 33413.85 | 30601.64 | 30590.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 14:15:00 | 33916.95 | 30660.62 | 30620.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 10:15:00 | 37413.25 | 37447.24 | 35511.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-22 09:15:00 | 37893.00 | 37457.67 | 35574.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-14 14:15:00 | 35999.85 | 37944.59 | 36806.16 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 13:15:00 | 34931.25 | 36209.67 | 36213.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 34468.00 | 35636.98 | 35856.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 11:15:00 | 35107.70 | 35091.40 | 35494.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-10 12:15:00 | 34663.00 | 35087.13 | 35490.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-31 09:15:00 | 35319.95 | 34320.16 | 34852.53 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 14:15:00 | 29975.00 | 28764.41 | 28762.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-25 13:15:00 | 30250.00 | 28890.08 | 28827.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 29295.00 | 29295.38 | 29076.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 29610.00 | 29272.86 | 29074.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 29100.00 | 29290.57 | 29095.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-13 15:15:00 | 29080.00 | 29288.48 | 29095.41 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 14:15:00 | 28360.00 | 29248.06 | 29249.52 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 29670.00 | 29174.71 | 29174.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 29970.00 | 29187.70 | 29181.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 30215.00 | 30311.00 | 29895.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-07 12:15:00 | 30595.00 | 30313.03 | 29900.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 30405.00 | 30575.89 | 30217.36 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-28 12:15:00 | 30695.00 | 30575.01 | 30222.25 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 30455.00 | 30689.37 | 30345.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-05 11:15:00 | 30185.00 | 30679.11 | 30345.46 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 28975.00 | 30257.79 | 30263.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 28805.00 | 30230.94 | 30249.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 29645.00 | 29637.48 | 29872.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-23 13:15:00 | 29595.00 | 29645.22 | 29862.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 29645.00 | 29644.53 | 29859.08 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 30055.00 | 29650.09 | 29854.49 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 36040.00 | 30034.29 | 30009.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 36375.00 | 30210.12 | 30098.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 34040.00 | 34148.25 | 32905.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-08 09:15:00 | 34500.00 | 34151.75 | 32913.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 34435.00 | 34969.32 | 34144.64 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-09 11:15:00 | 34920.00 | 34963.08 | 34149.70 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-09 14:15:00 | 34045.00 | 34945.10 | 34152.76 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 13:15:00 | 33260.00 | 34783.94 | 34789.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 14:15:00 | 33080.00 | 34766.98 | 34780.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 10:15:00 | 32005.00 | 31995.99 | 32908.83 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-04 11:15:00 | 30911.05 | 2024-03-27 14:15:00 | 31597.00 | EXIT_EMA400 | -685.95 |
| BUY | 2024-07-22 09:15:00 | 37893.00 | 2024-08-14 14:15:00 | 35999.85 | EXIT_EMA400 | -1893.15 |
| SELL | 2024-10-10 12:15:00 | 34663.00 | 2024-10-31 09:15:00 | 35319.95 | EXIT_EMA400 | -656.95 |
| BUY | 2025-05-12 09:15:00 | 29610.00 | 2025-05-13 15:15:00 | 29080.00 | EXIT_EMA400 | -530.00 |
| BUY | 2025-08-07 12:15:00 | 30595.00 | 2025-09-05 11:15:00 | 30185.00 | EXIT_EMA400 | -410.00 |
| BUY | 2025-08-28 12:15:00 | 30695.00 | 2025-09-05 11:15:00 | 30185.00 | EXIT_EMA400 | -510.00 |
| SELL | 2025-10-23 13:15:00 | 29595.00 | 2025-10-27 09:15:00 | 30055.00 | EXIT_EMA400 | -460.00 |
| BUY | 2025-12-08 09:15:00 | 34500.00 | 2026-01-09 14:15:00 | 34045.00 | EXIT_EMA400 | -455.00 |
| BUY | 2026-01-09 11:15:00 | 34920.00 | 2026-01-09 14:15:00 | 34045.00 | EXIT_EMA400 | -875.00 |
