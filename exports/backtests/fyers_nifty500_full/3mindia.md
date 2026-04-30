# 3M India Ltd. (3MINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 33365.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 8 |
| ENTRY1 | 5 |
| ENTRY2 | 5 |
| EXIT | 5 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / EMA400 exits:** 3 / 7
- **Total realized P&L (per unit):** -707.25
- **Avg P&L per closed trade:** -70.73

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 11:15:00 | 35018.00 | 36015.42 | 36019.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 13:15:00 | 34800.30 | 35793.34 | 35896.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 35749.90 | 35673.34 | 35816.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-27 10:15:00 | 35172.80 | 35668.36 | 35813.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 34700.00 | 34288.24 | 34826.10 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-31 09:15:00 | 35319.95 | 34316.52 | 34821.96 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 29850.00 | 28750.17 | 28746.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 30000.00 | 28762.61 | 28752.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 28750.00 | 29289.15 | 29069.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 15:15:00 | 29200.00 | 29269.79 | 29066.23 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 29200.00 | 29269.79 | 29066.23 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-12 09:15:00 | 29610.00 | 29273.18 | 29068.94 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 29100.00 | 29290.86 | 29089.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-14 09:15:00 | 29455.00 | 29290.61 | 29091.80 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 29130.00 | 29293.18 | 29097.05 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-15 13:15:00 | 29480.00 | 29293.65 | 29104.02 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 29665.00 | 29588.74 | 29321.67 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-28 14:15:00 | 28850.00 | 29581.39 | 29319.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 14:15:00 | 28365.00 | 29244.88 | 29244.98 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 29670.00 | 29171.44 | 29170.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 29970.00 | 29184.59 | 29177.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 30215.00 | 30310.71 | 29893.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-07 12:15:00 | 30595.00 | 30312.74 | 29898.66 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 30405.00 | 30578.96 | 30217.96 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-28 12:15:00 | 30695.00 | 30577.99 | 30222.83 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 30380.00 | 30689.49 | 30344.90 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-05 11:15:00 | 30185.00 | 30679.23 | 30344.87 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 28975.00 | 30258.44 | 30263.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 28805.00 | 30231.58 | 30249.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 29645.00 | 29638.07 | 29872.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-24 12:15:00 | 29575.00 | 29644.06 | 29855.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 30055.00 | 29650.57 | 29854.50 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 36045.00 | 30033.74 | 30009.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 36375.00 | 30209.28 | 30098.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 34040.00 | 34147.01 | 32904.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-08 09:15:00 | 34500.00 | 34150.52 | 32912.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 34435.00 | 34968.16 | 34143.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-09 11:15:00 | 34920.00 | 34961.94 | 34148.32 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-09 14:15:00 | 34045.00 | 34944.00 | 34151.41 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 13:15:00 | 33260.00 | 34784.25 | 34791.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 14:15:00 | 33080.00 | 34767.29 | 34783.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 31985.00 | 31979.99 | 32888.38 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-27 10:15:00 | 35172.80 | 2024-10-22 12:15:00 | 33250.96 | TARGET | 1921.84 |
| BUY | 2025-05-09 15:15:00 | 29200.00 | 2025-05-12 09:15:00 | 29601.31 | TARGET | 401.31 |
| BUY | 2025-05-14 09:15:00 | 29455.00 | 2025-05-21 15:15:00 | 30544.59 | TARGET | 1089.59 |
| BUY | 2025-05-12 09:15:00 | 29610.00 | 2025-05-28 14:15:00 | 28850.00 | EXIT_EMA400 | -760.00 |
| BUY | 2025-05-15 13:15:00 | 29480.00 | 2025-05-28 14:15:00 | 28850.00 | EXIT_EMA400 | -630.00 |
| BUY | 2025-08-07 12:15:00 | 30595.00 | 2025-09-05 11:15:00 | 30185.00 | EXIT_EMA400 | -410.00 |
| BUY | 2025-08-28 12:15:00 | 30695.00 | 2025-09-05 11:15:00 | 30185.00 | EXIT_EMA400 | -510.00 |
| SELL | 2025-10-24 12:15:00 | 29575.00 | 2025-10-27 09:15:00 | 30055.00 | EXIT_EMA400 | -480.00 |
| BUY | 2025-12-08 09:15:00 | 34500.00 | 2026-01-09 14:15:00 | 34045.00 | EXIT_EMA400 | -455.00 |
| BUY | 2026-01-09 11:15:00 | 34920.00 | 2026-01-09 14:15:00 | 34045.00 | EXIT_EMA400 | -875.00 |
