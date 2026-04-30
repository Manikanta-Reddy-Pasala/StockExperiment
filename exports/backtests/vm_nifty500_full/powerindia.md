# Hitachi Energy India Ltd. (POWERINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 33550.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 7 |
| ENTRY1 | 5 |
| ENTRY2 | 6 |
| EXIT | 5 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / EMA400 exits:** 5 / 6
- **Total realized P&L (per unit):** 911.14
- **Avg P&L per closed trade:** 82.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 13:15:00 | 3975.00 | 4240.57 | 4241.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 15:15:00 | 3969.40 | 4235.25 | 4239.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 11:15:00 | 4200.90 | 4176.31 | 4206.38 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 12:15:00 | 4562.90 | 4233.19 | 4232.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 14:15:00 | 4588.05 | 4261.99 | 4247.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-25 09:15:00 | 4271.45 | 4309.75 | 4274.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-02 12:15:00 | 4368.90 | 4271.75 | 4260.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 11:15:00 | 4365.80 | 4348.74 | 4307.55 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-11-15 12:15:00 | 4375.80 | 4349.01 | 4307.89 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 11197.20 | 12110.99 | 11092.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-22 09:15:00 | 11362.95 | 12094.58 | 11094.14 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 11405.95 | 12018.85 | 11104.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-24 09:15:00 | 11832.25 | 11996.40 | 11110.88 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-06 10:15:00 | 11300.05 | 11919.84 | 11309.49 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 09:15:00 | 12005.05 | 13148.03 | 13150.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-25 14:15:00 | 11609.85 | 13092.28 | 13122.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 12613.20 | 12603.50 | 12819.53 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 11:15:00 | 14351.50 | 12947.99 | 12945.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 14540.75 | 13060.71 | 13003.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 15:15:00 | 13805.00 | 13846.71 | 13482.72 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 11487.05 | 13251.70 | 13257.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 11339.95 | 13214.88 | 13238.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 13518.60 | 12607.97 | 12904.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-31 11:15:00 | 12547.30 | 12610.39 | 12902.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 12547.30 | 12610.39 | 12902.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-31 12:15:00 | 12375.85 | 12608.06 | 12900.35 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 12800.00 | 12608.54 | 12897.68 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-03 09:15:00 | 11189.80 | 12596.32 | 12888.68 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-04 10:15:00 | 12205.70 | 11747.04 | 12187.70 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 13592.00 | 12311.86 | 12309.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 10:15:00 | 13682.00 | 12325.49 | 12316.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 17000.00 | 17074.88 | 15672.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 10:15:00 | 17446.00 | 17075.70 | 15714.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 19260.00 | 19980.05 | 19214.61 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-26 12:15:00 | 19150.00 | 19971.79 | 19214.29 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 17785.00 | 19137.31 | 19139.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 17557.00 | 19099.86 | 19120.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 10:15:00 | 18020.00 | 17972.08 | 18418.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 10:15:00 | 17872.00 | 17972.16 | 18403.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-04 09:15:00 | 20236.00 | 17967.43 | 18373.64 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 21865.00 | 18735.95 | 18726.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 22172.00 | 19781.04 | 19311.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 20560.00 | 20898.89 | 20110.83 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 18630.00 | 19672.52 | 19676.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 18410.00 | 19659.96 | 19670.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 19340.00 | 19173.95 | 19383.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 14:15:00 | 18460.00 | 19207.03 | 19388.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 18460.00 | 19207.03 | 19388.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-09 09:15:00 | 17808.00 | 19185.59 | 19376.31 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 18562.00 | 17921.41 | 18525.11 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 14:15:00 | 22474.00 | 18967.14 | 18949.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 22565.00 | 19036.67 | 18984.94 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-15 12:15:00 | 4375.80 | 2023-11-16 10:15:00 | 4579.53 | TARGET | 203.73 |
| BUY | 2023-11-02 12:15:00 | 4368.90 | 2023-11-20 12:15:00 | 4693.43 | TARGET | 324.53 |
| BUY | 2024-07-22 09:15:00 | 11362.95 | 2024-07-25 10:15:00 | 12169.37 | TARGET | 806.42 |
| BUY | 2024-07-24 09:15:00 | 11832.25 | 2024-08-06 10:15:00 | 11300.05 | EXIT_EMA400 | -532.20 |
| SELL | 2025-01-31 11:15:00 | 12547.30 | 2025-02-03 09:15:00 | 11480.25 | TARGET | 1067.04 |
| SELL | 2025-01-31 12:15:00 | 12375.85 | 2025-02-17 11:15:00 | 10802.34 | TARGET | 1573.51 |
| SELL | 2025-02-03 09:15:00 | 11189.80 | 2025-03-04 10:15:00 | 12205.70 | EXIT_EMA400 | -1015.90 |
| BUY | 2025-06-13 10:15:00 | 17446.00 | 2025-08-26 12:15:00 | 19150.00 | EXIT_EMA400 | 1704.00 |
| SELL | 2025-10-31 10:15:00 | 17872.00 | 2025-11-04 09:15:00 | 20236.00 | EXIT_EMA400 | -2364.00 |
| SELL | 2026-01-08 14:15:00 | 18460.00 | 2026-01-30 09:15:00 | 18562.00 | EXIT_EMA400 | -102.00 |
| SELL | 2026-01-09 09:15:00 | 17808.00 | 2026-01-30 09:15:00 | 18562.00 | EXIT_EMA400 | -754.00 |
