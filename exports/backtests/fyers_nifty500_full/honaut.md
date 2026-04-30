# Honeywell Automation India Ltd. (HONAUT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 31100.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 1566.12
- **Avg P&L per closed trade:** 313.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 14:15:00 | 52086.05 | 53291.39 | 53291.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 15:15:00 | 51865.05 | 53277.20 | 53284.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 09:15:00 | 50139.00 | 49752.76 | 50775.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 09:15:00 | 49255.50 | 49807.06 | 50642.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-18 12:15:00 | 50641.65 | 49807.92 | 50602.20 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 37200.00 | 35757.34 | 35752.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 10:15:00 | 37375.00 | 35787.64 | 35767.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 37685.00 | 37749.30 | 37071.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 38430.00 | 37745.36 | 37142.93 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 13:15:00 | 38655.00 | 39733.68 | 38857.95 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 35700.00 | 38345.30 | 38350.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 35015.00 | 36225.95 | 36610.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 35965.00 | 35806.77 | 36325.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-04 15:15:00 | 34765.00 | 35579.17 | 36069.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 34560.00 | 34017.69 | 34818.05 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-09 09:15:00 | 33595.00 | 34030.28 | 34721.18 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 33820.00 | 33211.32 | 33993.88 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-01 13:15:00 | 33070.00 | 33218.43 | 33963.12 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 33685.00 | 33209.14 | 33921.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 14:15:00 | 33925.00 | 33237.85 | 33918.70 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-17 09:15:00 | 49255.50 | 2024-10-18 12:15:00 | 50641.65 | EXIT_EMA400 | -1386.15 |
| BUY | 2025-06-24 09:15:00 | 38430.00 | 2025-07-28 13:15:00 | 38655.00 | EXIT_EMA400 | 225.00 |
| SELL | 2025-12-04 15:15:00 | 34765.00 | 2026-01-27 13:15:00 | 30852.73 | TARGET | 3912.27 |
| SELL | 2026-01-09 09:15:00 | 33595.00 | 2026-02-03 14:15:00 | 33925.00 | EXIT_EMA400 | -330.00 |
| SELL | 2026-02-01 13:15:00 | 33070.00 | 2026-02-03 14:15:00 | 33925.00 | EXIT_EMA400 | -855.00 |
