# Page Industries Ltd. (PAGEIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 36785.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -5518.65
- **Avg P&L per closed trade:** -919.77

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 15:15:00 | 37671.65 | 39328.79 | 39336.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 37480.35 | 39275.86 | 39310.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 13:15:00 | 38261.25 | 38233.34 | 38648.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-13 09:15:00 | 37790.30 | 38231.38 | 38641.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 37657.20 | 37806.98 | 38229.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-12-04 10:15:00 | 37500.00 | 37803.93 | 38225.57 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 37899.80 | 37607.05 | 37997.71 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-12-18 09:15:00 | 38187.50 | 37629.68 | 37995.77 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 15:15:00 | 38000.00 | 35575.09 | 35570.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 38261.00 | 35791.23 | 35684.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 09:15:00 | 39919.50 | 40613.74 | 39456.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-12 13:15:00 | 40852.00 | 40566.64 | 39494.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 40441.60 | 41273.85 | 40431.11 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-06 14:15:00 | 40301.10 | 41243.09 | 40432.28 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 43603.30 | 45902.68 | 45912.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 12:15:00 | 42961.40 | 45849.48 | 45885.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 41600.00 | 41541.01 | 42861.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 40990.00 | 42116.69 | 42783.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-09 14:15:00 | 42758.05 | 42048.05 | 42686.15 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 45970.00 | 43177.28 | 43171.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 47015.00 | 44163.37 | 43731.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 46080.00 | 46090.53 | 45110.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-29 14:15:00 | 46575.00 | 46088.91 | 45143.71 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 45530.00 | 46129.96 | 45523.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-18 10:15:00 | 45200.00 | 46120.71 | 45521.43 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 43840.00 | 46514.64 | 46523.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 43630.00 | 46485.94 | 46509.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 46215.00 | 45978.57 | 46225.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 09:15:00 | 44855.00 | 46010.17 | 46210.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 45485.00 | 45104.56 | 45570.60 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-17 11:15:00 | 45595.00 | 45112.90 | 45570.14 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 38205.00 | 33917.35 | 33900.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 14:15:00 | 38460.00 | 34157.24 | 34022.33 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-13 09:15:00 | 37790.30 | 2023-12-18 09:15:00 | 38187.50 | EXIT_EMA400 | -397.20 |
| SELL | 2023-12-04 10:15:00 | 37500.00 | 2023-12-18 09:15:00 | 38187.50 | EXIT_EMA400 | -687.50 |
| BUY | 2024-08-12 13:15:00 | 40852.00 | 2024-09-06 14:15:00 | 40301.10 | EXIT_EMA400 | -550.90 |
| SELL | 2025-04-07 09:15:00 | 40990.00 | 2025-04-09 14:15:00 | 42758.05 | EXIT_EMA400 | -1768.05 |
| BUY | 2025-05-29 14:15:00 | 46575.00 | 2025-06-18 10:15:00 | 45200.00 | EXIT_EMA400 | -1375.00 |
| SELL | 2025-08-28 09:15:00 | 44855.00 | 2025-09-17 11:15:00 | 45595.00 | EXIT_EMA400 | -740.00 |
