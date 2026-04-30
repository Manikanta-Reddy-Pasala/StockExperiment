# Dixon Technologies (India) Ltd. (DIXON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 11166.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -2108.00
- **Avg P&L per closed trade:** -351.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 13:15:00 | 14761.25 | 16196.79 | 16197.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 14511.85 | 16156.63 | 16177.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 14280.00 | 14093.33 | 14666.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 09:15:00 | 13990.00 | 14121.82 | 14661.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 13990.00 | 14121.82 | 14661.31 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 13892.00 | 14119.53 | 14657.47 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-11 11:15:00 | 14273.00 | 13630.55 | 14196.07 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 11:15:00 | 16335.00 | 14596.64 | 14594.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-25 12:15:00 | 16501.00 | 14615.58 | 14603.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 15169.00 | 15371.06 | 15045.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 16136.00 | 15366.33 | 15054.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 15690.00 | 15768.92 | 15344.87 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-22 09:15:00 | 15211.00 | 15754.96 | 15352.39 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 14528.00 | 15130.79 | 15131.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 14505.00 | 15124.56 | 15128.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 14823.00 | 14707.16 | 14881.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-01 12:15:00 | 14397.00 | 14695.48 | 14861.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 14797.00 | 14694.68 | 14857.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-02 11:15:00 | 14915.00 | 14697.50 | 14857.29 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 15716.00 | 14982.81 | 14979.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 15849.00 | 14999.52 | 14988.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 13:15:00 | 16137.00 | 16182.94 | 15765.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-18 09:15:00 | 16800.00 | 16144.87 | 15804.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 09:15:00 | 16890.00 | 17642.46 | 17052.88 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 15517.00 | 16809.08 | 16812.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 15456.00 | 16770.05 | 16793.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 11446.00 | 11421.63 | 12550.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-23 09:15:00 | 10854.00 | 11459.67 | 12166.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 10814.00 | 10436.24 | 10908.87 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 11:15:00 | 10945.00 | 10441.30 | 10909.05 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-25 09:15:00 | 13990.00 | 2025-04-11 11:15:00 | 14273.00 | EXIT_EMA400 | -283.00 |
| SELL | 2025-03-25 10:15:00 | 13892.00 | 2025-04-11 11:15:00 | 14273.00 | EXIT_EMA400 | -381.00 |
| BUY | 2025-05-12 09:15:00 | 16136.00 | 2025-05-22 09:15:00 | 15211.00 | EXIT_EMA400 | -925.00 |
| SELL | 2025-07-01 12:15:00 | 14397.00 | 2025-07-02 11:15:00 | 14915.00 | EXIT_EMA400 | -518.00 |
| BUY | 2025-08-18 09:15:00 | 16800.00 | 2025-09-29 09:15:00 | 16890.00 | EXIT_EMA400 | 90.00 |
| SELL | 2026-02-23 09:15:00 | 10854.00 | 2026-04-15 11:15:00 | 10945.00 | EXIT_EMA400 | -91.00 |
