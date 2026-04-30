# Neuland Laboratories Ltd. (NEULANDLAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 14902.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 2152.23
- **Avg P&L per closed trade:** 538.06

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 13700.10 | 14287.95 | 14289.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 12:15:00 | 13345.75 | 14210.00 | 14248.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 12:15:00 | 14166.05 | 14130.72 | 14202.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-24 14:15:00 | 13825.75 | 14126.57 | 14199.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 14082.50 | 14006.23 | 14128.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 15:15:00 | 14170.00 | 13996.95 | 14115.48 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 10:15:00 | 13274.00 | 12131.71 | 12128.54 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 11646.00 | 12166.22 | 12167.58 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 14761.00 | 12174.37 | 12163.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 14848.00 | 12320.77 | 12237.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 13229.00 | 13276.14 | 12854.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-20 11:15:00 | 13849.00 | 13169.89 | 12934.69 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 13192.00 | 13301.43 | 13043.10 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-29 10:15:00 | 13334.00 | 13292.09 | 13048.45 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-26 11:15:00 | 13957.00 | 14662.52 | 14051.65 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 13:15:00 | 15009.00 | 15950.67 | 15954.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 14646.00 | 15772.86 | 15859.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 14:15:00 | 14123.00 | 13925.62 | 14582.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-10 11:15:00 | 13361.00 | 13915.37 | 14563.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-06 11:15:00 | 13296.00 | 12586.65 | 13167.08 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 13:15:00 | 15153.00 | 13558.68 | 13555.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 15388.00 | 13608.47 | 13580.43 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-24 14:15:00 | 13825.75 | 2025-01-28 10:15:00 | 12703.17 | TARGET | 1122.58 |
| BUY | 2025-08-29 10:15:00 | 13334.00 | 2025-09-03 11:15:00 | 14190.66 | TARGET | 856.66 |
| BUY | 2025-08-20 11:15:00 | 13849.00 | 2025-09-26 11:15:00 | 13957.00 | EXIT_EMA400 | 108.00 |
| SELL | 2026-02-10 11:15:00 | 13361.00 | 2026-04-06 11:15:00 | 13296.00 | EXIT_EMA400 | 65.00 |
