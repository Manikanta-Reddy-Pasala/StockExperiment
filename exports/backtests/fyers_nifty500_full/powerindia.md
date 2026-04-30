# Hitachi Energy India Ltd. (POWERINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 33550.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -127.26
- **Avg P&L per closed trade:** -18.18

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 14:15:00 | 11494.20 | 13171.17 | 13179.20 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 14392.65 | 12960.52 | 12958.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 14527.85 | 13058.99 | 13008.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 15:15:00 | 13805.00 | 13845.11 | 13486.43 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 11461.50 | 13268.28 | 13268.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 11339.95 | 13213.76 | 13240.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 13518.60 | 12607.69 | 12907.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-31 12:15:00 | 12375.85 | 12607.76 | 12902.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 12800.00 | 12608.28 | 12899.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-01 12:15:00 | 12101.30 | 12610.43 | 12893.86 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 12230.95 | 12426.40 | 12740.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-10 12:15:00 | 12105.55 | 12418.65 | 12731.81 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-04 10:15:00 | 12205.70 | 11743.85 | 12180.82 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 13592.00 | 12311.65 | 12307.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 10:15:00 | 13682.00 | 12325.29 | 12314.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 17000.00 | 17071.46 | 15669.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 10:15:00 | 17446.00 | 17072.46 | 15711.10 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 19260.00 | 19980.46 | 19214.80 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-26 12:15:00 | 19150.00 | 19972.20 | 19214.48 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 17785.00 | 19138.66 | 19140.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 17557.00 | 19101.17 | 19121.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 10:15:00 | 18021.00 | 17971.80 | 18419.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 10:15:00 | 17872.00 | 17971.82 | 18403.63 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-04 09:15:00 | 20225.00 | 17966.26 | 18373.23 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 21865.00 | 18733.88 | 18725.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 22172.00 | 19779.36 | 19310.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 20560.00 | 20898.99 | 20110.66 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 18630.00 | 19671.87 | 19675.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 18415.00 | 19659.37 | 19669.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 19340.00 | 19173.71 | 19383.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 14:15:00 | 18460.00 | 19207.37 | 19388.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 18460.00 | 19207.37 | 19388.92 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-09 09:15:00 | 17810.00 | 19185.39 | 19376.08 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 18010.00 | 17891.90 | 18531.61 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 18562.00 | 17919.15 | 18523.43 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 22483.00 | 18916.79 | 18911.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 22565.00 | 19056.13 | 18982.16 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-31 12:15:00 | 12375.85 | 2025-02-17 11:15:00 | 10795.56 | TARGET | 1580.29 |
| SELL | 2025-02-01 12:15:00 | 12101.30 | 2025-03-04 10:15:00 | 12205.70 | EXIT_EMA400 | -104.40 |
| SELL | 2025-02-10 12:15:00 | 12105.55 | 2025-03-04 10:15:00 | 12205.70 | EXIT_EMA400 | -100.15 |
| BUY | 2025-06-13 10:15:00 | 17446.00 | 2025-08-26 12:15:00 | 19150.00 | EXIT_EMA400 | 1704.00 |
| SELL | 2025-10-31 10:15:00 | 17872.00 | 2025-11-04 09:15:00 | 20225.00 | EXIT_EMA400 | -2353.00 |
| SELL | 2026-01-08 14:15:00 | 18460.00 | 2026-01-30 09:15:00 | 18562.00 | EXIT_EMA400 | -102.00 |
| SELL | 2026-01-09 09:15:00 | 17810.00 | 2026-01-30 09:15:00 | 18562.00 | EXIT_EMA400 | -752.00 |
