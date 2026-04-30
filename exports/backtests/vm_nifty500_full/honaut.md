# Honeywell Automation India Ltd. (HONAUT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 31030.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 4 |
| EXIT | 5 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 8496.49
- **Avg P&L per closed trade:** 944.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 38439.00 | 37204.95 | 37203.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 12:15:00 | 38701.00 | 37244.76 | 37223.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 15:15:00 | 37600.00 | 37866.18 | 37594.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-14 14:15:00 | 37976.20 | 37852.67 | 37604.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 37687.10 | 37852.48 | 37607.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-15 14:15:00 | 38165.85 | 37858.95 | 37616.43 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-02-20 11:15:00 | 37369.00 | 37898.94 | 37658.77 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-03-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 14:15:00 | 37430.00 | 37620.71 | 37620.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 09:15:00 | 37250.15 | 37614.84 | 37617.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 12:15:00 | 37868.00 | 37615.52 | 37618.10 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 12:15:00 | 38044.00 | 37623.26 | 37621.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 38338.05 | 37645.16 | 37633.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 14:15:00 | 55179.95 | 55575.96 | 52639.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-23 10:15:00 | 55746.45 | 55350.58 | 52760.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-05 09:15:00 | 53035.30 | 54845.27 | 53161.04 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 12:15:00 | 50980.00 | 52399.07 | 52400.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 13:15:00 | 50800.00 | 52383.16 | 52392.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 09:15:00 | 50139.00 | 49749.50 | 50613.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 09:15:00 | 49255.50 | 49807.70 | 50508.06 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-18 11:15:00 | 50481.45 | 49800.31 | 50473.21 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 37195.00 | 35757.32 | 35756.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 37200.00 | 35771.67 | 35764.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 37685.00 | 37751.27 | 37075.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 38430.00 | 37747.40 | 37146.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 13:15:00 | 38655.00 | 39734.59 | 38859.63 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 35585.00 | 38343.73 | 38350.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 35015.00 | 36224.96 | 36610.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 35970.00 | 35805.01 | 36324.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-05 09:15:00 | 34755.00 | 35574.01 | 36064.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 34560.00 | 34018.15 | 34818.53 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-09 09:15:00 | 33595.00 | 34031.08 | 34721.81 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 33820.00 | 33210.99 | 33994.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-02 09:15:00 | 32605.00 | 33211.40 | 33974.98 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 33685.00 | 33209.97 | 33947.81 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-04 12:15:00 | 33200.00 | 33248.04 | 33931.47 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-15 11:15:00 | 30320.00 | 28863.99 | 30180.90 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-14 14:15:00 | 37976.20 | 2024-02-20 11:15:00 | 37369.00 | EXIT_EMA400 | -607.20 |
| BUY | 2024-02-15 14:15:00 | 38165.85 | 2024-02-20 11:15:00 | 37369.00 | EXIT_EMA400 | -796.85 |
| BUY | 2024-07-23 10:15:00 | 55746.45 | 2024-08-05 09:15:00 | 53035.30 | EXIT_EMA400 | -2711.15 |
| SELL | 2024-10-17 09:15:00 | 49255.50 | 2024-10-18 11:15:00 | 50481.45 | EXIT_EMA400 | -1225.95 |
| BUY | 2025-06-24 09:15:00 | 38430.00 | 2025-07-28 13:15:00 | 38655.00 | EXIT_EMA400 | 225.00 |
| SELL | 2025-12-05 09:15:00 | 34755.00 | 2026-01-27 13:15:00 | 30827.17 | TARGET | 3927.83 |
| SELL | 2026-02-04 12:15:00 | 33200.00 | 2026-02-20 09:15:00 | 31005.58 | TARGET | 2194.42 |
| SELL | 2026-01-09 09:15:00 | 33595.00 | 2026-03-04 09:15:00 | 30214.56 | TARGET | 3380.44 |
| SELL | 2026-02-02 09:15:00 | 32605.00 | 2026-03-19 14:15:00 | 28495.06 | TARGET | 4109.94 |
