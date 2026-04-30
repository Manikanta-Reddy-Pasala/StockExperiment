# Blue Dart Express Ltd. (BLUEDART.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 5454.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -752.26
- **Avg P&L per closed trade:** -125.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 7861.50 | 8165.35 | 8165.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 7760.00 | 8067.41 | 8110.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 14:15:00 | 7698.00 | 7665.18 | 7832.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-19 14:15:00 | 7484.40 | 7692.60 | 7803.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-25 10:15:00 | 6359.00 | 6020.42 | 6307.95 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 15:15:00 | 6685.00 | 6343.59 | 6342.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 15:15:00 | 6731.50 | 6365.79 | 6353.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 6656.00 | 6698.95 | 6562.71 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 6142.00 | 6499.81 | 6501.46 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 6845.00 | 6488.76 | 6488.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 6901.50 | 6575.34 | 6538.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 6666.50 | 6696.77 | 6619.20 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 5820.00 | 6551.66 | 6554.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 10:15:00 | 5794.00 | 6544.12 | 6551.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 5871.00 | 5846.71 | 6031.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 09:15:00 | 5817.00 | 5846.49 | 6026.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 5800.50 | 5838.52 | 6009.77 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-23 10:15:00 | 5767.00 | 5837.81 | 6008.56 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 5952.50 | 5808.63 | 5966.26 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-01 09:15:00 | 5682.50 | 5808.24 | 5960.67 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-29 09:15:00 | 6136.00 | 5638.18 | 5781.97 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 6378.00 | 5904.62 | 5903.66 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 5734.00 | 5927.99 | 5928.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 5720.00 | 5908.78 | 5918.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 5574.00 | 5553.83 | 5677.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-31 09:15:00 | 5513.00 | 5553.43 | 5676.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-01 09:15:00 | 5780.50 | 5554.19 | 5672.43 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 11:15:00 | 5765.00 | 5608.26 | 5608.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 12:15:00 | 5793.50 | 5610.11 | 5609.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 5640.00 | 5654.26 | 5633.23 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 09:15:00 | 5439.00 | 5617.81 | 5618.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 5420.00 | 5605.11 | 5612.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 12:15:00 | 5164.50 | 5143.11 | 5305.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 14:15:00 | 5077.80 | 5142.32 | 5298.16 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-17 09:15:00 | 5379.30 | 5150.76 | 5290.47 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-19 14:15:00 | 7484.40 | 2024-12-31 09:15:00 | 6526.16 | TARGET | 958.24 |
| SELL | 2025-09-19 09:15:00 | 5817.00 | 2025-10-29 09:15:00 | 6136.00 | EXIT_EMA400 | -319.00 |
| SELL | 2025-09-23 10:15:00 | 5767.00 | 2025-10-29 09:15:00 | 6136.00 | EXIT_EMA400 | -369.00 |
| SELL | 2025-10-01 09:15:00 | 5682.50 | 2025-10-29 09:15:00 | 6136.00 | EXIT_EMA400 | -453.50 |
| SELL | 2025-12-31 09:15:00 | 5513.00 | 2026-01-01 09:15:00 | 5780.50 | EXIT_EMA400 | -267.50 |
| SELL | 2026-04-13 14:15:00 | 5077.80 | 2026-04-17 09:15:00 | 5379.30 | EXIT_EMA400 | -301.50 |
