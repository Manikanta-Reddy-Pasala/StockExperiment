# Bayer Cropscience Ltd. (BAYERCROP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 4764.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 1
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -597.44
- **Avg P&L per closed trade:** -85.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 10:15:00 | 5109.70 | 5659.19 | 5660.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-14 13:15:00 | 5080.15 | 5643.00 | 5652.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 5396.95 | 5391.04 | 5501.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-10 11:15:00 | 5239.90 | 5451.90 | 5484.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 5444.80 | 5417.73 | 5462.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-15 14:15:00 | 5415.55 | 5418.11 | 5462.22 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-05-16 09:15:00 | 5482.50 | 5418.61 | 5462.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 6079.60 | 5454.57 | 5452.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 6117.35 | 5479.34 | 5465.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 09:15:00 | 6605.30 | 6649.95 | 6366.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-03 09:15:00 | 6699.95 | 6391.90 | 6366.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 6492.25 | 6399.11 | 6371.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-04 11:15:00 | 6625.00 | 6402.45 | 6373.28 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-22 14:15:00 | 6390.00 | 6537.18 | 6466.83 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 10:15:00 | 5649.95 | 6449.27 | 6449.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 12:15:00 | 5638.40 | 6433.32 | 6441.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 6119.00 | 6109.26 | 6249.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-06 14:15:00 | 6031.40 | 6106.70 | 6231.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 6209.70 | 6105.34 | 6221.63 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-11 09:15:00 | 6245.00 | 6106.73 | 6221.75 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 5717.50 | 4939.51 | 4938.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 13:15:00 | 5745.40 | 4947.53 | 4942.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 14:15:00 | 5994.50 | 6198.52 | 5943.88 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 5273.00 | 5793.98 | 5796.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 5266.00 | 5788.73 | 5793.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 15:15:00 | 5113.20 | 5108.75 | 5281.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-17 09:15:00 | 5088.90 | 5108.55 | 5280.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-09 10:15:00 | 4561.80 | 4445.29 | 4529.13 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 4752.70 | 4589.32 | 4588.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 12:15:00 | 4800.10 | 4593.06 | 4590.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4623.50 | 4634.44 | 4614.38 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 13:15:00 | 4514.50 | 4598.64 | 4598.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 4506.30 | 4597.73 | 4598.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 4584.80 | 4560.36 | 4577.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-23 10:15:00 | 4490.00 | 4563.60 | 4578.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4535.00 | 4559.81 | 4576.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-24 13:15:00 | 4594.00 | 4559.26 | 4575.80 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 4777.30 | 4586.18 | 4585.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 14:15:00 | 4818.00 | 4592.34 | 4588.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 4698.80 | 4724.21 | 4668.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-30 11:15:00 | 4759.50 | 4714.24 | 4670.26 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-10 11:15:00 | 5239.90 | 2024-05-16 09:15:00 | 5482.50 | EXIT_EMA400 | -242.60 |
| SELL | 2024-05-15 14:15:00 | 5415.55 | 2024-05-16 09:15:00 | 5482.50 | EXIT_EMA400 | -66.95 |
| BUY | 2024-10-03 09:15:00 | 6699.95 | 2024-10-22 14:15:00 | 6390.00 | EXIT_EMA400 | -309.95 |
| BUY | 2024-10-04 11:15:00 | 6625.00 | 2024-10-22 14:15:00 | 6390.00 | EXIT_EMA400 | -235.00 |
| SELL | 2024-12-06 14:15:00 | 6031.40 | 2024-12-11 09:15:00 | 6245.00 | EXIT_EMA400 | -213.60 |
| SELL | 2025-10-17 09:15:00 | 5088.90 | 2025-11-11 13:15:00 | 4514.24 | TARGET | 574.66 |
| SELL | 2026-03-23 10:15:00 | 4490.00 | 2026-03-24 13:15:00 | 4594.00 | EXIT_EMA400 | -104.00 |
