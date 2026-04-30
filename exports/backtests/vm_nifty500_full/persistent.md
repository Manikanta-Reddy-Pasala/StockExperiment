# Persistent Systems Ltd. (PERSISTENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 4800.00
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
| ENTRY1 | 8 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 1
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** 1376.55
- **Avg P&L per closed trade:** 152.95

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 14:15:00 | 2555.43 | 2468.30 | 2468.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 2581.90 | 2470.20 | 2469.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 12:15:00 | 2818.52 | 2825.13 | 2724.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-10 10:15:00 | 2857.82 | 2824.72 | 2726.47 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 4089.85 | 4168.10 | 4003.46 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-14 11:15:00 | 4093.50 | 4166.54 | 4004.31 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-03-19 09:15:00 | 4006.45 | 4158.36 | 4014.80 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 09:15:00 | 3500.00 | 3973.99 | 3974.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 14:15:00 | 3463.75 | 3920.76 | 3946.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 3581.10 | 3569.29 | 3699.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-24 14:15:00 | 3555.40 | 3568.95 | 3693.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 3669.90 | 3570.57 | 3692.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-05-27 11:15:00 | 3715.25 | 3572.01 | 3692.15 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 3971.10 | 3716.42 | 3715.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 4026.00 | 3754.02 | 3734.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 4468.80 | 4561.73 | 4307.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-05 14:15:00 | 4557.25 | 4558.03 | 4312.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 5140.95 | 5335.88 | 5139.64 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-23 09:15:00 | 5674.00 | 5337.49 | 5142.39 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-01-13 13:15:00 | 6012.65 | 6259.67 | 6037.23 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 14:15:00 | 5529.00 | 6003.20 | 6003.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 13:15:00 | 5485.90 | 5910.57 | 5953.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 12:15:00 | 5475.00 | 5450.57 | 5639.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-01 09:15:00 | 5372.85 | 5479.80 | 5627.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-24 09:15:00 | 5308.00 | 5058.86 | 5306.19 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 5764.50 | 5411.53 | 5411.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 5935.50 | 5551.34 | 5499.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 5873.00 | 5890.17 | 5744.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 12:15:00 | 5946.00 | 5885.94 | 5753.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 5787.00 | 5883.93 | 5753.32 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-09 09:15:00 | 5736.00 | 5881.47 | 5753.38 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 5174.00 | 5681.02 | 5681.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 5159.00 | 5670.71 | 5676.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 5346.00 | 5341.68 | 5459.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-05 09:15:00 | 5151.00 | 5354.42 | 5431.19 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 13:15:00 | 5402.00 | 5310.19 | 5398.33 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 5850.70 | 5368.69 | 5366.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 5918.10 | 5391.96 | 5378.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 12:15:00 | 6140.50 | 6182.75 | 5952.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-12 09:15:00 | 6266.50 | 6176.49 | 5961.38 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-21 09:15:00 | 6119.50 | 6317.64 | 6187.71 | Close below EMA400 |

### Cycle 8 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 5738.50 | 6125.62 | 6126.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 5703.00 | 6110.14 | 6118.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 5039.00 | 4962.70 | 5276.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-24 10:15:00 | 4907.80 | 5179.16 | 5288.72 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-10 10:15:00 | 2857.82 | 2023-11-21 09:15:00 | 3251.88 | TARGET | 394.05 |
| BUY | 2024-03-14 11:15:00 | 4093.50 | 2024-03-19 09:15:00 | 4006.45 | EXIT_EMA400 | -87.05 |
| SELL | 2024-05-24 14:15:00 | 3555.40 | 2024-05-27 11:15:00 | 3715.25 | EXIT_EMA400 | -159.85 |
| BUY | 2024-08-05 14:15:00 | 4557.25 | 2024-09-03 11:15:00 | 5292.08 | TARGET | 734.83 |
| BUY | 2024-10-23 09:15:00 | 5674.00 | 2025-01-13 13:15:00 | 6012.65 | EXIT_EMA400 | 338.65 |
| SELL | 2025-04-01 09:15:00 | 5372.85 | 2025-04-04 09:15:00 | 4608.93 | TARGET | 763.92 |
| BUY | 2025-07-08 12:15:00 | 5946.00 | 2025-07-09 09:15:00 | 5736.00 | EXIT_EMA400 | -210.00 |
| SELL | 2025-09-05 09:15:00 | 5151.00 | 2025-09-10 13:15:00 | 5402.00 | EXIT_EMA400 | -251.00 |
| BUY | 2025-12-12 09:15:00 | 6266.50 | 2026-01-21 09:15:00 | 6119.50 | EXIT_EMA400 | -147.00 |
