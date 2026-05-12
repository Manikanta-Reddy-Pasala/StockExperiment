# Persistent Systems Ltd. (PERSISTENT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 5115.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 0 |
| ALERT3 | 58 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 42 |
| PARTIAL | 13 |
| TARGET_HIT | 7 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 33
- **Target hits / Stop hits / Partials:** 7 / 39 / 13
- **Avg / median % per leg:** 0.20% / -2.24%
- **Sum % (uncompounded):** 12.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 8 | 25.0% | 4 | 24 | 4 | -0.80% | -25.7% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 24 | 0 | 0.0% | 0 | 24 | 0 | -3.57% | -85.8% |
| SELL (all) | 27 | 18 | 66.7% | 3 | 15 | 9 | 1.40% | 37.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 18 | 66.7% | 3 | 15 | 9 | 1.40% | 37.8% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 51 | 18 | 35.3% | 3 | 39 | 9 | -0.94% | -48.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 14:15:00 | 2464.68 | 4714.05 | 4719.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 09:15:00 | 2448.93 | 4143.09 | 4412.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 09:15:00 | 2712.20 | 2399.12 | 3007.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 10:00:00 | 2712.20 | 2399.12 | 3007.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 2983.05 | 2530.26 | 2992.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 11:45:00 | 2990.50 | 2530.26 | 2992.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 15:15:00 | 2979.83 | 2576.10 | 2991.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 09:15:00 | 3001.90 | 2576.10 | 2991.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 2998.88 | 2580.31 | 2991.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 09:30:00 | 3012.48 | 2580.31 | 2991.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 10:15:00 | 3005.00 | 2584.53 | 2991.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 10:30:00 | 3002.80 | 2584.53 | 2991.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 12:15:00 | 2995.00 | 2592.66 | 2991.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-11 13:15:00 | 2989.58 | 2592.66 | 2991.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 13:15:00 | 2840.10 | 2780.31 | 2967.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-28 13:15:00 | 2863.95 | 2780.31 | 2967.37 | SL hit (close>static) qty=0.50 sl=2780.31 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 12:15:00 | 3124.70 | 2974.71 | 2974.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 14:15:00 | 3146.88 | 2977.85 | 2975.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 10:15:00 | 4155.35 | 4173.96 | 3931.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-26 11:00:00 | 4155.35 | 4173.96 | 3931.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 4088.68 | 4171.17 | 4017.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 11:30:00 | 4105.10 | 4169.54 | 4018.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 13:00:00 | 4100.98 | 4168.86 | 4018.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 13:45:00 | 4095.50 | 4168.12 | 4018.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 13:45:00 | 4095.75 | 4164.01 | 4026.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 4006.45 | 4161.02 | 4027.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 4006.45 | 4161.02 | 4027.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 4027.03 | 4159.68 | 4027.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 3959.05 | 4149.57 | 4026.16 | SL hit (close<static) qty=1.00 sl=3963.08 alert=retest2 |

### Cycle 3 — SELL (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 14:15:00 | 3504.50 | 3984.19 | 3985.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 10:15:00 | 3297.40 | 3644.88 | 3780.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 3581.40 | 3567.89 | 3699.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 12:00:00 | 3581.40 | 3567.89 | 3699.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 3669.90 | 3569.29 | 3692.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:00:00 | 3669.90 | 3569.29 | 3692.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 3715.10 | 3570.74 | 3692.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:00:00 | 3715.10 | 3570.74 | 3692.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 3727.50 | 3572.30 | 3692.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:15:00 | 3752.95 | 3572.30 | 3692.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 3683.15 | 3583.81 | 3693.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 3649.25 | 3586.01 | 3693.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:15:00 | 3466.79 | 3585.87 | 3685.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 3284.33 | 3557.91 | 3663.25 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 3971.10 | 3716.14 | 3715.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 4026.00 | 3753.64 | 3734.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 4468.80 | 4561.35 | 4307.75 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 15:00:00 | 4557.25 | 4557.63 | 4312.13 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 09:15:00 | 4610.70 | 4557.48 | 4313.27 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 14:15:00 | 4546.20 | 4558.19 | 4319.66 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 09:15:00 | 4598.40 | 4557.01 | 4321.44 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:15:00 | 4773.51 | 4584.44 | 4367.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 12:15:00 | 4785.11 | 4598.81 | 4385.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 09:15:00 | 4841.23 | 4606.13 | 4393.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 09:15:00 | 4828.32 | 4606.13 | 4393.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-08-26 09:15:00 | 5012.98 | 4711.64 | 4491.96 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 5 — SELL (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 13:15:00 | 5501.65 | 6002.47 | 6004.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 5439.90 | 5885.79 | 5940.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 12:15:00 | 5475.00 | 5448.30 | 5638.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-24 13:00:00 | 5475.00 | 5448.30 | 5638.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 5555.00 | 5453.34 | 5632.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:30:00 | 5608.30 | 5453.34 | 5632.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 5653.00 | 5465.42 | 5629.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:00:00 | 5653.00 | 5465.42 | 5629.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 5633.65 | 5467.10 | 5629.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:30:00 | 5662.00 | 5467.10 | 5629.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 5645.85 | 5472.35 | 5629.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 5624.05 | 5472.35 | 5629.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 5622.50 | 5475.03 | 5629.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:45:00 | 5644.45 | 5475.03 | 5629.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 5650.40 | 5476.77 | 5629.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:45:00 | 5648.25 | 5476.77 | 5629.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 5556.95 | 5477.57 | 5628.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:45:00 | 5529.00 | 5478.10 | 5628.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:00:00 | 5518.25 | 5478.50 | 5627.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 12:15:00 | 5252.55 | 5471.27 | 5620.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 12:15:00 | 5242.34 | 5471.27 | 5620.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-03 09:15:00 | 4976.10 | 5446.71 | 5599.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 5764.50 | 5411.24 | 5410.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 5936.00 | 5551.39 | 5499.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 5873.00 | 5890.37 | 5744.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 5848.50 | 5890.37 | 5744.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 5787.00 | 5884.09 | 5753.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 5844.00 | 5754.73 | 5714.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 5808.00 | 5755.59 | 5715.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 5714.00 | 5755.66 | 5716.47 | SL hit (close<static) qty=1.00 sl=5735.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 5174.00 | 5681.19 | 5681.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 5159.00 | 5670.87 | 5676.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 5346.00 | 5342.19 | 5460.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 5346.00 | 5342.19 | 5460.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 5464.00 | 5345.64 | 5451.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 5486.00 | 5345.64 | 5451.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 5494.00 | 5347.11 | 5451.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 5494.00 | 5347.11 | 5451.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 5498.00 | 5348.62 | 5451.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:45:00 | 5477.50 | 5349.97 | 5451.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 13:15:00 | 5475.00 | 5349.97 | 5451.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 13:45:00 | 5478.00 | 5351.27 | 5452.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:15:00 | 5203.62 | 5354.26 | 5431.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:15:00 | 5201.25 | 5354.26 | 5431.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:15:00 | 5204.10 | 5354.26 | 5431.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 5369.50 | 5306.81 | 5398.46 | SL hit (close>ema200) qty=0.50 sl=5306.81 alert=retest2 |

### Cycle 8 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 5851.40 | 5368.54 | 5366.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 5918.10 | 5391.71 | 5378.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 12:15:00 | 6140.50 | 6182.49 | 5952.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 13:00:00 | 6140.50 | 6182.49 | 5952.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 6120.00 | 6317.90 | 6187.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 6105.50 | 6317.90 | 6187.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 6069.50 | 6315.42 | 6187.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 6042.00 | 6315.42 | 6187.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 6149.00 | 6310.77 | 6186.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:00:00 | 6149.00 | 6310.77 | 6186.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 6239.00 | 6310.06 | 6187.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 6364.50 | 6309.16 | 6187.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 13:45:00 | 6260.00 | 6307.02 | 6189.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 13:15:00 | 6135.00 | 6301.50 | 6190.46 | SL hit (close<static) qty=1.00 sl=6142.50 alert=retest2 |

### Cycle 9 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 5723.00 | 6125.83 | 6127.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 5703.00 | 6102.95 | 6115.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 5039.00 | 4963.44 | 5276.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 10:00:00 | 5039.00 | 4963.44 | 5276.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 5233.80 | 4979.72 | 5266.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:30:00 | 5224.20 | 4979.72 | 5266.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 5273.20 | 4985.06 | 5266.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 5117.30 | 5208.24 | 5309.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 4861.44 | 5182.09 | 5287.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 5066.80 | 5033.71 | 5173.12 | SL hit (close>ema200) qty=0.50 sl=5033.71 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-09-11 13:15:00 | 2989.58 | 2023-09-28 13:15:00 | 2840.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-11 13:15:00 | 2989.58 | 2023-09-28 13:15:00 | 2863.95 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2023-10-25 10:15:00 | 2949.53 | 2023-10-27 09:15:00 | 3039.48 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-03-14 11:30:00 | 4105.10 | 2024-03-20 09:15:00 | 3959.05 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-03-14 13:00:00 | 4100.98 | 2024-03-20 09:15:00 | 3959.05 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-03-14 13:45:00 | 4095.50 | 2024-03-20 09:15:00 | 3959.05 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2024-03-18 13:45:00 | 4095.75 | 2024-03-20 09:15:00 | 3959.05 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2024-03-21 09:15:00 | 4110.63 | 2024-04-01 10:15:00 | 3939.55 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2024-03-26 09:45:00 | 4035.13 | 2024-04-01 10:15:00 | 3939.55 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-03-26 11:00:00 | 4030.00 | 2024-04-01 10:15:00 | 3939.55 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-03-27 10:00:00 | 4045.00 | 2024-04-01 10:15:00 | 3939.55 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-04-04 11:15:00 | 4068.75 | 2024-04-08 13:15:00 | 3924.25 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-04-04 13:30:00 | 4045.05 | 2024-04-08 13:15:00 | 3924.25 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-05-29 09:15:00 | 3649.25 | 2024-05-31 09:15:00 | 3466.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 09:15:00 | 3649.25 | 2024-06-04 12:15:00 | 3284.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-06 12:30:00 | 3676.70 | 2024-06-07 09:15:00 | 3863.65 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2024-06-06 14:15:00 | 3674.25 | 2024-06-07 09:15:00 | 3863.65 | STOP_HIT | 1.00 | -5.15% |
| BUY | retest1 | 2024-08-05 15:00:00 | 4557.25 | 2024-08-13 09:15:00 | 4773.51 | PARTIAL | 0.50 | 4.75% |
| BUY | retest1 | 2024-08-06 09:15:00 | 4610.70 | 2024-08-14 12:15:00 | 4785.11 | PARTIAL | 0.50 | 3.78% |
| BUY | retest1 | 2024-08-06 14:15:00 | 4546.20 | 2024-08-16 09:15:00 | 4841.23 | PARTIAL | 0.50 | 6.49% |
| BUY | retest1 | 2024-08-07 09:15:00 | 4598.40 | 2024-08-16 09:15:00 | 4828.32 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-05 15:00:00 | 4557.25 | 2024-08-26 09:15:00 | 5012.98 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-08-06 09:15:00 | 4610.70 | 2024-08-26 09:15:00 | 5000.82 | TARGET_HIT | 0.50 | 8.46% |
| BUY | retest1 | 2024-08-06 14:15:00 | 4546.20 | 2024-08-27 09:15:00 | 5058.24 | TARGET_HIT | 0.50 | 11.26% |
| BUY | retest1 | 2024-08-07 09:15:00 | 4598.40 | 2024-08-29 14:15:00 | 5071.77 | TARGET_HIT | 0.50 | 10.29% |
| BUY | retest2 | 2025-01-20 14:45:00 | 6131.70 | 2025-01-21 11:15:00 | 5948.00 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-01-21 09:15:00 | 6132.60 | 2025-01-21 11:15:00 | 5948.00 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-01-23 09:30:00 | 6153.35 | 2025-01-28 09:15:00 | 5944.00 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-01-27 13:30:00 | 6098.65 | 2025-01-28 09:15:00 | 5944.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-01-29 15:00:00 | 6282.80 | 2025-02-01 12:15:00 | 5882.10 | STOP_HIT | 1.00 | -6.38% |
| BUY | retest2 | 2025-01-30 09:30:00 | 6287.40 | 2025-02-01 12:15:00 | 5882.10 | STOP_HIT | 1.00 | -6.45% |
| BUY | retest2 | 2025-02-06 09:45:00 | 6263.20 | 2025-02-11 09:15:00 | 5954.00 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest2 | 2025-02-06 11:30:00 | 6267.95 | 2025-02-11 09:15:00 | 5954.00 | STOP_HIT | 1.00 | -5.01% |
| SELL | retest2 | 2025-03-28 13:45:00 | 5529.00 | 2025-04-01 12:15:00 | 5252.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 15:00:00 | 5518.25 | 2025-04-01 12:15:00 | 5242.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 13:45:00 | 5529.00 | 2025-04-03 09:15:00 | 4976.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-28 15:00:00 | 5518.25 | 2025-04-03 09:15:00 | 4966.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-05 10:15:00 | 5522.00 | 2025-05-12 09:15:00 | 5707.50 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-05-06 09:45:00 | 5511.50 | 2025-05-12 09:15:00 | 5707.50 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2025-07-22 09:15:00 | 5844.00 | 2025-07-22 14:15:00 | 5714.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-07-22 10:30:00 | 5808.00 | 2025-07-22 14:15:00 | 5714.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-08-25 12:45:00 | 5477.50 | 2025-09-05 09:15:00 | 5203.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 13:15:00 | 5475.00 | 2025-09-05 09:15:00 | 5201.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 13:45:00 | 5478.00 | 2025-09-05 09:15:00 | 5204.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 12:45:00 | 5477.50 | 2025-09-10 09:15:00 | 5369.50 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2025-08-25 13:15:00 | 5475.00 | 2025-09-10 09:15:00 | 5369.50 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2025-08-25 13:45:00 | 5478.00 | 2025-09-10 09:15:00 | 5369.50 | STOP_HIT | 0.50 | 1.98% |
| SELL | retest2 | 2025-09-22 09:15:00 | 5302.50 | 2025-09-26 11:15:00 | 5037.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 5302.50 | 2025-10-06 10:15:00 | 5220.00 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2025-10-13 09:15:00 | 5289.90 | 2025-10-15 09:15:00 | 5725.00 | STOP_HIT | 1.00 | -8.23% |
| SELL | retest2 | 2025-10-14 10:30:00 | 5333.60 | 2025-10-15 09:15:00 | 5725.00 | STOP_HIT | 1.00 | -7.34% |
| SELL | retest2 | 2025-10-14 13:15:00 | 5349.40 | 2025-10-15 09:15:00 | 5725.00 | STOP_HIT | 1.00 | -7.02% |
| SELL | retest2 | 2025-10-14 15:00:00 | 5348.70 | 2025-10-15 09:15:00 | 5725.00 | STOP_HIT | 1.00 | -7.04% |
| BUY | retest2 | 2026-01-22 09:15:00 | 6364.50 | 2026-01-23 13:15:00 | 6135.00 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2026-01-22 13:45:00 | 6260.00 | 2026-01-23 13:15:00 | 6135.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-01-28 09:30:00 | 6278.00 | 2026-01-29 09:15:00 | 6027.00 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2026-02-03 09:15:00 | 6299.00 | 2026-02-04 09:15:00 | 5921.50 | STOP_HIT | 1.00 | -5.99% |
| SELL | retest2 | 2026-04-22 09:15:00 | 5117.30 | 2026-04-24 11:15:00 | 4861.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 09:15:00 | 5117.30 | 2026-05-08 09:15:00 | 5066.80 | STOP_HIT | 0.50 | 0.99% |
