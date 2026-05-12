# HEROMOTOCO (HEROMOTOCO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 5325.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 5 |
| TARGET_HIT | 10 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 15
- **Target hits / Stop hits / Partials:** 10 / 15 / 5
- **Avg / median % per leg:** 3.12% / 5.00%
- **Sum % (uncompounded):** 93.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 5 | 1 | 0 | 8.15% | 48.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 5 | 83.3% | 5 | 1 | 0 | 8.15% | 48.9% |
| SELL (all) | 24 | 10 | 41.7% | 5 | 14 | 5 | 1.86% | 44.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 10 | 41.7% | 5 | 14 | 5 | 1.86% | 44.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 15 | 50.0% | 10 | 15 | 5 | 3.12% | 93.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 5216.70 | 5528.57 | 5529.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 15:15:00 | 5133.95 | 5509.50 | 5520.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 4225.90 | 4190.11 | 4409.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 10:00:00 | 4225.90 | 4190.11 | 4409.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 4445.80 | 4204.26 | 4405.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 11:15:00 | 4280.20 | 4212.79 | 4405.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:15:00 | 4266.00 | 4216.21 | 4401.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 11:45:00 | 4281.65 | 4222.91 | 4390.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:45:00 | 4277.10 | 4226.62 | 4381.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 4066.19 | 4214.93 | 4366.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 4052.70 | 4214.93 | 4366.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 4067.57 | 4214.93 | 4366.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 4063.25 | 4214.93 | 4366.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 13:15:00 | 3852.18 | 4174.65 | 4332.08 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 4314.90 | 3887.56 | 3887.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 4354.60 | 3892.20 | 3889.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 4256.90 | 4257.48 | 4149.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 15:15:00 | 4247.20 | 4257.48 | 4149.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 4203.00 | 4279.86 | 4205.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 4232.00 | 4279.26 | 4205.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 4232.60 | 4278.24 | 4205.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 12:15:00 | 4237.30 | 4278.24 | 4205.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 15:00:00 | 4234.10 | 4315.20 | 4248.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 4257.80 | 4312.39 | 4252.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 4265.40 | 4312.39 | 4252.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 4227.50 | 4311.55 | 4252.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:00:00 | 4227.50 | 4311.55 | 4252.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 4230.10 | 4310.74 | 4252.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:30:00 | 4260.70 | 4309.45 | 4252.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 4218.00 | 4307.95 | 4251.93 | SL hit (close<static) qty=1.00 sl=4220.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 5379.00 | 5703.60 | 5705.16 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 15:15:00 | 5754.50 | 5699.06 | 5698.99 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 5668.00 | 5698.75 | 5698.83 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 5744.00 | 5699.29 | 5699.10 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 5682.00 | 5698.82 | 5698.87 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 5778.00 | 5699.61 | 5699.27 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 5614.00 | 5698.86 | 5698.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 5590.00 | 5696.43 | 5697.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 5653.00 | 5619.35 | 5654.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 5653.00 | 5619.35 | 5654.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 5653.00 | 5619.35 | 5654.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 5653.00 | 5619.35 | 5654.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 5780.00 | 5620.95 | 5654.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:00:00 | 5780.00 | 5620.95 | 5654.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 5813.00 | 5622.86 | 5655.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:45:00 | 5815.50 | 5622.86 | 5655.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 5607.00 | 5642.68 | 5663.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 11:45:00 | 5572.50 | 5641.87 | 5662.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 5465.50 | 5639.17 | 5660.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 5560.00 | 5623.26 | 5651.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 15:00:00 | 5571.50 | 5622.74 | 5650.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 5685.00 | 5604.26 | 5638.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 5685.00 | 5604.26 | 5638.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 5655.50 | 5604.77 | 5639.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:30:00 | 5640.00 | 5605.03 | 5638.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 14:15:00 | 5704.00 | 5606.53 | 5639.39 | SL hit (close>static) qty=1.00 sl=5686.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-03 11:15:00 | 4280.20 | 2025-02-12 09:15:00 | 4066.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 10:15:00 | 4266.00 | 2025-02-12 09:15:00 | 4052.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 11:45:00 | 4281.65 | 2025-02-12 09:15:00 | 4067.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 10:45:00 | 4277.10 | 2025-02-12 09:15:00 | 4063.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-03 11:15:00 | 4280.20 | 2025-02-14 13:15:00 | 3852.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-04 10:15:00 | 4266.00 | 2025-02-14 13:15:00 | 3853.48 | TARGET_HIT | 0.50 | 9.67% |
| SELL | retest2 | 2025-02-06 11:45:00 | 4281.65 | 2025-02-17 09:15:00 | 3839.40 | TARGET_HIT | 0.50 | 10.33% |
| SELL | retest2 | 2025-02-10 10:45:00 | 4277.10 | 2025-02-17 09:15:00 | 3849.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-22 10:30:00 | 3827.50 | 2025-04-23 12:15:00 | 3884.50 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-04-22 11:15:00 | 3830.60 | 2025-04-23 12:15:00 | 3884.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-04-22 12:00:00 | 3830.00 | 2025-04-23 12:15:00 | 3884.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-04-30 10:15:00 | 3826.00 | 2025-05-02 09:15:00 | 3896.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-05-02 11:45:00 | 3831.20 | 2025-05-12 09:15:00 | 3920.80 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-05-06 11:15:00 | 3833.50 | 2025-05-12 09:15:00 | 3920.80 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-05-08 09:30:00 | 3836.40 | 2025-05-12 09:15:00 | 3920.80 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-05-08 10:30:00 | 3831.00 | 2025-05-12 09:15:00 | 3920.80 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-05-09 13:15:00 | 3841.10 | 2025-05-12 09:15:00 | 3920.80 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-07-14 10:15:00 | 4232.00 | 2025-07-31 09:15:00 | 4218.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-07-14 11:45:00 | 4232.60 | 2025-08-07 14:15:00 | 4655.20 | TARGET_HIT | 1.00 | 9.98% |
| BUY | retest2 | 2025-07-14 12:15:00 | 4237.30 | 2025-08-07 14:15:00 | 4655.86 | TARGET_HIT | 1.00 | 9.88% |
| BUY | retest2 | 2025-07-25 15:00:00 | 4234.10 | 2025-08-07 14:15:00 | 4661.03 | TARGET_HIT | 1.00 | 10.08% |
| BUY | retest2 | 2025-07-30 14:30:00 | 4260.70 | 2025-08-07 14:15:00 | 4657.51 | TARGET_HIT | 1.00 | 9.31% |
| BUY | retest2 | 2025-07-31 11:15:00 | 4256.00 | 2025-08-08 09:15:00 | 4681.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-02 11:45:00 | 5572.50 | 2026-03-10 14:15:00 | 5704.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-03-04 09:15:00 | 5465.50 | 2026-03-10 14:15:00 | 5704.00 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-03-05 13:45:00 | 5560.00 | 2026-03-10 14:15:00 | 5704.00 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-03-05 15:00:00 | 5571.50 | 2026-03-10 14:15:00 | 5704.00 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-03-10 12:30:00 | 5640.00 | 2026-03-10 15:15:00 | 5716.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-03-11 10:30:00 | 5627.00 | 2026-03-13 09:15:00 | 5345.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:30:00 | 5627.00 | 2026-03-23 10:15:00 | 5064.30 | TARGET_HIT | 0.50 | 10.00% |
