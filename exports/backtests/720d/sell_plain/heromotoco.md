# HEROMOTOCO (HEROMOTOCO)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 5356.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 14 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 9 / 5
- **Target hits / Stop hits / Partials:** 0 / 10 / 4
- **Avg / median % per leg:** 6.23% / 11.37%
- **Sum % (uncompounded):** 87.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 9 | 64.3% | 0 | 10 | 4 | 6.23% | 87.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 9 | 64.3% | 0 | 10 | 4 | 6.23% | 87.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 9 | 64.3% | 0 | 10 | 4 | 6.23% | 87.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 5292.20 | 5386.82 | 5387.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 5017.70 | 5373.94 | 5380.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 10:15:00 | 5323.00 | 5320.21 | 5349.49 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 11:15:00 | 5375.00 | 5320.75 | 5349.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 5375.00 | 5320.75 | 5349.62 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-08-28 09:15:00 | 5289.60 | 5331.31 | 5351.77 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:15:00 | 5290.00 | 5330.30 | 5351.06 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-08-28 14:15:00 | 5314.20 | 5329.67 | 5350.44 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-08-29 09:15:00 | 5328.40 | 5329.38 | 5350.08 | ENTRY2 sustain failed after 1140m |
| Stop hit — per-position SL triggered | 2024-08-30 09:15:00 | 5481.05 | 5331.89 | 5350.63 | SL hit (close>static) qty=1.00 sl=5381.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-17 09:15:00 | 5111.45 | 5633.24 | 5592.81 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:15:00 | 5206.05 | 5624.53 | 5588.83 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-10-22 09:15:00 | 5200.10 | 5553.25 | 5554.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 5200.10 | 5553.25 | 5554.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 5177.30 | 5546.00 | 5551.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 4225.90 | 4190.11 | 4410.33 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 4445.80 | 4204.26 | 4406.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 4445.80 | 4204.26 | 4406.84 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-03 11:15:00 | 4273.30 | 4213.40 | 4405.50 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 13:15:00 | 4236.65 | 4214.27 | 4404.03 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-04 10:15:00 | 4236.75 | 4216.42 | 4401.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 12:15:00 | 4237.55 | 4217.00 | 4399.81 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-06 11:15:00 | 4279.15 | 4222.91 | 4391.33 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 13:15:00 | 4227.75 | 4223.24 | 4389.82 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-10 10:15:00 | 4268.95 | 4226.62 | 4382.63 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 12:15:00 | 4190.50 | 4226.35 | 4380.94 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-04 09:15:00 | 3601.15 | 3977.27 | 4170.30 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-04 09:15:00 | 3601.92 | 3977.27 | 4170.30 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-04 09:15:00 | 3593.59 | 3977.27 | 4170.30 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-04 09:15:00 | 3561.92 | 3977.27 | 4170.30 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 3746.95 | 3725.31 | 3925.26 | SL hit (close>ema200) qty=0.50 sl=3725.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 3746.95 | 3725.31 | 3925.26 | SL hit (close>ema200) qty=0.50 sl=3725.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 3746.95 | 3725.31 | 3925.26 | SL hit (close>ema200) qty=0.50 sl=3725.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 3746.95 | 3725.31 | 3925.26 | SL hit (close>ema200) qty=0.50 sl=3725.31 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 3857.50 | 3710.95 | 3842.39 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-30 13:15:00 | 3814.20 | 3779.40 | 3851.47 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-30 14:15:00 | 3823.90 | 3779.85 | 3851.33 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-02 11:15:00 | 3814.10 | 3782.51 | 3851.27 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 13:15:00 | 3753.60 | 3781.70 | 3850.17 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-05-06 12:15:00 | 3887.00 | 3783.43 | 3846.70 | SL hit (close>static) qty=1.00 sl=3865.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-08 12:15:00 | 3810.80 | 3793.23 | 3847.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:15:00 | 3804.10 | 3793.20 | 3846.97 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-05-09 11:15:00 | 3874.90 | 3795.09 | 3846.86 | SL hit (close>static) qty=1.00 sl=3865.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 5379.00 | 5703.60 | 5705.16 | EMA200 below EMA400 |

### Cycle 4 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 5668.00 | 5698.75 | 5698.83 | EMA200 below EMA400 |

### Cycle 5 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 5682.00 | 5698.82 | 5698.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 5646.00 | 5700.52 | 5699.75 | Break + close below crossover candle low |

### Cycle 6 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 5614.00 | 5698.86 | 5698.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 5590.00 | 5696.43 | 5697.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 5653.00 | 5619.35 | 5654.25 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 5653.00 | 5619.35 | 5654.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 5653.00 | 5619.35 | 5654.25 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-04 09:15:00 | 5477.50 | 5637.56 | 5659.81 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:15:00 | 5440.50 | 5633.93 | 5657.77 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-03-04 14:15:00 | 5499.00 | 5629.74 | 5655.30 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-03-05 09:15:00 | 5525.50 | 5627.32 | 5653.83 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2026-03-09 09:15:00 | 5427.50 | 5613.66 | 5644.99 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 11:15:00 | 5443.50 | 5609.83 | 5642.75 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5685.00 | 5604.26 | 5638.93 | SL hit (close>static) qty=1.00 sl=5667.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5685.00 | 5604.26 | 5638.93 | SL hit (close>static) qty=1.00 sl=5667.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-12 09:15:00 | 5419.50 | 5605.11 | 5637.25 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 11:15:00 | 5454.50 | 5602.04 | 5635.39 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-08-28 11:15:00 | 5290.00 | 2024-08-30 09:15:00 | 5481.05 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2024-10-17 11:15:00 | 5206.05 | 2024-10-22 09:15:00 | 5200.10 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-02-03 13:15:00 | 4236.65 | 2025-03-04 09:15:00 | 3601.15 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-04 12:15:00 | 4237.55 | 2025-03-04 09:15:00 | 3601.92 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-06 13:15:00 | 4227.75 | 2025-03-04 09:15:00 | 3593.59 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-10 12:15:00 | 4190.50 | 2025-03-04 09:15:00 | 3561.92 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-03 13:15:00 | 4236.65 | 2025-03-27 10:15:00 | 3746.95 | STOP_HIT | 0.50 | 11.56% |
| SELL | retest2 | 2025-02-04 12:15:00 | 4237.55 | 2025-03-27 10:15:00 | 3746.95 | STOP_HIT | 0.50 | 11.58% |
| SELL | retest2 | 2025-02-06 13:15:00 | 4227.75 | 2025-03-27 10:15:00 | 3746.95 | STOP_HIT | 0.50 | 11.37% |
| SELL | retest2 | 2025-02-10 12:15:00 | 4190.50 | 2025-03-27 10:15:00 | 3746.95 | STOP_HIT | 0.50 | 10.58% |
| SELL | retest2 | 2025-05-02 13:15:00 | 3753.60 | 2025-05-06 12:15:00 | 3887.00 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-05-08 14:15:00 | 3804.10 | 2025-05-09 11:15:00 | 3874.90 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-03-04 11:15:00 | 5440.50 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2026-03-09 11:15:00 | 5443.50 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -4.44% |
