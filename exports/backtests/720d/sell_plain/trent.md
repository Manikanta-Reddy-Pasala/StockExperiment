# TRENT (TRENT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 4298.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 3 |
| PENDING | 18 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 2 |
| ENTRY2 | 10 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / Stop hits / Partials:** 0 / 12 / 2
- **Avg / median % per leg:** 1.22% / -0.76%
- **Sum % (uncompounded):** 17.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 5 | 35.7% | 0 | 12 | 2 | 1.22% | 17.1% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 6.79% | 20.4% |
| SELL @ 3rd Alert (retest2) | 11 | 3 | 27.3% | 0 | 10 | 1 | -0.30% | -3.3% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 6.79% | 20.4% |
| retest2 (combined) | 11 | 3 | 27.3% | 0 | 10 | 1 | -0.30% | -3.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 6479.60 | 6989.22 | 6989.49 | EMA200 below EMA400 |

### Cycle 2 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 6547.95 | 6952.20 | 6953.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 6505.90 | 6947.76 | 6951.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 6199.95 | 6173.19 | 6472.12 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-02-04 09:15:00 | 5806.50 | 6166.79 | 6454.30 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 11:15:00 | 5729.05 | 6158.51 | 6447.28 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-27 09:15:00 | 4869.69 | 5490.81 | 5914.73 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 5237.00 | 5211.80 | 5598.29 | SL hit (close>ema200) qty=0.50 sl=5211.80 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 5488.50 | 5226.15 | 5509.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-28 11:15:00 | 5426.80 | 5228.14 | 5509.13 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:15:00 | 5336.15 | 5230.24 | 5507.39 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-01 09:15:00 | 5564.00 | 5235.34 | 5505.83 | SL hit (close>static) qty=1.00 sl=5525.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 4543.60 | 5319.21 | 5515.70 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 4626.25 | 5305.70 | 5506.96 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-04-30 09:15:00 | 5193.00 | 5190.90 | 5349.70 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:15:00 | 5220.50 | 5191.55 | 5348.44 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 5540.00 | 5243.92 | 5330.04 | SL hit (close>static) qty=1.00 sl=5525.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 5540.00 | 5243.92 | 5330.04 | SL hit (close>static) qty=1.00 sl=5525.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-20 12:15:00 | 5463.00 | 5300.47 | 5351.60 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 14:15:00 | 5443.50 | 5303.43 | 5352.58 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 5441.50 | 5304.81 | 5353.03 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-21 11:15:00 | 5420.00 | 5309.22 | 5354.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 5417.50 | 5311.48 | 5355.22 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-22 09:15:00 | 5354.50 | 5314.56 | 5356.12 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 5386.50 | 5315.62 | 5356.24 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-23 11:15:00 | 5431.00 | 5320.36 | 5357.24 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-23 12:15:00 | 5442.00 | 5321.57 | 5357.66 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-23 15:15:00 | 5428.00 | 5325.01 | 5358.86 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-26 09:15:00 | 5436.50 | 5326.12 | 5359.25 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 5458.50 | 5327.44 | 5359.74 | SL hit (close>static) qty=1.00 sl=5453.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 5458.50 | 5327.44 | 5359.74 | SL hit (close>static) qty=1.00 sl=5453.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 14:15:00 | 5528.50 | 5334.83 | 5362.83 | SL hit (close>static) qty=1.00 sl=5525.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-08 09:15:00 | 5432.00 | 5798.08 | 5670.41 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:15:00 | 5385.50 | 5789.90 | 5667.57 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 5481.50 | 5772.53 | 5661.79 | SL hit (close>static) qty=1.00 sl=5453.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-09 12:15:00 | 5425.00 | 5762.90 | 5658.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 5407.00 | 5756.02 | 5656.17 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 5363.50 | 5581.54 | 5582.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 5363.50 | 5581.54 | 5582.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 5345.00 | 5576.95 | 5579.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 5400.50 | 5362.00 | 5451.89 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-07 12:15:00 | 5311.00 | 5362.14 | 5447.59 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 14:15:00 | 5306.00 | 5360.54 | 5445.94 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-08 15:15:00 | 5308.00 | 5357.09 | 5440.82 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-11 09:15:00 | 5385.00 | 5357.37 | 5440.55 | ENTRY1 sustain failed after 3960m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 5436.00 | 5358.15 | 5440.52 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 5477.00 | 5360.58 | 5440.52 | SL hit (close>ema400) qty=1.00 sl=5440.52 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-12 14:15:00 | 5360.00 | 5364.17 | 5439.23 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-13 09:15:00 | 5420.00 | 5364.77 | 5438.78 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2025-08-14 14:15:00 | 5374.50 | 5369.17 | 5436.75 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-18 09:15:00 | 5563.50 | 5371.03 | 5437.01 | ENTRY2 sustain failed after 5460m |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 5362.50 | 5403.78 | 5443.06 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:15:00 | 5360.00 | 5402.84 | 5442.20 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 5456.50 | 5389.40 | 5429.57 | SL hit (close>static) qty=1.00 sl=5454.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-03 12:15:00 | 5369.00 | 5390.25 | 5428.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-03 13:15:00 | 5385.00 | 5390.20 | 5428.22 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-08 14:15:00 | 5320.50 | 5415.28 | 5437.63 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 5265.50 | 5412.73 | 5436.13 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1140m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-11-10 09:15:00 | 4475.68 | 4788.31 | 4945.12 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 4300.90 | 4227.70 | 4432.27 | SL hit (close>ema200) qty=0.50 sl=4227.70 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-02-04 11:15:00 | 5729.05 | 2025-02-27 09:15:00 | 4869.69 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2025-02-04 11:15:00 | 5729.05 | 2025-03-18 09:15:00 | 5237.00 | STOP_HIT | 0.50 | 8.59% |
| SELL | retest2 | 2025-03-28 13:15:00 | 5336.15 | 2025-04-01 09:15:00 | 5564.00 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-04-07 11:15:00 | 4626.25 | 2025-05-15 13:15:00 | 5540.00 | STOP_HIT | 1.00 | -19.75% |
| SELL | retest2 | 2025-04-30 11:15:00 | 5220.50 | 2025-05-15 13:15:00 | 5540.00 | STOP_HIT | 1.00 | -6.12% |
| SELL | retest2 | 2025-05-20 14:15:00 | 5443.50 | 2025-05-26 10:15:00 | 5458.50 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-05-21 13:15:00 | 5417.50 | 2025-05-26 10:15:00 | 5458.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-05-22 11:15:00 | 5386.50 | 2025-05-26 14:15:00 | 5528.50 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-07-08 11:15:00 | 5385.50 | 2025-07-09 09:15:00 | 5481.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-07-09 14:15:00 | 5407.00 | 2025-07-22 14:15:00 | 5363.50 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest1 | 2025-08-07 14:15:00 | 5306.00 | 2025-08-11 13:15:00 | 5477.00 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-08-26 11:15:00 | 5360.00 | 2025-09-02 11:15:00 | 5456.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-09-09 09:15:00 | 5265.50 | 2025-11-10 09:15:00 | 4475.68 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-09-09 09:15:00 | 5265.50 | 2025-12-24 12:15:00 | 4300.90 | STOP_HIT | 0.50 | 18.32% |
