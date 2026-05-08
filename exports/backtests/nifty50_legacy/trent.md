# TRENT (TRENT)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4998 bars)
- **Last close:** 4242.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 7 |
| PENDING | 25 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 3 |
| ENTRY2 | 14 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 13
- **Target hits / Stop hits / Partials:** 0 / 17 / 3
- **Avg / median % per leg:** 1.29% / -1.94%
- **Sum % (uncompounded):** 25.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.10% | -6.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.10% | -6.2% |
| SELL (all) | 18 | 7 | 38.9% | 0 | 15 | 3 | 1.77% | 31.9% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 4.43% | 17.7% |
| SELL @ 3rd Alert (retest2) | 14 | 5 | 35.7% | 0 | 12 | 2 | 1.02% | 14.2% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 4.43% | 17.7% |
| retest2 (combined) | 16 | 5 | 31.2% | 0 | 14 | 2 | 0.50% | 8.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 13:15:00 | 6462.60 | 6971.31 | 6972.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 15:15:00 | 6420.00 | 6960.99 | 6967.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 6891.15 | 6867.59 | 6909.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 14:15:00 | 6958.55 | 6868.50 | 6909.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 6958.55 | 6868.50 | 6909.88 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-10 12:15:00 | 6820.85 | 6880.01 | 6912.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-10 13:15:00 | 6834.30 | 6879.55 | 6911.88 | ENTRY2 sustain failed after 60m |

### Cycle 2 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 7116.85 | 6935.07 | 6934.89 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 6880.00 | 6934.51 | 6934.63 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 7008.40 | 6935.39 | 6935.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 11:15:00 | 7027.05 | 6936.30 | 6935.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 14:15:00 | 6931.65 | 6964.80 | 6950.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 6931.65 | 6964.80 | 6950.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 6931.65 | 6964.80 | 6950.82 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-12-31 13:15:00 | 7122.00 | 6966.00 | 6951.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 7133.35 | 6967.66 | 6952.73 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-07 09:15:00 | 7064.55 | 7021.65 | 6983.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-07 10:15:00 | 7024.35 | 7021.68 | 6984.05 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 6870.00 | 7018.66 | 6983.27 | SL hit (close<static) qty=1.00 sl=6923.85 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 6505.75 | 6948.63 | 6950.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 6407.40 | 6935.90 | 6943.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 6185.00 | 6183.09 | 6480.55 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-03 12:15:00 | 6099.10 | 6181.68 | 6476.88 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-03 13:15:00 | 6120.00 | 6181.07 | 6475.10 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-03 15:15:00 | 6100.00 | 6179.90 | 6471.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 09:15:00 | 5806.50 | 6176.18 | 6468.27 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 12:15:00 | 4935.52 | 5721.43 | 6117.53 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 5236.80 | 5213.67 | 5603.92 | SL hit (close>ema200) qty=0.50 sl=5213.67 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 5488.50 | 5227.36 | 5513.86 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-03-28 11:15:00 | 5425.35 | 5229.33 | 5513.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 5331.35 | 5230.35 | 5512.51 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-01 09:15:00 | 5564.45 | 5236.66 | 5510.10 | SL hit (close>static) qty=1.00 sl=5525.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 4543.90 | 5320.50 | 5519.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 10:15:00 | 4647.85 | 5313.81 | 5515.23 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-30 09:15:00 | 5193.50 | 5191.59 | 5352.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 10:15:00 | 5229.00 | 5191.96 | 5351.59 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 5540.00 | 5244.50 | 5331.92 | SL hit (close>static) qty=1.00 sl=5525.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 5540.00 | 5244.50 | 5331.92 | SL hit (close>static) qty=1.00 sl=5525.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-20 12:15:00 | 5462.50 | 5300.91 | 5353.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 13:15:00 | 5457.50 | 5302.47 | 5353.80 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 5443.50 | 5303.87 | 5354.25 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-05-21 11:15:00 | 5420.00 | 5309.70 | 5356.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 5429.50 | 5310.89 | 5356.57 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-22 09:15:00 | 5354.50 | 5315.01 | 5357.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 10:15:00 | 5350.00 | 5315.36 | 5357.70 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-23 11:15:00 | 5431.00 | 5320.79 | 5358.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-23 12:15:00 | 5442.00 | 5321.99 | 5359.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-23 15:15:00 | 5428.00 | 5325.42 | 5360.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-26 09:15:00 | 5436.50 | 5326.52 | 5360.77 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 5495.00 | 5329.50 | 5361.92 | SL hit (close>static) qty=1.00 sl=5474.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 5495.00 | 5329.50 | 5361.92 | SL hit (close>static) qty=1.00 sl=5474.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 14:15:00 | 5529.00 | 5335.22 | 5364.32 | SL hit (close>static) qty=1.00 sl=5525.95 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 5642.00 | 5391.28 | 5390.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 5679.00 | 5442.59 | 5418.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 5541.50 | 5551.49 | 5482.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 5541.50 | 5551.49 | 5482.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5541.50 | 5551.49 | 5482.88 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-13 13:15:00 | 5572.50 | 5551.36 | 5484.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:15:00 | 5584.00 | 5551.68 | 5484.67 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 5444.00 | 5831.78 | 5680.82 | SL hit (close<static) qty=1.00 sl=5481.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 5363.50 | 5581.56 | 5582.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 5345.00 | 5577.02 | 5580.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 5404.00 | 5361.93 | 5452.09 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-07 12:15:00 | 5311.00 | 5362.27 | 5447.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 13:15:00 | 5257.50 | 5361.22 | 5446.93 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-08 14:15:00 | 5319.50 | 5357.59 | 5441.70 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 15:15:00 | 5315.00 | 5357.16 | 5441.07 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 5436.00 | 5358.22 | 5440.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 5473.50 | 5360.61 | 5440.74 | SL hit (close>ema400) qty=1.00 sl=5440.74 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 5473.50 | 5360.61 | 5440.74 | SL hit (close>ema400) qty=1.00 sl=5440.74 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-12 14:15:00 | 5360.00 | 5364.34 | 5439.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:15:00 | 5365.50 | 5364.35 | 5439.14 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-14 14:15:00 | 5374.50 | 5369.30 | 5437.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 15:15:00 | 5362.00 | 5369.22 | 5436.63 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 5563.50 | 5371.16 | 5437.26 | SL hit (close>static) qty=1.00 sl=5454.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 5563.50 | 5371.16 | 5437.26 | SL hit (close>static) qty=1.00 sl=5454.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 5362.50 | 5403.85 | 5443.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:15:00 | 5352.50 | 5403.34 | 5442.81 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 5456.50 | 5389.52 | 5429.77 | SL hit (close>static) qty=1.00 sl=5454.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-03 12:15:00 | 5369.00 | 5390.36 | 5428.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-03 13:15:00 | 5385.00 | 5390.30 | 5428.41 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-08 14:15:00 | 5320.50 | 5415.54 | 5437.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 5308.00 | 5414.47 | 5437.25 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4511.80 | 4788.55 | 4945.39 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 4302.10 | 4227.58 | 4432.26 | SL hit (close>ema200) qty=0.50 sl=4227.58 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 4424.20 | 4243.04 | 4402.70 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-05 09:15:00 | 4377.30 | 4251.06 | 4402.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-05 10:15:00 | 4423.50 | 4252.77 | 4402.92 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 4118.80 | 4260.22 | 4402.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 4101.60 | 4258.64 | 4400.78 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 13:15:00 | 3486.36 | 3871.40 | 4000.39 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 3782.30 | 3645.36 | 3819.33 | SL hit (close>ema200) qty=0.50 sl=3645.36 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 4372.50 | 3843.29 | 3877.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-22 10:15:00 | 4411.00 | 3848.94 | 3880.44 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 4267.90 | 3881.14 | 3895.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 4296.10 | 3885.27 | 3897.86 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 4226.50 | 3912.22 | 3911.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 10:15:00 | 4226.50 | 3912.22 | 3911.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 13:15:00 | 4282.20 | 3922.31 | 3916.24 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-31 14:15:00 | 7133.35 | 2025-01-07 14:15:00 | 6870.00 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest1 | 2025-02-04 09:15:00 | 5806.50 | 2025-02-18 12:15:00 | 4935.52 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2025-02-04 09:15:00 | 5806.50 | 2025-03-18 09:15:00 | 5236.80 | STOP_HIT | 0.50 | 9.81% |
| SELL | retest2 | 2025-03-28 12:15:00 | 5331.35 | 2025-04-01 09:15:00 | 5564.45 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2025-04-07 10:15:00 | 4647.85 | 2025-05-15 13:15:00 | 5540.00 | STOP_HIT | 1.00 | -19.19% |
| SELL | retest2 | 2025-04-30 10:15:00 | 5229.00 | 2025-05-15 13:15:00 | 5540.00 | STOP_HIT | 1.00 | -5.95% |
| SELL | retest2 | 2025-05-20 13:15:00 | 5457.50 | 2025-05-26 11:15:00 | 5495.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-05-21 12:15:00 | 5429.50 | 2025-05-26 11:15:00 | 5495.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-05-22 10:15:00 | 5350.00 | 2025-05-26 14:15:00 | 5529.00 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-06-13 14:15:00 | 5584.00 | 2025-07-04 13:15:00 | 5444.00 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest1 | 2025-08-07 13:15:00 | 5257.50 | 2025-08-11 13:15:00 | 5473.50 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest1 | 2025-08-08 15:15:00 | 5315.00 | 2025-08-11 13:15:00 | 5473.50 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-08-12 15:15:00 | 5365.50 | 2025-08-18 09:15:00 | 5563.50 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-08-14 15:15:00 | 5362.00 | 2025-08-18 09:15:00 | 5563.50 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2025-08-26 10:15:00 | 5352.50 | 2025-09-02 11:15:00 | 5456.50 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-09-08 15:15:00 | 5308.00 | 2025-11-10 09:15:00 | 4511.80 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-09-08 15:15:00 | 5308.00 | 2025-12-24 12:15:00 | 4302.10 | STOP_HIT | 0.50 | 18.95% |
| SELL | retest2 | 2026-01-06 10:15:00 | 4101.60 | 2026-03-13 13:15:00 | 3486.36 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-01-06 10:15:00 | 4101.60 | 2026-04-06 09:15:00 | 3782.30 | STOP_HIT | 0.50 | 7.78% |
| SELL | retest2 | 2026-04-23 10:15:00 | 4296.10 | 2026-04-24 10:15:00 | 4226.50 | STOP_HIT | 1.00 | 1.62% |
