# SBIN (SBIN)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1092.90
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
| PENDING | 21 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 11
- **Target hits / Stop hits / Partials:** 0 / 14 / 0
- **Avg / median % per leg:** -1.21% / -1.18%
- **Sum % (uncompounded):** -16.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 3 | 21.4% | 0 | 14 | 0 | -1.21% | -16.9% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 6 | 0 | -0.66% | -4.0% |
| SELL @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 0 | 8 | 0 | -1.62% | -13.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 6 | 0 | -0.66% | -4.0% |
| retest2 (combined) | 8 | 1 | 12.5% | 0 | 8 | 0 | -1.62% | -13.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 796.85 | 841.18 | 841.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 10:15:00 | 789.45 | 824.01 | 830.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 12:15:00 | 804.65 | 803.74 | 815.86 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-09-23 15:15:00 | 800.05 | 803.69 | 815.66 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:15:00 | 801.20 | 803.67 | 815.59 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2024-09-24 14:15:00 | 798.55 | 803.56 | 815.24 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:15:00 | 792.05 | 803.39 | 815.04 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 1140m) |
| Cross detected — sustain check pending | 2024-09-27 12:15:00 | 800.95 | 802.50 | 813.62 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-09-27 13:15:00 | 801.55 | 802.49 | 813.56 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-30 09:15:00 | 793.20 | 802.40 | 813.35 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 11:15:00 | 792.70 | 802.19 | 813.13 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2024-10-03 11:15:00 | 794.00 | 801.26 | 811.91 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 13:15:00 | 790.65 | 801.02 | 811.68 | SELL ENTRY1 attempt 4/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 806.20 | 798.39 | 807.97 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-10-14 11:15:00 | 803.80 | 798.44 | 807.95 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 13:15:00 | 802.10 | 798.52 | 807.90 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-10-15 12:15:00 | 805.35 | 798.88 | 807.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 14:15:00 | 804.30 | 798.99 | 807.77 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-10-16 11:15:00 | 804.75 | 799.25 | 807.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-10-16 12:15:00 | 805.80 | 799.31 | 807.72 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-16 13:15:00 | 805.45 | 799.37 | 807.70 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-10-16 15:15:00 | 805.90 | 799.49 | 807.68 | ENTRY2 sustain failed after 120m |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 811.00 | 799.61 | 807.70 | SL hit (close>ema400) qty=1.00 sl=807.70 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 811.00 | 799.61 | 807.70 | SL hit (close>ema400) qty=1.00 sl=807.70 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 811.00 | 799.61 | 807.70 | SL hit (close>ema400) qty=1.00 sl=807.70 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 811.00 | 799.61 | 807.70 | SL hit (close>ema400) qty=1.00 sl=807.70 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 811.00 | 799.61 | 807.70 | SL hit (close>static) qty=1.00 sl=809.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 811.00 | 799.61 | 807.70 | SL hit (close>static) qty=1.00 sl=809.25 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-22 11:15:00 | 798.10 | 802.36 | 808.28 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 13:15:00 | 800.50 | 802.27 | 808.18 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 810.95 | 798.88 | 805.42 | SL hit (close>static) qty=1.00 sl=809.25 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-14 10:15:00 | 803.85 | 818.52 | 814.98 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-11-14 11:15:00 | 810.05 | 818.43 | 814.96 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-14 13:15:00 | 804.20 | 818.18 | 814.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-11-14 15:15:00 | 805.85 | 817.92 | 814.77 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-11-18 09:15:00 | 802.55 | 817.77 | 814.71 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-11-18 10:15:00 | 805.85 | 817.65 | 814.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-19 14:15:00 | 802.10 | 816.95 | 814.46 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 767.95 | 816.31 | 814.17 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 2580m) |
| Stop hit — per-position SL triggered | 2024-11-22 11:15:00 | 809.50 | 813.79 | 812.97 | SL hit (close>static) qty=1.00 sl=809.25 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 819.05 | 813.81 | 812.99 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-27 12:15:00 | 808.15 | 834.31 | 829.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:15:00 | 799.45 | 833.68 | 828.87 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 799.15 | 824.44 | 824.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 13:15:00 | 799.15 | 824.44 | 824.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 793.50 | 822.50 | 823.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 10:15:00 | 775.90 | 774.39 | 790.84 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-02-01 11:15:00 | 762.85 | 774.28 | 790.70 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 13:15:00 | 765.10 | 774.00 | 790.40 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-05 11:15:00 | 769.20 | 773.11 | 788.43 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 13:15:00 | 768.40 | 773.04 | 788.24 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 746.85 | 732.08 | 748.88 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-19 14:15:00 | 744.75 | 732.20 | 748.86 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 744.90 | 732.45 | 748.82 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Stop hit — per-position SL triggered | 2025-03-20 14:15:00 | 749.35 | 733.19 | 748.79 | SL hit (close>ema400) qty=1.00 sl=748.79 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-20 14:15:00 | 749.35 | 733.19 | 748.79 | SL hit (close>ema400) qty=1.00 sl=748.79 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-20 14:15:00 | 749.35 | 733.19 | 748.79 | SL hit (close>static) qty=1.00 sl=749.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 738.40 | 752.58 | 755.39 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-07 10:15:00 | 746.50 | 752.52 | 755.35 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-07 11:15:00 | 735.40 | 752.35 | 755.25 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 13:15:00 | 745.00 | 752.15 | 755.12 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 764.15 | 752.19 | 755.10 | SL hit (close>static) qty=1.00 sl=749.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-09 13:15:00 | 745.40 | 753.13 | 755.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:15:00 | 742.50 | 752.92 | 755.30 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 751.25 | 752.90 | 755.28 | SL hit (close>static) qty=1.00 sl=749.30 alert=retest2 |

### Cycle 3 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 1017.35 | 1070.19 | 1070.41 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-09-24 09:15:00 | 801.20 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest1 | 2024-09-25 09:15:00 | 792.05 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest1 | 2024-09-30 11:15:00 | 792.70 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest1 | 2024-10-03 13:15:00 | 790.65 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-10-14 13:15:00 | 802.10 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-10-15 14:15:00 | 804.30 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-10-22 13:15:00 | 800.50 | 2024-10-29 11:15:00 | 810.95 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-11-21 09:15:00 | 767.95 | 2024-11-22 11:15:00 | 809.50 | STOP_HIT | 1.00 | -5.41% |
| SELL | retest2 | 2024-12-27 14:15:00 | 799.45 | 2025-01-02 13:15:00 | 799.15 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest1 | 2025-02-01 13:15:00 | 765.10 | 2025-03-20 14:15:00 | 749.35 | STOP_HIT | 1.00 | 2.06% |
| SELL | retest1 | 2025-02-05 13:15:00 | 768.40 | 2025-03-20 14:15:00 | 749.35 | STOP_HIT | 1.00 | 2.48% |
| SELL | retest2 | 2025-03-20 09:15:00 | 744.90 | 2025-03-20 14:15:00 | 749.35 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-04-07 13:15:00 | 745.00 | 2025-04-08 09:15:00 | 764.15 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-04-09 15:15:00 | 742.50 | 2025-04-11 09:15:00 | 751.25 | STOP_HIT | 1.00 | -1.18% |
