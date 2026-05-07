# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 950.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 4 |
| PENDING | 18 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 6 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 10
- **Target hits / Stop hits / Partials:** 0 / 14 / 4
- **Avg / median % per leg:** 3.41% / -0.69%
- **Sum % (uncompounded):** 61.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 0 | 14 | 4 | 3.41% | 61.4% |
| BUY @ 2nd Alert (retest1) | 12 | 8 | 66.7% | 0 | 8 | 4 | 7.58% | 90.9% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -4.92% | -29.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 8 | 66.7% | 0 | 8 | 4 | 7.58% | 90.9% |
| retest2 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -4.92% | -29.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 1464.35 | 1419.30 | 1419.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 09:15:00 | 1491.00 | 1426.78 | 1423.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 12:15:00 | 1434.55 | 1437.06 | 1429.20 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-09-25 14:15:00 | 1440.85 | 1437.09 | 1429.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 15:15:00 | 1441.00 | 1437.13 | 1429.35 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1441.15 | 1439.94 | 1431.65 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 1431.10 | 1439.85 | 1431.64 | SL hit (close<ema400) qty=1.00 sl=1431.64 alert=retest1 |

### Cycle 2 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 866.20 | 822.24 | 822.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 882.00 | 828.17 | 825.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 845.10 | 850.17 | 839.76 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-21 13:15:00 | 852.95 | 850.09 | 839.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 14:15:00 | 859.65 | 850.19 | 839.98 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-23 12:15:00 | 850.95 | 849.86 | 840.40 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 13:15:00 | 851.75 | 849.88 | 840.46 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-23 15:15:00 | 851.25 | 849.89 | 840.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:15:00 | 855.55 | 849.95 | 840.64 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 835.75 | 849.79 | 840.88 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 835.75 | 849.79 | 840.88 | SL hit (close<ema400) qty=1.00 sl=840.88 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 835.75 | 849.79 | 840.88 | SL hit (close<ema400) qty=1.00 sl=840.88 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 835.75 | 849.79 | 840.88 | SL hit (close<ema400) qty=1.00 sl=840.88 alert=retest1 |

### Cycle 3 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 799.00 | 771.97 | 771.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 820.50 | 772.97 | 772.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 832.00 | 833.00 | 813.70 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-11 09:15:00 | 838.20 | 833.05 | 813.82 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-11 10:15:00 | 833.20 | 833.05 | 813.92 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-11 11:15:00 | 841.50 | 833.14 | 814.06 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-11 12:15:00 | 835.15 | 833.16 | 814.16 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-12 09:15:00 | 848.10 | 833.36 | 814.64 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:15:00 | 840.00 | 833.43 | 814.77 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-18 09:15:00 | 838.50 | 835.60 | 818.30 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:15:00 | 838.45 | 835.63 | 818.40 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-19 11:15:00 | 836.15 | 835.56 | 819.04 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 12:15:00 | 838.90 | 835.60 | 819.14 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-30 13:15:00 | 839.50 | 839.88 | 824.75 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 14:15:00 | 841.60 | 839.89 | 824.83 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 886.55 | 888.15 | 862.21 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-27 14:15:00 | 894.60 | 888.14 | 862.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 893.00 | 888.19 | 862.99 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-28 11:15:00 | 894.10 | 888.22 | 863.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 895.15 | 888.29 | 863.54 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-02 11:15:00 | 891.85 | 890.33 | 867.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-02 12:15:00 | 888.50 | 890.32 | 867.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-02 13:15:00 | 893.25 | 890.34 | 867.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:15:00 | 911.00 | 890.55 | 868.21 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-26 09:15:00 | 964.22 | 916.60 | 894.77 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-26 14:15:00 | 964.73 | 918.32 | 896.18 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-26 15:15:00 | 966.00 | 918.75 | 896.50 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-26 15:15:00 | 967.84 | 918.75 | 896.50 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 920.75 | 922.91 | 900.29 | SL hit (close<ema200) qty=0.50 sl=922.91 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 920.75 | 922.91 | 900.29 | SL hit (close<ema200) qty=0.50 sl=922.91 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 920.75 | 922.91 | 900.29 | SL hit (close<ema200) qty=0.50 sl=922.91 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 920.75 | 922.91 | 900.29 | SL hit (close<ema200) qty=0.50 sl=922.91 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 898.55 | 920.78 | 902.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 897.60 | 920.55 | 902.28 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 894.35 | 920.29 | 902.24 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-10 13:15:00 | 901.40 | 920.10 | 902.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-10 14:15:00 | 898.05 | 919.88 | 902.21 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.51 | 901.28 | SL hit (close<static) qty=1.00 sl=851.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.51 | 901.28 | SL hit (close<static) qty=1.00 sl=851.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.51 | 901.28 | SL hit (close<static) qty=1.00 sl=851.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.51 | 901.28 | SL hit (close<static) qty=1.00 sl=851.15 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-27 14:15:00 | 899.85 | 844.45 | 854.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 900.45 | 845.01 | 854.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 889.25 | 845.45 | 854.65 | SL hit (close<static) qty=1.00 sl=890.95 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-29 09:15:00 | 918.10 | 848.61 | 855.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 916.10 | 849.29 | 856.24 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 908.20 | 862.52 | 862.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 908.20 | 862.52 | 862.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 913.70 | 863.03 | 862.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-25 15:15:00 | 1441.00 | 2024-10-01 10:15:00 | 1431.10 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2025-07-21 14:15:00 | 859.65 | 2025-07-25 09:15:00 | 835.75 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest1 | 2025-07-23 13:15:00 | 851.75 | 2025-07-25 09:15:00 | 835.75 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest1 | 2025-07-24 09:15:00 | 855.55 | 2025-07-25 09:15:00 | 835.75 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest1 | 2025-12-12 10:15:00 | 840.00 | 2026-02-26 09:15:00 | 964.22 | PARTIAL | 0.50 | 14.79% |
| BUY | retest1 | 2025-12-18 10:15:00 | 838.45 | 2026-02-26 14:15:00 | 964.73 | PARTIAL | 0.50 | 15.06% |
| BUY | retest1 | 2025-12-19 12:15:00 | 838.90 | 2026-02-26 15:15:00 | 966.00 | PARTIAL | 0.50 | 15.15% |
| BUY | retest1 | 2025-12-30 14:15:00 | 841.60 | 2026-02-26 15:15:00 | 967.84 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-12-12 10:15:00 | 840.00 | 2026-03-04 09:15:00 | 920.75 | STOP_HIT | 0.50 | 9.61% |
| BUY | retest1 | 2025-12-18 10:15:00 | 838.45 | 2026-03-04 09:15:00 | 920.75 | STOP_HIT | 0.50 | 9.82% |
| BUY | retest1 | 2025-12-19 12:15:00 | 838.90 | 2026-03-04 09:15:00 | 920.75 | STOP_HIT | 0.50 | 9.76% |
| BUY | retest1 | 2025-12-30 14:15:00 | 841.60 | 2026-03-04 09:15:00 | 920.75 | STOP_HIT | 0.50 | 9.40% |
| BUY | retest2 | 2026-01-27 15:15:00 | 893.00 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest2 | 2026-01-28 12:15:00 | 895.15 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.45% |
| BUY | retest2 | 2026-02-02 14:15:00 | 911.00 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -8.07% |
| BUY | retest2 | 2026-03-10 11:15:00 | 897.60 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.70% |
| BUY | retest2 | 2026-04-27 15:15:00 | 900.45 | 2026-04-28 09:15:00 | 889.25 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-04-29 10:15:00 | 916.10 | 2026-05-05 11:15:00 | 908.20 | STOP_HIT | 1.00 | -0.86% |
