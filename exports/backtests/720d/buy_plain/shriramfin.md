# SHRIRAMFIN (SHRIRAMFIN)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1014.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 6 |
| PENDING | 18 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 13
- **Target hits / Stop hits / Partials:** 0 / 16 / 0
- **Avg / median % per leg:** -4.22% / -4.64%
- **Sum % (uncompounded):** -67.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 3 | 18.8% | 0 | 16 | 0 | -4.22% | -67.5% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.97% | -9.9% |
| BUY @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 0 | 14 | 0 | -4.11% | -57.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.97% | -9.9% |
| retest2 (combined) | 14 | 3 | 21.4% | 0 | 14 | 0 | -4.11% | -57.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 631.55 | 575.50 | 575.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 642.90 | 577.27 | 576.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 606.40 | 635.05 | 614.80 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 606.40 | 635.05 | 614.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 606.40 | 635.05 | 614.80 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 644.40 | 633.50 | 614.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 640.40 | 633.57 | 614.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-09 11:15:00 | 632.15 | 633.77 | 615.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 632.50 | 633.75 | 615.76 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 640.35 | 633.57 | 616.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 641.80 | 633.65 | 616.15 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-07 11:15:00 | 633.10 | 643.14 | 630.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:15:00 | 632.60 | 643.04 | 630.93 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 630.20 | 642.62 | 631.02 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 11:15:00 | 646.00 | 638.85 | 629.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 12:15:00 | 641.30 | 638.88 | 629.96 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-13 11:15:00 | 642.50 | 639.06 | 630.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 12:15:00 | 639.45 | 639.06 | 630.36 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-14 09:15:00 | 654.80 | 639.07 | 630.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:15:00 | 647.50 | 639.16 | 630.62 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-02 11:15:00 | 638.90 | 649.51 | 640.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 643.30 | 649.45 | 640.43 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 642.40 | 649.27 | 640.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-03 09:15:00 | 645.75 | 649.23 | 640.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-03 10:15:00 | 644.10 | 649.18 | 640.52 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-03 11:15:00 | 646.30 | 649.15 | 640.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 12:15:00 | 652.00 | 649.18 | 640.60 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 639.10 | 649.00 | 640.97 | SL hit (close<static) qty=1.00 sl=639.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-05 11:15:00 | 648.75 | 648.93 | 641.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:15:00 | 652.65 | 648.96 | 641.07 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-21 10:15:00 | 648.40 | 670.70 | 663.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:15:00 | 646.85 | 670.46 | 663.60 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-23 12:15:00 | 647.15 | 667.08 | 662.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 13:15:00 | 650.25 | 666.91 | 662.28 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 654.30 | 666.78 | 662.24 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 637.75 | 666.18 | 662.01 | SL hit (close<static) qty=1.00 sl=639.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 637.75 | 666.18 | 662.01 | SL hit (close<static) qty=1.00 sl=639.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 637.75 | 666.18 | 662.01 | SL hit (close<static) qty=1.00 sl=639.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 613.05 | 664.17 | 661.11 | SL hit (close<static) qty=1.00 sl=627.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 613.05 | 664.17 | 661.11 | SL hit (close<static) qty=1.00 sl=627.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 613.05 | 664.17 | 661.11 | SL hit (close<static) qty=1.00 sl=627.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 613.05 | 664.17 | 661.11 | SL hit (close<static) qty=1.00 sl=627.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 594.90 | 629.34 | 639.50 | SL hit (close<static) qty=1.00 sl=596.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 594.90 | 629.34 | 639.50 | SL hit (close<static) qty=1.00 sl=596.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 594.90 | 629.34 | 639.50 | SL hit (close<static) qty=1.00 sl=596.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 594.90 | 629.34 | 639.50 | SL hit (close<static) qty=1.00 sl=596.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 662.90 | 620.95 | 625.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 664.80 | 621.39 | 625.43 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 666.20 | 629.15 | 629.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 666.20 | 629.15 | 629.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 670.55 | 630.67 | 629.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 964.50 | 974.04 | 916.30 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 1004.60 | 973.18 | 917.84 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 10:15:00 | 1008.00 | 973.53 | 918.29 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-05 15:15:00 | 994.00 | 977.64 | 925.47 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 09:15:00 | 987.80 | 977.74 | 925.78 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-02-06 14:15:00 | 1004.40 | 978.33 | 927.36 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 15:15:00 | 1001.00 | 978.55 | 927.73 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1000.00 | 1038.87 | 987.84 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 954.60 | 1035.44 | 991.13 | SL hit (close<ema400) qty=1.00 sl=991.13 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 954.60 | 1035.44 | 991.13 | SL hit (close<ema400) qty=1.00 sl=991.13 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 1045.50 | 1032.23 | 991.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 1046.60 | 1032.37 | 991.49 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 984.80 | 1031.09 | 996.08 | SL hit (close<static) qty=1.00 sl=987.40 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1027.50 | 974.98 | 974.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1029.25 | 978.48 | 976.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 993.30 | 998.77 | 988.40 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 1021.10 | 999.02 | 988.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1021.10 | 999.02 | 988.72 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-04-08 10:15:00 | 640.40 | 2025-06-05 09:15:00 | 639.10 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-04-09 12:15:00 | 632.50 | 2025-07-24 10:15:00 | 637.75 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2025-04-11 10:15:00 | 641.80 | 2025-07-24 10:15:00 | 637.75 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-05-07 12:15:00 | 632.60 | 2025-07-24 10:15:00 | 637.75 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-05-12 12:15:00 | 641.30 | 2025-07-25 09:15:00 | 613.05 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-05-13 12:15:00 | 639.45 | 2025-07-25 09:15:00 | 613.05 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2025-05-14 10:15:00 | 647.50 | 2025-07-25 09:15:00 | 613.05 | STOP_HIT | 1.00 | -5.32% |
| BUY | retest2 | 2025-06-02 12:15:00 | 643.30 | 2025-07-25 09:15:00 | 613.05 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest2 | 2025-06-03 12:15:00 | 652.00 | 2025-08-26 14:15:00 | 594.90 | STOP_HIT | 1.00 | -8.76% |
| BUY | retest2 | 2025-06-05 12:15:00 | 652.65 | 2025-08-26 14:15:00 | 594.90 | STOP_HIT | 1.00 | -8.85% |
| BUY | retest2 | 2025-07-21 11:15:00 | 646.85 | 2025-08-26 14:15:00 | 594.90 | STOP_HIT | 1.00 | -8.03% |
| BUY | retest2 | 2025-07-23 13:15:00 | 650.25 | 2025-08-26 14:15:00 | 594.90 | STOP_HIT | 1.00 | -8.51% |
| BUY | retest2 | 2025-10-06 10:15:00 | 664.80 | 2025-10-08 14:15:00 | 666.20 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest1 | 2026-02-03 10:15:00 | 1008.00 | 2026-03-09 09:15:00 | 954.60 | STOP_HIT | 1.00 | -5.30% |
| BUY | retest1 | 2026-02-06 15:15:00 | 1001.00 | 2026-03-09 09:15:00 | 954.60 | STOP_HIT | 1.00 | -4.64% |
| BUY | retest2 | 2026-03-10 11:15:00 | 1046.60 | 2026-03-16 10:15:00 | 984.80 | STOP_HIT | 1.00 | -5.90% |
