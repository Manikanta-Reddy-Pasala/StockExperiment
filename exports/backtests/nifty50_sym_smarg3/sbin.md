# SBIN (SBIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1096.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 18 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 1 |
| ENTRY2 | 8 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 5
- **Target hits / Stop hits / Partials:** 3 / 6 / 3
- **Avg / median % per leg:** 10.41% / 15.00%
- **Sum % (uncompounded):** 124.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 3 | 6 | 3 | 10.41% | 124.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.78% | -2.8% |
| BUY @ 3rd Alert (retest2) | 11 | 7 | 63.6% | 3 | 5 | 3 | 11.61% | 127.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.78% | -2.8% |
| retest2 (combined) | 11 | 7 | 63.6% | 3 | 5 | 3 | 11.61% | 127.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 15:15:00 | 603.00 | 583.57 | 583.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-21 10:15:00 | 604.20 | 585.22 | 584.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 587.20 | 589.07 | 586.73 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 587.20 | 589.07 | 586.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 587.20 | 589.07 | 586.73 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-10-06 10:15:00 | 597.10 | 589.31 | 587.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-06 11:15:00 | 593.90 | 589.36 | 587.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-06 13:15:00 | 595.15 | 589.47 | 587.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 14:15:00 | 594.70 | 589.52 | 587.15 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-09 09:15:00 | 585.45 | 589.53 | 587.18 | SL hit qty=1.00 sl=585.45 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-11 09:15:00 | 595.00 | 589.37 | 587.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-11 10:15:00 | 592.55 | 589.41 | 587.28 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-11 11:15:00 | 594.85 | 589.46 | 587.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-11 12:15:00 | 593.55 | 589.50 | 587.35 | ENTRY2 sustain failed after 60m |
| CROSSOVER_SKIP | 2023-10-19 11:15:00 | 572.10 | 585.61 | 585.66 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2023-12-04 15:15:00 | 595.80 | 572.26 | 575.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 09:15:00 | 608.40 | 572.62 | 575.64 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2023-12-07 13:15:00 | 610.65 | 578.59 | 578.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 13:15:00 | 610.65 | 578.59 | 578.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 14:15:00 | 614.05 | 581.05 | 579.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 14:15:00 | 624.90 | 625.82 | 610.33 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-12 13:15:00 | 632.15 | 625.51 | 611.64 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 14:15:00 | 632.90 | 625.59 | 611.74 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-18 11:15:00 | 632.00 | 627.55 | 614.44 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-18 12:15:00 | 628.10 | 627.56 | 614.50 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-19 09:15:00 | 632.70 | 627.64 | 614.80 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-19 10:15:00 | 628.85 | 627.65 | 614.87 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 613.10 | 627.48 | 615.29 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 615.29 | 627.48 | 615.29 | SL hit qty=1.00 sl=615.29 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-29 13:15:00 | 627.75 | 624.92 | 615.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-29 14:15:00 | 623.50 | 624.91 | 615.29 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-30 10:15:00 | 627.90 | 624.93 | 615.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 11:15:00 | 632.75 | 625.01 | 615.53 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-31 09:15:00 | 628.90 | 625.18 | 615.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 10:15:00 | 632.00 | 625.25 | 615.93 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-02-09 15:15:00 | 727.66 | 643.67 | 628.12 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-02-09 15:15:00 | 726.80 | 643.67 | 628.12 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-04-29 13:15:00 | 822.58 | 763.10 | 739.38 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2024-04-29 13:15:00 | 821.60 | 763.10 | 739.38 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2024-08-27 10:15:00 | 815.30 | 829.61 | 829.63 | HTF filter: close above htf_sma |

### Cycle 3 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 856.30 | 808.89 | 808.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 860.25 | 811.57 | 810.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 10:15:00 | 814.95 | 818.72 | 814.20 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 11:15:00 | 816.60 | 818.70 | 814.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 816.60 | 818.70 | 814.21 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-13 12:15:00 | 817.60 | 818.69 | 814.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-13 13:15:00 | 815.00 | 818.65 | 814.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-22 13:15:00 | 819.10 | 813.67 | 812.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-22 14:15:00 | 816.50 | 813.70 | 812.30 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-25 09:15:00 | 844.00 | 814.01 | 812.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 10:15:00 | 846.05 | 814.33 | 812.64 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-20 14:15:00 | 813.60 | 839.91 | 830.69 | SL hit qty=1.00 sl=813.60 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-23 09:15:00 | 818.00 | 839.43 | 830.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:15:00 | 821.35 | 839.25 | 830.50 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-24 12:15:00 | 813.60 | 837.58 | 830.03 | SL hit qty=1.00 sl=813.60 alert=retest2 |
| CROSSOVER_SKIP | 2025-01-02 14:15:00 | 801.25 | 824.16 | 824.16 | HTF filter: close above htf_sma |

### Cycle 4 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 819.10 | 757.67 | 757.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 822.65 | 758.32 | 757.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 779.10 | 781.81 | 772.06 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 776.20 | 781.71 | 772.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 776.20 | 781.71 | 772.11 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-07 10:15:00 | 778.45 | 781.37 | 772.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-07 11:15:00 | 776.50 | 781.32 | 772.24 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-08 09:15:00 | 783.75 | 781.10 | 772.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:15:00 | 781.50 | 781.11 | 772.40 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-08 14:15:00 | 771.10 | 780.91 | 772.47 | SL hit qty=1.00 sl=771.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-09 14:15:00 | 780.20 | 780.42 | 772.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 780.00 | 780.41 | 772.55 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-20 09:15:00 | 897.00 | 860.84 | 843.50 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2026-01-05 10:15:00 | 1014.00 | 969.10 | 947.35 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2026-04-07 11:15:00 | 1020.45 | 1069.63 | 1069.75 | HTF filter: close above htf_sma |

### Cycle 5 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 1111.00 | 1069.30 | 1069.16 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-06 14:15:00 | 594.70 | 2023-10-09 09:15:00 | 585.45 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2023-12-05 09:15:00 | 608.40 | 2023-12-07 13:15:00 | 610.65 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest1 | 2024-01-12 14:15:00 | 632.90 | 2024-01-23 11:15:00 | 615.29 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2024-01-30 11:15:00 | 632.75 | 2024-02-09 15:15:00 | 727.66 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-01-31 10:15:00 | 632.00 | 2024-02-09 15:15:00 | 726.80 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-01-30 11:15:00 | 632.75 | 2024-04-29 13:15:00 | 822.58 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2024-01-31 10:15:00 | 632.00 | 2024-04-29 13:15:00 | 821.60 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2024-11-25 10:15:00 | 846.05 | 2024-12-20 14:15:00 | 813.60 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2024-12-23 10:15:00 | 821.35 | 2024-12-24 12:15:00 | 813.60 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-05-08 10:15:00 | 781.50 | 2025-05-08 14:15:00 | 771.10 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-05-09 15:15:00 | 780.00 | 2025-10-20 09:15:00 | 897.00 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-05-09 15:15:00 | 780.00 | 2026-01-05 10:15:00 | 1014.00 | TARGET_HIT | 0.50 | 30.00% |
