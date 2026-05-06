# HDFCLIFE (HDFCLIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 606.35
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 8 |
| PENDING | 25 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 4 |
| ENTRY2 | 15 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 3 / 15
- **Target hits / Stop hits / Partials:** 0 / 17 / 1
- **Avg / median % per leg:** 0.82% / -0.87%
- **Sum % (uncompounded):** 14.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 0 | 14 | 1 | 1.32% | 19.8% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 14.00% | 28.0% |
| BUY @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 0 | 13 | 0 | -0.63% | -8.2% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.67% | -5.0% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.67% | -5.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | 4.60% | 23.0% |
| retest2 (combined) | 13 | 1 | 7.7% | 0 | 13 | 0 | -0.63% | -8.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 15:15:00 | 639.95 | 636.19 | 636.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-23 10:15:00 | 644.45 | 636.28 | 636.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 12:15:00 | 635.85 | 636.30 | 636.24 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 12:15:00 | 635.85 | 636.30 | 636.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 12:15:00 | 635.85 | 636.30 | 636.24 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2023-10-23 14:15:00 | 624.90 | 636.17 | 636.17 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2023-11-17 09:15:00 | 657.20 | 627.04 | 629.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 10:15:00 | 652.75 | 627.29 | 629.94 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-21 15:15:00 | 667.90 | 632.51 | 632.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 15:15:00 | 667.90 | 632.51 | 632.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 671.70 | 632.90 | 632.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 665.55 | 665.73 | 654.25 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2023-12-20 10:15:00 | 671.00 | 665.83 | 654.47 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-12-20 11:15:00 | 668.55 | 665.86 | 654.54 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 652.00 | 665.68 | 654.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 652.00 | 665.68 | 654.56 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-01-15 14:15:00 | 614.20 | 649.06 | 649.09 | HTF filter: close above htf_sma |

### Cycle 3 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 11:15:00 | 632.40 | 613.68 | 613.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 634.00 | 613.88 | 613.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 619.95 | 620.49 | 617.60 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 619.95 | 620.49 | 617.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 619.95 | 620.49 | 617.60 | EMA400 retest candle locked |

### Cycle 4 — SELL (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 09:15:00 | 598.15 | 615.36 | 615.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 13:15:00 | 595.40 | 614.67 | 615.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 10:15:00 | 576.10 | 575.45 | 588.51 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-05-29 09:15:00 | 567.40 | 575.42 | 588.11 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:15:00 | 568.00 | 575.35 | 588.01 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-10 14:15:00 | 569.55 | 565.83 | 579.19 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 15:15:00 | 567.90 | 565.86 | 579.13 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-11 13:15:00 | 571.35 | 566.23 | 579.00 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 14:15:00 | 572.10 | 566.29 | 578.96 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-12 14:15:00 | 572.25 | 566.83 | 578.80 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-12 15:15:00 | 572.45 | 566.88 | 578.77 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 588.55 | 567.10 | 578.81 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 578.81 | 567.10 | 578.81 | SL hit qty=1.00 sl=578.81 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 578.81 | 567.10 | 578.81 | SL hit qty=1.00 sl=578.81 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 578.81 | 567.10 | 578.81 | SL hit qty=1.00 sl=578.81 alert=retest1 |
| CROSSOVER_SKIP | 2024-07-05 14:15:00 | 607.45 | 585.31 | 585.21 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2024-11-25 14:15:00 | 686.05 | 704.75 | 704.81 | HTF filter: close above htf_sma |

### Cycle 5 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 681.50 | 634.11 | 633.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 692.35 | 646.97 | 640.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 12:15:00 | 656.90 | 658.18 | 647.84 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-04-07 13:15:00 | 663.15 | 658.23 | 647.92 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-07 14:15:00 | 666.35 | 658.31 | 648.01 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-05-23 09:15:00 | 766.30 | 724.51 | 700.00 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 760.15 | 776.57 | 756.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-14 10:15:00 | 760.65 | 776.41 | 756.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-14 11:15:00 | 760.15 | 776.25 | 756.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-14 12:15:00 | 760.45 | 776.09 | 756.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 763.25 | 775.96 | 756.70 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 753.00 | 775.19 | 756.88 | SL hit qty=1.00 sl=753.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-16 09:15:00 | 763.95 | 774.54 | 756.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 760.75 | 774.41 | 756.94 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 753.00 | 773.44 | 756.96 | SL hit qty=1.00 sl=753.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-22 12:15:00 | 764.50 | 768.42 | 756.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:15:00 | 765.65 | 768.40 | 756.21 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-23 10:15:00 | 760.60 | 768.15 | 756.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:15:00 | 762.75 | 768.09 | 756.35 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 757.30 | 767.40 | 756.63 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 753.00 | 767.26 | 756.61 | SL hit qty=1.00 sl=753.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 753.00 | 767.26 | 756.61 | SL hit qty=1.00 sl=753.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-25 11:15:00 | 760.70 | 767.06 | 756.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:15:00 | 764.75 | 767.04 | 756.66 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 756.10 | 766.37 | 757.11 | SL hit qty=1.00 sl=756.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 760.45 | 766.20 | 757.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-30 10:15:00 | 752.25 | 766.06 | 757.09 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-31 13:15:00 | 760.40 | 765.11 | 757.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-31 14:15:00 | 755.45 | 765.01 | 757.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 10:15:00 | 759.00 | 759.07 | 755.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-08 11:15:00 | 758.20 | 759.06 | 755.14 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 12:15:00 | 761.15 | 759.08 | 755.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 759.50 | 759.09 | 755.19 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 756.10 | 759.07 | 755.24 | SL hit qty=1.00 sl=756.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-11 12:15:00 | 760.90 | 759.06 | 755.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 764.25 | 759.11 | 755.34 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 13:15:00 | 756.10 | 774.65 | 767.07 | SL hit qty=1.00 sl=756.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-05 10:15:00 | 760.00 | 773.94 | 766.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 761.25 | 773.82 | 766.84 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2025-09-05 12:15:00 | 761.60 | 773.69 | 766.81 | max_alert3_locks_per_cycle=2 reached — end cycle |
| CROSSOVER_SKIP | 2025-10-09 10:15:00 | 743.15 | 766.39 | 766.41 | HTF filter: close above htf_sma |

### Cycle 6 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 785.45 | 759.54 | 759.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 786.70 | 759.81 | 759.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 760.75 | 761.94 | 760.81 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 760.75 | 761.94 | 760.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 760.75 | 761.94 | 760.81 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-01 14:15:00 | 767.55 | 761.98 | 760.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 766.75 | 762.03 | 760.88 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 759.10 | 761.98 | 760.86 | SL hit qty=1.00 sl=759.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-05 11:15:00 | 767.55 | 760.60 | 760.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 770.00 | 760.69 | 760.29 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 759.10 | 761.53 | 760.75 | SL hit qty=1.00 sl=759.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-09 14:15:00 | 764.15 | 761.44 | 760.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 762.90 | 761.45 | 760.73 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 759.10 | 761.50 | 760.76 | SL hit qty=1.00 sl=759.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-19 12:15:00 | 766.90 | 762.96 | 761.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 764.75 | 762.98 | 761.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 765.20 | 763.00 | 761.89 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-19 15:15:00 | 766.45 | 763.04 | 761.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 766.45 | 763.07 | 761.94 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-12-22 13:15:00 | 761.30 | 763.06 | 761.95 | SL hit qty=1.00 sl=761.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 759.10 | 763.02 | 761.95 | SL hit qty=1.00 sl=759.10 alert=retest2 |
| CROSSOVER_SKIP | 2025-12-29 14:15:00 | 747.05 | 761.00 | 761.04 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 773.20 | 758.45 | 759.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 772.95 | 758.59 | 759.64 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 761.30 | 760.39 | 760.51 | SL hit qty=1.00 sl=761.30 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-11-17 10:15:00 | 652.75 | 2023-11-21 15:15:00 | 667.90 | STOP_HIT | 1.00 | 2.32% |
| SELL | retest1 | 2024-05-29 10:15:00 | 568.00 | 2024-06-13 09:15:00 | 578.81 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest1 | 2024-06-10 15:15:00 | 567.90 | 2024-06-13 09:15:00 | 578.81 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest1 | 2024-06-11 14:15:00 | 572.10 | 2024-06-13 09:15:00 | 578.81 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2025-04-07 14:15:00 | 666.35 | 2025-05-23 09:15:00 | 766.30 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-04-07 14:15:00 | 666.35 | 2025-07-15 12:15:00 | 753.00 | STOP_HIT | 0.50 | 13.00% |
| BUY | retest2 | 2025-07-14 13:15:00 | 763.25 | 2025-07-17 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-07-16 10:15:00 | 760.75 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-22 13:15:00 | 765.65 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-23 11:15:00 | 762.75 | 2025-07-29 14:15:00 | 756.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-25 12:15:00 | 764.75 | 2025-08-11 09:15:00 | 756.10 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-08-08 13:15:00 | 759.50 | 2025-09-04 13:15:00 | 756.10 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-08-11 13:15:00 | 764.25 | 2025-12-02 09:15:00 | 759.10 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-05 11:15:00 | 761.25 | 2025-12-09 09:15:00 | 759.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-12-01 15:15:00 | 766.75 | 2025-12-10 09:15:00 | 759.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-05 12:15:00 | 770.00 | 2025-12-22 13:15:00 | 761.30 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-09 15:15:00 | 762.90 | 2025-12-23 09:15:00 | 759.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-19 13:15:00 | 764.75 | 2026-01-08 09:15:00 | 761.30 | STOP_HIT | 1.00 | -0.45% |
