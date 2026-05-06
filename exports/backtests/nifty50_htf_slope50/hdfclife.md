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
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 6 |
| ALERT3 | 15 |
| PENDING | 51 |
| PENDING_CANCEL | 13 |
| ENTRY1 | 1 |
| ENTRY2 | 37 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 34
- **Target hits / Stop hits / Partials:** 0 / 38 / 2
- **Avg / median % per leg:** 0.41% / -0.85%
- **Sum % (uncompounded):** 16.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 6 | 15.0% | 0 | 38 | 2 | 0.41% | 16.5% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 14.00% | 28.0% |
| BUY @ 3rd Alert (retest2) | 38 | 4 | 10.5% | 0 | 37 | 1 | -0.30% | -11.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 14.00% | 28.0% |
| retest2 (combined) | 38 | 4 | 10.5% | 0 | 37 | 1 | -0.30% | -11.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 623.10 | 636.23 | 636.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 13:15:00 | 620.55 | 635.47 | 635.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 640.20 | 634.85 | 635.55 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 640.20 | 634.85 | 635.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 640.20 | 634.85 | 635.55 | EMA400 retest candle locked |

### Cycle 2 — BUY (started 2023-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 15:15:00 | 639.95 | 636.19 | 636.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-23 10:15:00 | 644.45 | 636.28 | 636.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 12:15:00 | 635.85 | 636.30 | 636.24 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 12:15:00 | 635.85 | 636.30 | 636.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 12:15:00 | 635.85 | 636.30 | 636.24 | EMA400 retest candle locked |

### Cycle 3 — SELL (started 2023-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 14:15:00 | 624.90 | 636.17 | 636.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 09:15:00 | 613.05 | 635.83 | 636.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 626.65 | 625.73 | 629.68 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 626.65 | 625.73 | 629.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 626.65 | 625.73 | 629.68 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-11-13 12:15:00 | 619.05 | 625.56 | 629.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-13 13:15:00 | 622.05 | 625.52 | 629.36 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-13 14:15:00 | 620.45 | 625.47 | 629.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-13 15:15:00 | 621.05 | 625.43 | 629.28 | ENTRY2 sustain failed after 60m |

### Cycle 4 — BUY (started 2023-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 15:15:00 | 667.90 | 632.51 | 632.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 671.70 | 632.90 | 632.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 665.55 | 665.73 | 654.25 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2023-12-20 10:15:00 | 671.00 | 665.83 | 654.47 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-12-20 11:15:00 | 668.55 | 665.86 | 654.54 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 652.00 | 665.68 | 654.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 652.00 | 665.68 | 654.56 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-01-15 14:15:00 | 614.20 | 649.06 | 649.09 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 5 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 11:15:00 | 632.40 | 613.68 | 613.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 634.00 | 613.88 | 613.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 619.95 | 620.49 | 617.60 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 619.95 | 620.49 | 617.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 619.95 | 620.49 | 617.60 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-04-24 09:15:00 | 598.15 | 615.36 | 615.42 | slope filter: EMA200 not falling 0.50% over 350 bars |
| CROSSOVER_SKIP | 2024-07-05 14:15:00 | 607.45 | 585.31 | 585.21 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2024-07-09 09:15:00 | 624.10 | 587.91 | 586.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 10:15:00 | 623.65 | 588.27 | 586.73 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 617.40 | 590.66 | 588.00 | SL hit qty=1.00 sl=617.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-10 11:15:00 | 630.20 | 591.05 | 588.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:15:00 | 637.05 | 591.51 | 588.46 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-08-26 09:15:00 | 732.61 | 684.11 | 655.55 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2024-11-25 14:15:00 | 686.05 | 704.75 | 704.81 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2024-12-03 09:15:00 | 637.05 | 692.39 | 698.21 | SL hit qty=0.50 sl=637.05 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-19 14:15:00 | 623.15 | 657.22 | 674.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 15:15:00 | 624.10 | 656.89 | 674.73 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-20 11:15:00 | 624.45 | 655.86 | 673.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-20 12:15:00 | 620.85 | 655.51 | 673.68 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-20 13:15:00 | 623.00 | 655.19 | 673.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-20 14:15:00 | 622.35 | 654.86 | 673.17 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-20 15:15:00 | 623.75 | 654.55 | 672.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-23 09:15:00 | 620.00 | 654.21 | 672.66 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2024-12-23 09:15:00 | 617.40 | 654.21 | 672.66 | SL hit qty=1.00 sl=617.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-23 10:15:00 | 624.00 | 653.91 | 672.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 622.85 | 653.60 | 672.17 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 620.55 | 653.27 | 671.91 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-24 09:15:00 | 624.05 | 652.05 | 670.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-24 10:15:00 | 623.50 | 651.77 | 670.69 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-24 11:15:00 | 626.05 | 651.51 | 670.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 12:15:00 | 625.20 | 651.25 | 670.25 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-26 09:15:00 | 625.70 | 650.17 | 669.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2024-12-26 09:15:00 | 620.25 | 650.17 | 669.33 | SL hit qty=1.00 sl=620.25 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 10:15:00 | 628.55 | 649.96 | 669.12 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 617.40 | 646.80 | 666.29 | SL hit qty=1.00 sl=617.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 620.25 | 646.80 | 666.29 | SL hit qty=1.00 sl=620.25 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-16 09:15:00 | 653.10 | 625.04 | 646.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 10:15:00 | 653.30 | 625.32 | 646.18 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 620.25 | 627.52 | 644.76 | SL hit qty=1.00 sl=620.25 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-23 12:15:00 | 624.60 | 626.69 | 643.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-23 13:15:00 | 623.10 | 626.65 | 643.39 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-29 13:15:00 | 627.50 | 623.77 | 639.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 14:15:00 | 628.45 | 623.82 | 639.61 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 628.15 | 623.86 | 639.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-30 10:15:00 | 633.65 | 624.01 | 639.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 11:15:00 | 631.05 | 624.08 | 639.43 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-30 12:15:00 | 627.15 | 624.11 | 639.37 | SL hit qty=1.00 sl=627.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-30 13:15:00 | 631.00 | 624.18 | 639.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 14:15:00 | 635.20 | 624.29 | 639.31 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 620.25 | 625.19 | 639.11 | SL hit qty=1.00 sl=620.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 627.15 | 625.19 | 639.11 | SL hit qty=1.00 sl=627.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-05 10:15:00 | 635.45 | 624.98 | 637.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 11:15:00 | 633.25 | 625.06 | 637.97 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-05 12:15:00 | 627.15 | 625.11 | 637.93 | SL hit qty=1.00 sl=627.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-06 09:15:00 | 631.60 | 625.25 | 637.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 10:15:00 | 634.55 | 625.34 | 637.73 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 634.90 | 625.43 | 637.71 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-06 12:15:00 | 638.00 | 625.56 | 637.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 13:15:00 | 636.30 | 625.67 | 637.71 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-06 14:15:00 | 633.10 | 625.75 | 637.69 | SL hit qty=1.00 sl=633.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 627.15 | 625.87 | 637.63 | SL hit qty=1.00 sl=627.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-07 14:15:00 | 636.90 | 626.23 | 637.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 15:15:00 | 637.10 | 626.34 | 637.52 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 633.10 | 626.43 | 637.51 | SL hit qty=1.00 sl=633.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-10 13:15:00 | 637.40 | 626.71 | 637.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 14:15:00 | 635.90 | 626.80 | 637.42 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-10 15:15:00 | 633.10 | 626.86 | 637.40 | SL hit qty=1.00 sl=633.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-13 09:15:00 | 636.20 | 626.60 | 636.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-13 10:15:00 | 638.55 | 626.72 | 636.51 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 635.45 | 626.92 | 636.51 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-13 13:15:00 | 633.10 | 626.94 | 636.47 | SL hit qty=1.00 sl=633.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-18 14:15:00 | 641.45 | 624.66 | 629.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 15:15:00 | 640.50 | 624.82 | 629.52 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-24 10:15:00 | 681.50 | 634.11 | 633.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-03-24 10:15:00)

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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 761.60 | 773.69 | 766.81 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 756.10 | 773.14 | 766.67 | SL hit qty=1.00 sl=756.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-10 09:15:00 | 771.15 | 771.10 | 766.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 774.00 | 771.13 | 766.09 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 760.00 | 774.17 | 769.57 | SL hit qty=1.00 sl=760.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-25 13:15:00 | 763.65 | 774.06 | 769.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 14:15:00 | 765.05 | 773.97 | 769.52 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-29 10:15:00 | 760.00 | 773.27 | 769.38 | SL hit qty=1.00 sl=760.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-01 14:15:00 | 763.25 | 770.62 | 768.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 15:15:00 | 765.50 | 770.56 | 768.31 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 760.00 | 770.47 | 768.27 | SL hit qty=1.00 sl=760.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 14:15:00 | 763.30 | 769.17 | 767.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 763.05 | 769.11 | 767.71 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 760.00 | 769.02 | 767.67 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 760.00 | 769.02 | 767.67 | SL hit qty=1.00 sl=760.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-09 10:15:00 | 743.15 | 766.39 | 766.41 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-11-12 09:15:00 | 770.75 | 751.00 | 755.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:15:00 | 775.00 | 751.24 | 755.97 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-17 14:15:00 | 770.25 | 756.68 | 758.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-17 15:15:00 | 768.50 | 756.80 | 758.38 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 759.70 | 757.24 | 758.54 | SL hit qty=1.00 sl=759.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-26 09:15:00 | 780.30 | 758.82 | 759.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 779.85 | 759.03 | 759.27 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 785.45 | 759.54 | 759.53 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 785.45 | 759.54 | 759.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-11-26 12:15:00)

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
| CROSSOVER_SKIP | 2025-12-29 14:15:00 | 747.05 | 761.00 | 761.04 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 773.20 | 758.45 | 759.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 772.95 | 758.59 | 759.64 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 761.30 | 760.39 | 760.51 | SL hit qty=1.00 sl=761.30 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-09 10:15:00 | 623.65 | 2024-07-10 10:15:00 | 617.40 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-07-10 12:15:00 | 637.05 | 2024-08-26 09:15:00 | 732.61 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-07-10 12:15:00 | 637.05 | 2024-12-03 09:15:00 | 637.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2024-12-19 15:15:00 | 624.10 | 2024-12-23 09:15:00 | 617.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-12-23 11:15:00 | 622.85 | 2024-12-26 09:15:00 | 620.25 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-12-24 12:15:00 | 625.20 | 2024-12-30 09:15:00 | 617.40 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-12-26 10:15:00 | 628.55 | 2024-12-30 09:15:00 | 620.25 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-01-16 10:15:00 | 653.30 | 2025-01-22 09:15:00 | 620.25 | STOP_HIT | 1.00 | -5.06% |
| BUY | retest2 | 2025-01-29 14:15:00 | 628.45 | 2025-01-30 12:15:00 | 627.15 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-01-30 11:15:00 | 631.05 | 2025-02-03 09:15:00 | 620.25 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-01-30 14:15:00 | 635.20 | 2025-02-03 09:15:00 | 627.15 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-02-05 11:15:00 | 633.25 | 2025-02-05 12:15:00 | 627.15 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-02-06 10:15:00 | 634.55 | 2025-02-06 14:15:00 | 633.10 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-02-06 13:15:00 | 636.30 | 2025-02-07 09:15:00 | 627.15 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-02-07 15:15:00 | 637.10 | 2025-02-10 09:15:00 | 633.10 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-02-10 14:15:00 | 635.90 | 2025-02-10 15:15:00 | 633.10 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-02-13 10:15:00 | 638.55 | 2025-02-13 13:15:00 | 633.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-03-18 15:15:00 | 640.50 | 2025-03-24 10:15:00 | 681.50 | STOP_HIT | 1.00 | 6.40% |
| BUY | retest1 | 2025-04-07 14:15:00 | 666.35 | 2025-05-23 09:15:00 | 766.30 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-04-07 14:15:00 | 666.35 | 2025-07-15 12:15:00 | 753.00 | STOP_HIT | 0.50 | 13.00% |
| BUY | retest2 | 2025-07-14 13:15:00 | 763.25 | 2025-07-17 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-07-16 10:15:00 | 760.75 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-22 13:15:00 | 765.65 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-23 11:15:00 | 762.75 | 2025-07-29 14:15:00 | 756.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-25 12:15:00 | 764.75 | 2025-08-11 09:15:00 | 756.10 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-08-08 13:15:00 | 759.50 | 2025-09-04 13:15:00 | 756.10 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-08-11 13:15:00 | 764.25 | 2025-09-08 09:15:00 | 756.10 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-09-05 11:15:00 | 761.25 | 2025-09-25 12:15:00 | 760.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-09-10 10:15:00 | 774.00 | 2025-09-29 10:15:00 | 760.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-09-25 14:15:00 | 765.05 | 2025-10-03 09:15:00 | 760.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-10-01 15:15:00 | 765.50 | 2025-10-07 09:15:00 | 760.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-10-06 15:15:00 | 763.05 | 2025-11-19 09:15:00 | 759.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-11-12 10:15:00 | 775.00 | 2025-11-26 12:15:00 | 785.45 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2025-11-26 10:15:00 | 779.85 | 2025-11-26 12:15:00 | 785.45 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-12-01 15:15:00 | 766.75 | 2025-12-02 09:15:00 | 759.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-05 12:15:00 | 770.00 | 2025-12-09 09:15:00 | 759.10 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-09 15:15:00 | 762.90 | 2025-12-10 09:15:00 | 759.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-19 13:15:00 | 764.75 | 2025-12-22 13:15:00 | 761.30 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-22 09:15:00 | 766.45 | 2025-12-23 09:15:00 | 759.10 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-01-06 10:15:00 | 772.95 | 2026-01-08 09:15:00 | 761.30 | STOP_HIT | 1.00 | -1.51% |
