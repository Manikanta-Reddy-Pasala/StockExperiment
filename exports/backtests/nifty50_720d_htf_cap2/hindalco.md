# HINDALCO (HINDALCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1045.80
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 5 |
| ALERT3 | 7 |
| PENDING | 14 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 2 / 11 / 2
- **Avg / median % per leg:** 5.20% / -2.19%
- **Sum % (uncompounded):** 78.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 2 | 10 | 2 | 5.86% | 82.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 2 | 10 | 2 | 5.86% | 82.1% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.07% | -4.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.07% | -4.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 6 | 40.0% | 2 | 11 | 2 | 5.20% | 78.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 14:15:00 | 568.70 | 539.01 | 539.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 14:15:00 | 570.85 | 540.92 | 539.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 637.10 | 655.58 | 624.35 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 636.00 | 655.12 | 624.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 636.00 | 655.12 | 624.43 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-04 13:15:00 | 650.65 | 655.07 | 624.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 14:15:00 | 653.90 | 655.06 | 624.71 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| CROSSOVER_SKIP | 2024-08-09 13:15:00 | 625.10 | 657.77 | 657.81 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2024-08-19 09:15:00 | 653.80 | 650.01 | 653.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 650.60 | 650.02 | 653.59 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-23 10:15:00 | 692.50 | 656.61 | 656.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-23 10:15:00 | 692.50 | 656.61 | 656.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 10:15:00 | 692.50 | 656.61 | 656.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 694.30 | 658.33 | 657.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 671.70 | 672.76 | 665.90 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 671.70 | 672.76 | 665.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 671.70 | 672.76 | 665.90 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-09-12 14:15:00 | 677.10 | 668.18 | 664.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 676.20 | 668.26 | 664.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-16 09:15:00 | 680.10 | 668.94 | 665.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 10:15:00 | 678.20 | 669.03 | 665.44 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-25 14:15:00 | 678.60 | 717.17 | 702.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 15:15:00 | 678.05 | 716.78 | 702.72 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-04 14:15:00 | 674.70 | 707.63 | 700.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:15:00 | 674.00 | 707.30 | 699.87 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 685.10 | 707.07 | 699.80 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-05 13:15:00 | 696.35 | 706.34 | 699.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:15:00 | 698.35 | 706.26 | 699.57 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 661.40 | 705.79 | 699.62 | SL hit qty=1.00 sl=661.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 661.40 | 705.79 | 699.62 | SL hit qty=1.00 sl=661.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 661.40 | 705.79 | 699.62 | SL hit qty=1.00 sl=661.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 661.40 | 705.79 | 699.62 | SL hit qty=1.00 sl=661.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 675.00 | 705.79 | 699.62 | SL hit qty=1.00 sl=675.00 alert=retest2 |
| CROSSOVER_SKIP | 2024-11-12 13:15:00 | 654.10 | 694.16 | 694.17 | HTF filter: close above htf_sma |

### Cycle 3 — BUY (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 09:15:00 | 691.25 | 623.74 | 623.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 14:15:00 | 691.90 | 626.81 | 625.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 668.00 | 668.49 | 651.82 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 652.85 | 667.42 | 652.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 652.85 | 667.42 | 652.32 | EMA400 retest candle locked |

### Cycle 4 — SELL (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 09:15:00 | 612.20 | 640.19 | 640.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-16 10:15:00 | 606.50 | 638.08 | 639.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 647.70 | 630.88 | 634.53 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 647.70 | 630.88 | 634.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 647.70 | 630.88 | 634.53 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-02 11:15:00 | 627.55 | 630.88 | 634.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-02 12:15:00 | 629.60 | 630.87 | 634.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-08 11:15:00 | 626.80 | 631.49 | 634.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:15:00 | 624.60 | 631.42 | 634.30 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 650.05 | 630.72 | 633.78 | SL hit qty=1.00 sl=650.05 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 656.90 | 636.36 | 636.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 659.50 | 637.20 | 636.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.73 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.73 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-09 09:15:00 | 651.95 | 642.78 | 640.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 651.50 | 642.87 | 640.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-10 09:15:00 | 656.85 | 643.40 | 641.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 10:15:00 | 658.25 | 643.55 | 641.34 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-12 14:15:00 | 651.65 | 645.71 | 642.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 15:15:00 | 651.05 | 645.77 | 642.71 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 636.30 | 645.49 | 642.69 | SL hit qty=1.00 sl=636.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 636.30 | 645.49 | 642.69 | SL hit qty=1.00 sl=636.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 636.30 | 645.49 | 642.69 | SL hit qty=1.00 sl=636.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-20 10:15:00 | 653.85 | 645.27 | 642.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 652.25 | 645.34 | 643.00 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 644.20 | 645.40 | 643.09 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-23 11:15:00 | 657.35 | 645.51 | 643.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:15:00 | 664.70 | 645.70 | 643.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-12 09:15:00 | 750.09 | 714.32 | 697.31 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-30 09:15:00 | 764.41 | 733.76 | 714.95 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2025-10-28 09:15:00 | 847.93 | 770.79 | 746.12 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-12-19 09:15:00 | 864.11 | 818.99 | 798.93 | Target hit (30%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 14:15:00 | 653.90 | 2024-08-23 10:15:00 | 692.50 | STOP_HIT | 1.00 | 5.90% |
| BUY | retest2 | 2024-08-19 10:15:00 | 650.60 | 2024-08-23 10:15:00 | 692.50 | STOP_HIT | 1.00 | 6.44% |
| BUY | retest2 | 2024-09-12 15:15:00 | 676.20 | 2024-11-07 09:15:00 | 661.40 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-09-16 10:15:00 | 678.20 | 2024-11-07 09:15:00 | 661.40 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-10-25 15:15:00 | 678.05 | 2024-11-07 09:15:00 | 661.40 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2024-11-04 15:15:00 | 674.00 | 2024-11-07 09:15:00 | 661.40 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-11-05 14:15:00 | 698.35 | 2024-11-07 09:15:00 | 675.00 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2025-05-08 12:15:00 | 624.60 | 2025-05-12 09:15:00 | 650.05 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2025-06-09 10:15:00 | 651.50 | 2025-06-16 09:15:00 | 636.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-06-10 10:15:00 | 658.25 | 2025-06-16 09:15:00 | 636.30 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2025-06-12 15:15:00 | 651.05 | 2025-06-16 09:15:00 | 636.30 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-06-20 11:15:00 | 652.25 | 2025-09-12 09:15:00 | 750.09 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-06-23 12:15:00 | 664.70 | 2025-09-30 09:15:00 | 764.41 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-06-20 11:15:00 | 652.25 | 2025-10-28 09:15:00 | 847.93 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-06-23 12:15:00 | 664.70 | 2025-12-19 09:15:00 | 864.11 | TARGET_HIT | 0.50 | 30.00% |
