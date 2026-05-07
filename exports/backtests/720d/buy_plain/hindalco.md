# HINDALCO (HINDALCO)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1057.00
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
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 10 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 7 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 2
- **Target hits / Stop hits / Partials:** 4 / 4 / 5
- **Avg / median % per leg:** 15.79% / 15.21%
- **Sum % (uncompounded):** 205.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 11 | 84.6% | 4 | 4 | 5 | 15.79% | 205.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -6.50% | -6.5% |
| BUY @ 3rd Alert (retest2) | 12 | 11 | 91.7% | 4 | 3 | 5 | 17.64% | 211.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -6.50% | -6.5% |
| retest2 (combined) | 12 | 11 | 91.7% | 4 | 3 | 5 | 17.64% | 211.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 12:15:00 | 700.55 | 668.46 | 668.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 704.50 | 670.53 | 669.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 671.70 | 673.00 | 670.79 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 671.70 | 673.00 | 670.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 671.70 | 673.00 | 670.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-09-12 14:15:00 | 676.95 | 668.34 | 668.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 674.60 | 668.40 | 668.74 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-16 09:15:00 | 679.90 | 669.07 | 669.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2024-09-16 10:15:00 | 678.20 | 669.16 | 669.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 10:15:00 | 678.20 | 669.16 | 669.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 11:15:00 | 681.00 | 669.28 | 669.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 14:15:00 | 719.10 | 722.03 | 704.99 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-10-23 09:15:00 | 730.00 | 722.09 | 705.19 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 10:15:00 | 728.15 | 722.15 | 705.30 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 680.85 | 721.65 | 705.55 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 680.85 | 721.65 | 705.55 | SL hit (close<ema400) qty=1.00 sl=705.55 alert=retest1 |

### Cycle 3 — BUY (started 2025-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 15:15:00 | 680.45 | 622.89 | 622.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 691.25 | 623.57 | 623.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 668.05 | 668.42 | 651.55 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 652.85 | 667.37 | 652.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 652.85 | 667.37 | 652.07 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-15 12:15:00 | 660.20 | 634.03 | 635.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:15:00 | 662.80 | 634.31 | 635.21 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 657.50 | 636.11 | 636.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 657.50 | 636.11 | 636.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 659.50 | 637.15 | 636.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.66 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.66 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-09 09:15:00 | 651.95 | 642.78 | 640.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 651.55 | 642.86 | 640.86 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-09 14:15:00 | 651.05 | 643.19 | 641.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-09 15:15:00 | 650.00 | 643.26 | 641.11 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-10 09:15:00 | 656.85 | 643.39 | 641.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 10:15:00 | 658.25 | 643.54 | 641.28 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-12 14:15:00 | 651.65 | 645.70 | 642.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 15:15:00 | 651.00 | 645.75 | 642.65 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-20 10:15:00 | 653.85 | 645.25 | 642.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 652.20 | 645.32 | 642.94 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 644.20 | 645.38 | 643.03 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-23 11:15:00 | 657.40 | 645.49 | 643.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:15:00 | 664.70 | 645.68 | 643.22 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-08 09:15:00 | 749.28 | 704.99 | 690.41 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-08 09:15:00 | 748.65 | 704.99 | 690.41 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-09 09:15:00 | 750.03 | 707.51 | 692.19 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-12 13:15:00 | 756.99 | 715.75 | 698.36 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-25 09:15:00 | 764.40 | 730.38 | 711.27 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit — 30% from entry | 2025-10-28 09:15:00 | 847.01 | 770.78 | 746.11 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-10-28 09:15:00 | 855.73 | 770.78 | 746.11 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-10-28 09:15:00 | 846.30 | 770.78 | 746.11 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-10-28 09:15:00 | 847.86 | 770.78 | 746.11 | Target hit (30%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 777.40 | 796.94 | 765.32 | SL hit (close<ema200) qty=0.50 sl=796.94 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-09-12 15:15:00 | 674.60 | 2024-09-16 10:15:00 | 678.20 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest1 | 2024-10-23 10:15:00 | 728.15 | 2024-10-24 09:15:00 | 680.85 | STOP_HIT | 1.00 | -6.50% |
| BUY | retest2 | 2025-05-15 13:15:00 | 662.80 | 2025-05-16 14:15:00 | 657.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-06-09 10:15:00 | 651.55 | 2025-09-08 09:15:00 | 749.28 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-06-10 10:15:00 | 658.25 | 2025-09-08 09:15:00 | 748.65 | PARTIAL | 0.50 | 13.73% |
| BUY | retest2 | 2025-06-12 15:15:00 | 651.00 | 2025-09-09 09:15:00 | 750.03 | PARTIAL | 0.50 | 15.21% |
| BUY | retest2 | 2025-06-20 11:15:00 | 652.20 | 2025-09-12 13:15:00 | 756.99 | PARTIAL | 0.50 | 16.07% |
| BUY | retest2 | 2025-06-23 12:15:00 | 664.70 | 2025-09-25 09:15:00 | 764.40 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-06-09 10:15:00 | 651.55 | 2025-10-28 09:15:00 | 847.01 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-06-10 10:15:00 | 658.25 | 2025-10-28 09:15:00 | 855.73 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-06-12 15:15:00 | 651.00 | 2025-10-28 09:15:00 | 846.30 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-06-20 11:15:00 | 652.20 | 2025-10-28 09:15:00 | 847.86 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-06-23 12:15:00 | 664.70 | 2025-11-06 09:15:00 | 777.40 | STOP_HIT | 0.50 | 16.96% |
