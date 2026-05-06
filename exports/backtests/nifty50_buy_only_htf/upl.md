# UPL (UPL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 660.00
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
| ALERT2_SKIP | 2 |
| ALERT3 | 6 |
| PENDING | 27 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 14 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 15
- **Target hits / Stop hits / Partials:** 0 / 21 / 2
- **Avg / median % per leg:** 0.88% / -0.79%
- **Sum % (uncompounded):** 20.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 8 | 72.7% | 0 | 9 | 2 | 3.45% | 38.0% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 7 | 0 | 0.30% | 2.1% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 2 | 2 | 8.97% | 35.9% |
| SELL (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.48% | -17.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.48% | -17.7% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 7 | 0 | 0.30% | 2.1% |
| retest2 (combined) | 16 | 4 | 25.0% | 0 | 14 | 2 | 1.13% | 18.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 15:15:00 | 532.25 | 576.66 | 576.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 522.90 | 573.28 | 575.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 562.75 | 561.57 | 567.91 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 14:15:00 | 567.35 | 561.69 | 567.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 567.35 | 561.69 | 567.88 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-08 10:15:00 | 562.00 | 562.22 | 567.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 558.15 | 562.18 | 567.80 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 568.10 | 552.05 | 560.42 | SL hit qty=1.00 sl=568.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-26 09:15:00 | 558.35 | 553.73 | 560.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 10:15:00 | 556.10 | 553.75 | 560.85 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-03 14:15:00 | 563.00 | 552.79 | 559.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 15:15:00 | 563.10 | 552.89 | 559.06 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-04 09:15:00 | 568.10 | 553.07 | 559.11 | SL hit qty=1.00 sl=568.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-04 09:15:00 | 568.10 | 553.07 | 559.11 | SL hit qty=1.00 sl=568.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-05 09:15:00 | 555.50 | 553.93 | 559.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 10:15:00 | 555.00 | 553.94 | 559.32 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 558.15 | 554.04 | 559.29 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-09 09:15:00 | 552.35 | 554.66 | 559.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 554.15 | 554.66 | 559.33 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-09 13:15:00 | 555.35 | 554.69 | 559.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:15:00 | 556.00 | 554.71 | 559.26 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 559.80 | 554.65 | 559.03 | SL hit qty=1.00 sl=559.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 559.80 | 554.65 | 559.03 | SL hit qty=1.00 sl=559.80 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-11 13:15:00 | 554.45 | 554.81 | 559.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 14:15:00 | 551.95 | 554.78 | 558.99 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-21 09:15:00 | 559.80 | 536.85 | 541.77 | SL hit qty=1.00 sl=559.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-23 13:15:00 | 555.35 | 538.32 | 542.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-23 14:15:00 | 558.45 | 538.52 | 542.19 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-24 09:15:00 | 554.70 | 538.87 | 542.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-24 10:15:00 | 556.45 | 539.05 | 542.40 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-24 12:15:00 | 554.45 | 539.38 | 542.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:15:00 | 553.15 | 539.51 | 542.59 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 551.80 | 539.64 | 542.63 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-27 09:15:00 | 539.15 | 539.75 | 542.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 10:15:00 | 538.90 | 539.75 | 542.64 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-29 15:15:00 | 554.65 | 540.62 | 542.83 | SL hit qty=1.00 sl=554.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 568.10 | 540.94 | 542.98 | SL hit qty=1.00 sl=568.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 559.80 | 540.94 | 542.98 | SL hit qty=1.00 sl=559.80 alert=retest2 |

### Cycle 2 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 607.40 | 545.03 | 544.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 09:15:00 | 626.50 | 546.42 | 545.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 11:15:00 | 611.60 | 612.38 | 590.26 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-03-05 09:15:00 | 623.45 | 612.50 | 590.87 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:15:00 | 628.20 | 612.66 | 591.05 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-11 10:15:00 | 616.40 | 615.92 | 595.64 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:15:00 | 616.00 | 615.92 | 595.74 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-17 10:15:00 | 616.05 | 614.63 | 596.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 11:15:00 | 620.85 | 614.69 | 597.11 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 606.55 | 634.27 | 615.96 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 615.96 | 634.27 | 615.96 | SL hit qty=1.00 sl=615.96 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 615.96 | 634.27 | 615.96 | SL hit qty=1.00 sl=615.96 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 615.96 | 634.27 | 615.96 | SL hit qty=1.00 sl=615.96 alert=retest1 |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 630.75 | 630.36 | 615.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 639.85 | 630.46 | 615.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-30 11:15:00 | 625.00 | 645.40 | 640.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:15:00 | 625.50 | 645.21 | 640.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-22 09:15:00 | 719.32 | 665.11 | 655.12 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-25 09:15:00 | 735.83 | 697.71 | 683.76 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 13:15:00 | 650.95 | 688.13 | 688.14 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 13:15:00 | 650.95 | 688.13 | 688.14 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 650.95 | 688.13 | 688.14 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 719.75 | 685.32 | 685.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 15:15:00 | 723.45 | 685.70 | 685.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 736.25 | 740.97 | 724.37 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-10 09:15:00 | 748.00 | 740.93 | 725.23 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:15:00 | 746.80 | 740.99 | 725.34 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-12 09:15:00 | 749.45 | 741.28 | 726.47 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-12 10:15:00 | 744.90 | 741.32 | 726.57 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-12 14:15:00 | 748.95 | 741.47 | 726.93 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 15:15:00 | 748.35 | 741.54 | 727.04 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-15 10:15:00 | 747.80 | 741.63 | 727.23 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 11:15:00 | 753.35 | 741.75 | 727.36 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-17 14:15:00 | 746.95 | 743.50 | 729.46 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-17 15:15:00 | 743.55 | 743.50 | 729.53 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-18 11:15:00 | 747.25 | 743.50 | 729.74 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 12:15:00 | 747.05 | 743.54 | 729.83 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 763.70 | 775.31 | 757.94 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 757.94 | 775.31 | 757.94 | SL hit qty=1.00 sl=757.94 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 757.94 | 775.31 | 757.94 | SL hit qty=1.00 sl=757.94 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 757.94 | 775.31 | 757.94 | SL hit qty=1.00 sl=757.94 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 757.94 | 775.31 | 757.94 | SL hit qty=1.00 sl=757.94 alert=retest1 |

### Cycle 5 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 678.40 | 744.68 | 744.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 657.25 | 743.55 | 744.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 633.75 | 629.93 | 662.86 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 11:15:00 | 628.00 | 629.92 | 662.69 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-08 12:15:00 | 637.40 | 629.99 | 662.56 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 627.30 | 632.12 | 660.87 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 635.15 | 632.15 | 660.75 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 14:15:00 | 659.50 | 634.07 | 660.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 659.50 | 634.07 | 660.20 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-16 10:15:00 | 656.30 | 634.80 | 660.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 653.65 | 634.99 | 660.14 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-16 15:15:00 | 661.50 | 635.85 | 660.08 | SL hit qty=1.00 sl=661.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-20 14:15:00 | 657.10 | 638.32 | 660.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 15:15:00 | 655.95 | 638.50 | 660.26 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-21 09:15:00 | 661.50 | 638.72 | 660.26 | SL hit qty=1.00 sl=661.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-21 12:15:00 | 654.70 | 639.30 | 660.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:15:00 | 653.65 | 639.45 | 660.20 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 661.50 | 641.90 | 655.45 | SL hit qty=1.00 sl=661.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-08 11:15:00 | 558.15 | 2024-11-22 13:15:00 | 568.10 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-11-26 10:15:00 | 556.10 | 2024-12-04 09:15:00 | 568.10 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-12-03 15:15:00 | 563.10 | 2024-12-04 09:15:00 | 568.10 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-12-05 10:15:00 | 555.00 | 2024-12-11 09:15:00 | 559.80 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-09 10:15:00 | 554.15 | 2024-12-11 09:15:00 | 559.80 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-12-09 14:15:00 | 556.00 | 2025-01-21 09:15:00 | 559.80 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-12-11 14:15:00 | 551.95 | 2025-01-29 15:15:00 | 554.65 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-01-24 13:15:00 | 553.15 | 2025-01-30 09:15:00 | 568.10 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-01-27 10:15:00 | 538.90 | 2025-01-30 09:15:00 | 559.80 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest1 | 2025-03-05 10:15:00 | 628.20 | 2025-04-07 09:15:00 | 615.96 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest1 | 2025-03-11 11:15:00 | 616.00 | 2025-04-07 09:15:00 | 615.96 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest1 | 2025-03-17 11:15:00 | 620.85 | 2025-04-07 09:15:00 | 615.96 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-04-11 10:15:00 | 639.85 | 2025-07-22 09:15:00 | 719.32 | PARTIAL | 0.50 | 12.42% |
| BUY | retest2 | 2025-05-30 12:15:00 | 625.50 | 2025-08-25 09:15:00 | 735.83 | PARTIAL | 0.50 | 17.64% |
| BUY | retest2 | 2025-04-11 10:15:00 | 639.85 | 2025-09-30 13:15:00 | 650.95 | STOP_HIT | 0.50 | 1.73% |
| BUY | retest2 | 2025-05-30 12:15:00 | 625.50 | 2025-09-30 13:15:00 | 650.95 | STOP_HIT | 0.50 | 4.07% |
| BUY | retest1 | 2025-12-10 10:15:00 | 746.80 | 2026-01-20 09:15:00 | 757.94 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest1 | 2025-12-12 15:15:00 | 748.35 | 2026-01-20 09:15:00 | 757.94 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest1 | 2025-12-15 11:15:00 | 753.35 | 2026-01-20 09:15:00 | 757.94 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest1 | 2025-12-18 12:15:00 | 747.05 | 2026-01-20 09:15:00 | 757.94 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2026-04-16 11:15:00 | 653.65 | 2026-04-16 15:15:00 | 661.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-04-20 15:15:00 | 655.95 | 2026-04-21 09:15:00 | 661.50 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-04-21 13:15:00 | 653.65 | 2026-05-06 14:15:00 | 661.50 | STOP_HIT | 1.00 | -1.20% |
