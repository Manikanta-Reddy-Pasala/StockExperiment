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
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 16 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 5 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 6
- **Target hits / Stop hits / Partials:** 0 / 12 / 2
- **Avg / median % per leg:** 4.03% / 1.28%
- **Sum % (uncompounded):** 56.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 8 | 72.7% | 0 | 9 | 2 | 5.43% | 59.7% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 7 | 0 | 0.30% | 2.1% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 2 | 2 | 14.40% | 57.6% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.08% | -3.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.08% | -3.2% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 7 | 0 | 0.30% | 2.1% |
| retest2 (combined) | 7 | 4 | 57.1% | 0 | 5 | 2 | 7.77% | 54.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-01-31 14:15:00)

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
| CROSSOVER_SKIP | 2025-09-30 13:15:00 | 650.95 | 688.13 | 688.14 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-10-30 14:15:00 | 719.75 | 685.32 | 685.27 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 14:15:00 | 719.75 | 685.32 | 685.27 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-30 14:15:00)

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

### Cycle 3 — SELL (started 2026-02-02 12:15:00)

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
| BUY | retest1 | 2025-03-05 10:15:00 | 628.20 | 2025-04-07 09:15:00 | 615.96 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest1 | 2025-03-11 11:15:00 | 616.00 | 2025-04-07 09:15:00 | 615.96 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest1 | 2025-03-17 11:15:00 | 620.85 | 2025-04-07 09:15:00 | 615.96 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-04-11 10:15:00 | 639.85 | 2025-07-22 09:15:00 | 719.32 | PARTIAL | 0.50 | 12.42% |
| BUY | retest2 | 2025-05-30 12:15:00 | 625.50 | 2025-08-25 09:15:00 | 735.83 | PARTIAL | 0.50 | 17.64% |
| BUY | retest2 | 2025-04-11 10:15:00 | 639.85 | 2025-10-30 14:15:00 | 719.75 | STOP_HIT | 0.50 | 12.49% |
| BUY | retest2 | 2025-05-30 12:15:00 | 625.50 | 2025-10-30 14:15:00 | 719.75 | STOP_HIT | 0.50 | 15.07% |
| BUY | retest1 | 2025-12-10 10:15:00 | 746.80 | 2026-01-20 09:15:00 | 757.94 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest1 | 2025-12-12 15:15:00 | 748.35 | 2026-01-20 09:15:00 | 757.94 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest1 | 2025-12-15 11:15:00 | 753.35 | 2026-01-20 09:15:00 | 757.94 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest1 | 2025-12-18 12:15:00 | 747.05 | 2026-01-20 09:15:00 | 757.94 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2026-04-16 11:15:00 | 653.65 | 2026-04-16 15:15:00 | 661.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-04-20 15:15:00 | 655.95 | 2026-04-21 09:15:00 | 661.50 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-04-21 13:15:00 | 653.65 | 2026-05-06 14:15:00 | 661.50 | STOP_HIT | 1.00 | -1.20% |
