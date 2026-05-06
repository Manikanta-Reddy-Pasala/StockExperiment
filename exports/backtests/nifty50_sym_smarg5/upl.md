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
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 11 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 3
- **Target hits / Stop hits / Partials:** 0 / 9 / 2
- **Avg / median % per leg:** 5.43% / 1.46%
- **Sum % (uncompounded):** 59.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 8 | 72.7% | 0 | 9 | 2 | 5.43% | 59.7% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 7 | 0 | 0.30% | 2.1% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 2 | 2 | 14.40% | 57.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 7 | 0 | 0.30% | 2.1% |
| retest2 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 14.40% | 57.6% |

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
| CROSSOVER_SKIP | 2026-02-02 12:15:00 | 678.40 | 744.68 | 744.93 | HTF filter: close above htf_sma |


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
