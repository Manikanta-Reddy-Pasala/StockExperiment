# UPL (UPL)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 644.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 20 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 11 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 10 / 8
- **Target hits / Stop hits / Partials:** 1 / 12 / 5
- **Avg / median % per leg:** 1.70% / 2.69%
- **Sum % (uncompounded):** 30.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.01% | 32.1% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.01% | 32.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 2 | 20.0% | 1 | 8 | 1 | -0.15% | -1.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 1 | 8 | 1 | -0.15% | -1.5% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.01% | 32.1% |
| retest2 (combined) | 10 | 2 | 20.0% | 1 | 8 | 1 | -0.15% | -1.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 15:15:00 | 679.75 | 684.26 | 684.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 678.30 | 683.97 | 684.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 682.15 | 680.94 | 682.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 682.15 | 680.94 | 682.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 682.15 | 680.94 | 682.43 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-17 13:15:00 | 674.10 | 680.81 | 682.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 14:15:00 | 674.25 | 680.74 | 682.24 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 686.75 | 680.73 | 682.16 | SL hit (close>static) qty=1.00 sl=686.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-23 14:15:00 | 674.55 | 680.72 | 682.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 674.15 | 680.65 | 682.08 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 698.05 | 680.32 | 681.77 | SL hit (close>static) qty=1.00 sl=686.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 718.85 | 683.34 | 683.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 15:15:00 | 723.45 | 685.70 | 684.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 736.25 | 741.00 | 723.95 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-10 09:15:00 | 748.00 | 740.93 | 724.83 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:15:00 | 746.80 | 740.99 | 724.94 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-12 09:15:00 | 749.45 | 741.23 | 726.07 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-12 10:15:00 | 744.90 | 741.27 | 726.16 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-12 14:15:00 | 749.00 | 741.42 | 726.54 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-12 15:15:00 | 746.60 | 741.47 | 726.64 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-15 10:15:00 | 747.80 | 741.57 | 726.84 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 11:15:00 | 753.35 | 741.68 | 726.97 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-17 14:15:00 | 747.15 | 743.45 | 729.10 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-17 15:15:00 | 743.55 | 743.45 | 729.17 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-18 11:15:00 | 747.25 | 743.45 | 729.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 12:15:00 | 747.05 | 743.49 | 729.47 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-19 13:15:00 | 749.15 | 743.53 | 730.05 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 14:15:00 | 752.20 | 743.62 | 730.16 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 784.14 | 746.14 | 732.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 784.40 | 746.14 | 732.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 791.02 | 754.97 | 738.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 789.81 | 754.97 | 738.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 772.45 | 773.27 | 753.35 | SL hit (close<ema200) qty=0.50 sl=773.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 772.45 | 773.27 | 753.35 | SL hit (close<ema200) qty=0.50 sl=773.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 772.45 | 773.27 | 753.35 | SL hit (close<ema200) qty=0.50 sl=773.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 772.45 | 773.27 | 753.35 | SL hit (close<ema200) qty=0.50 sl=773.27 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 763.70 | 775.21 | 757.72 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 680.75 | 744.85 | 744.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 663.85 | 744.04 | 744.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 745.45 | 739.26 | 741.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 10:15:00 | 745.45 | 739.26 | 741.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 745.45 | 739.26 | 741.99 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-03 12:15:00 | 736.80 | 739.27 | 741.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:15:00 | 736.90 | 739.25 | 741.95 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 757.70 | 739.45 | 742.01 | SL hit (close>static) qty=1.00 sl=746.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-06 15:15:00 | 736.95 | 740.84 | 742.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 736.55 | 740.79 | 742.48 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-02-10 10:15:00 | 754.55 | 741.15 | 742.59 | SL hit (close>static) qty=1.00 sl=746.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 732.70 | 742.04 | 742.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:15:00 | 737.15 | 742.00 | 742.90 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 749.65 | 740.59 | 742.07 | SL hit (close>static) qty=1.00 sl=746.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-23 09:15:00 | 657.25 | 742.07 | 742.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 652.55 | 741.18 | 742.27 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 13:15:00 | 619.92 | 707.21 | 723.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-30 09:15:00 | 587.29 | 644.02 | 675.42 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 11:15:00 | 659.40 | 633.16 | 659.77 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-16 10:15:00 | 656.30 | 634.65 | 659.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 653.75 | 634.84 | 659.71 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 670.55 | 636.05 | 659.71 | SL hit (close>static) qty=1.00 sl=659.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-20 13:15:00 | 657.05 | 638.94 | 659.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 14:15:00 | 657.10 | 639.12 | 659.92 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-21 09:15:00 | 660.95 | 639.50 | 659.90 | SL hit (close>static) qty=1.00 sl=659.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-21 12:15:00 | 654.65 | 640.06 | 659.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:15:00 | 653.65 | 640.20 | 659.85 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 660.10 | 642.29 | 655.21 | SL hit (close>static) qty=1.00 sl=659.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-07 09:15:00 | 653.20 | 642.58 | 655.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 10:15:00 | 657.05 | 642.73 | 655.24 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 652.50 | 642.93 | 655.22 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-05-07 14:15:00 | 649.65 | 643.10 | 655.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-07 15:15:00 | 652.00 | 643.19 | 655.16 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-08 09:15:00 | 647.85 | 643.23 | 655.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-08 10:15:00 | 652.30 | 643.32 | 655.11 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-08 11:15:00 | 644.65 | 643.34 | 655.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 12:15:00 | 644.50 | 643.35 | 655.01 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-17 14:15:00 | 674.25 | 2025-10-23 09:15:00 | 686.75 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-10-23 15:15:00 | 674.15 | 2025-10-28 12:15:00 | 698.05 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest1 | 2025-12-10 10:15:00 | 746.80 | 2025-12-23 09:15:00 | 784.14 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-15 11:15:00 | 753.35 | 2025-12-23 09:15:00 | 784.40 | PARTIAL | 0.50 | 4.12% |
| BUY | retest1 | 2025-12-18 12:15:00 | 747.05 | 2025-12-30 14:15:00 | 791.02 | PARTIAL | 0.50 | 5.89% |
| BUY | retest1 | 2025-12-19 14:15:00 | 752.20 | 2025-12-30 14:15:00 | 789.81 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-10 10:15:00 | 746.80 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 3.43% |
| BUY | retest1 | 2025-12-15 11:15:00 | 753.35 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 2.54% |
| BUY | retest1 | 2025-12-18 12:15:00 | 747.05 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 3.40% |
| BUY | retest1 | 2025-12-19 14:15:00 | 752.20 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 2.69% |
| SELL | retest2 | 2026-02-03 13:15:00 | 736.90 | 2026-02-04 09:15:00 | 757.70 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2026-02-09 09:15:00 | 736.55 | 2026-02-10 10:15:00 | 754.55 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-02-13 10:15:00 | 737.15 | 2026-02-18 09:15:00 | 749.65 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-02-23 10:15:00 | 652.55 | 2026-03-02 13:15:00 | 619.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:15:00 | 652.55 | 2026-03-30 09:15:00 | 587.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 11:15:00 | 653.75 | 2026-04-17 09:15:00 | 670.55 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-04-20 14:15:00 | 657.10 | 2026-04-21 09:15:00 | 660.95 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-04-21 13:15:00 | 653.65 | 2026-05-06 14:15:00 | 660.10 | STOP_HIT | 1.00 | -0.99% |
