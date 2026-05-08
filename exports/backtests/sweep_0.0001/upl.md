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
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 7 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 0
- **Target hits / Stop hits / Partials:** 0 / 4 / 4
- **Avg / median % per leg:** 4.01% / 4.12%
- **Sum % (uncompounded):** 32.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.01% | 32.1% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.01% | 32.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.01% | 32.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-29 15:15:00)

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
| CROSSOVER_SKIP | 2026-02-01 13:15:00 | 680.75 | 744.85 | 744.89 | min_gap filter: gap=0.005% < 0.010% |
| TREND_RESET | 2026-02-01 13:15:00 | 680.75 | 744.85 | 744.89 | EMA inversion without crossover edge (EMA200=744.85 EMA400=744.89) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-10 10:15:00 | 746.80 | 2025-12-23 09:15:00 | 784.14 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-15 11:15:00 | 753.35 | 2025-12-23 09:15:00 | 784.40 | PARTIAL | 0.50 | 4.12% |
| BUY | retest1 | 2025-12-18 12:15:00 | 747.05 | 2025-12-30 14:15:00 | 791.02 | PARTIAL | 0.50 | 5.89% |
| BUY | retest1 | 2025-12-19 14:15:00 | 752.20 | 2025-12-30 14:15:00 | 789.81 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-10 10:15:00 | 746.80 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 3.43% |
| BUY | retest1 | 2025-12-15 11:15:00 | 753.35 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 2.54% |
| BUY | retest1 | 2025-12-18 12:15:00 | 747.05 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 3.40% |
| BUY | retest1 | 2025-12-19 14:15:00 | 752.20 | 2026-01-09 14:15:00 | 772.45 | STOP_HIT | 0.50 | 2.69% |
