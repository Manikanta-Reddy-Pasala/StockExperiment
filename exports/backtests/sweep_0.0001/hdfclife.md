# HDFCLIFE (HDFCLIFE)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 619.60
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
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 4 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -2.33% / -2.00%
- **Sum % (uncompounded):** -9.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.33% | -9.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.33% | -9.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.33% | -9.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 789.20 | 765.37 | 765.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 793.20 | 766.82 | 766.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 09:15:00 | 769.10 | 773.81 | 770.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 769.10 | 773.81 | 770.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 769.10 | 773.81 | 770.14 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-01 10:15:00 | 778.50 | 773.91 | 770.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 784.00 | 774.01 | 770.41 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-04 09:15:00 | 787.60 | 775.19 | 771.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:15:00 | 779.10 | 775.23 | 771.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 763.55 | 775.11 | 771.36 | SL hit (close<static) qty=1.00 sl=766.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 763.55 | 775.11 | 771.36 | SL hit (close<static) qty=1.00 sl=766.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-12 12:15:00 | 780.00 | 771.82 | 770.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:15:00 | 779.60 | 771.90 | 770.22 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 766.25 | 772.39 | 770.64 | SL hit (close<static) qty=1.00 sl=766.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-18 10:15:00 | 783.65 | 772.37 | 770.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 11:15:00 | 788.30 | 772.53 | 770.76 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 769.05 | 774.86 | 772.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 764.55 | 774.57 | 772.18 | SL hit (close<static) qty=1.00 sl=766.65 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-03 12:15:00 | 760.20 | 770.17 | 770.21 | min_gap filter: gap=0.005% < 0.010% |
| TREND_RESET | 2025-10-03 12:15:00 | 760.20 | 770.17 | 770.21 | EMA inversion without crossover edge (EMA200=770.17 EMA400=770.21) — end cycle |
| CROSSOVER_SKIP | 2025-11-27 10:15:00 | 782.05 | 760.78 | 760.75 | min_gap filter: gap=0.004% < 0.010% |
| CROSSOVER_SKIP | 2025-12-04 10:15:00 | 750.95 | 760.80 | 760.83 | min_gap filter: gap=0.005% < 0.010% |
| CROSSOVER_SKIP | 2025-12-05 14:15:00 | 768.40 | 760.90 | 760.88 | min_gap filter: gap=0.003% < 0.010% |
| CROSSOVER_SKIP | 2025-12-29 10:15:00 | 743.55 | 761.59 | 761.62 | min_gap filter: gap=0.004% < 0.010% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-01 11:15:00 | 784.00 | 2025-09-04 11:15:00 | 763.55 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-09-04 10:15:00 | 779.10 | 2025-09-04 11:15:00 | 763.55 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-09-12 13:15:00 | 779.60 | 2025-09-17 12:15:00 | 766.25 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-09-18 11:15:00 | 788.30 | 2025-09-25 09:15:00 | 764.55 | STOP_HIT | 1.00 | -3.01% |
