# SHRIRAMFIN (SHRIRAMFIN)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1003.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 4 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 2.65% / 4.27%
- **Sum % (uncompounded):** 13.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 2.65% | 13.2% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.79% | 19.1% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.90% | -5.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.79% | 19.1% |
| retest2 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.90% | -5.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 670.85 | 631.06 | 630.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 14:15:00 | 672.95 | 636.50 | 633.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 964.50 | 974.04 | 916.36 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 1004.60 | 973.18 | 917.90 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 10:15:00 | 1008.00 | 973.53 | 918.35 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-05 15:15:00 | 994.00 | 977.64 | 925.52 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 09:15:00 | 987.80 | 977.74 | 925.83 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-02-06 14:15:00 | 1004.40 | 978.33 | 927.41 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 15:15:00 | 1001.00 | 978.55 | 927.78 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 13:15:00 | 1051.05 | 981.37 | 930.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 14:15:00 | 1058.40 | 982.19 | 931.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-02-26 09:15:00 | 1101.10 | 1031.31 | 978.57 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1000.00 | 1038.87 | 987.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 1000.00 | 1038.87 | 987.87 | SL hit (close<ema200) qty=0.50 sl=1038.87 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 1045.50 | 1032.23 | 991.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 1046.60 | 1032.37 | 991.51 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 984.80 | 1031.09 | 996.10 | SL hit (close<static) qty=1.00 sl=987.40 alert=retest2 |
| CROSSOVER_SKIP | 2026-04-02 11:15:00 | 870.90 | 975.53 | 975.59 | min_gap filter: gap=0.007% < 0.010% |
| TREND_RESET | 2026-04-02 11:15:00 | 870.90 | 975.53 | 975.59 | EMA inversion without crossover edge (EMA200=975.53 EMA400=975.59) — end cycle |

### Cycle 2 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1027.50 | 974.98 | 974.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1029.25 | 978.48 | 976.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 993.30 | 998.77 | 988.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 1021.10 | 999.02 | 988.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1021.10 | 999.02 | 988.73 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-03 10:15:00 | 1008.00 | 2026-02-09 13:15:00 | 1051.05 | PARTIAL | 0.50 | 4.27% |
| BUY | retest1 | 2026-02-06 15:15:00 | 1001.00 | 2026-02-09 14:15:00 | 1058.40 | PARTIAL | 0.50 | 5.73% |
| BUY | retest1 | 2026-02-03 10:15:00 | 1008.00 | 2026-02-26 09:15:00 | 1101.10 | TARGET_HIT | 0.50 | 9.24% |
| BUY | retest1 | 2026-02-06 15:15:00 | 1001.00 | 2026-03-04 09:15:00 | 1000.00 | STOP_HIT | 0.50 | -0.10% |
| BUY | retest2 | 2026-03-10 11:15:00 | 1046.60 | 2026-03-16 10:15:00 | 984.80 | STOP_HIT | 1.00 | -5.90% |
