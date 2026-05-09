# NSE:POWERGRID-EQ (NSE:POWERGRID-EQ)

## Backtest Summary

- **Window:** 2024-04-04 09:15:00 → 2026-05-08 15:15:00 (3612 bars)
- **Last close:** 313.90
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
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 1.72% / -0.44%
- **Sum % (uncompounded):** 8.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.72% | 8.6% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.35% | -1.4% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.35% | -1.4% |
| retest2 (combined) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.48 | 269.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.80 | 269.95 | 269.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.75 | 290.18 | 282.80 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 293.25 | 290.22 | 282.89 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-09 11:30:00 | 292.50 | 290.22 | 282.89 | ENTRY1 sustain failed after 15m (15m): close back below level |
| Cross detected — sustain check pending | 2026-03-09 13:15:00 | 293.00 | 290.26 | 282.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 13:30:00 | 293.20 | 290.26 | 282.99 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 15m on 15m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 307.86 | 292.36 | 284.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 291.90 | 292.94 | 285.50 | SL hit (close<ema200) qty=0.50 sl=292.94 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-16 11:15:00 | 296.10 | 292.98 | 285.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 11:30:00 | 296.15 | 292.98 | 285.56 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 15m on 15m) |
| Cross detected — sustain check pending | 2026-03-27 11:15:00 | 295.60 | 295.23 | 288.66 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:30:00 | 294.60 | 295.23 | 288.66 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 15m on 15m) |
| Cross detected — sustain check pending | 2026-04-01 15:15:00 | 293.30 | 295.23 | 289.23 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-02 09:15:00 | 285.65 | 295.23 | 289.23 | ENTRY1 sustain failed after 1080m (15m): close back below level |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-02 15:15:00 | 290.95 | 294.71 | 289.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 292.60 | 294.71 | 289.17 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 5400m on 15m) |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 290.65 | 294.51 | 289.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-06 12:30:00 | 289.55 | 294.51 | 289.18 | ENTRY2 sustain failed after 15m (15m): close back below level |
| Target hit | 2026-04-20 10:15:00 | 321.86 | 299.38 | 293.20 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-03-09 13:30:00 | 293.20 | 2026-03-13 09:15:00 | 307.86 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-03-09 13:30:00 | 293.20 | 2026-03-16 10:15:00 | 291.90 | STOP_HIT | 0.50 | -0.44% |
| BUY | retest1 | 2026-03-16 11:30:00 | 296.15 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest1 | 2026-03-27 11:30:00 | 294.60 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2026-04-06 09:15:00 | 292.60 | 2026-04-20 10:15:00 | 321.86 | TARGET_HIT | 1.00 | 10.00% |
