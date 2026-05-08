# POWERGRID (POWERGRID)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 313.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty @ 15% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 15%, trail SL → EMA200
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
| PENDING_CANCEL | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -3.25% / -3.24%
- **Sum % (uncompounded):** -9.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.25% | -9.7% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.25% | -9.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.25% | -9.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 291.25 | 269.52 | 269.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 292.80 | 269.76 | 269.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.80 | 290.27 | 282.94 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 293.25 | 290.31 | 283.04 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-09 12:15:00 | 291.90 | 290.33 | 283.08 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-09 13:15:00 | 293.05 | 290.36 | 283.13 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 14:15:00 | 295.55 | 290.41 | 283.19 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-16 11:15:00 | 296.05 | 293.05 | 285.68 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 12:15:00 | 297.00 | 293.09 | 285.74 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 11:15:00 | 295.60 | 295.29 | 288.77 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 12:15:00 | 296.25 | 295.30 | 288.80 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-01 15:15:00 | 293.30 | 295.28 | 289.33 | ENTRY1 cross detected — sustain check pending (15m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 286.65 | 295.19 | 289.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.65 | 295.19 | 289.31 | SL hit (close<ema400) qty=1.00 sl=289.31 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.65 | 295.19 | 289.31 | SL hit (close<ema400) qty=1.00 sl=289.31 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.65 | 295.19 | 289.31 | SL hit (close<ema400) qty=1.00 sl=289.31 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-06 09:15:00 | 291.75 | 294.72 | 289.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-06 10:15:00 | 287.95 | 294.65 | 289.26 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 290.65 | 294.55 | 289.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 293.45 | 294.54 | 289.29 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-03-09 14:15:00 | 295.55 | 2026-04-02 09:15:00 | 286.65 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest1 | 2026-03-16 12:15:00 | 297.00 | 2026-04-02 09:15:00 | 286.65 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest1 | 2026-03-27 12:15:00 | 296.25 | 2026-04-02 09:15:00 | 286.65 | STOP_HIT | 1.00 | -3.24% |
