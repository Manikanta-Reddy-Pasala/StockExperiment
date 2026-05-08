# BPCL (BPCL)

## Backtest Summary

- **Window:** 2025-11-10 09:15:00 → 2026-05-08 15:15:00 (854 bars)
- **Last close:** 303.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty @ 15% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 15%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 7 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -2.11% / -1.94%
- **Sum % (uncompounded):** -6.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.11% | -6.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.28% | -4.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.28% | -4.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 10:15:00 | 383.20 | 365.20 | 365.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 386.55 | 365.98 | 365.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 09:15:00 | 370.35 | 370.91 | 368.32 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-16 14:15:00 | 373.80 | 370.95 | 368.40 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 15:15:00 | 374.45 | 370.98 | 368.43 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 367.85 | 370.94 | 368.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 367.85 | 370.94 | 368.44 | SL hit (close<ema400) qty=1.00 sl=368.44 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-17 12:15:00 | 371.70 | 370.93 | 368.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:15:00 | 373.15 | 370.95 | 368.48 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 365.90 | 371.53 | 368.98 | SL hit (close<static) qty=1.00 sl=366.55 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-23 10:15:00 | 375.05 | 371.09 | 368.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 372.20 | 371.10 | 368.88 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 362.45 | 373.35 | 370.51 | SL hit (close<static) qty=1.00 sl=366.55 alert=retest2 |

### Cycle 2 — SELL (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 14:15:00 | 330.75 | 368.04 | 368.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 09:15:00 | 328.00 | 367.27 | 367.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.40 | 307.35 | 326.77 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-24 09:15:00 | 301.90 | 309.29 | 324.00 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:15:00 | 305.65 | 309.26 | 323.91 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 303.75 | 309.20 | 322.17 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 15:15:00 | 304.60 | 309.16 | 322.08 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-07 11:15:00 | 305.65 | 307.49 | 319.31 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-07 12:15:00 | 308.45 | 307.50 | 319.25 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-08 09:15:00 | 303.00 | 307.48 | 319.01 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:15:00 | 305.20 | 307.46 | 318.94 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 15:15:00 | 374.45 | 2026-02-17 10:15:00 | 367.85 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-02-17 13:15:00 | 373.15 | 2026-02-19 15:15:00 | 365.90 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-02-23 11:15:00 | 372.20 | 2026-03-04 09:15:00 | 362.45 | STOP_HIT | 1.00 | -2.62% |
