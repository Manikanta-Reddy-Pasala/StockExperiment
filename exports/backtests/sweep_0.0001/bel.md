# BEL (BEL)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 439.50
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
| ALERT3 | 3 |
| PENDING | 14 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 0 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 0
- **Avg / median % per leg:** -2.19% / -1.57%
- **Sum % (uncompounded):** -19.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.19% | -19.7% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.67% | -6.7% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.59% | -13.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.67% | -6.7% |
| retest2 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.59% | -13.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 409.40 | 386.40 | 386.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 412.00 | 395.17 | 391.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 407.00 | 407.56 | 400.97 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-30 15:15:00 | 410.20 | 407.67 | 401.28 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:15:00 | 414.00 | 407.73 | 401.34 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-10-31 12:15:00 | 420.50 | 407.91 | 401.53 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 13:15:00 | 424.70 | 408.08 | 401.64 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-06 12:15:00 | 410.60 | 409.85 | 403.20 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 13:15:00 | 410.35 | 409.85 | 403.23 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-07 10:15:00 | 411.35 | 409.82 | 403.35 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 11:15:00 | 411.25 | 409.84 | 403.39 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 408.05 | 416.02 | 409.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.02 | 409.11 | SL hit (close<ema400) qty=1.00 sl=409.11 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.02 | 409.11 | SL hit (close<ema400) qty=1.00 sl=409.11 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.02 | 409.11 | SL hit (close<ema400) qty=1.00 sl=409.11 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.02 | 409.11 | SL hit (close<ema400) qty=1.00 sl=409.11 alert=retest1 |
| Cross detected — sustain check pending | 2025-11-25 11:15:00 | 410.95 | 415.35 | 409.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 411.25 | 415.31 | 409.05 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-26 09:15:00 | 411.85 | 415.14 | 409.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 411.90 | 415.11 | 409.10 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-28 14:15:00 | 411.55 | 414.85 | 409.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 411.55 | 414.81 | 409.50 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 404.80 | 414.59 | 409.79 | SL hit (close<static) qty=1.00 sl=407.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 404.80 | 414.59 | 409.79 | SL hit (close<static) qty=1.00 sl=407.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 404.80 | 414.59 | 409.79 | SL hit (close<static) qty=1.00 sl=407.60 alert=retest2 |
| CROSSOVER_SKIP | 2025-12-12 14:15:00 | 390.00 | 406.30 | 406.34 | min_gap filter: gap=0.009% < 0.010% |
| TREND_RESET | 2025-12-12 14:15:00 | 390.00 | 406.30 | 406.34 | EMA inversion without crossover edge (EMA200=406.30 EMA400=406.34) — end cycle |

### Cycle 2 — BUY (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 12:15:00 | 419.20 | 404.08 | 404.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 423.80 | 408.70 | 406.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 440.90 | 444.04 | 433.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 432.80 | 443.62 | 433.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 432.80 | 443.62 | 433.40 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-17 13:15:00 | 440.20 | 442.24 | 433.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 439.50 | 442.21 | 433.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 410.20 | 440.55 | 433.38 | SL hit (close<static) qty=1.00 sl=425.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 438.40 | 429.85 | 429.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 440.30 | 429.96 | 429.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-28 09:15:00 | 439.95 | 439.99 | 435.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-28 10:15:00 | 437.25 | 439.96 | 435.47 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-29 09:15:00 | 438.70 | 439.74 | 435.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 438.85 | 439.74 | 435.51 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 438.00 | 438.24 | 435.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 437.90 | 438.24 | 435.26 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 436.40 | 438.22 | 435.28 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-05-06 14:15:00 | 438.15 | 438.21 | 435.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 15:15:00 | 438.10 | 438.21 | 435.32 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 434.25 | 438.17 | 435.32 | SL hit (close<static) qty=1.00 sl=435.15 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-07 11:15:00 | 437.85 | 438.15 | 435.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:15:00 | 438.75 | 438.15 | 435.35 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-31 09:15:00 | 414.00 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-10-31 13:15:00 | 424.70 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest1 | 2025-11-06 13:15:00 | 410.35 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2025-11-07 11:15:00 | 411.25 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-25 12:15:00 | 411.25 | 2025-12-03 10:15:00 | 404.80 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-11-26 10:15:00 | 411.90 | 2025-12-03 10:15:00 | 404.80 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-28 15:15:00 | 411.55 | 2025-12-03 10:15:00 | 404.80 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-03-17 14:15:00 | 439.50 | 2026-03-23 09:15:00 | 410.20 | STOP_HIT | 1.00 | -6.67% |
| BUY | retest2 | 2026-04-09 10:15:00 | 440.30 | 2026-05-07 09:15:00 | 434.25 | STOP_HIT | 1.00 | -1.37% |
