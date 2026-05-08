# BEL (BEL)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 439.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 4 |
| PENDING | 15 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 4 / 7
- **Target hits / Stop hits / Partials:** 4 / 7 / 0
- **Avg / median % per leg:** 1.56% / -2.49%
- **Sum % (uncompounded):** 17.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 4 | 7 | 0 | 1.56% | 17.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 4 | 36.4% | 4 | 7 | 0 | 1.56% | 17.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 4 | 36.4% | 4 | 7 | 0 | 1.56% | 17.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 11:15:00 | 383.65 | 401.90 | 401.91 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 413.30 | 401.35 | 401.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 10:15:00 | 414.35 | 401.70 | 401.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-21 09:15:00 | 404.25 | 407.55 | 404.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 404.25 | 407.55 | 404.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 404.25 | 407.55 | 404.99 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-22 09:15:00 | 412.25 | 407.30 | 404.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 411.80 | 407.34 | 404.99 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-23 15:15:00 | 412.30 | 408.06 | 405.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 416.60 | 408.15 | 405.55 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 5400m) |
| Target hit | 2026-01-28 14:15:00 | 452.98 | 410.07 | 406.68 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 458.26 | 435.20 | 425.66 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 412.00 | 438.17 | 431.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-24 10:15:00 | 409.10 | 437.88 | 431.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-24 11:15:00 | 409.60 | 437.60 | 431.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 12:15:00 | 415.90 | 437.39 | 431.68 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 410.45 | 434.81 | 430.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 413.40 | 434.59 | 430.60 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 438.55 | 429.77 | 428.62 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-09 11:15:00 | 442.60 | 430.00 | 428.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 441.80 | 430.12 | 428.81 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 443.35 | 430.55 | 429.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 442.40 | 430.66 | 429.12 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 11:15:00 | 444.85 | 431.54 | 429.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 444.80 | 431.68 | 429.71 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2026-04-16 09:15:00 | 454.74 | 433.21 | 430.60 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-17 09:15:00 | 457.49 | 434.66 | 431.42 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-24 13:15:00 | 442.25 | 439.42 | 434.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 14:15:00 | 444.50 | 439.47 | 434.63 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 435.35 | 439.41 | 434.72 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-28 09:15:00 | 439.95 | 439.32 | 434.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 10:15:00 | 437.20 | 439.29 | 434.78 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-29 09:15:00 | 438.70 | 439.12 | 434.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 438.85 | 439.12 | 434.84 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 437.65 | 439.08 | 434.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 437.50 | 439.07 | 434.92 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 438.94 | 434.88 | SL hit (close<static) qty=1.00 sl=427.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 438.94 | 434.88 | SL hit (close<static) qty=1.00 sl=427.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 438.94 | 434.88 | SL hit (close<static) qty=1.00 sl=427.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 438.94 | 434.88 | SL hit (close<static) qty=1.00 sl=427.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 438.94 | 434.88 | SL hit (close<static) qty=1.00 sl=433.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 438.94 | 434.88 | SL hit (close<static) qty=1.00 sl=433.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 426.60 | 438.94 | 434.88 | SL hit (close<static) qty=1.00 sl=433.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 438.00 | 437.78 | 434.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 437.85 | 437.78 | 434.68 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 434.30 | 437.73 | 434.75 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-05-07 15:15:00 | 441.00 | 437.77 | 434.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 442.60 | 437.82 | 434.90 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-05-08 14:15:00 | 439.80 | 437.91 | 435.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 15:15:00 | 439.65 | 437.93 | 435.04 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-22 10:15:00 | 411.80 | 2026-01-28 14:15:00 | 452.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-27 09:15:00 | 416.60 | 2026-03-04 09:15:00 | 458.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 12:15:00 | 415.90 | 2026-04-16 09:15:00 | 454.74 | TARGET_HIT | 1.00 | 9.34% |
| BUY | retest2 | 2026-03-27 11:15:00 | 413.40 | 2026-04-17 09:15:00 | 457.49 | TARGET_HIT | 1.00 | 10.67% |
| BUY | retest2 | 2026-04-09 12:15:00 | 441.80 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2026-04-10 10:15:00 | 442.40 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2026-04-13 12:15:00 | 444.80 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2026-04-24 14:15:00 | 444.50 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2026-04-28 10:15:00 | 437.20 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-04-29 10:15:00 | 438.85 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2026-04-29 15:15:00 | 437.50 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -2.49% |
