# ONGC (ONGC)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 279.20
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
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 11 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 6 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** 1.26% / -1.20%
- **Sum % (uncompounded):** 8.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 3.05% | 12.2% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.58% | -7.2% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 0 | 1 | 1 | 9.69% | 19.4% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.13% | -3.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.13% | -3.4% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.58% | -7.2% |
| retest2 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | 3.20% | 16.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 237.70 | 244.23 | 244.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 231.90 | 243.93 | 244.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 239.61 | 238.69 | 240.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 240.38 | 238.72 | 240.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 240.38 | 238.72 | 240.85 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-01 10:15:00 | 239.00 | 238.74 | 240.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:15:00 | 238.65 | 238.74 | 240.82 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 241.52 | 238.75 | 240.77 | SL hit (close>static) qty=1.00 sl=241.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-05 09:15:00 | 236.85 | 238.84 | 240.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 238.20 | 238.84 | 240.74 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 241.05 | 238.81 | 240.67 | SL hit (close>static) qty=1.00 sl=241.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-06 10:15:00 | 239.32 | 238.81 | 240.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-06 11:15:00 | 240.89 | 238.84 | 240.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-07 13:15:00 | 238.60 | 239.00 | 240.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 14:15:00 | 239.14 | 239.00 | 240.66 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 241.54 | 238.11 | 239.99 | SL hit (close>static) qty=1.00 sl=241.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 263.56 | 241.37 | 241.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 268.00 | 241.85 | 241.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 09:15:00 | 269.30 | 269.53 | 261.10 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 269.70 | 269.54 | 261.14 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-10 11:15:00 | 268.30 | 269.52 | 261.18 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-10 12:15:00 | 270.05 | 269.53 | 261.22 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 13:15:00 | 269.95 | 269.53 | 261.27 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-10 15:15:00 | 270.20 | 269.54 | 261.35 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-11 09:15:00 | 269.05 | 269.53 | 261.39 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-03-11 11:15:00 | 272.00 | 269.55 | 261.48 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 12:15:00 | 271.55 | 269.57 | 261.53 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 264.90 | 269.33 | 262.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 261.05 | 269.18 | 262.10 | SL hit (close<ema400) qty=1.00 sl=262.10 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 261.05 | 269.18 | 262.10 | SL hit (close<ema400) qty=1.00 sl=262.10 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-19 10:15:00 | 269.50 | 268.12 | 262.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 269.25 | 268.13 | 262.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-23 09:15:00 | 267.95 | 268.13 | 262.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 10:15:00 | 266.90 | 268.12 | 262.61 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 270.10 | 268.04 | 262.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 271.20 | 268.08 | 262.78 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:15:00 | 306.93 | 282.12 | 274.76 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 12:15:00 | 283.75 | 285.22 | 277.49 | SL hit (close<ema200) qty=0.50 sl=285.22 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-01 11:15:00 | 238.65 | 2026-01-02 10:15:00 | 241.52 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-05 10:15:00 | 238.20 | 2026-01-06 09:15:00 | 241.05 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-07 14:15:00 | 239.14 | 2026-01-13 11:15:00 | 241.54 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest1 | 2026-03-10 13:15:00 | 269.95 | 2026-03-16 11:15:00 | 261.05 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest1 | 2026-03-11 12:15:00 | 271.55 | 2026-03-16 11:15:00 | 261.05 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-03-19 11:15:00 | 269.25 | 2026-04-29 10:15:00 | 306.93 | PARTIAL | 0.50 | 14.00% |
| BUY | retest2 | 2026-03-19 11:15:00 | 269.25 | 2026-05-06 12:15:00 | 283.75 | STOP_HIT | 0.50 | 5.39% |
