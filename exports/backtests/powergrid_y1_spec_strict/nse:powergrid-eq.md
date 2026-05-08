# NSE:POWERGRID-EQ (NSE:POWERGRID-EQ)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 313.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 3 |
| PENDING | 18 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 6 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 6
- **Target hits / Stop hits / Partials:** 2 / 10 / 4
- **Avg / median % per leg:** 2.03% / 2.06%
- **Sum % (uncompounded):** 32.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.04% | 10.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.26% | -9.8% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| SELL (all) | 11 | 8 | 72.7% | 0 | 7 | 4 | 2.02% | 22.2% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.22% | 19.3% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.58% | 2.9% |
| retest1 (combined) | 9 | 6 | 66.7% | 0 | 6 | 3 | 1.06% | 9.5% |
| retest2 (combined) | 7 | 4 | 57.1% | 2 | 4 | 1 | 3.27% | 22.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 14:15:00 | 288.00 | 294.62 | 294.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 286.50 | 294.47 | 294.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 291.15 | 291.01 | 292.52 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-18 14:15:00 | 290.30 | 291.00 | 292.51 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-18 15:15:00 | 290.70 | 291.00 | 292.50 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-19 09:15:00 | 289.50 | 290.98 | 292.49 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:15:00 | 289.15 | 290.96 | 292.47 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:15:00 | 274.69 | 288.06 | 290.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 287.15 | 286.56 | 289.59 | SL hit (close>ema200) qty=0.50 sl=286.56 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 288.30 | 285.77 | 288.47 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-25 14:15:00 | 284.00 | 286.95 | 288.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 284.20 | 286.92 | 288.30 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 289.45 | 285.66 | 287.39 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 285.15 | 285.68 | 287.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 285.55 | 285.68 | 287.38 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 289.55 | 285.75 | 287.36 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-08 10:15:00 | 285.55 | 285.98 | 287.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-08 11:15:00 | 285.80 | 285.98 | 287.41 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-08 13:15:00 | 284.70 | 285.96 | 287.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:15:00 | 285.25 | 285.96 | 287.38 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 13:15:00 | 285.40 | 285.91 | 287.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-09 14:15:00 | 286.05 | 285.92 | 287.31 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 290.20 | 285.96 | 287.32 | SL hit (close>static) qty=1.00 sl=288.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-13 13:15:00 | 285.60 | 286.16 | 287.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-13 14:15:00 | 285.75 | 286.15 | 287.34 | ENTRY2 sustain failed after 60m |

### Cycle 2 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 294.40 | 288.17 | 288.15 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 278.50 | 288.17 | 288.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 271.35 | 287.82 | 288.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 268.65 | 268.51 | 274.00 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-23 14:15:00 | 267.10 | 268.49 | 273.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 15:15:00 | 267.00 | 268.48 | 273.87 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-26 12:15:00 | 267.50 | 268.45 | 273.57 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 13:15:00 | 266.80 | 268.44 | 273.54 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 269.10 | 267.18 | 272.06 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-06 11:15:00 | 267.40 | 267.61 | 271.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 268.05 | 267.62 | 271.94 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 254.65 | 266.35 | 270.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 253.65 | 263.47 | 268.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 253.46 | 263.47 | 268.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 261.50 | 261.21 | 266.18 | SL hit (close>ema200) qty=0.50 sl=261.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 261.50 | 261.21 | 266.18 | SL hit (close>ema200) qty=0.50 sl=261.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 261.50 | 261.21 | 266.18 | SL hit (close>ema200) qty=0.50 sl=261.21 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.48 | 269.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.80 | 269.95 | 269.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.75 | 290.18 | 282.80 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 293.25 | 290.22 | 282.89 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-09 12:15:00 | 291.85 | 290.24 | 282.94 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-09 13:15:00 | 293.00 | 290.26 | 282.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 14:15:00 | 295.55 | 290.32 | 283.05 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-16 11:15:00 | 296.10 | 292.98 | 285.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 12:15:00 | 297.00 | 293.02 | 285.61 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 11:15:00 | 295.60 | 295.23 | 288.66 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 12:15:00 | 296.25 | 295.24 | 288.70 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-01 15:15:00 | 293.30 | 295.23 | 289.23 | ENTRY1 cross detected — sustain check pending (15m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-02 15:15:00 | 290.95 | 294.71 | 289.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 291.80 | 294.68 | 289.18 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 5400m) |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 290.65 | 294.51 | 289.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 293.45 | 294.50 | 289.20 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2026-04-20 10:15:00 | 320.98 | 299.38 | 293.20 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-20 10:15:00 | 322.80 | 299.38 | 293.20 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-08-19 10:15:00 | 289.15 | 2025-08-28 14:15:00 | 274.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-08-19 10:15:00 | 289.15 | 2025-09-02 10:15:00 | 287.15 | STOP_HIT | 0.50 | 0.69% |
| SELL | retest2 | 2025-09-25 15:15:00 | 284.20 | 2025-10-03 14:15:00 | 289.45 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-10-06 10:15:00 | 285.55 | 2025-10-07 09:15:00 | 289.55 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-10-08 14:15:00 | 285.25 | 2025-10-10 09:15:00 | 290.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest1 | 2025-12-23 15:15:00 | 267.00 | 2026-01-12 09:15:00 | 254.65 | PARTIAL | 0.50 | 4.63% |
| SELL | retest1 | 2025-12-26 13:15:00 | 266.80 | 2026-01-20 14:15:00 | 253.65 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-06 12:15:00 | 268.05 | 2026-01-20 14:15:00 | 253.46 | PARTIAL | 0.50 | 5.44% |
| SELL | retest1 | 2025-12-23 15:15:00 | 267.00 | 2026-01-29 15:15:00 | 261.50 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2025-12-26 13:15:00 | 266.80 | 2026-01-29 15:15:00 | 261.50 | STOP_HIT | 0.50 | 1.99% |
| SELL | retest2 | 2026-01-06 12:15:00 | 268.05 | 2026-01-29 15:15:00 | 261.50 | STOP_HIT | 0.50 | 2.44% |
| BUY | retest1 | 2026-03-09 14:15:00 | 295.55 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2026-03-16 12:15:00 | 297.00 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest1 | 2026-03-27 12:15:00 | 296.25 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2026-04-06 09:15:00 | 291.80 | 2026-04-20 10:15:00 | 320.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:15:00 | 293.45 | 2026-04-20 10:15:00 | 322.80 | TARGET_HIT | 1.00 | 10.00% |
