# ONGC (ONGC)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 279.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 12 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 3 / 6 / 0
- **Avg / median % per leg:** 1.54% / -1.54%
- **Sum % (uncompounded):** 13.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 3 | 3 | 0 | 3.33% | 20.0% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.35% | -10.0% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.04% | -6.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.04% | -6.1% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.35% | -10.0% |
| retest2 (combined) | 6 | 3 | 50.0% | 3 | 3 | 0 | 3.98% | 23.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 241.95 | 239.10 | 239.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 10:15:00 | 243.45 | 239.14 | 239.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 247.20 | 248.70 | 245.28 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-11 15:15:00 | 249.90 | 248.69 | 245.36 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:15:00 | 253.00 | 248.73 | 245.40 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 247.75 | 249.14 | 245.84 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-19 13:15:00 | 249.80 | 248.89 | 246.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-19 14:15:00 | 249.05 | 248.89 | 246.11 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-20 10:15:00 | 250.10 | 248.90 | 246.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-20 11:15:00 | 249.40 | 248.91 | 246.17 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 245.90 | 248.74 | 246.26 | SL hit (close<ema400) qty=1.00 sl=246.26 alert=retest1 |

### Cycle 2 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 239.25 | 244.91 | 244.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 14:15:00 | 238.14 | 244.57 | 244.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 239.61 | 238.70 | 241.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 240.18 | 238.74 | 241.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 240.18 | 238.74 | 241.05 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-01 12:15:00 | 237.93 | 238.73 | 241.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:15:00 | 237.90 | 238.72 | 241.00 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 241.55 | 238.75 | 240.97 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-05 09:15:00 | 236.83 | 238.84 | 240.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 238.20 | 238.84 | 240.94 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 241.88 | 238.91 | 240.86 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-07 13:15:00 | 238.60 | 239.00 | 240.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-07 14:15:00 | 239.14 | 239.00 | 240.84 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-08 09:15:00 | 236.20 | 238.97 | 240.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:15:00 | 234.48 | 238.93 | 240.78 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 241.58 | 238.11 | 240.15 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 263.31 | 241.57 | 241.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 268.08 | 241.84 | 241.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 09:15:00 | 269.30 | 269.70 | 261.39 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-11 11:15:00 | 272.00 | 269.71 | 261.76 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 12:15:00 | 271.50 | 269.72 | 261.81 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-12 09:15:00 | 271.05 | 269.77 | 261.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:15:00 | 272.60 | 269.80 | 262.04 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 264.90 | 269.46 | 262.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | SL hit (close<ema400) qty=1.00 sl=262.36 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | SL hit (close<ema400) qty=1.00 sl=262.36 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-19 10:15:00 | 269.50 | 268.22 | 262.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 269.40 | 268.23 | 262.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-23 09:15:00 | 267.90 | 268.22 | 262.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 10:15:00 | 266.80 | 268.20 | 262.82 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 270.20 | 268.12 | 262.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 271.30 | 268.15 | 262.98 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2026-04-28 09:15:00 | 293.48 | 280.80 | 274.04 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 11:15:00 | 296.34 | 281.11 | 274.26 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 13:15:00 | 298.43 | 281.43 | 274.49 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-12 09:15:00 | 253.00 | 2025-11-24 10:15:00 | 245.90 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-01-01 13:15:00 | 237.90 | 2026-01-02 10:15:00 | 241.55 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-05 10:15:00 | 238.20 | 2026-01-06 14:15:00 | 241.88 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-01-08 10:15:00 | 234.48 | 2026-01-13 11:15:00 | 241.58 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2026-03-11 12:15:00 | 271.50 | 2026-03-16 10:15:00 | 262.20 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest1 | 2026-03-12 10:15:00 | 272.60 | 2026-03-16 10:15:00 | 262.20 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2026-03-19 11:15:00 | 269.40 | 2026-04-28 09:15:00 | 293.48 | TARGET_HIT | 1.00 | 8.94% |
| BUY | retest2 | 2026-03-23 10:15:00 | 266.80 | 2026-04-28 11:15:00 | 296.34 | TARGET_HIT | 1.00 | 11.07% |
| BUY | retest2 | 2026-03-24 10:15:00 | 271.30 | 2026-04-28 13:15:00 | 298.43 | TARGET_HIT | 1.00 | 10.00% |
