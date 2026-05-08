# ONGC (ONGC)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 279.20
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
| PENDING | 11 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 3 / 5 / 0
- **Avg / median % per leg:** 2.43% / -1.00%
- **Sum % (uncompounded):** 19.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 3 | 2 | 0 | 4.57% | 22.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.58% | -7.2% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.13% | -3.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.13% | -3.4% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.58% | -7.2% |
| retest2 (combined) | 6 | 3 | 50.0% | 3 | 3 | 0 | 4.43% | 26.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 232.46 | 242.22 | 242.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 232.09 | 241.84 | 242.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 239.61 | 238.69 | 240.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 240.38 | 238.72 | 240.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 240.38 | 238.72 | 240.15 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-01 10:15:00 | 239.00 | 238.74 | 240.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:15:00 | 238.65 | 238.74 | 240.14 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 241.52 | 238.75 | 240.10 | SL hit (close>static) qty=1.00 sl=241.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-05 09:15:00 | 236.85 | 238.84 | 240.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 238.20 | 238.84 | 240.10 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 241.05 | 238.81 | 240.05 | SL hit (close>static) qty=1.00 sl=241.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-06 10:15:00 | 239.32 | 238.81 | 240.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-06 11:15:00 | 240.89 | 238.84 | 240.05 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-07 13:15:00 | 238.60 | 239.00 | 240.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 14:15:00 | 239.14 | 239.00 | 240.07 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 241.54 | 238.11 | 239.47 | SL hit (close>static) qty=1.00 sl=241.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 13:15:00 | 245.30 | 240.57 | 240.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 247.02 | 240.73 | 240.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 09:15:00 | 269.30 | 269.53 | 260.96 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 269.70 | 269.54 | 261.00 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-10 11:15:00 | 268.30 | 269.52 | 261.04 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-10 12:15:00 | 270.05 | 269.53 | 261.08 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 13:15:00 | 269.95 | 269.53 | 261.12 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-10 15:15:00 | 270.20 | 269.54 | 261.21 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-11 09:15:00 | 269.05 | 269.53 | 261.25 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-03-11 11:15:00 | 272.00 | 269.55 | 261.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 12:15:00 | 271.55 | 269.57 | 261.39 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 264.90 | 269.33 | 261.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 261.05 | 269.18 | 261.97 | SL hit (close<ema400) qty=1.00 sl=261.97 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 261.05 | 269.18 | 261.97 | SL hit (close<ema400) qty=1.00 sl=261.97 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-19 10:15:00 | 269.50 | 268.12 | 262.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 269.25 | 268.13 | 262.13 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-23 09:15:00 | 267.95 | 268.13 | 262.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 10:15:00 | 266.90 | 268.12 | 262.51 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 270.10 | 268.04 | 262.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 271.20 | 268.08 | 262.68 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2026-04-28 09:15:00 | 293.59 | 280.64 | 273.68 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 11:15:00 | 296.18 | 280.95 | 273.91 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 13:15:00 | 298.32 | 281.27 | 274.14 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-01 11:15:00 | 238.65 | 2026-01-02 10:15:00 | 241.52 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-05 10:15:00 | 238.20 | 2026-01-06 09:15:00 | 241.05 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-07 14:15:00 | 239.14 | 2026-01-13 11:15:00 | 241.54 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest1 | 2026-03-10 13:15:00 | 269.95 | 2026-03-16 11:15:00 | 261.05 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest1 | 2026-03-11 12:15:00 | 271.55 | 2026-03-16 11:15:00 | 261.05 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-03-19 11:15:00 | 269.25 | 2026-04-28 09:15:00 | 293.59 | TARGET_HIT | 1.00 | 9.04% |
| BUY | retest2 | 2026-03-23 10:15:00 | 266.90 | 2026-04-28 11:15:00 | 296.18 | TARGET_HIT | 1.00 | 10.97% |
| BUY | retest2 | 2026-03-24 10:15:00 | 271.20 | 2026-04-28 13:15:00 | 298.32 | TARGET_HIT | 1.00 | 10.00% |
