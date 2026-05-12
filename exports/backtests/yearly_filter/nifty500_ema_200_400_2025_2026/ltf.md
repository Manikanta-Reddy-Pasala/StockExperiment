# L&T Finance Ltd. (LTF)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 302.85
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
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 0
- **Avg / median % per leg:** 3.36% / 10.00%
- **Sum % (uncompounded):** 13.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.29% | -6.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.29% | -6.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 2 | 50.0% | 2 | 2 | 0 | 3.36% | 13.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 12:15:00 | 273.00 | 290.92 | 290.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 271.40 | 290.73 | 290.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 274.88 | 263.91 | 273.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 274.88 | 263.91 | 273.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 274.88 | 263.91 | 273.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:15:00 | 276.80 | 263.91 | 273.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 277.09 | 264.04 | 273.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:30:00 | 275.16 | 264.04 | 273.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 275.55 | 264.84 | 273.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:00:00 | 275.55 | 264.84 | 273.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 275.23 | 265.36 | 273.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:30:00 | 273.59 | 266.13 | 273.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 273.34 | 266.59 | 273.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 282.45 | 266.82 | 273.81 | SL hit (close>static) qty=1.00 sl=277.19 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 280.07 | 278.49 | 278.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 286.70 | 278.60 | 278.54 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-11 14:00:00 | 196.23 | 2025-08-19 10:15:00 | 215.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-12 09:45:00 | 197.54 | 2025-08-19 14:15:00 | 217.29 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-13 09:30:00 | 273.59 | 2026-04-15 09:15:00 | 282.45 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2026-04-13 14:45:00 | 273.34 | 2026-04-15 09:15:00 | 282.45 | STOP_HIT | 1.00 | -3.33% |
