# Tata Capital Ltd. (TATACAP)

## Backtest Summary

- **Window:** 2025-10-13 09:15:00 → 2026-05-11 15:15:00 (982 bars)
- **Last close:** 314.45
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
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.16% / -0.90%
- **Sum % (uncompounded):** -4.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.16% | -4.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.16% | -4.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.16% | -4.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 318.10 | 339.75 | 339.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 312.30 | 339.27 | 339.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 14:15:00 | 327.20 | 325.83 | 331.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 15:00:00 | 327.20 | 325.83 | 331.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 326.50 | 320.47 | 327.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 15:00:00 | 326.50 | 320.47 | 327.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 326.50 | 320.53 | 327.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:15:00 | 326.60 | 320.53 | 327.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 326.80 | 320.59 | 327.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:30:00 | 327.10 | 320.59 | 327.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 328.15 | 320.67 | 327.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:30:00 | 328.75 | 320.67 | 327.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 329.80 | 320.76 | 327.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:45:00 | 329.75 | 320.76 | 327.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 327.45 | 320.96 | 327.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 327.45 | 320.96 | 327.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 326.50 | 321.01 | 327.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 327.90 | 321.01 | 327.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 328.55 | 321.09 | 327.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:30:00 | 329.10 | 321.09 | 327.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 327.90 | 321.16 | 327.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 11:30:00 | 326.50 | 321.21 | 327.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 325.30 | 321.42 | 327.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 330.50 | 321.78 | 326.90 | SL hit (close>static) qty=1.00 sl=328.70 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-10 11:30:00 | 326.50 | 2026-04-15 12:15:00 | 330.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-04-13 09:15:00 | 325.30 | 2026-04-15 12:15:00 | 330.50 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-04-28 12:00:00 | 326.85 | 2026-04-29 12:15:00 | 329.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-04-28 13:45:00 | 326.85 | 2026-04-29 12:15:00 | 329.80 | STOP_HIT | 1.00 | -0.90% |
