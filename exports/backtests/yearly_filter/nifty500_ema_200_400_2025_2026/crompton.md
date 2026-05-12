# Crompton Greaves Consumer Electricals Ltd. (CROMPTON)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 293.55
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 3
- **Avg / median % per leg:** 2.07% / -1.45%
- **Sum % (uncompounded):** 29.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.12% | -12.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.12% | -12.7% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 5.22% | 41.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 6 | 75.0% | 3 | 2 | 3 | 5.22% | 41.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 6 | 42.9% | 3 | 8 | 3 | 2.07% | 29.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 348.30 | 345.21 | 345.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 350.00 | 345.37 | 345.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 347.45 | 347.47 | 346.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:45:00 | 348.00 | 347.47 | 346.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 345.55 | 347.45 | 346.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 345.55 | 347.45 | 346.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 348.50 | 347.46 | 346.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:30:00 | 347.90 | 347.46 | 346.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 344.50 | 347.43 | 346.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 344.50 | 347.43 | 346.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 343.95 | 347.40 | 346.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 343.95 | 347.40 | 346.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 346.85 | 347.04 | 346.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:45:00 | 347.30 | 347.04 | 346.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:00:00 | 347.50 | 347.29 | 346.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 347.20 | 347.29 | 346.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 347.35 | 347.26 | 346.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 347.35 | 347.26 | 346.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 347.05 | 347.26 | 346.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 345.30 | 347.24 | 346.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 345.30 | 347.24 | 346.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 344.55 | 347.21 | 346.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 344.25 | 347.21 | 346.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-19 11:15:00 | 341.85 | 347.16 | 346.43 | SL hit (close<static) qty=1.00 sl=344.35 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 334.45 | 346.86 | 346.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 330.95 | 344.78 | 345.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 329.85 | 329.39 | 335.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 15:00:00 | 329.85 | 329.39 | 335.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 333.55 | 327.38 | 333.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 333.55 | 327.38 | 333.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 331.15 | 327.42 | 333.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:30:00 | 329.35 | 327.43 | 333.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:00:00 | 330.55 | 327.47 | 333.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 335.35 | 327.87 | 333.14 | SL hit (close>static) qty=1.00 sl=334.40 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 277.85 | 250.24 | 250.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 278.80 | 250.53 | 250.37 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-16 13:45:00 | 347.30 | 2025-06-19 11:15:00 | 341.85 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-18 11:00:00 | 347.50 | 2025-06-19 11:15:00 | 341.85 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-06-18 11:30:00 | 347.20 | 2025-06-19 11:15:00 | 341.85 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-06-18 15:15:00 | 347.35 | 2025-06-19 11:15:00 | 341.85 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-07-04 10:45:00 | 354.90 | 2025-07-08 13:15:00 | 343.60 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2025-07-04 15:00:00 | 355.05 | 2025-07-08 13:15:00 | 343.60 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-08-29 12:30:00 | 329.35 | 2025-09-01 14:15:00 | 335.35 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-08-29 14:00:00 | 330.55 | 2025-09-01 14:15:00 | 335.35 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-04 11:15:00 | 330.55 | 2025-09-12 14:15:00 | 314.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 13:15:00 | 330.50 | 2025-09-12 14:15:00 | 313.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-09 09:15:00 | 323.75 | 2025-09-22 14:15:00 | 307.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 11:15:00 | 330.55 | 2025-09-25 10:15:00 | 297.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-04 13:15:00 | 330.50 | 2025-09-25 10:15:00 | 297.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-09 09:15:00 | 323.75 | 2025-09-26 11:15:00 | 291.38 | TARGET_HIT | 0.50 | 10.00% |
