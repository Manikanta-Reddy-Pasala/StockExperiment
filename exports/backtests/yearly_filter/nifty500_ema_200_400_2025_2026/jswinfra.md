# JSW Infrastructure Ltd. (JSWINFRA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-11 15:15:00 (3605 bars)
- **Last close:** 286.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 16
- **Target hits / Stop hits / Partials:** 0 / 19 / 3
- **Avg / median % per leg:** -0.41% / -1.19%
- **Sum % (uncompounded):** -9.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.97% | -23.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.97% | -23.6% |
| SELL (all) | 10 | 6 | 60.0% | 0 | 7 | 3 | 1.45% | 14.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 0 | 7 | 3 | 1.45% | 14.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 6 | 27.3% | 0 | 19 | 3 | -0.41% | -9.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 300.00 | 304.29 | 304.29 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 310.30 | 304.31 | 304.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 12:15:00 | 312.85 | 304.40 | 304.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 318.30 | 319.78 | 313.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 12:00:00 | 318.30 | 319.78 | 313.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 313.60 | 319.71 | 313.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 313.60 | 319.71 | 313.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 313.35 | 319.65 | 313.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:15:00 | 313.00 | 319.65 | 313.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 313.05 | 319.58 | 313.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:30:00 | 312.80 | 319.58 | 313.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 315.75 | 319.55 | 313.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:45:00 | 316.90 | 319.21 | 313.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:30:00 | 316.25 | 319.14 | 313.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 10:45:00 | 317.25 | 319.13 | 313.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:45:00 | 316.20 | 319.07 | 313.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 308.50 | 318.87 | 313.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 308.50 | 318.87 | 313.80 | SL hit (close<static) qty=1.00 sl=311.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 297.65 | 310.74 | 310.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 295.00 | 310.02 | 310.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 275.25 | 274.89 | 283.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 275.25 | 274.89 | 283.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 283.30 | 275.37 | 283.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:45:00 | 283.80 | 275.37 | 283.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 284.45 | 275.46 | 283.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 284.45 | 275.46 | 283.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 284.30 | 275.55 | 283.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:15:00 | 285.00 | 275.55 | 283.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 285.30 | 276.13 | 283.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 285.30 | 276.13 | 283.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 284.00 | 277.38 | 283.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 284.00 | 277.38 | 283.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 284.10 | 277.45 | 283.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 281.50 | 277.45 | 283.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 284.15 | 277.58 | 283.47 | SL hit (close>static) qty=1.00 sl=284.10 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 14:15:00 | 274.71 | 261.75 | 261.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 279.20 | 262.05 | 261.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 266.01 | 266.66 | 264.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 266.01 | 266.66 | 264.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-29 09:15:00 | 307.45 | 2025-08-01 14:15:00 | 303.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-07-29 09:45:00 | 307.20 | 2025-08-01 14:15:00 | 303.80 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-29 12:30:00 | 307.45 | 2025-08-01 14:15:00 | 303.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-07-30 15:15:00 | 306.90 | 2025-08-01 14:15:00 | 303.80 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-31 11:15:00 | 307.65 | 2025-08-06 13:15:00 | 300.95 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-08-04 09:15:00 | 307.95 | 2025-08-06 13:15:00 | 300.95 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-08-18 09:15:00 | 308.50 | 2025-08-26 09:15:00 | 301.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-08-22 11:45:00 | 307.00 | 2025-08-26 09:15:00 | 301.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-01 14:45:00 | 316.90 | 2025-10-06 09:15:00 | 308.50 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-10-03 09:30:00 | 316.25 | 2025-10-06 09:15:00 | 308.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-10-03 10:45:00 | 317.25 | 2025-10-06 09:15:00 | 308.50 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-10-03 12:45:00 | 316.20 | 2025-10-06 09:15:00 | 308.50 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-12-30 09:15:00 | 281.50 | 2025-12-30 10:15:00 | 284.15 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-31 14:00:00 | 283.05 | 2025-12-31 14:15:00 | 284.60 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-01-05 10:30:00 | 282.55 | 2026-01-09 09:15:00 | 268.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 12:15:00 | 283.00 | 2026-01-09 09:15:00 | 268.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:30:00 | 282.55 | 2026-01-19 10:15:00 | 276.20 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2026-01-05 12:15:00 | 283.00 | 2026-01-19 10:15:00 | 276.20 | STOP_HIT | 0.50 | 2.40% |
| SELL | retest2 | 2026-03-06 15:00:00 | 268.25 | 2026-03-09 09:15:00 | 254.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:00:00 | 268.25 | 2026-03-09 14:15:00 | 264.00 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2026-04-15 11:45:00 | 268.42 | 2026-04-17 09:15:00 | 275.66 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-15 13:15:00 | 268.94 | 2026-04-17 09:15:00 | 275.66 | STOP_HIT | 1.00 | -2.50% |
