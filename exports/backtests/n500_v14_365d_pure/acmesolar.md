# ACME Solar Holdings Ltd. (ACMESOLAR)

## Backtest Summary

- **Window:** 2024-11-13 09:15:00 → 2026-05-08 15:15:00 (2557 bars)
- **Last close:** 283.00
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
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 19
- **Target hits / Stop hits / Partials:** 0 / 19 / 0
- **Avg / median % per leg:** -2.02% / -1.38%
- **Sum % (uncompounded):** -38.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 0 | 0.0% | 0 | 19 | 0 | -2.02% | -38.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 0 | 0.0% | 0 | 19 | 0 | -2.02% | -38.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 0 | 0.0% | 0 | 19 | 0 | -2.02% | -38.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 10:15:00 | 224.15 | 210.25 | 210.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 236.08 | 211.09 | 210.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 12:15:00 | 241.90 | 241.92 | 231.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:00:00 | 241.90 | 241.92 | 231.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 288.45 | 295.24 | 284.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 288.45 | 295.24 | 284.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 283.75 | 294.74 | 284.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 283.75 | 294.74 | 284.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 283.00 | 294.62 | 284.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 281.25 | 294.62 | 284.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 281.35 | 294.49 | 284.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 285.70 | 289.84 | 283.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 11:30:00 | 284.55 | 289.74 | 283.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:45:00 | 285.25 | 289.32 | 283.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:15:00 | 284.20 | 289.24 | 283.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 284.00 | 289.19 | 283.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 285.30 | 289.15 | 283.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 282.50 | 289.03 | 283.78 | SL hit (close<static) qty=1.00 sl=283.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 285.70 | 288.98 | 283.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 283.00 | 288.81 | 283.77 | SL hit (close<static) qty=1.00 sl=283.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:45:00 | 285.65 | 288.52 | 283.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:45:00 | 287.45 | 288.33 | 283.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 284.35 | 288.29 | 283.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 283.20 | 288.29 | 283.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 283.75 | 288.25 | 283.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:00:00 | 283.75 | 288.25 | 283.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 283.75 | 288.20 | 283.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:30:00 | 283.65 | 288.20 | 283.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 283.95 | 288.16 | 283.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:30:00 | 283.75 | 288.16 | 283.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 283.60 | 288.11 | 283.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 283.90 | 288.11 | 283.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 283.90 | 288.07 | 283.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 286.25 | 288.07 | 283.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 284.25 | 287.99 | 283.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 282.10 | 287.93 | 283.81 | SL hit (close<static) qty=1.00 sl=283.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 282.10 | 287.93 | 283.81 | SL hit (close<static) qty=1.00 sl=283.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 282.10 | 287.93 | 283.81 | SL hit (close<static) qty=1.00 sl=283.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 282.10 | 287.93 | 283.81 | SL hit (close<static) qty=1.00 sl=283.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 274.85 | 285.57 | 283.13 | SL hit (close<static) qty=1.00 sl=275.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 274.85 | 285.57 | 283.13 | SL hit (close<static) qty=1.00 sl=275.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 274.85 | 285.57 | 283.13 | SL hit (close<static) qty=1.00 sl=275.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 274.85 | 285.57 | 283.13 | SL hit (close<static) qty=1.00 sl=275.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 14:00:00 | 285.20 | 285.35 | 283.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 13:15:00 | 283.05 | 285.35 | 283.13 | SL hit (close<static) qty=1.00 sl=283.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 286.45 | 285.30 | 283.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 283.65 | 285.29 | 283.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:15:00 | 282.65 | 285.29 | 283.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 283.00 | 285.26 | 283.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 283.00 | 285.26 | 283.12 | SL hit (close<static) qty=1.00 sl=283.10 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 282.50 | 285.26 | 283.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 283.55 | 285.25 | 283.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 283.60 | 285.25 | 283.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 282.95 | 285.22 | 283.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 282.55 | 285.22 | 283.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 282.50 | 285.20 | 283.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 281.55 | 285.20 | 283.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 282.40 | 285.17 | 283.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 282.55 | 285.17 | 283.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 283.45 | 285.15 | 283.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 282.05 | 285.15 | 283.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 284.80 | 285.15 | 283.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:30:00 | 287.70 | 284.86 | 283.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 288.95 | 284.94 | 283.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 13:30:00 | 287.80 | 285.00 | 283.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:00:00 | 288.15 | 285.00 | 283.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 282.35 | 285.05 | 283.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 282.35 | 285.05 | 283.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 282.85 | 285.03 | 283.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 14:00:00 | 283.10 | 284.99 | 283.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 281.30 | 284.95 | 283.33 | SL hit (close<static) qty=1.00 sl=282.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 283.30 | 284.92 | 283.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:15:00 | 283.25 | 284.90 | 283.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 280.25 | 284.85 | 283.30 | SL hit (close<static) qty=1.00 sl=282.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 280.25 | 284.85 | 283.30 | SL hit (close<static) qty=1.00 sl=282.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 279.35 | 284.80 | 283.28 | SL hit (close<static) qty=1.00 sl=280.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 279.35 | 284.80 | 283.28 | SL hit (close<static) qty=1.00 sl=280.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 279.35 | 284.80 | 283.28 | SL hit (close<static) qty=1.00 sl=280.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 279.35 | 284.80 | 283.28 | SL hit (close<static) qty=1.00 sl=280.10 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 265.75 | 281.79 | 281.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 14:15:00 | 264.85 | 281.62 | 281.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 12:15:00 | 235.64 | 235.64 | 248.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 11:15:00 | 228.22 | 221.25 | 230.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 228.22 | 221.25 | 230.79 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 248.26 | 231.77 | 231.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 249.26 | 232.40 | 232.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 283.60 | 286.11 | 269.96 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-01 10:15:00 | 285.70 | 2025-10-08 14:15:00 | 282.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-01 11:30:00 | 284.55 | 2025-10-09 11:15:00 | 283.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-10-06 11:45:00 | 285.25 | 2025-10-14 11:15:00 | 282.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-10-08 11:15:00 | 284.20 | 2025-10-14 11:15:00 | 282.10 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-08 12:45:00 | 285.30 | 2025-10-14 11:15:00 | 282.10 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-09 09:15:00 | 285.70 | 2025-10-14 11:15:00 | 282.10 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-10 09:45:00 | 285.65 | 2025-10-20 13:15:00 | 274.85 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-10-13 09:45:00 | 287.45 | 2025-10-20 13:15:00 | 274.85 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest2 | 2025-10-14 09:15:00 | 286.25 | 2025-10-20 13:15:00 | 274.85 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest2 | 2025-10-14 10:30:00 | 284.25 | 2025-10-20 13:15:00 | 274.85 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2025-10-21 14:00:00 | 285.20 | 2025-10-23 13:15:00 | 283.05 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-24 09:15:00 | 286.45 | 2025-10-24 10:15:00 | 283.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-10-29 12:30:00 | 287.70 | 2025-10-31 14:15:00 | 281.30 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-10-30 12:00:00 | 288.95 | 2025-11-03 10:15:00 | 280.25 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-10-30 13:30:00 | 287.80 | 2025-11-03 10:15:00 | 280.25 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-10-30 14:00:00 | 288.15 | 2025-11-03 11:15:00 | 279.35 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-10-31 14:00:00 | 283.10 | 2025-11-03 11:15:00 | 279.35 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-11-03 09:15:00 | 283.30 | 2025-11-03 11:15:00 | 279.35 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-11-03 10:15:00 | 283.25 | 2025-11-03 11:15:00 | 279.35 | STOP_HIT | 1.00 | -1.38% |
