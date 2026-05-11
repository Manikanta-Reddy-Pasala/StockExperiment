# Honasa Consumer Ltd. (HONASA)

## Backtest Summary

- **Window:** 2023-11-07 09:15:00 → 2026-05-08 15:15:00 (4316 bars)
- **Last close:** 358.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 4 |
| ALERT3 | 54 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 8 |
| TARGET_HIT | 6 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 31
- **Target hits / Stop hits / Partials:** 6 / 39 / 8
- **Avg / median % per leg:** 0.60% / -1.33%
- **Sum % (uncompounded):** 31.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 6 | 31.6% | 6 | 13 | 0 | 1.48% | 28.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 6 | 31.6% | 6 | 13 | 0 | 1.48% | 28.1% |
| SELL (all) | 34 | 16 | 47.1% | 0 | 26 | 8 | 0.10% | 3.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 16 | 47.1% | 0 | 26 | 8 | 0.10% | 3.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 22 | 41.5% | 6 | 39 | 8 | 0.60% | 31.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 09:15:00 | 379.00 | 418.44 | 418.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 375.00 | 411.21 | 414.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 13:15:00 | 402.00 | 398.67 | 406.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-28 13:30:00 | 402.00 | 398.67 | 406.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 414.30 | 398.90 | 406.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-01 11:00:00 | 408.20 | 398.99 | 406.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-02 09:45:00 | 407.00 | 399.44 | 406.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-02 11:45:00 | 407.60 | 399.55 | 406.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-02 15:15:00 | 408.00 | 399.89 | 406.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 410.70 | 400.08 | 407.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:00:00 | 410.70 | 400.08 | 407.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 412.90 | 400.20 | 407.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:30:00 | 412.55 | 400.20 | 407.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 408.15 | 401.71 | 407.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 13:15:00 | 406.95 | 401.85 | 407.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-05 13:15:00 | 410.45 | 401.93 | 407.38 | SL hit (close>static) qty=1.00 sl=409.95 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 14:15:00 | 431.95 | 408.33 | 408.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 441.00 | 419.57 | 415.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 421.70 | 422.54 | 417.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 421.70 | 422.54 | 417.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 421.70 | 422.54 | 417.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 421.70 | 422.54 | 417.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 430.05 | 422.62 | 417.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 10:00:00 | 439.40 | 422.79 | 417.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:00:00 | 433.35 | 428.90 | 421.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 433.95 | 429.83 | 422.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 10:00:00 | 433.70 | 429.87 | 422.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 423.05 | 430.24 | 423.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:00:00 | 423.05 | 430.24 | 423.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 423.30 | 430.17 | 423.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 14:45:00 | 425.20 | 430.13 | 423.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 421.10 | 430.00 | 423.74 | SL hit (close<static) qty=1.00 sl=421.25 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 10:15:00 | 427.30 | 471.38 | 471.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 13:15:00 | 416.85 | 455.61 | 462.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 14:15:00 | 233.73 | 230.96 | 257.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 15:00:00 | 233.73 | 230.96 | 257.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 234.44 | 220.52 | 234.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 13:00:00 | 234.44 | 220.52 | 234.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 235.36 | 220.67 | 234.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 13:30:00 | 236.90 | 220.67 | 234.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 235.18 | 220.81 | 234.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:30:00 | 236.22 | 220.81 | 234.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 234.40 | 220.95 | 234.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 234.00 | 220.95 | 234.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 234.60 | 221.08 | 234.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:30:00 | 231.92 | 223.00 | 234.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 15:15:00 | 237.00 | 223.56 | 234.77 | SL hit (close>static) qty=1.00 sl=236.70 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 250.18 | 236.51 | 236.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 255.46 | 238.65 | 237.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 297.45 | 297.65 | 279.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 297.45 | 297.65 | 279.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 287.55 | 299.72 | 288.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 287.55 | 299.72 | 288.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 287.65 | 299.60 | 288.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 289.40 | 299.28 | 288.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 286.30 | 298.22 | 289.53 | SL hit (close<static) qty=1.00 sl=286.60 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 264.90 | 283.81 | 283.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 10:15:00 | 264.10 | 283.10 | 283.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 294.45 | 276.78 | 279.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 294.45 | 276.78 | 279.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 294.45 | 276.78 | 279.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:30:00 | 299.80 | 276.78 | 279.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 284.45 | 277.63 | 280.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 284.30 | 277.63 | 280.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 283.20 | 277.69 | 280.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:30:00 | 282.65 | 277.72 | 280.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 09:30:00 | 281.45 | 278.08 | 280.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 290.50 | 278.58 | 280.46 | SL hit (close>static) qty=1.00 sl=288.65 alert=retest2 |

### Cycle 6 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 299.55 | 282.28 | 282.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 302.90 | 283.03 | 282.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 297.00 | 297.02 | 292.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 12:00:00 | 297.00 | 297.02 | 292.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 284.85 | 296.89 | 292.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 284.95 | 296.89 | 292.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 284.70 | 296.77 | 292.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 14:45:00 | 285.35 | 296.24 | 292.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 277.55 | 295.94 | 291.95 | SL hit (close<static) qty=1.00 sl=282.75 alert=retest2 |

### Cycle 7 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 276.80 | 289.41 | 289.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 276.50 | 289.08 | 289.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 283.75 | 283.40 | 286.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 283.75 | 283.40 | 286.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 283.75 | 283.40 | 286.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 283.75 | 283.40 | 286.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 285.30 | 283.42 | 286.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 283.65 | 283.42 | 286.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 287.00 | 283.46 | 286.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 287.00 | 283.46 | 286.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 288.55 | 283.51 | 286.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 288.55 | 283.51 | 286.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 284.55 | 283.58 | 286.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 284.85 | 283.58 | 286.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 278.10 | 281.08 | 284.23 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 293.05 | 286.26 | 286.25 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 279.80 | 286.24 | 286.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 279.30 | 286.11 | 286.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 14:15:00 | 272.00 | 271.78 | 277.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 272.00 | 271.78 | 277.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 276.25 | 271.96 | 277.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 275.85 | 271.96 | 277.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 275.80 | 271.90 | 276.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 275.80 | 271.90 | 276.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 276.20 | 271.95 | 276.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 271.10 | 271.95 | 276.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 283.90 | 272.06 | 276.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 283.90 | 272.06 | 276.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 286.70 | 272.21 | 276.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 288.40 | 272.21 | 276.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 13:15:00 | 296.00 | 280.42 | 280.39 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 271.50 | 280.87 | 280.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 267.35 | 280.39 | 280.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 13:15:00 | 280.55 | 277.70 | 279.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 13:15:00 | 280.55 | 277.70 | 279.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 280.55 | 277.70 | 279.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 283.20 | 277.70 | 279.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 278.30 | 277.71 | 279.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:30:00 | 280.85 | 277.71 | 279.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 275.55 | 277.42 | 278.93 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 292.65 | 280.01 | 279.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 295.50 | 280.59 | 280.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 10:15:00 | 291.10 | 293.84 | 288.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 11:00:00 | 291.10 | 293.84 | 288.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 289.25 | 293.74 | 288.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 13:15:00 | 290.00 | 293.54 | 288.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 294.20 | 293.40 | 288.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 11:00:00 | 289.70 | 293.37 | 288.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 280.35 | 293.46 | 288.86 | SL hit (close<static) qty=1.00 sl=283.00 alert=retest2 |

### Cycle 13 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 280.80 | 285.77 | 285.79 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 289.50 | 285.82 | 285.81 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 281.55 | 285.76 | 285.78 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 291.25 | 285.81 | 285.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 293.30 | 285.88 | 285.83 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-01 11:00:00 | 408.20 | 2024-04-05 13:15:00 | 410.45 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-04-02 09:45:00 | 407.00 | 2024-04-15 09:15:00 | 387.79 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2024-04-02 11:45:00 | 407.60 | 2024-04-15 09:15:00 | 386.65 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2024-04-02 15:15:00 | 408.00 | 2024-04-15 09:15:00 | 387.22 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2024-04-05 13:15:00 | 406.95 | 2024-04-15 09:15:00 | 387.60 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2024-04-05 14:15:00 | 406.45 | 2024-04-15 09:15:00 | 386.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-08 11:30:00 | 407.05 | 2024-04-15 09:15:00 | 386.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-08 12:15:00 | 407.00 | 2024-04-15 09:15:00 | 386.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-02 09:45:00 | 407.00 | 2024-04-22 09:15:00 | 400.10 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2024-04-02 11:45:00 | 407.60 | 2024-04-22 09:15:00 | 400.10 | STOP_HIT | 0.50 | 1.84% |
| SELL | retest2 | 2024-04-02 15:15:00 | 408.00 | 2024-04-22 09:15:00 | 400.10 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2024-04-05 13:15:00 | 406.95 | 2024-04-22 09:15:00 | 400.10 | STOP_HIT | 0.50 | 1.68% |
| SELL | retest2 | 2024-04-05 14:15:00 | 406.45 | 2024-04-22 09:15:00 | 400.10 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2024-04-08 11:30:00 | 407.05 | 2024-04-22 09:15:00 | 400.10 | STOP_HIT | 0.50 | 1.71% |
| SELL | retest2 | 2024-04-08 12:15:00 | 407.00 | 2024-04-22 09:15:00 | 400.10 | STOP_HIT | 0.50 | 1.70% |
| BUY | retest2 | 2024-06-05 10:00:00 | 439.40 | 2024-06-20 09:15:00 | 421.10 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2024-06-11 12:00:00 | 433.35 | 2024-07-04 10:15:00 | 466.90 | TARGET_HIT | 1.00 | 7.74% |
| BUY | retest2 | 2024-06-13 09:15:00 | 433.95 | 2024-07-04 10:15:00 | 467.45 | TARGET_HIT | 1.00 | 7.72% |
| BUY | retest2 | 2024-06-13 10:00:00 | 433.70 | 2024-07-04 14:15:00 | 483.34 | TARGET_HIT | 1.00 | 11.45% |
| BUY | retest2 | 2024-06-19 14:45:00 | 425.20 | 2024-07-04 14:15:00 | 476.69 | TARGET_HIT | 1.00 | 12.11% |
| BUY | retest2 | 2024-06-20 12:45:00 | 424.45 | 2024-07-04 14:15:00 | 477.35 | TARGET_HIT | 1.00 | 12.46% |
| BUY | retest2 | 2024-06-20 13:15:00 | 424.95 | 2024-07-04 14:15:00 | 477.07 | TARGET_HIT | 1.00 | 12.26% |
| SELL | retest2 | 2025-03-27 09:30:00 | 231.92 | 2025-03-27 15:15:00 | 237.00 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-03-28 10:00:00 | 232.53 | 2025-04-03 09:15:00 | 237.37 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-04-01 09:15:00 | 232.35 | 2025-04-03 09:15:00 | 237.37 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-04-01 09:45:00 | 232.14 | 2025-04-03 09:15:00 | 237.37 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-04-01 15:00:00 | 233.89 | 2025-04-03 09:15:00 | 237.37 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-04-02 09:30:00 | 234.12 | 2025-04-03 09:15:00 | 237.37 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-04-04 11:00:00 | 234.68 | 2025-04-07 09:15:00 | 222.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 11:00:00 | 234.68 | 2025-04-11 10:15:00 | 226.22 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2025-04-17 12:30:00 | 234.88 | 2025-04-21 09:15:00 | 237.37 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-04-21 12:30:00 | 232.87 | 2025-04-24 09:15:00 | 241.80 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-04-21 13:15:00 | 232.99 | 2025-04-24 09:15:00 | 241.80 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-04-21 15:15:00 | 233.08 | 2025-04-24 09:15:00 | 241.80 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-04-22 14:30:00 | 232.24 | 2025-04-24 09:15:00 | 241.80 | STOP_HIT | 1.00 | -4.12% |
| SELL | retest2 | 2025-04-23 10:00:00 | 230.49 | 2025-04-24 09:15:00 | 241.80 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2025-04-25 09:45:00 | 230.19 | 2025-04-29 10:15:00 | 240.15 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2025-04-25 15:15:00 | 230.07 | 2025-04-29 10:15:00 | 240.15 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2025-04-28 09:45:00 | 231.49 | 2025-04-29 10:15:00 | 240.15 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-07-14 15:00:00 | 289.40 | 2025-07-18 13:15:00 | 286.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-08-14 12:30:00 | 282.65 | 2025-08-20 10:15:00 | 290.50 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-08-19 09:30:00 | 281.45 | 2025-08-20 10:15:00 | 290.50 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-09-26 14:45:00 | 285.35 | 2025-09-29 09:15:00 | 277.55 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-10-01 13:45:00 | 285.80 | 2025-10-03 09:15:00 | 282.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-10-03 13:30:00 | 285.65 | 2025-10-14 09:15:00 | 280.30 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-10-08 12:15:00 | 285.50 | 2025-10-14 09:15:00 | 280.30 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-03-04 13:15:00 | 290.00 | 2026-03-09 09:15:00 | 280.35 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-03-05 09:15:00 | 294.20 | 2026-03-09 09:15:00 | 280.35 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest2 | 2026-03-05 11:00:00 | 289.70 | 2026-03-09 09:15:00 | 280.35 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-03-10 09:15:00 | 290.50 | 2026-03-11 15:15:00 | 286.90 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-03-11 09:15:00 | 292.15 | 2026-03-11 15:15:00 | 286.90 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-03-11 14:30:00 | 291.10 | 2026-03-12 12:15:00 | 281.40 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-03-18 14:30:00 | 294.65 | 2026-03-18 15:15:00 | 280.00 | STOP_HIT | 1.00 | -4.97% |
