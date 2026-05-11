# Gujarat State Petronet Ltd. (GSPL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 289.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 7 |
| ALERT3 | 83 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 75 |
| PARTIAL | 7 |
| TARGET_HIT | 10 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 84 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 66
- **Target hits / Stop hits / Partials:** 10 / 67 / 7
- **Avg / median % per leg:** 0.06% / -1.70%
- **Sum % (uncompounded):** 5.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 5 | 14.3% | 5 | 30 | 0 | 0.13% | 4.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.86% | -1.7% |
| BUY @ 3rd Alert (retest2) | 33 | 5 | 15.2% | 5 | 28 | 0 | 0.19% | 6.1% |
| SELL (all) | 49 | 13 | 26.5% | 5 | 37 | 7 | 0.02% | 0.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 49 | 13 | 26.5% | 5 | 37 | 7 | 0.02% | 0.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.86% | -1.7% |
| retest2 (combined) | 82 | 18 | 22.0% | 10 | 65 | 7 | 0.08% | 6.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 09:15:00 | 279.35 | 287.19 | 287.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 10:15:00 | 278.50 | 287.10 | 287.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 09:15:00 | 280.35 | 279.60 | 282.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 09:15:00 | 280.35 | 279.60 | 282.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 280.35 | 279.60 | 282.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 09:30:00 | 280.95 | 279.60 | 282.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 286.25 | 279.57 | 282.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:30:00 | 290.10 | 279.57 | 282.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 10:15:00 | 283.30 | 279.61 | 282.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 11:15:00 | 280.70 | 279.61 | 282.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 13:15:00 | 280.10 | 279.63 | 282.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 14:45:00 | 280.00 | 279.64 | 282.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-08 11:15:00 | 288.65 | 280.03 | 282.22 | SL hit (close>static) qty=1.00 sl=286.30 alert=retest2 |

### Cycle 2 — BUY (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 13:15:00 | 288.10 | 283.28 | 283.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 14:15:00 | 290.45 | 283.58 | 283.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 15:15:00 | 285.30 | 285.50 | 284.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 09:15:00 | 287.45 | 285.50 | 284.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 12:15:00 | 286.00 | 285.57 | 284.64 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 12:15:00 | 285.00 | 285.57 | 284.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 12:30:00 | 285.25 | 285.57 | 284.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 13:15:00 | 284.25 | 285.55 | 284.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-20 13:15:00 | 284.25 | 285.55 | 284.64 | SL hit (close<ema400) qty=1.00 sl=284.64 alert=retest1 |

### Cycle 3 — SELL (started 2023-10-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 15:15:00 | 274.90 | 283.78 | 283.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 15:15:00 | 273.00 | 282.68 | 283.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 09:15:00 | 283.45 | 277.87 | 280.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 09:15:00 | 283.45 | 277.87 | 280.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 283.45 | 277.87 | 280.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 11:45:00 | 279.80 | 277.94 | 280.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 10:45:00 | 280.35 | 277.35 | 279.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 10:45:00 | 279.80 | 277.52 | 279.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 10:00:00 | 280.45 | 276.70 | 278.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 279.25 | 276.73 | 278.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:30:00 | 280.25 | 276.73 | 278.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 280.00 | 276.76 | 278.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:30:00 | 280.80 | 276.76 | 278.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 12:15:00 | 278.80 | 276.78 | 278.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 13:30:00 | 278.70 | 276.79 | 278.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 14:15:00 | 282.95 | 276.85 | 278.86 | SL hit (close>static) qty=1.00 sl=280.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 12:15:00 | 287.05 | 280.49 | 280.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 288.30 | 280.75 | 280.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 282.70 | 286.01 | 283.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 282.70 | 286.01 | 283.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 282.70 | 286.01 | 283.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 282.70 | 286.01 | 283.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 284.30 | 286.00 | 283.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:15:00 | 282.30 | 286.00 | 283.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 282.30 | 285.96 | 283.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 09:15:00 | 285.20 | 285.96 | 283.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 288.20 | 285.98 | 283.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 10:45:00 | 288.50 | 285.99 | 283.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 11:30:00 | 288.75 | 286.02 | 283.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 14:00:00 | 288.30 | 286.08 | 283.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-01 09:15:00 | 317.35 | 290.27 | 286.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-04-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 13:15:00 | 295.00 | 349.73 | 349.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 14:15:00 | 293.15 | 349.17 | 349.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 10:15:00 | 303.90 | 297.21 | 310.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-12 11:00:00 | 303.90 | 297.21 | 310.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 307.85 | 299.17 | 309.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:45:00 | 309.60 | 299.17 | 309.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 306.60 | 299.62 | 309.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 306.25 | 299.62 | 309.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:45:00 | 305.70 | 299.75 | 309.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 305.95 | 299.88 | 309.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 14:00:00 | 305.90 | 300.27 | 309.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 307.70 | 300.88 | 307.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 312.00 | 301.11 | 307.59 | SL hit (close>static) qty=1.00 sl=311.30 alert=retest2 |

### Cycle 6 — BUY (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 10:15:00 | 330.65 | 311.48 | 311.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 333.15 | 313.16 | 312.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 323.40 | 326.18 | 320.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 09:45:00 | 322.65 | 326.18 | 320.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 322.95 | 326.11 | 320.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:15:00 | 323.55 | 326.11 | 320.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 12:15:00 | 317.65 | 325.99 | 320.53 | SL hit (close<static) qty=1.00 sl=320.30 alert=retest2 |

### Cycle 7 — SELL (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 13:15:00 | 328.85 | 385.30 | 385.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 325.50 | 384.70 | 385.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 15:15:00 | 367.95 | 364.50 | 372.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 09:15:00 | 372.45 | 364.50 | 372.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 370.50 | 364.56 | 372.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 12:30:00 | 366.35 | 364.67 | 372.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 14:15:00 | 376.00 | 365.13 | 372.66 | SL hit (close>static) qty=1.00 sl=373.35 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 321.20 | 310.56 | 310.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 15:15:00 | 325.95 | 311.33 | 310.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 328.55 | 328.76 | 321.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 13:45:00 | 328.50 | 328.76 | 321.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 327.25 | 330.61 | 325.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 325.60 | 330.61 | 325.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 326.60 | 330.33 | 325.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 325.60 | 330.33 | 325.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 326.10 | 330.29 | 325.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:30:00 | 326.30 | 330.29 | 325.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 326.00 | 330.24 | 325.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:30:00 | 325.75 | 330.24 | 325.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 325.10 | 330.19 | 325.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 326.30 | 330.19 | 325.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 323.50 | 330.13 | 325.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 323.50 | 330.13 | 325.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 318.25 | 330.01 | 325.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 318.25 | 330.01 | 325.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 319.50 | 326.85 | 324.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 319.50 | 326.85 | 324.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 325.90 | 326.47 | 324.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 325.75 | 326.47 | 324.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 326.15 | 329.17 | 326.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 326.00 | 329.17 | 326.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 328.25 | 329.16 | 326.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 330.15 | 329.15 | 326.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 325.40 | 328.97 | 326.60 | SL hit (close<static) qty=1.00 sl=326.20 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 303.20 | 325.92 | 325.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 299.15 | 321.50 | 323.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 309.00 | 304.95 | 311.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 309.00 | 304.95 | 311.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 309.00 | 304.95 | 311.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 312.50 | 304.95 | 311.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 311.60 | 305.01 | 311.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 314.50 | 305.01 | 311.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 311.40 | 305.08 | 311.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 313.00 | 305.08 | 311.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 311.55 | 305.14 | 311.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 311.60 | 305.14 | 311.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 312.25 | 305.21 | 311.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 312.25 | 305.21 | 311.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 311.25 | 305.27 | 311.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 311.80 | 305.27 | 311.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 308.00 | 305.30 | 311.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 305.50 | 305.30 | 311.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 307.10 | 305.32 | 311.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 306.80 | 305.46 | 311.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:00:00 | 307.00 | 305.56 | 311.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 313.40 | 305.69 | 311.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 313.40 | 305.69 | 311.00 | SL hit (close>static) qty=1.00 sl=311.50 alert=retest2 |

### Cycle 10 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 319.00 | 313.70 | 313.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 320.75 | 314.22 | 313.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 09:15:00 | 314.35 | 314.46 | 314.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 314.35 | 314.46 | 314.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 316.75 | 314.48 | 314.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:30:00 | 317.30 | 314.50 | 314.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 12:30:00 | 317.60 | 314.54 | 314.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 13:00:00 | 318.50 | 314.91 | 314.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 317.15 | 314.96 | 314.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 314.90 | 314.96 | 314.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 314.55 | 314.96 | 314.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 314.25 | 314.95 | 314.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 314.25 | 314.95 | 314.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 314.15 | 314.94 | 314.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:30:00 | 314.55 | 314.95 | 314.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:00:00 | 315.95 | 314.95 | 314.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 311.25 | 314.91 | 314.36 | SL hit (close<static) qty=1.00 sl=312.50 alert=retest2 |

### Cycle 11 — SELL (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 14:15:00 | 312.35 | 313.92 | 313.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 310.90 | 313.88 | 313.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 307.05 | 301.58 | 306.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 307.05 | 301.58 | 306.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 307.05 | 301.58 | 306.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 308.60 | 301.58 | 306.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 313.85 | 301.70 | 306.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 313.85 | 301.70 | 306.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 311.95 | 301.80 | 306.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:00:00 | 306.00 | 301.85 | 306.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 290.70 | 300.26 | 304.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 294.70 | 292.78 | 299.18 | SL hit (close>ema200) qty=0.50 sl=292.78 alert=retest2 |

### Cycle 12 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 311.65 | 302.20 | 302.16 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 13:15:00 | 294.40 | 302.26 | 302.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 289.35 | 301.75 | 302.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 09:15:00 | 305.65 | 301.20 | 301.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 305.65 | 301.20 | 301.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 305.65 | 301.20 | 301.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 305.65 | 301.20 | 301.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 302.55 | 301.21 | 301.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 301.70 | 301.21 | 301.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 301.00 | 301.33 | 301.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 298.00 | 301.53 | 301.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 300.30 | 301.55 | 301.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 302.00 | 301.56 | 301.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 302.00 | 301.56 | 301.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 299.50 | 301.54 | 301.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 295.45 | 301.52 | 301.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:45:00 | 299.10 | 301.18 | 301.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 15:15:00 | 303.00 | 301.20 | 301.63 | SL hit (close>static) qty=1.00 sl=302.40 alert=retest2 |

### Cycle 14 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 314.55 | 301.96 | 301.95 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 13:15:00 | 285.45 | 302.74 | 302.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 283.05 | 301.60 | 302.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 275.04 | 255.05 | 270.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 275.04 | 255.05 | 270.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 275.04 | 255.05 | 270.23 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-09-06 11:15:00 | 280.70 | 2023-09-08 11:15:00 | 288.65 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2023-09-06 13:15:00 | 280.10 | 2023-09-08 11:15:00 | 288.65 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2023-09-06 14:45:00 | 280.00 | 2023-09-08 11:15:00 | 288.65 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2023-09-12 12:45:00 | 280.95 | 2023-09-20 14:15:00 | 287.05 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest1 | 2023-10-20 09:15:00 | 287.45 | 2023-10-20 13:15:00 | 284.25 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest1 | 2023-10-20 12:15:00 | 286.00 | 2023-10-20 13:15:00 | 284.25 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2023-11-07 11:45:00 | 279.80 | 2023-11-28 14:15:00 | 282.95 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-11-15 10:45:00 | 280.35 | 2023-12-01 09:15:00 | 290.75 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2023-11-16 10:45:00 | 279.80 | 2023-12-01 09:15:00 | 290.75 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2023-11-28 10:00:00 | 280.45 | 2023-12-01 09:15:00 | 290.75 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2023-11-28 13:30:00 | 278.70 | 2023-12-01 09:15:00 | 290.75 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2023-12-21 10:45:00 | 288.50 | 2024-01-01 09:15:00 | 317.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-21 11:30:00 | 288.75 | 2024-01-01 09:15:00 | 317.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-21 14:00:00 | 288.30 | 2024-01-01 09:15:00 | 317.13 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-21 10:15:00 | 306.25 | 2024-07-05 09:15:00 | 312.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-06-21 11:45:00 | 305.70 | 2024-07-05 09:15:00 | 312.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-06-21 14:15:00 | 305.95 | 2024-07-05 09:15:00 | 312.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-06-24 14:00:00 | 305.90 | 2024-07-05 09:15:00 | 312.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-08-13 11:15:00 | 323.55 | 2024-08-13 12:15:00 | 317.65 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-08-19 09:15:00 | 326.35 | 2024-08-19 14:15:00 | 319.95 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-08-20 10:15:00 | 323.40 | 2024-08-26 09:15:00 | 355.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 13:45:00 | 323.80 | 2024-08-26 09:15:00 | 356.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-18 10:30:00 | 392.25 | 2024-10-24 13:15:00 | 388.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-10-22 09:30:00 | 392.55 | 2024-10-24 13:15:00 | 388.70 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-10-23 12:30:00 | 392.10 | 2024-10-24 13:15:00 | 388.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-10-23 14:45:00 | 392.45 | 2024-10-24 13:15:00 | 388.70 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-10-24 09:15:00 | 393.50 | 2024-10-25 09:15:00 | 383.95 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-10-24 09:45:00 | 391.10 | 2024-10-25 09:15:00 | 383.95 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-10-24 11:30:00 | 390.55 | 2024-10-25 09:15:00 | 383.95 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-10-24 13:00:00 | 390.60 | 2024-10-25 09:15:00 | 383.95 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-10-28 15:15:00 | 393.20 | 2024-10-30 15:15:00 | 387.45 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-10-30 09:15:00 | 394.00 | 2024-10-30 15:15:00 | 387.45 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-10-30 12:30:00 | 392.55 | 2024-10-30 15:15:00 | 387.45 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-11-07 09:15:00 | 392.20 | 2024-11-08 09:15:00 | 384.20 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-11-07 14:00:00 | 392.40 | 2024-11-08 09:15:00 | 384.20 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-12-04 12:30:00 | 366.35 | 2024-12-05 14:15:00 | 376.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-12-18 12:00:00 | 366.40 | 2024-12-20 10:15:00 | 373.45 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-12-18 13:30:00 | 365.50 | 2024-12-20 10:15:00 | 373.45 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-12-19 09:15:00 | 363.80 | 2024-12-20 10:15:00 | 373.45 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-01-03 15:00:00 | 367.05 | 2025-01-10 09:15:00 | 348.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 15:00:00 | 367.05 | 2025-01-17 09:15:00 | 382.85 | STOP_HIT | 0.50 | -4.30% |
| SELL | retest2 | 2025-01-17 15:00:00 | 362.85 | 2025-01-24 15:15:00 | 349.17 | PARTIAL | 0.50 | 3.77% |
| SELL | retest2 | 2025-01-20 10:30:00 | 367.55 | 2025-01-24 15:15:00 | 348.84 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2025-01-20 11:30:00 | 367.20 | 2025-01-27 09:15:00 | 344.71 | PARTIAL | 0.50 | 6.13% |
| SELL | retest2 | 2025-01-21 10:15:00 | 360.60 | 2025-01-27 09:15:00 | 343.85 | PARTIAL | 0.50 | 4.64% |
| SELL | retest2 | 2025-01-22 11:00:00 | 361.95 | 2025-01-27 11:15:00 | 342.57 | PARTIAL | 0.50 | 5.35% |
| SELL | retest2 | 2025-01-17 15:00:00 | 362.85 | 2025-02-03 09:15:00 | 330.80 | TARGET_HIT | 0.50 | 8.83% |
| SELL | retest2 | 2025-01-20 10:30:00 | 367.55 | 2025-02-07 13:15:00 | 330.48 | TARGET_HIT | 0.50 | 10.09% |
| SELL | retest2 | 2025-01-20 11:30:00 | 367.20 | 2025-02-10 09:15:00 | 326.57 | TARGET_HIT | 0.50 | 11.07% |
| SELL | retest2 | 2025-01-21 10:15:00 | 360.60 | 2025-02-10 11:15:00 | 324.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-22 11:00:00 | 361.95 | 2025-02-10 11:15:00 | 325.75 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-11 09:15:00 | 330.15 | 2025-07-14 09:15:00 | 325.40 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-16 13:30:00 | 330.90 | 2025-07-21 11:15:00 | 325.75 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-21 13:45:00 | 329.85 | 2025-07-25 10:15:00 | 325.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-21 14:15:00 | 330.00 | 2025-07-25 10:15:00 | 325.20 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-12 09:15:00 | 305.50 | 2025-09-16 09:15:00 | 313.40 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-09-12 10:45:00 | 307.10 | 2025-09-16 09:15:00 | 313.40 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-09-15 10:00:00 | 306.80 | 2025-09-16 09:15:00 | 313.40 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-09-15 13:00:00 | 307.00 | 2025-09-16 09:15:00 | 313.40 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-29 12:30:00 | 306.70 | 2025-10-01 12:15:00 | 315.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-09-29 14:45:00 | 307.10 | 2025-10-01 12:15:00 | 315.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-09-30 09:30:00 | 307.50 | 2025-10-01 12:15:00 | 315.00 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-09-30 10:15:00 | 306.90 | 2025-10-01 12:15:00 | 315.00 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-10-15 11:30:00 | 317.30 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-10-15 12:30:00 | 317.60 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-16 13:00:00 | 318.50 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-10-17 09:30:00 | 317.15 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-10-17 13:30:00 | 314.55 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-17 14:00:00 | 315.95 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-24 09:15:00 | 315.25 | 2025-10-24 13:15:00 | 313.05 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-29 10:15:00 | 315.20 | 2025-10-29 15:15:00 | 313.30 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-10-29 11:45:00 | 318.35 | 2025-10-30 09:15:00 | 310.55 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-12-02 13:00:00 | 306.00 | 2025-12-08 14:15:00 | 290.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 13:00:00 | 306.00 | 2025-12-19 15:15:00 | 294.70 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2026-01-01 10:00:00 | 309.55 | 2026-01-02 11:15:00 | 318.55 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2026-01-09 09:15:00 | 309.10 | 2026-01-09 09:15:00 | 311.65 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-29 11:15:00 | 301.70 | 2026-02-03 15:15:00 | 303.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-01-30 09:15:00 | 301.00 | 2026-02-03 15:15:00 | 303.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-02-01 09:15:00 | 298.00 | 2026-02-06 10:15:00 | 304.65 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-02-01 12:15:00 | 300.30 | 2026-02-06 10:15:00 | 304.65 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-02-02 09:15:00 | 295.45 | 2026-02-06 14:15:00 | 305.95 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2026-02-03 10:45:00 | 299.10 | 2026-02-06 14:15:00 | 305.95 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-02-04 09:15:00 | 298.80 | 2026-02-06 14:15:00 | 305.95 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-02-05 09:15:00 | 298.70 | 2026-02-06 14:15:00 | 305.95 | STOP_HIT | 1.00 | -2.43% |
