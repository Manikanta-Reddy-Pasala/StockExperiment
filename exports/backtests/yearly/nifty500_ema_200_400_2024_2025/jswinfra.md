# JSW Infrastructure Ltd. (JSWINFRA)

## Backtest Summary

- **Window:** 2023-10-03 09:15:00 → 2026-05-08 15:15:00 (4484 bars)
- **Last close:** 284.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 0 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 44 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 39
- **Target hits / Stop hits / Partials:** 0 / 45 / 6
- **Avg / median % per leg:** -1.46% / -2.18%
- **Sum % (uncompounded):** -74.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 0 | 0.0% | 0 | 23 | 0 | -2.65% | -60.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.53% | -1.5% |
| BUY @ 3rd Alert (retest2) | 22 | 0 | 0.0% | 0 | 22 | 0 | -2.70% | -59.3% |
| SELL (all) | 28 | 12 | 42.9% | 0 | 22 | 6 | -0.49% | -13.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 12 | 42.9% | 0 | 22 | 6 | -0.49% | -13.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.53% | -1.5% |
| retest2 (combined) | 50 | 12 | 24.0% | 0 | 44 | 6 | -1.46% | -73.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 09:15:00 | 292.75 | 321.58 | 321.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 14:15:00 | 289.25 | 320.09 | 320.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 316.00 | 314.93 | 318.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 14:30:00 | 315.70 | 314.93 | 318.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 320.35 | 315.00 | 318.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:45:00 | 320.40 | 315.00 | 318.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 318.30 | 315.03 | 318.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 14:30:00 | 316.35 | 315.15 | 318.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 15:15:00 | 316.00 | 315.15 | 318.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 10:00:00 | 316.20 | 315.17 | 318.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-01 17:15:00 | 323.15 | 315.18 | 317.90 | SL hit (close>static) qty=1.00 sl=320.80 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 13:15:00 | 326.25 | 315.13 | 315.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 15:15:00 | 329.35 | 315.41 | 315.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 15:15:00 | 317.20 | 317.29 | 316.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 09:15:00 | 319.50 | 317.29 | 316.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 319.45 | 317.31 | 316.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:45:00 | 316.85 | 317.31 | 316.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 317.85 | 317.36 | 316.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:45:00 | 317.30 | 317.36 | 316.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 314.60 | 317.35 | 316.39 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-19 09:15:00 | 314.60 | 317.35 | 316.39 | SL hit (close<ema400) qty=1.00 sl=316.39 alert=retest1 |

### Cycle 3 — SELL (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 09:15:00 | 293.80 | 316.23 | 316.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 288.45 | 308.95 | 312.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 09:15:00 | 259.45 | 259.23 | 277.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:45:00 | 259.90 | 259.23 | 277.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 270.60 | 256.99 | 271.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:30:00 | 271.00 | 256.99 | 271.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 271.95 | 257.14 | 271.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:30:00 | 272.00 | 257.14 | 271.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 269.50 | 257.26 | 271.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 11:30:00 | 272.60 | 257.26 | 271.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 272.25 | 257.41 | 271.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:45:00 | 272.70 | 257.41 | 271.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 270.90 | 257.54 | 271.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:30:00 | 272.80 | 257.54 | 271.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 270.75 | 257.68 | 271.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:45:00 | 271.35 | 257.68 | 271.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 267.85 | 257.89 | 271.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 14:15:00 | 264.50 | 258.25 | 271.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 15:00:00 | 264.15 | 258.73 | 271.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 263.55 | 258.88 | 271.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 262.65 | 258.91 | 271.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 269.00 | 260.20 | 270.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 280.15 | 260.67 | 270.63 | SL hit (close>static) qty=1.00 sl=273.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 323.55 | 278.27 | 278.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 326.75 | 309.63 | 304.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 10:15:00 | 308.80 | 311.32 | 305.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 11:00:00 | 308.80 | 311.32 | 305.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 304.90 | 311.17 | 305.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:00:00 | 304.90 | 311.17 | 305.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 305.70 | 311.11 | 305.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 307.45 | 311.05 | 305.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:45:00 | 307.20 | 311.00 | 305.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:30:00 | 307.45 | 310.88 | 305.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 15:15:00 | 306.90 | 310.79 | 306.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 306.90 | 310.75 | 306.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 305.40 | 310.75 | 306.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 305.25 | 310.70 | 306.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 307.65 | 310.66 | 306.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 303.80 | 310.16 | 306.08 | SL hit (close<static) qty=1.00 sl=304.25 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 300.00 | 304.29 | 304.29 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-09 11:15:00)

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

### Cycle 7 — SELL (started 2025-10-23 11:15:00)

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

### Cycle 8 — BUY (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 14:15:00 | 274.71 | 261.75 | 261.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 279.20 | 262.05 | 261.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 266.01 | 266.66 | 264.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 266.01 | 266.66 | 264.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-30 14:30:00 | 316.35 | 2024-11-01 17:15:00 | 323.15 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-10-30 15:15:00 | 316.00 | 2024-11-01 17:15:00 | 323.15 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-10-31 10:00:00 | 316.20 | 2024-11-01 17:15:00 | 323.15 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-11-04 09:15:00 | 314.55 | 2024-11-06 09:15:00 | 320.40 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-11-05 09:30:00 | 309.60 | 2024-11-06 09:15:00 | 320.40 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2024-11-05 15:00:00 | 310.40 | 2024-11-13 09:15:00 | 298.82 | PARTIAL | 0.50 | 3.73% |
| SELL | retest2 | 2024-11-08 11:15:00 | 310.00 | 2024-11-13 09:15:00 | 294.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 11:00:00 | 310.35 | 2024-11-13 09:15:00 | 294.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-05 15:00:00 | 310.40 | 2024-11-26 09:15:00 | 308.65 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2024-11-08 11:15:00 | 310.00 | 2024-11-26 09:15:00 | 308.65 | STOP_HIT | 0.50 | 0.44% |
| SELL | retest2 | 2024-11-11 11:00:00 | 310.35 | 2024-11-26 09:15:00 | 308.65 | STOP_HIT | 0.50 | 0.55% |
| SELL | retest2 | 2024-11-27 15:00:00 | 309.45 | 2024-11-28 09:15:00 | 313.90 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-11-29 11:15:00 | 309.90 | 2024-12-02 09:15:00 | 318.10 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-11-29 13:45:00 | 309.85 | 2024-12-02 09:15:00 | 318.10 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest1 | 2024-12-18 09:15:00 | 319.50 | 2024-12-19 09:15:00 | 314.60 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-12-27 11:45:00 | 318.15 | 2025-01-09 12:15:00 | 316.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-12-27 12:15:00 | 318.35 | 2025-01-09 12:15:00 | 316.50 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-12-27 13:15:00 | 318.35 | 2025-01-10 09:15:00 | 306.50 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2024-12-27 13:45:00 | 321.30 | 2025-01-10 09:15:00 | 306.50 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2025-01-01 09:15:00 | 320.00 | 2025-01-10 09:15:00 | 306.50 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2025-01-01 10:00:00 | 320.05 | 2025-01-10 09:15:00 | 306.50 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2025-01-01 11:15:00 | 320.10 | 2025-01-10 09:15:00 | 306.50 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2025-01-07 09:30:00 | 320.90 | 2025-01-10 09:15:00 | 306.50 | STOP_HIT | 1.00 | -4.49% |
| BUY | retest2 | 2025-01-07 11:30:00 | 320.30 | 2025-01-10 09:15:00 | 306.50 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2025-01-07 15:00:00 | 321.80 | 2025-01-10 09:15:00 | 306.50 | STOP_HIT | 1.00 | -4.75% |
| SELL | retest2 | 2025-03-10 14:15:00 | 264.50 | 2025-03-19 09:15:00 | 280.15 | STOP_HIT | 1.00 | -5.92% |
| SELL | retest2 | 2025-03-11 15:00:00 | 264.15 | 2025-03-19 09:15:00 | 280.15 | STOP_HIT | 1.00 | -6.06% |
| SELL | retest2 | 2025-03-12 10:15:00 | 263.55 | 2025-03-19 09:15:00 | 280.15 | STOP_HIT | 1.00 | -6.30% |
| SELL | retest2 | 2025-03-12 10:45:00 | 262.65 | 2025-03-19 09:15:00 | 280.15 | STOP_HIT | 1.00 | -6.66% |
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
