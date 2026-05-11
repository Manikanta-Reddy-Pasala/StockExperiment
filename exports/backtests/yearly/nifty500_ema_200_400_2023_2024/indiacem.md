# India Cements Ltd. (INDIACEM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 408.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 56 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 60 |
| PARTIAL | 9 |
| TARGET_HIT | 13 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 47
- **Target hits / Stop hits / Partials:** 13 / 47 / 9
- **Avg / median % per leg:** 0.86% / -1.60%
- **Sum % (uncompounded):** 59.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 3 | 18.8% | 3 | 13 | 0 | -0.22% | -3.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 3 | 18.8% | 3 | 13 | 0 | -0.22% | -3.5% |
| SELL (all) | 53 | 19 | 35.8% | 10 | 34 | 9 | 1.19% | 62.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 53 | 19 | 35.8% | 10 | 34 | 9 | 1.19% | 62.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 69 | 22 | 31.9% | 13 | 47 | 9 | 0.86% | 59.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 11:15:00 | 209.95 | 228.91 | 229.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 205.25 | 228.68 | 228.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 12:15:00 | 219.35 | 217.60 | 221.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-16 12:30:00 | 219.90 | 217.60 | 221.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 220.45 | 217.69 | 221.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 13:45:00 | 219.15 | 217.76 | 221.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 14:30:00 | 218.05 | 217.76 | 221.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 13:45:00 | 218.70 | 217.81 | 221.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 09:45:00 | 218.80 | 217.83 | 221.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 221.00 | 217.90 | 221.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:30:00 | 221.20 | 217.90 | 221.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 11:15:00 | 219.45 | 217.91 | 221.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 13:00:00 | 219.05 | 217.92 | 221.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 09:15:00 | 226.30 | 218.07 | 221.08 | SL hit (close>static) qty=1.00 sl=221.90 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 15:15:00 | 253.00 | 223.81 | 223.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 254.35 | 224.12 | 223.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 246.90 | 247.30 | 238.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-21 09:15:00 | 245.75 | 247.30 | 238.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 246.85 | 255.93 | 248.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 244.40 | 255.93 | 248.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 249.60 | 255.87 | 248.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 11:45:00 | 250.95 | 255.82 | 248.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 14:30:00 | 250.30 | 255.63 | 248.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:15:00 | 251.85 | 255.56 | 248.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-20 12:45:00 | 251.25 | 255.21 | 248.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 248.70 | 255.05 | 248.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 248.70 | 255.05 | 248.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 246.65 | 254.97 | 248.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 11:15:00 | 246.60 | 254.97 | 248.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 243.20 | 254.85 | 248.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 243.20 | 254.85 | 248.57 | SL hit (close<static) qty=1.00 sl=245.30 alert=retest2 |

### Cycle 3 — SELL (started 2024-02-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 13:15:00 | 236.70 | 245.93 | 245.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-15 14:15:00 | 235.70 | 245.82 | 245.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 247.60 | 245.03 | 245.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 247.60 | 245.03 | 245.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 247.60 | 245.03 | 245.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 10:00:00 | 247.60 | 245.03 | 245.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 246.85 | 245.05 | 245.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 14:30:00 | 243.15 | 245.05 | 245.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 15:15:00 | 248.90 | 245.15 | 245.52 | SL hit (close>static) qty=1.00 sl=248.15 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 263.78 | 219.39 | 219.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 11:15:00 | 264.80 | 219.84 | 219.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 14:15:00 | 361.90 | 362.19 | 342.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-20 15:00:00 | 361.90 | 362.19 | 342.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 352.10 | 361.57 | 353.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 12:45:00 | 357.80 | 360.97 | 353.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 09:45:00 | 357.40 | 360.60 | 353.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 12:00:00 | 355.85 | 360.64 | 354.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 12:30:00 | 356.65 | 360.59 | 354.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 355.30 | 360.14 | 355.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:00:00 | 355.30 | 360.14 | 355.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 355.45 | 360.10 | 355.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 14:00:00 | 357.35 | 359.24 | 355.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 14:30:00 | 356.65 | 359.03 | 355.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 11:15:00 | 355.05 | 358.89 | 355.18 | SL hit (close<static) qty=1.00 sl=355.10 alert=retest2 |

### Cycle 5 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 332.60 | 354.85 | 354.91 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 376.45 | 354.54 | 354.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 378.90 | 356.09 | 355.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 368.25 | 370.43 | 364.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 368.25 | 370.43 | 364.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 355.85 | 370.28 | 364.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 355.85 | 370.28 | 364.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 359.40 | 370.17 | 364.67 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 297.70 | 359.71 | 359.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 276.10 | 357.00 | 358.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 12:15:00 | 280.90 | 279.98 | 303.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 12:30:00 | 280.65 | 279.98 | 303.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 12:15:00 | 287.95 | 278.27 | 288.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 13:15:00 | 285.75 | 278.27 | 288.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 09:15:00 | 294.40 | 278.57 | 288.08 | SL hit (close>static) qty=1.00 sl=289.25 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 13:15:00 | 315.10 | 292.78 | 292.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 319.00 | 294.97 | 293.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 15:15:00 | 312.65 | 315.32 | 306.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 09:15:00 | 317.60 | 315.32 | 306.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 319.00 | 327.17 | 317.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 319.00 | 327.17 | 317.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 319.80 | 327.10 | 317.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 317.80 | 327.10 | 317.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 316.30 | 326.83 | 317.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 315.50 | 326.83 | 317.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 317.60 | 326.74 | 317.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 316.65 | 326.74 | 317.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 315.05 | 326.62 | 317.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 315.05 | 326.62 | 317.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 312.85 | 326.48 | 317.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:45:00 | 312.95 | 326.48 | 317.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 316.60 | 324.58 | 317.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 316.60 | 324.58 | 317.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 315.65 | 324.49 | 317.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:00:00 | 315.65 | 324.49 | 317.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 312.35 | 324.37 | 317.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 312.35 | 324.37 | 317.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 337.20 | 339.25 | 330.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:00:00 | 343.00 | 339.26 | 330.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:30:00 | 342.85 | 339.28 | 330.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 358.85 | 339.31 | 330.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 09:15:00 | 377.30 | 356.96 | 345.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 12:15:00 | 387.10 | 432.68 | 432.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 382.55 | 426.25 | 429.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 376.75 | 375.97 | 394.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 376.75 | 375.97 | 394.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 393.20 | 377.82 | 392.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:00:00 | 393.20 | 377.82 | 392.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 395.90 | 378.00 | 392.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:15:00 | 396.00 | 378.00 | 392.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 392.40 | 398.32 | 400.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 391.15 | 398.32 | 400.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:45:00 | 391.70 | 398.26 | 400.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 391.10 | 397.96 | 400.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 391.75 | 397.69 | 399.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 392.55 | 397.46 | 399.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 411.10 | 397.64 | 399.74 | SL hit (close>static) qty=1.00 sl=404.85 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-17 13:45:00 | 219.15 | 2023-11-28 09:15:00 | 226.30 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2023-11-17 14:30:00 | 218.05 | 2023-11-28 09:15:00 | 226.30 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2023-11-20 13:45:00 | 218.70 | 2023-11-28 09:15:00 | 226.30 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2023-11-21 09:45:00 | 218.80 | 2023-11-28 09:15:00 | 226.30 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2023-11-24 13:00:00 | 219.05 | 2023-11-28 09:15:00 | 226.30 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2024-01-18 11:45:00 | 250.95 | 2024-01-23 11:15:00 | 243.20 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2024-01-18 14:30:00 | 250.30 | 2024-01-23 11:15:00 | 243.20 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-01-19 09:15:00 | 251.85 | 2024-01-23 11:15:00 | 243.20 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2024-01-20 12:45:00 | 251.25 | 2024-01-23 11:15:00 | 243.20 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-02-21 14:30:00 | 243.15 | 2024-02-22 15:15:00 | 248.90 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-02-23 13:30:00 | 244.20 | 2024-02-28 10:15:00 | 231.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-26 09:15:00 | 243.40 | 2024-02-28 10:15:00 | 231.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-26 12:30:00 | 242.70 | 2024-02-28 10:15:00 | 230.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-23 13:30:00 | 244.20 | 2024-03-06 11:15:00 | 219.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-26 09:15:00 | 243.40 | 2024-03-11 15:15:00 | 219.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-26 12:30:00 | 242.70 | 2024-03-12 09:15:00 | 218.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-03 09:15:00 | 224.05 | 2024-04-08 10:15:00 | 231.45 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2024-04-04 10:30:00 | 225.95 | 2024-04-08 10:15:00 | 231.45 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-04-08 09:45:00 | 226.25 | 2024-04-08 10:15:00 | 231.45 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-04-08 13:15:00 | 226.85 | 2024-04-08 14:15:00 | 231.30 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-04-09 13:30:00 | 225.10 | 2024-04-10 13:15:00 | 229.30 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-04-10 09:15:00 | 225.30 | 2024-04-10 13:15:00 | 229.30 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-04-10 10:00:00 | 225.75 | 2024-04-10 13:15:00 | 229.30 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-04-10 12:15:00 | 225.25 | 2024-04-10 13:15:00 | 229.30 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-04-12 15:15:00 | 224.05 | 2024-04-24 09:15:00 | 228.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-04-16 11:00:00 | 224.50 | 2024-04-24 09:15:00 | 228.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-04-16 11:45:00 | 224.35 | 2024-04-24 09:15:00 | 228.50 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-04-16 13:00:00 | 223.90 | 2024-04-24 09:15:00 | 228.50 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-04-18 14:15:00 | 224.70 | 2024-04-26 10:15:00 | 228.40 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-04-19 13:15:00 | 224.80 | 2024-04-26 10:15:00 | 228.40 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-04-19 14:00:00 | 224.85 | 2024-05-03 09:15:00 | 227.60 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-04-22 09:30:00 | 224.70 | 2024-05-07 09:15:00 | 213.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-25 12:00:00 | 225.15 | 2024-05-07 09:15:00 | 213.56 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2024-04-25 14:45:00 | 225.50 | 2024-05-07 09:15:00 | 213.61 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2024-04-30 14:45:00 | 225.50 | 2024-05-07 09:15:00 | 213.46 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2024-05-03 11:15:00 | 224.50 | 2024-05-07 10:15:00 | 213.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-22 09:30:00 | 224.70 | 2024-05-09 15:15:00 | 202.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-25 12:00:00 | 225.15 | 2024-05-09 15:15:00 | 202.32 | TARGET_HIT | 0.50 | 10.14% |
| SELL | retest2 | 2024-04-25 14:45:00 | 225.50 | 2024-05-09 15:15:00 | 202.37 | TARGET_HIT | 0.50 | 10.26% |
| SELL | retest2 | 2024-04-30 14:45:00 | 225.50 | 2024-05-09 15:15:00 | 202.23 | TARGET_HIT | 0.50 | 10.32% |
| SELL | retest2 | 2024-05-03 11:15:00 | 224.50 | 2024-05-09 15:15:00 | 202.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 14:30:00 | 215.35 | 2024-06-04 09:15:00 | 204.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 14:30:00 | 215.35 | 2024-06-04 11:15:00 | 193.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 202.00 | 2024-06-04 11:15:00 | 181.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-07 11:45:00 | 214.60 | 2024-06-10 09:15:00 | 222.03 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-10-28 12:45:00 | 357.80 | 2024-11-21 11:15:00 | 355.05 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-10-30 09:45:00 | 357.40 | 2024-11-21 11:15:00 | 355.05 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-11-08 12:00:00 | 355.85 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-11-08 12:30:00 | 356.65 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-11-18 14:00:00 | 357.35 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-11-19 14:30:00 | 356.65 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-11-22 11:45:00 | 356.65 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-11-25 09:15:00 | 358.95 | 2024-12-09 09:15:00 | 346.20 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-12-11 09:15:00 | 349.50 | 2024-12-11 10:15:00 | 344.70 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-04-16 13:15:00 | 285.75 | 2025-04-17 09:15:00 | 294.40 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-04-21 09:15:00 | 285.90 | 2025-04-22 15:15:00 | 289.50 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-04-21 11:15:00 | 286.70 | 2025-04-22 15:15:00 | 289.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-04-22 11:15:00 | 285.00 | 2025-04-22 15:15:00 | 289.50 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-04-23 09:15:00 | 287.65 | 2025-04-24 09:15:00 | 291.15 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-04-23 14:00:00 | 288.05 | 2025-04-24 09:15:00 | 291.15 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-04-23 15:15:00 | 288.00 | 2025-04-24 09:15:00 | 291.15 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-04-25 09:45:00 | 287.00 | 2025-04-28 12:15:00 | 290.05 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-21 14:00:00 | 343.00 | 2025-08-18 09:15:00 | 377.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-21 14:30:00 | 342.85 | 2025-08-18 09:15:00 | 377.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-22 09:15:00 | 358.85 | 2025-08-21 12:15:00 | 394.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 10:15:00 | 391.15 | 2026-05-07 09:15:00 | 411.10 | STOP_HIT | 1.00 | -5.10% |
| SELL | retest2 | 2026-04-30 10:45:00 | 391.70 | 2026-05-07 09:15:00 | 411.10 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2026-05-04 12:00:00 | 391.10 | 2026-05-07 09:15:00 | 411.10 | STOP_HIT | 1.00 | -5.11% |
| SELL | retest2 | 2026-05-05 09:15:00 | 391.75 | 2026-05-07 09:15:00 | 411.10 | STOP_HIT | 1.00 | -4.94% |
