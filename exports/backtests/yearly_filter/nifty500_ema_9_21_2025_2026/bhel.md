# Bharat Heavy Electricals Ltd. (BHEL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 403.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 61 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 23 |
| ALERT3 | 147 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 48 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 29
- **Target hits / Stop hits / Partials:** 4 / 47 / 6
- **Avg / median % per leg:** 1.11% / -0.14%
- **Sum % (uncompounded):** 63.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 7 | 35.0% | 4 | 15 | 1 | 1.98% | 39.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.99% | 11.9% |
| BUY @ 3rd Alert (retest2) | 16 | 5 | 31.2% | 3 | 13 | 0 | 1.73% | 27.6% |
| SELL (all) | 37 | 21 | 56.8% | 0 | 32 | 5 | 0.64% | 23.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 21 | 56.8% | 0 | 32 | 5 | 0.64% | 23.7% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.99% | 11.9% |
| retest2 (combined) | 53 | 26 | 49.1% | 3 | 45 | 5 | 0.97% | 51.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 229.90 | 221.16 | 220.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 230.95 | 223.12 | 221.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 235.63 | 236.92 | 232.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:00:00 | 235.63 | 236.92 | 232.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 245.00 | 248.69 | 246.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:00:00 | 245.00 | 248.69 | 246.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 244.70 | 247.89 | 246.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 244.70 | 247.89 | 246.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 247.24 | 246.59 | 246.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 246.63 | 246.59 | 246.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 247.70 | 247.16 | 246.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 246.40 | 247.16 | 246.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 246.52 | 247.03 | 246.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 246.52 | 247.03 | 246.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 244.68 | 246.56 | 246.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 244.06 | 246.56 | 246.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 244.65 | 246.18 | 246.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 243.23 | 246.18 | 246.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 246.20 | 246.75 | 246.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 246.20 | 246.75 | 246.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 248.55 | 247.11 | 246.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 248.90 | 247.76 | 247.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:45:00 | 250.01 | 247.90 | 247.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 13:15:00 | 262.52 | 263.58 | 263.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 262.52 | 263.58 | 263.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 259.77 | 262.82 | 263.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 261.35 | 261.28 | 262.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:45:00 | 261.10 | 261.28 | 262.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 261.85 | 261.39 | 262.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:45:00 | 262.15 | 261.39 | 262.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 257.65 | 260.62 | 261.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 256.30 | 259.09 | 260.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:00:00 | 255.80 | 259.09 | 260.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 255.70 | 254.12 | 254.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 258.75 | 255.65 | 255.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 258.75 | 255.65 | 255.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 260.65 | 257.94 | 256.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 260.20 | 260.26 | 258.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 258.05 | 260.26 | 258.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 258.75 | 259.96 | 258.70 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 257.20 | 258.21 | 258.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 255.55 | 257.68 | 258.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 254.10 | 253.74 | 255.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 254.10 | 253.74 | 255.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 254.65 | 253.26 | 254.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 254.65 | 253.26 | 254.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 255.50 | 253.71 | 254.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 255.50 | 253.71 | 254.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 254.70 | 253.91 | 254.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:15:00 | 254.00 | 253.91 | 254.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 254.20 | 253.96 | 254.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 253.75 | 253.23 | 253.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 253.10 | 250.64 | 250.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 253.10 | 250.64 | 250.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 254.95 | 251.91 | 251.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 264.40 | 264.99 | 263.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 264.40 | 264.99 | 263.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 263.90 | 264.77 | 263.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 266.40 | 264.77 | 263.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:30:00 | 264.85 | 265.23 | 264.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 264.45 | 264.66 | 264.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 262.00 | 264.25 | 264.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 262.00 | 264.25 | 264.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 261.75 | 263.75 | 264.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 261.65 | 259.73 | 260.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 261.65 | 259.73 | 260.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 261.65 | 259.73 | 260.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 262.15 | 259.73 | 260.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 262.60 | 260.31 | 261.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 262.60 | 260.31 | 261.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 259.55 | 260.24 | 260.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 258.40 | 260.01 | 260.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:30:00 | 258.30 | 258.21 | 258.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 263.50 | 259.27 | 258.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 263.50 | 259.27 | 258.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 264.30 | 260.28 | 259.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 261.30 | 261.43 | 260.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 10:15:00 | 260.85 | 261.31 | 260.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 260.85 | 261.31 | 260.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 260.85 | 261.31 | 260.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 260.50 | 261.20 | 260.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 260.60 | 261.20 | 260.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 261.20 | 261.20 | 260.68 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 258.40 | 260.13 | 260.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 12:15:00 | 256.85 | 257.59 | 258.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 14:15:00 | 257.65 | 257.47 | 258.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 15:00:00 | 257.65 | 257.47 | 258.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 255.60 | 257.14 | 257.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:00:00 | 254.60 | 256.63 | 257.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 15:00:00 | 254.60 | 255.46 | 256.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 254.00 | 254.88 | 256.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 15:15:00 | 254.35 | 253.52 | 253.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 254.35 | 253.52 | 253.46 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 252.95 | 253.41 | 253.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 252.05 | 253.14 | 253.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 13:15:00 | 250.50 | 250.34 | 251.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:00:00 | 250.50 | 250.34 | 251.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 251.30 | 250.53 | 251.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 251.30 | 250.53 | 251.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 250.55 | 250.53 | 251.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 250.90 | 250.53 | 251.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 249.15 | 250.26 | 251.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 247.45 | 249.95 | 250.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 235.08 | 238.28 | 241.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 239.10 | 237.84 | 240.73 | SL hit (close>ema200) qty=0.50 sl=237.84 alert=retest2 |

### Cycle 11 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 241.64 | 237.59 | 237.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 244.89 | 239.05 | 237.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 240.50 | 244.13 | 241.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 240.50 | 244.13 | 241.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 240.50 | 244.13 | 241.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 240.50 | 244.13 | 241.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 239.75 | 243.26 | 241.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 239.75 | 243.26 | 241.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 239.13 | 242.43 | 241.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 239.44 | 242.43 | 241.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 225.01 | 237.78 | 239.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 223.62 | 231.35 | 235.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 224.15 | 222.73 | 225.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 10:00:00 | 224.15 | 222.73 | 225.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 224.60 | 223.43 | 224.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:45:00 | 225.27 | 223.43 | 224.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 222.87 | 223.32 | 224.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 221.74 | 223.49 | 224.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:45:00 | 222.58 | 223.55 | 224.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 222.23 | 223.20 | 223.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 222.34 | 221.97 | 222.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 219.72 | 218.75 | 219.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:45:00 | 219.17 | 218.75 | 219.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 219.90 | 218.98 | 219.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 220.47 | 218.98 | 219.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 220.29 | 219.24 | 219.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 220.76 | 220.21 | 220.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 220.76 | 220.21 | 220.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 221.85 | 220.62 | 220.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 220.00 | 220.78 | 220.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 12:15:00 | 220.00 | 220.78 | 220.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 220.00 | 220.78 | 220.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 220.00 | 220.78 | 220.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 219.79 | 220.58 | 220.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 219.35 | 220.58 | 220.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 218.72 | 220.21 | 220.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 218.49 | 219.87 | 220.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 14:15:00 | 218.84 | 218.76 | 219.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 14:15:00 | 218.84 | 218.76 | 219.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 218.84 | 218.76 | 219.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 217.43 | 218.25 | 218.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:15:00 | 217.45 | 218.25 | 218.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:00:00 | 217.30 | 217.93 | 218.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 206.56 | 209.83 | 212.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 206.58 | 209.83 | 212.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 206.44 | 209.83 | 212.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 210.44 | 209.95 | 212.16 | SL hit (close>ema200) qty=0.50 sl=209.95 alert=retest2 |

### Cycle 15 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 212.70 | 211.54 | 211.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 214.50 | 212.13 | 211.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 214.99 | 215.12 | 214.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 214.99 | 215.12 | 214.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 216.09 | 215.32 | 214.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 214.66 | 215.32 | 214.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 215.01 | 215.89 | 215.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 214.91 | 215.89 | 215.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 214.11 | 215.53 | 214.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:15:00 | 213.78 | 215.53 | 214.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 212.07 | 214.84 | 214.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 212.28 | 214.84 | 214.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 210.85 | 214.04 | 214.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 210.08 | 212.12 | 213.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 212.15 | 211.92 | 212.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 212.15 | 211.92 | 212.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 212.15 | 211.92 | 212.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 212.15 | 211.92 | 212.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 212.90 | 212.19 | 212.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 215.15 | 212.19 | 212.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 216.25 | 213.00 | 213.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 216.25 | 213.00 | 213.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 216.39 | 213.68 | 213.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 218.00 | 214.54 | 213.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 215.88 | 216.01 | 215.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:30:00 | 216.11 | 216.01 | 215.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 216.18 | 216.04 | 215.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 215.39 | 216.04 | 215.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 215.24 | 215.88 | 215.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 215.24 | 215.88 | 215.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 215.38 | 215.78 | 215.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 215.22 | 215.78 | 215.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 214.65 | 215.56 | 215.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 214.65 | 215.56 | 215.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 214.60 | 215.36 | 215.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:15:00 | 214.95 | 215.36 | 215.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 216.95 | 215.62 | 215.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 217.51 | 215.99 | 215.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-19 10:15:00 | 239.26 | 235.37 | 233.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 236.00 | 237.72 | 237.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 234.15 | 235.91 | 236.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 233.92 | 232.46 | 233.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 10:15:00 | 233.92 | 232.46 | 233.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 233.92 | 232.46 | 233.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 233.92 | 232.46 | 233.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 232.27 | 232.42 | 233.59 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 236.70 | 234.08 | 234.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 15:15:00 | 238.85 | 237.01 | 235.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 243.93 | 244.61 | 243.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 243.93 | 244.61 | 243.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 243.93 | 244.61 | 243.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 243.85 | 244.61 | 243.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 243.38 | 244.36 | 243.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 243.23 | 244.36 | 243.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 243.66 | 244.22 | 243.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:00:00 | 245.00 | 244.38 | 243.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:30:00 | 244.51 | 244.48 | 243.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 240.70 | 243.66 | 243.46 | SL hit (close<static) qty=1.00 sl=243.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 240.44 | 243.01 | 243.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 238.68 | 241.76 | 242.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 239.45 | 239.42 | 240.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 240.67 | 239.42 | 240.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 240.64 | 239.67 | 240.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 240.64 | 239.67 | 240.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 240.39 | 239.81 | 240.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 241.00 | 239.81 | 240.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 240.80 | 240.01 | 240.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 240.80 | 240.01 | 240.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 241.75 | 240.36 | 240.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 241.75 | 240.36 | 240.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 240.65 | 240.42 | 240.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:15:00 | 240.42 | 240.42 | 240.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 239.30 | 240.19 | 240.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 15:15:00 | 236.30 | 235.66 | 235.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 236.30 | 235.66 | 235.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 236.75 | 235.88 | 235.71 | Break + close above crossover candle high |

### Cycle 22 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 232.53 | 235.43 | 235.57 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 237.65 | 234.92 | 234.76 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 234.85 | 235.18 | 235.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 231.81 | 234.51 | 234.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 235.01 | 233.67 | 234.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 235.01 | 233.67 | 234.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 235.01 | 233.67 | 234.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 235.85 | 233.67 | 234.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 233.69 | 233.68 | 234.27 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 235.60 | 234.66 | 234.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 237.50 | 235.58 | 235.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 12:15:00 | 263.10 | 263.53 | 258.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 263.10 | 263.53 | 258.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 261.65 | 265.22 | 263.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 261.65 | 265.22 | 263.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 260.45 | 264.26 | 263.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 260.20 | 264.26 | 263.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 259.95 | 262.55 | 262.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 257.10 | 261.16 | 261.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 261.25 | 261.17 | 261.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:45:00 | 260.50 | 261.17 | 261.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 262.65 | 261.47 | 261.95 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 264.70 | 262.50 | 262.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 267.55 | 263.93 | 263.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 285.90 | 286.43 | 282.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 281.95 | 284.99 | 282.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 281.95 | 284.99 | 282.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 281.95 | 284.99 | 282.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 282.00 | 284.39 | 282.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 282.00 | 284.39 | 282.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 283.05 | 283.61 | 282.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:00:00 | 285.10 | 283.17 | 282.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:15:00 | 285.80 | 283.49 | 282.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 10:30:00 | 284.65 | 284.76 | 283.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 283.60 | 286.56 | 286.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 283.60 | 286.56 | 286.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 12:15:00 | 279.90 | 282.73 | 284.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 282.05 | 280.71 | 282.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 282.05 | 280.71 | 282.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 282.05 | 280.71 | 282.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 282.05 | 280.71 | 282.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 282.50 | 281.07 | 282.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 284.40 | 281.07 | 282.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 282.15 | 281.28 | 282.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:00:00 | 281.60 | 281.35 | 282.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 282.95 | 281.78 | 282.47 | SL hit (close>static) qty=1.00 sl=282.85 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 286.45 | 283.42 | 283.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 289.00 | 284.53 | 283.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 15:15:00 | 290.20 | 290.62 | 289.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:15:00 | 293.55 | 290.62 | 289.43 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 291.50 | 291.64 | 290.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 290.65 | 291.64 | 290.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 290.65 | 291.44 | 290.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 290.80 | 291.44 | 290.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 290.85 | 291.33 | 290.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 288.25 | 290.71 | 290.40 | SL hit (close<ema400) qty=1.00 sl=290.40 alert=retest1 |

### Cycle 30 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 288.25 | 289.91 | 290.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 14:15:00 | 285.40 | 288.68 | 289.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 277.75 | 276.46 | 278.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 277.75 | 276.46 | 278.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 277.75 | 276.46 | 278.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 274.95 | 276.44 | 278.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 276.95 | 276.02 | 275.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 276.95 | 276.02 | 275.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 277.90 | 276.61 | 276.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 276.30 | 276.55 | 276.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 11:15:00 | 276.30 | 276.55 | 276.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 276.30 | 276.55 | 276.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:00:00 | 276.30 | 276.55 | 276.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 276.20 | 276.48 | 276.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:30:00 | 275.55 | 276.48 | 276.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 275.20 | 276.22 | 276.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 275.20 | 276.22 | 276.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 275.35 | 276.05 | 276.05 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 276.90 | 276.14 | 276.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 282.35 | 277.76 | 276.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 282.70 | 283.21 | 280.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:00:00 | 282.70 | 283.21 | 280.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 279.05 | 282.09 | 281.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 279.05 | 282.09 | 281.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 280.20 | 281.71 | 281.29 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 279.15 | 280.77 | 280.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 278.65 | 280.35 | 280.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 278.50 | 278.14 | 279.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 15:00:00 | 278.50 | 278.14 | 279.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 274.50 | 274.96 | 276.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 273.90 | 274.69 | 276.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:00:00 | 273.95 | 274.38 | 275.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 278.25 | 276.23 | 276.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 278.25 | 276.23 | 276.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 280.00 | 276.99 | 276.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 281.60 | 281.81 | 280.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 284.00 | 281.81 | 280.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 281.10 | 282.43 | 281.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 281.10 | 282.43 | 281.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 279.60 | 281.86 | 280.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 279.60 | 281.86 | 280.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 278.10 | 281.11 | 280.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 278.10 | 281.11 | 280.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 282.45 | 282.88 | 281.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 282.65 | 282.88 | 281.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 280.55 | 282.42 | 281.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 283.40 | 282.78 | 282.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 279.60 | 281.92 | 281.75 | SL hit (close<static) qty=1.00 sl=280.20 alert=retest2 |

### Cycle 36 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 279.95 | 281.52 | 281.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 279.00 | 280.54 | 281.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 279.80 | 279.72 | 280.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 11:15:00 | 279.80 | 279.72 | 280.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 279.80 | 279.72 | 280.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:30:00 | 280.50 | 279.72 | 280.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 283.20 | 280.43 | 280.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 283.20 | 280.43 | 280.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 283.20 | 280.98 | 280.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 284.15 | 281.95 | 281.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 296.80 | 297.73 | 294.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 296.80 | 297.73 | 294.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 296.15 | 299.03 | 297.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 296.15 | 299.03 | 297.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 296.95 | 298.61 | 297.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 295.50 | 298.61 | 297.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 296.45 | 298.18 | 297.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 298.40 | 298.18 | 297.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 294.75 | 300.33 | 299.57 | SL hit (close<static) qty=1.00 sl=296.40 alert=retest2 |

### Cycle 38 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 295.05 | 298.42 | 298.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 272.05 | 292.46 | 295.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 267.85 | 266.27 | 270.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:45:00 | 267.85 | 266.27 | 270.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 271.55 | 267.32 | 270.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 271.95 | 267.32 | 270.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 269.60 | 267.78 | 270.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 271.85 | 267.78 | 270.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 268.10 | 268.05 | 269.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 268.00 | 268.05 | 269.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 268.30 | 266.95 | 268.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 268.30 | 266.95 | 268.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 267.65 | 267.09 | 268.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:30:00 | 265.05 | 265.19 | 267.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 251.80 | 261.35 | 264.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 255.05 | 253.44 | 257.29 | SL hit (close>ema200) qty=0.50 sl=253.44 alert=retest2 |

### Cycle 39 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 254.80 | 249.84 | 249.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 259.85 | 252.54 | 250.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 255.10 | 261.26 | 259.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 255.10 | 261.26 | 259.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 255.10 | 261.26 | 259.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 255.10 | 261.26 | 259.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 256.80 | 260.37 | 258.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 255.45 | 260.37 | 258.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 252.25 | 257.67 | 257.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 250.70 | 255.02 | 256.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 254.45 | 254.37 | 255.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 254.50 | 254.37 | 255.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 258.85 | 255.27 | 256.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 258.85 | 255.27 | 256.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 258.40 | 255.89 | 256.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 264.10 | 255.89 | 256.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 261.75 | 257.07 | 256.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 273.00 | 269.23 | 265.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 269.50 | 270.13 | 266.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 269.50 | 270.13 | 266.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 265.05 | 268.46 | 267.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 265.05 | 268.46 | 267.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 265.40 | 267.85 | 267.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 264.50 | 267.85 | 267.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 264.50 | 266.85 | 266.91 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 272.30 | 267.70 | 267.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 274.05 | 268.97 | 267.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 261.15 | 271.52 | 271.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 261.15 | 271.52 | 271.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 261.15 | 271.52 | 271.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 260.30 | 271.52 | 271.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 260.35 | 269.29 | 270.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 259.25 | 265.90 | 268.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 09:15:00 | 263.55 | 263.25 | 266.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 09:30:00 | 264.35 | 263.25 | 266.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 260.50 | 259.84 | 262.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:45:00 | 260.00 | 259.84 | 262.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 260.10 | 258.37 | 259.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 260.10 | 258.37 | 259.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 260.60 | 258.82 | 260.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 261.25 | 258.82 | 260.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 262.40 | 260.10 | 260.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 261.35 | 260.10 | 260.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 262.90 | 261.04 | 260.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 11:15:00 | 264.00 | 261.63 | 261.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 262.60 | 262.65 | 261.86 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:15:00 | 264.30 | 262.65 | 261.86 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 263.10 | 263.50 | 262.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:15:00 | 261.00 | 263.50 | 262.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 261.00 | 263.00 | 262.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 261.00 | 263.00 | 262.59 | SL hit (close<ema400) qty=1.00 sl=262.59 alert=retest1 |

### Cycle 46 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 256.25 | 261.65 | 262.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 253.90 | 257.92 | 259.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 257.80 | 256.46 | 258.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 257.80 | 256.46 | 258.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 257.80 | 256.46 | 258.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 257.80 | 256.46 | 258.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 256.40 | 256.44 | 258.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 258.60 | 256.44 | 258.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 257.90 | 256.74 | 258.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 258.25 | 256.74 | 258.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 257.15 | 256.82 | 258.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 257.15 | 256.82 | 258.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 258.90 | 257.38 | 258.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 260.50 | 257.38 | 258.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 256.75 | 257.25 | 257.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 255.80 | 257.02 | 257.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 255.70 | 256.96 | 257.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:00:00 | 255.30 | 256.96 | 257.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:45:00 | 255.85 | 256.96 | 257.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 260.20 | 257.61 | 257.66 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-24 13:15:00 | 260.20 | 257.61 | 257.66 | SL hit (close>static) qty=1.00 sl=259.10 alert=retest2 |

### Cycle 47 — BUY (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 14:15:00 | 261.95 | 258.48 | 258.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 264.10 | 260.18 | 258.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 260.95 | 261.02 | 259.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 13:00:00 | 260.95 | 261.02 | 259.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 260.80 | 261.42 | 260.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 259.90 | 261.42 | 260.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 260.65 | 261.26 | 260.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:15:00 | 261.25 | 261.26 | 260.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 262.45 | 261.50 | 260.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:30:00 | 263.70 | 262.16 | 261.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 259.40 | 262.94 | 262.85 | SL hit (close<static) qty=1.00 sl=260.40 alert=retest2 |

### Cycle 48 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 260.90 | 262.53 | 262.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 253.95 | 260.69 | 261.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 252.40 | 252.01 | 255.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 252.40 | 252.01 | 255.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 253.60 | 252.98 | 255.14 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 262.85 | 257.30 | 256.66 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 249.70 | 256.00 | 256.76 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 258.30 | 256.22 | 256.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 259.95 | 257.35 | 256.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 257.60 | 257.95 | 257.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 257.60 | 257.95 | 257.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 256.75 | 257.71 | 257.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 256.75 | 257.71 | 257.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 255.20 | 257.21 | 256.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 255.20 | 257.21 | 256.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 255.80 | 256.93 | 256.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 253.75 | 256.93 | 256.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 255.10 | 256.56 | 256.66 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 263.70 | 257.99 | 257.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 269.55 | 260.30 | 258.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 262.65 | 264.29 | 261.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 262.65 | 264.29 | 261.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 262.65 | 264.29 | 261.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 262.65 | 264.29 | 261.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 261.80 | 263.79 | 261.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:45:00 | 262.00 | 263.79 | 261.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 262.80 | 263.59 | 261.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:15:00 | 261.20 | 263.59 | 261.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 258.20 | 262.51 | 261.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 258.20 | 262.51 | 261.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 258.30 | 261.67 | 261.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:00:00 | 258.30 | 261.67 | 261.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 258.25 | 260.56 | 260.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 254.80 | 259.41 | 260.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 13:15:00 | 255.85 | 253.67 | 255.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 13:15:00 | 255.85 | 253.67 | 255.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 255.85 | 253.67 | 255.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:45:00 | 255.40 | 253.67 | 255.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 256.70 | 254.28 | 255.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 257.05 | 254.28 | 255.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 255.40 | 254.50 | 255.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 258.80 | 254.50 | 255.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 258.70 | 255.34 | 255.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 260.40 | 255.34 | 255.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 259.00 | 256.07 | 256.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 259.75 | 257.26 | 256.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 253.60 | 257.58 | 257.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 253.60 | 257.58 | 257.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 253.60 | 257.58 | 257.12 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 254.45 | 256.45 | 256.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 253.15 | 255.79 | 256.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 263.25 | 255.79 | 255.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 263.25 | 255.79 | 255.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 263.25 | 255.79 | 255.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 263.25 | 255.79 | 255.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 264.80 | 257.59 | 256.76 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 251.45 | 257.80 | 257.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 250.00 | 255.11 | 256.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 254.80 | 254.06 | 255.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 254.80 | 254.06 | 255.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 254.80 | 254.06 | 255.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 253.75 | 254.06 | 255.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 258.40 | 256.34 | 256.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 258.40 | 256.34 | 256.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 266.10 | 258.52 | 257.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 262.20 | 262.30 | 259.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:45:00 | 262.40 | 262.30 | 259.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 257.40 | 261.30 | 259.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 257.40 | 261.30 | 259.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 257.10 | 260.46 | 259.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 256.50 | 260.46 | 259.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 256.40 | 258.96 | 259.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 254.70 | 257.69 | 258.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 252.68 | 251.04 | 253.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 252.68 | 251.04 | 253.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 252.68 | 251.04 | 253.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 254.42 | 251.04 | 253.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 252.86 | 251.40 | 253.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 253.07 | 251.40 | 253.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 255.50 | 252.22 | 253.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 255.50 | 252.22 | 253.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 256.71 | 253.12 | 254.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:45:00 | 256.80 | 253.12 | 254.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 251.71 | 252.95 | 253.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 243.69 | 252.64 | 253.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 14:15:00 | 253.91 | 248.94 | 248.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 14:15:00 | 253.91 | 248.94 | 248.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 262.81 | 252.44 | 250.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 15:15:00 | 287.00 | 287.40 | 281.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:15:00 | 294.13 | 287.40 | 281.78 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 14:15:00 | 308.84 | 297.00 | 291.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-04-20 09:15:00 | 323.54 | 315.04 | 305.80 | Target hit (10%) qty=0.50 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-22 09:30:00 | 248.90 | 2025-05-30 13:15:00 | 262.52 | STOP_HIT | 1.00 | 5.47% |
| BUY | retest2 | 2025-05-23 10:45:00 | 250.01 | 2025-05-30 13:15:00 | 262.52 | STOP_HIT | 1.00 | 5.00% |
| SELL | retest2 | 2025-06-03 11:30:00 | 256.30 | 2025-06-09 09:15:00 | 258.75 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-03 12:00:00 | 255.80 | 2025-06-09 09:15:00 | 258.75 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-06-06 10:30:00 | 255.70 | 2025-06-09 09:15:00 | 258.75 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-06-16 15:15:00 | 254.00 | 2025-06-23 10:15:00 | 253.10 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-06-17 11:30:00 | 254.20 | 2025-06-23 10:15:00 | 253.10 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-06-18 10:15:00 | 253.75 | 2025-06-23 10:15:00 | 253.10 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-06-30 09:15:00 | 266.40 | 2025-07-02 09:15:00 | 262.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-01 10:30:00 | 264.85 | 2025-07-02 09:15:00 | 262.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-01 13:15:00 | 264.45 | 2025-07-02 09:15:00 | 262.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-07 09:15:00 | 258.40 | 2025-07-09 10:15:00 | 263.50 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-07-09 09:30:00 | 258.30 | 2025-07-09 10:15:00 | 263.50 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-16 11:00:00 | 254.60 | 2025-07-21 15:15:00 | 254.35 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-07-16 15:00:00 | 254.60 | 2025-07-21 15:15:00 | 254.35 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-07-17 09:30:00 | 254.00 | 2025-07-21 15:15:00 | 254.35 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-07-25 09:30:00 | 247.45 | 2025-07-29 09:15:00 | 235.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 247.45 | 2025-07-29 12:15:00 | 239.10 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2025-08-13 11:45:00 | 221.74 | 2025-08-20 14:15:00 | 220.76 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-08-14 09:45:00 | 222.58 | 2025-08-20 14:15:00 | 220.76 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2025-08-14 10:30:00 | 222.23 | 2025-08-20 14:15:00 | 220.76 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2025-08-18 09:30:00 | 222.34 | 2025-08-20 14:15:00 | 220.76 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-08-25 11:30:00 | 217.43 | 2025-08-29 09:15:00 | 206.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 12:15:00 | 217.45 | 2025-08-29 09:15:00 | 206.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 14:00:00 | 217.30 | 2025-08-29 09:15:00 | 206.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 11:30:00 | 217.43 | 2025-08-29 10:15:00 | 210.44 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-08-25 12:15:00 | 217.45 | 2025-08-29 10:15:00 | 210.44 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2025-08-25 14:00:00 | 217.30 | 2025-08-29 10:15:00 | 210.44 | STOP_HIT | 0.50 | 3.16% |
| BUY | retest2 | 2025-09-10 11:00:00 | 217.51 | 2025-09-19 10:15:00 | 239.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-07 13:00:00 | 245.00 | 2025-10-08 10:15:00 | 240.70 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-10-07 14:30:00 | 244.51 | 2025-10-08 10:15:00 | 240.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-10 14:15:00 | 240.42 | 2025-10-16 15:15:00 | 236.30 | STOP_HIT | 1.00 | 1.71% |
| SELL | retest2 | 2025-10-10 15:00:00 | 239.30 | 2025-10-16 15:15:00 | 236.30 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2025-11-17 10:00:00 | 285.10 | 2025-11-21 09:15:00 | 283.60 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-11-17 12:15:00 | 285.80 | 2025-11-21 09:15:00 | 283.60 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-18 10:30:00 | 284.65 | 2025-11-21 09:15:00 | 283.60 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-11-25 13:00:00 | 281.60 | 2025-11-25 14:15:00 | 282.95 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-12-01 09:15:00 | 293.55 | 2025-12-02 10:15:00 | 288.25 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-12-08 09:45:00 | 274.95 | 2025-12-09 15:15:00 | 276.95 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-19 10:45:00 | 273.90 | 2025-12-22 11:15:00 | 278.25 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-12-19 13:00:00 | 273.95 | 2025-12-22 11:15:00 | 278.25 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-12-29 09:30:00 | 283.40 | 2025-12-29 11:15:00 | 279.60 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-01-07 09:15:00 | 298.40 | 2026-01-08 10:15:00 | 294.75 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-19 11:30:00 | 265.05 | 2026-01-20 09:15:00 | 251.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 11:30:00 | 265.05 | 2026-01-21 11:15:00 | 255.05 | STOP_HIT | 0.50 | 3.77% |
| BUY | retest1 | 2026-02-18 09:15:00 | 264.30 | 2026-02-18 15:15:00 | 261.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-02-23 11:45:00 | 255.80 | 2026-02-24 13:15:00 | 260.20 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-02-24 09:30:00 | 255.70 | 2026-02-24 13:15:00 | 260.20 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-02-24 10:00:00 | 255.30 | 2026-02-24 13:15:00 | 260.20 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-02-24 12:45:00 | 255.85 | 2026-02-24 13:15:00 | 260.20 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-02-26 14:30:00 | 263.70 | 2026-03-02 11:15:00 | 259.40 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-03-24 10:15:00 | 253.75 | 2026-03-24 14:15:00 | 258.40 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-04-02 09:15:00 | 243.69 | 2026-04-07 14:15:00 | 253.91 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest1 | 2026-04-15 09:15:00 | 294.13 | 2026-04-16 14:15:00 | 308.84 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-15 09:15:00 | 294.13 | 2026-04-20 09:15:00 | 323.54 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-30 09:30:00 | 346.42 | 2026-05-04 11:15:00 | 381.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 11:00:00 | 347.65 | 2026-05-04 11:15:00 | 382.42 | TARGET_HIT | 1.00 | 10.00% |
