# RBL Bank Ltd. (RBLBANK)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 343.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 171 |
| ALERT1 | 115 |
| ALERT2 | 115 |
| ALERT2_SKIP | 66 |
| ALERT3 | 311 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 112 |
| PARTIAL | 13 |
| TARGET_HIT | 6 |
| STOP_HIT | 110 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 129 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 83
- **Target hits / Stop hits / Partials:** 6 / 110 / 13
- **Avg / median % per leg:** 0.48% / -0.84%
- **Sum % (uncompounded):** 61.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 13 | 18.6% | 5 | 65 | 0 | -0.20% | -14.2% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.81% | -7.2% |
| BUY @ 3rd Alert (retest2) | 66 | 13 | 19.7% | 5 | 61 | 0 | -0.11% | -7.0% |
| SELL (all) | 59 | 33 | 55.9% | 1 | 45 | 13 | 1.28% | 75.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 59 | 33 | 55.9% | 1 | 45 | 13 | 1.28% | 75.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.81% | -7.2% |
| retest2 (combined) | 125 | 46 | 36.8% | 6 | 106 | 13 | 0.55% | 68.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 247.10 | 245.15 | 245.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 247.95 | 245.97 | 245.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 249.20 | 249.20 | 247.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:15:00 | 250.65 | 249.20 | 247.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:00:00 | 250.50 | 249.46 | 248.11 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 248.50 | 250.01 | 248.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-16 13:15:00 | 248.50 | 250.01 | 248.88 | SL hit (close<ema400) qty=1.00 sl=248.88 alert=retest1 |

### Cycle 2 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 250.20 | 251.15 | 251.25 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 252.55 | 251.49 | 251.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 12:15:00 | 255.35 | 253.81 | 252.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 253.75 | 254.28 | 253.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 253.75 | 254.28 | 253.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 253.75 | 254.28 | 253.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 253.75 | 254.28 | 253.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 253.75 | 254.17 | 253.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:15:00 | 254.80 | 254.17 | 253.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 12:15:00 | 255.00 | 254.24 | 253.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 15:15:00 | 253.00 | 254.15 | 253.76 | SL hit (close<static) qty=1.00 sl=253.05 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 10:15:00 | 251.80 | 253.21 | 253.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 250.35 | 252.64 | 253.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 14:15:00 | 252.05 | 251.96 | 252.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 15:00:00 | 252.05 | 251.96 | 252.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 251.65 | 251.71 | 252.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 251.65 | 251.71 | 252.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 250.55 | 251.48 | 252.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 12:00:00 | 250.20 | 251.22 | 252.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 13:45:00 | 250.05 | 250.73 | 251.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 257.15 | 248.05 | 247.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 257.15 | 248.05 | 247.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 263.50 | 252.71 | 250.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 250.50 | 256.59 | 253.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 250.50 | 256.59 | 253.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 250.50 | 256.59 | 253.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 247.45 | 256.59 | 253.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 235.15 | 252.30 | 251.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 235.15 | 252.30 | 251.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 223.25 | 246.49 | 249.25 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 248.75 | 242.45 | 242.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 12:15:00 | 251.60 | 247.39 | 245.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 252.40 | 253.11 | 250.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:15:00 | 252.90 | 253.11 | 250.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 258.10 | 259.07 | 258.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:45:00 | 258.28 | 259.07 | 258.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 257.45 | 258.74 | 257.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 259.95 | 258.74 | 257.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 10:45:00 | 258.65 | 259.82 | 259.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 254.12 | 262.81 | 263.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 254.12 | 262.81 | 263.62 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 263.60 | 259.02 | 258.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 268.64 | 262.65 | 260.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 263.40 | 263.86 | 262.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 15:00:00 | 263.40 | 263.86 | 262.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 262.98 | 263.68 | 262.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:30:00 | 263.60 | 263.65 | 262.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 11:15:00 | 260.25 | 263.17 | 263.05 | SL hit (close<static) qty=1.00 sl=262.01 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 259.05 | 262.35 | 262.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 15:15:00 | 258.00 | 260.46 | 261.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 266.80 | 261.73 | 262.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 266.80 | 261.73 | 262.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 266.80 | 261.73 | 262.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:00:00 | 266.80 | 261.73 | 262.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 266.20 | 262.62 | 262.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 12:15:00 | 268.00 | 264.36 | 263.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 263.80 | 266.13 | 264.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 10:15:00 | 263.80 | 266.13 | 264.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 263.80 | 266.13 | 264.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:00:00 | 263.80 | 266.13 | 264.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 264.80 | 265.86 | 264.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:45:00 | 265.05 | 265.53 | 264.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:00:00 | 265.80 | 265.58 | 264.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 259.50 | 264.47 | 264.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 259.50 | 264.47 | 264.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 254.05 | 259.09 | 261.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 14:15:00 | 245.80 | 244.13 | 246.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 15:00:00 | 245.80 | 244.13 | 246.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 246.00 | 244.50 | 246.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 246.75 | 244.50 | 246.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 247.35 | 245.07 | 246.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 248.75 | 245.07 | 246.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 248.20 | 245.70 | 246.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 248.20 | 245.70 | 246.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 245.00 | 246.31 | 246.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 243.80 | 246.31 | 246.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 245.15 | 246.07 | 246.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:45:00 | 240.35 | 242.68 | 243.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 11:30:00 | 240.80 | 242.52 | 243.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 15:00:00 | 240.85 | 242.08 | 243.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 09:45:00 | 240.70 | 241.34 | 242.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 243.55 | 241.79 | 242.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 243.55 | 241.79 | 242.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 241.30 | 241.69 | 242.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 240.60 | 242.17 | 242.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 09:15:00 | 228.33 | 236.23 | 238.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 09:15:00 | 228.76 | 236.23 | 238.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 09:15:00 | 228.81 | 236.23 | 238.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 09:15:00 | 228.66 | 236.23 | 238.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 09:15:00 | 228.57 | 236.23 | 238.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 232.60 | 231.74 | 234.22 | SL hit (close>ema200) qty=0.50 sl=231.74 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 240.30 | 235.10 | 234.85 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 13:15:00 | 234.20 | 236.06 | 236.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 233.05 | 234.91 | 235.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 15:15:00 | 216.00 | 213.66 | 216.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 15:15:00 | 216.00 | 213.66 | 216.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 216.00 | 213.66 | 216.16 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 13:15:00 | 217.21 | 216.33 | 216.21 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 214.94 | 216.24 | 216.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 214.20 | 215.11 | 215.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 207.33 | 207.19 | 209.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 14:00:00 | 207.33 | 207.19 | 209.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 211.28 | 207.91 | 209.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:45:00 | 210.50 | 207.91 | 209.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 210.00 | 208.33 | 209.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 12:15:00 | 208.75 | 208.61 | 209.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 09:15:00 | 214.33 | 209.77 | 209.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 214.33 | 209.77 | 209.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 10:15:00 | 216.25 | 211.06 | 210.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 226.45 | 228.96 | 225.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:30:00 | 225.97 | 228.96 | 225.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 224.34 | 227.13 | 225.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:45:00 | 224.84 | 227.13 | 225.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 224.55 | 226.62 | 225.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 226.45 | 226.62 | 225.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 12:15:00 | 225.25 | 225.92 | 225.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 226.34 | 228.20 | 228.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 226.34 | 228.20 | 228.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 224.90 | 226.94 | 227.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 227.14 | 226.98 | 227.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 227.14 | 226.98 | 227.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 227.10 | 227.01 | 227.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 229.06 | 227.01 | 227.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 230.77 | 227.76 | 227.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 230.77 | 227.76 | 227.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 230.53 | 228.31 | 228.11 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 225.64 | 228.09 | 228.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 222.48 | 225.80 | 226.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 213.73 | 211.34 | 213.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 213.73 | 211.34 | 213.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 213.73 | 211.34 | 213.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:00:00 | 213.73 | 211.34 | 213.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 213.86 | 211.85 | 213.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:30:00 | 214.60 | 211.85 | 213.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 215.24 | 212.88 | 213.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:00:00 | 215.24 | 212.88 | 213.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 214.65 | 213.23 | 213.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:45:00 | 215.14 | 213.23 | 213.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 213.76 | 213.40 | 213.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 213.05 | 213.40 | 213.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 213.52 | 213.42 | 213.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:30:00 | 212.02 | 212.90 | 213.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 215.99 | 213.30 | 212.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 215.99 | 213.30 | 212.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 216.09 | 214.20 | 213.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 214.19 | 214.26 | 213.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 15:00:00 | 214.19 | 214.26 | 213.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 214.35 | 215.15 | 214.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 14:30:00 | 216.10 | 214.94 | 214.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 13:15:00 | 212.79 | 215.11 | 215.00 | SL hit (close<static) qty=1.00 sl=214.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 212.22 | 214.53 | 214.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 208.00 | 212.00 | 213.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 210.90 | 210.70 | 212.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 211.95 | 210.70 | 212.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 211.35 | 210.83 | 212.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 212.50 | 210.83 | 212.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 213.19 | 211.37 | 212.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:00:00 | 213.19 | 211.37 | 212.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 213.50 | 211.80 | 212.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:30:00 | 213.49 | 211.80 | 212.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 212.72 | 212.10 | 212.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:30:00 | 213.38 | 212.10 | 212.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 212.76 | 212.49 | 212.49 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 10:15:00 | 211.40 | 212.27 | 212.39 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 213.94 | 212.62 | 212.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 15:15:00 | 214.84 | 213.27 | 212.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 212.18 | 213.05 | 212.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 212.18 | 213.05 | 212.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 212.18 | 213.05 | 212.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:30:00 | 212.70 | 213.05 | 212.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 211.65 | 212.77 | 212.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:45:00 | 211.58 | 212.77 | 212.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 212.00 | 212.62 | 212.62 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 212.07 | 212.51 | 212.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 13:15:00 | 211.84 | 212.38 | 212.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 207.91 | 206.81 | 208.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 207.91 | 206.81 | 208.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 207.91 | 206.81 | 208.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 207.91 | 206.81 | 208.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 208.79 | 207.20 | 208.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 208.96 | 207.20 | 208.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 209.48 | 207.66 | 208.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 210.23 | 207.66 | 208.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 210.68 | 208.26 | 208.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 210.68 | 208.26 | 208.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 208.84 | 208.62 | 208.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:00:00 | 208.84 | 208.62 | 208.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 207.50 | 208.40 | 208.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:30:00 | 208.98 | 208.40 | 208.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 206.63 | 207.88 | 208.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 10:30:00 | 205.42 | 207.37 | 208.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:15:00 | 205.60 | 206.44 | 207.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 195.32 | 198.88 | 200.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 195.15 | 198.15 | 200.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 194.69 | 193.76 | 196.51 | SL hit (close>ema200) qty=0.50 sl=193.76 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 200.79 | 197.14 | 196.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 09:15:00 | 203.01 | 200.70 | 199.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 15:15:00 | 205.72 | 206.35 | 204.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 10:00:00 | 205.27 | 206.14 | 204.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 207.51 | 208.92 | 207.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 207.70 | 208.92 | 207.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 208.05 | 208.74 | 207.64 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 203.29 | 206.72 | 206.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 14:15:00 | 202.00 | 205.77 | 206.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 206.18 | 205.45 | 206.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 206.18 | 205.45 | 206.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 206.18 | 205.45 | 206.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:00:00 | 206.18 | 205.45 | 206.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 205.23 | 205.41 | 206.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 184.73 | 205.84 | 206.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 13:15:00 | 175.49 | 186.22 | 195.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-22 15:15:00 | 166.26 | 172.18 | 181.19 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 170.81 | 167.96 | 167.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 12:15:00 | 171.54 | 168.68 | 167.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 11:15:00 | 173.05 | 175.12 | 173.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 11:15:00 | 173.05 | 175.12 | 173.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 173.05 | 175.12 | 173.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 173.05 | 175.12 | 173.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 172.67 | 174.63 | 173.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:30:00 | 172.60 | 174.63 | 173.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 170.22 | 173.75 | 172.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:45:00 | 170.23 | 173.75 | 172.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 170.48 | 173.09 | 172.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:30:00 | 169.50 | 173.09 | 172.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 15:15:00 | 168.55 | 172.19 | 172.26 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 175.90 | 172.93 | 172.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 18:15:00 | 176.60 | 173.66 | 172.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 171.08 | 173.15 | 172.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 171.08 | 173.15 | 172.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 171.08 | 173.15 | 172.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 171.08 | 173.15 | 172.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 170.30 | 172.58 | 172.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 170.30 | 172.58 | 172.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 169.00 | 171.86 | 172.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 167.06 | 170.03 | 171.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 170.88 | 169.95 | 170.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 170.88 | 169.95 | 170.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 170.88 | 169.95 | 170.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 170.88 | 169.95 | 170.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 171.49 | 170.26 | 170.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 171.49 | 170.26 | 170.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 171.00 | 170.41 | 170.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 172.15 | 170.41 | 170.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 171.25 | 170.59 | 170.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 171.25 | 170.59 | 170.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 171.85 | 170.84 | 170.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:30:00 | 171.55 | 170.84 | 170.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 173.77 | 171.42 | 171.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 10:15:00 | 174.31 | 172.99 | 172.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 172.90 | 173.07 | 172.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 13:15:00 | 172.90 | 173.07 | 172.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 172.90 | 173.07 | 172.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:00:00 | 172.90 | 173.07 | 172.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 171.40 | 172.74 | 172.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 171.40 | 172.74 | 172.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 170.80 | 172.35 | 172.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 169.60 | 172.35 | 172.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 168.74 | 171.63 | 171.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 167.94 | 170.89 | 171.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 155.17 | 154.62 | 157.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 12:45:00 | 154.63 | 154.62 | 157.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 156.40 | 154.80 | 156.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 156.40 | 154.80 | 156.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 156.44 | 155.13 | 156.22 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 160.86 | 157.59 | 157.16 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 14:15:00 | 156.45 | 157.56 | 157.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 15:15:00 | 156.00 | 157.25 | 157.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 157.39 | 157.28 | 157.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 157.39 | 157.28 | 157.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 157.39 | 157.28 | 157.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:30:00 | 157.39 | 157.28 | 157.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 157.13 | 157.25 | 157.43 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 160.65 | 158.06 | 157.74 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 157.62 | 158.08 | 158.14 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 158.81 | 158.25 | 158.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 160.20 | 158.58 | 158.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 158.55 | 158.58 | 158.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 158.55 | 158.58 | 158.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 158.55 | 158.58 | 158.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 158.70 | 158.58 | 158.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 158.38 | 158.54 | 158.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:30:00 | 158.16 | 158.54 | 158.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 157.37 | 158.30 | 158.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 157.37 | 158.30 | 158.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 157.59 | 158.16 | 158.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 09:15:00 | 156.61 | 157.76 | 158.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 11:15:00 | 155.40 | 154.30 | 155.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 11:15:00 | 155.40 | 154.30 | 155.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 155.40 | 154.30 | 155.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:00:00 | 155.40 | 154.30 | 155.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 154.82 | 154.40 | 155.55 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 11:15:00 | 156.77 | 156.09 | 156.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 12:15:00 | 157.30 | 156.34 | 156.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 172.75 | 173.31 | 169.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:15:00 | 176.88 | 174.01 | 171.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:15:00 | 178.84 | 174.51 | 171.97 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 172.89 | 174.77 | 172.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-09 13:15:00 | 172.89 | 174.77 | 172.99 | SL hit (close<ema400) qty=1.00 sl=172.99 alert=retest1 |

### Cycle 42 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 170.39 | 172.13 | 172.21 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 13:15:00 | 174.92 | 172.35 | 172.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 178.25 | 173.53 | 172.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 176.95 | 177.62 | 175.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 09:45:00 | 176.99 | 177.62 | 175.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 174.89 | 176.94 | 175.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 174.89 | 176.94 | 175.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 175.05 | 176.56 | 175.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 14:00:00 | 175.59 | 176.37 | 175.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 14:15:00 | 173.14 | 175.72 | 175.60 | SL hit (close<static) qty=1.00 sl=174.76 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 173.80 | 175.34 | 175.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 171.39 | 174.55 | 175.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 173.36 | 173.29 | 174.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:30:00 | 173.51 | 173.29 | 174.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 173.11 | 173.29 | 173.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:15:00 | 172.10 | 173.73 | 174.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 163.49 | 166.34 | 168.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 15:15:00 | 165.10 | 164.77 | 166.55 | SL hit (close>ema200) qty=0.50 sl=164.77 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 163.16 | 159.76 | 159.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 09:15:00 | 164.61 | 161.33 | 160.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 14:15:00 | 163.00 | 163.13 | 161.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 15:00:00 | 163.00 | 163.13 | 161.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 161.90 | 162.88 | 161.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 160.25 | 162.88 | 161.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 158.88 | 162.08 | 161.42 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 13:15:00 | 159.60 | 161.09 | 161.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 157.19 | 159.61 | 160.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 12:15:00 | 158.18 | 157.74 | 158.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 13:00:00 | 158.18 | 157.74 | 158.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 157.18 | 157.71 | 158.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 155.80 | 157.92 | 158.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 159.21 | 157.65 | 157.91 | SL hit (close>static) qty=1.00 sl=158.84 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 158.71 | 158.14 | 158.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 10:15:00 | 159.90 | 158.66 | 158.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 14:15:00 | 166.90 | 167.01 | 164.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-07 15:00:00 | 166.90 | 167.01 | 164.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 165.00 | 166.61 | 164.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 160.35 | 166.61 | 164.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 162.37 | 165.76 | 164.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:30:00 | 160.91 | 165.76 | 164.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 161.56 | 164.92 | 164.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:45:00 | 162.07 | 164.92 | 164.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 160.55 | 164.05 | 164.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 11:15:00 | 160.04 | 161.75 | 162.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 152.50 | 151.65 | 154.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 10:00:00 | 152.50 | 151.65 | 154.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 154.12 | 152.47 | 154.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 154.12 | 152.47 | 154.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 153.31 | 152.64 | 153.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:15:00 | 155.60 | 152.64 | 153.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 155.60 | 153.23 | 154.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 155.20 | 153.23 | 154.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 154.66 | 153.51 | 154.14 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 158.06 | 154.91 | 154.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 158.92 | 156.36 | 155.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 157.50 | 157.92 | 156.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 157.50 | 157.92 | 156.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 157.50 | 157.92 | 156.91 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 155.20 | 156.37 | 156.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 153.60 | 155.64 | 156.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 13:15:00 | 155.00 | 154.78 | 155.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 13:15:00 | 155.00 | 154.78 | 155.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 155.00 | 154.78 | 155.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:30:00 | 156.80 | 154.78 | 155.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 157.85 | 155.39 | 155.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:45:00 | 157.08 | 155.39 | 155.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 15:15:00 | 157.79 | 155.87 | 155.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 12:15:00 | 159.90 | 157.21 | 156.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 15:15:00 | 157.30 | 157.87 | 157.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 15:15:00 | 157.30 | 157.87 | 157.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 157.30 | 157.87 | 157.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 155.85 | 157.87 | 157.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 154.14 | 157.12 | 156.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 154.93 | 157.12 | 156.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 153.45 | 156.39 | 156.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 153.11 | 155.73 | 156.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 155.20 | 154.90 | 155.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 155.20 | 154.90 | 155.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 154.69 | 154.86 | 155.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 153.80 | 154.86 | 155.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 10:15:00 | 156.55 | 155.34 | 155.64 | SL hit (close>static) qty=1.00 sl=156.10 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 156.40 | 155.81 | 155.78 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 15:15:00 | 155.45 | 155.74 | 155.75 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 09:15:00 | 157.15 | 156.02 | 155.88 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 154.55 | 155.90 | 155.90 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 14:15:00 | 155.94 | 155.91 | 155.90 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 155.20 | 155.76 | 155.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 150.17 | 154.65 | 155.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 151.25 | 151.19 | 152.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 09:15:00 | 151.25 | 151.19 | 152.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 151.25 | 151.19 | 152.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 10:15:00 | 151.05 | 151.19 | 152.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 10:45:00 | 151.07 | 151.29 | 152.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 156.05 | 152.74 | 153.19 | SL hit (close>static) qty=1.00 sl=155.20 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 156.75 | 153.54 | 153.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 158.00 | 154.96 | 154.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 164.86 | 164.91 | 162.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 164.86 | 164.91 | 162.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 164.86 | 164.91 | 162.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 164.19 | 164.91 | 162.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 164.51 | 164.83 | 162.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 164.51 | 164.83 | 162.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 165.74 | 165.01 | 163.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 166.94 | 165.40 | 163.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 167.41 | 165.68 | 164.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:00:00 | 166.77 | 166.12 | 164.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 167.20 | 165.43 | 164.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 168.12 | 169.48 | 168.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 168.12 | 169.48 | 168.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 167.60 | 169.10 | 168.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 167.60 | 169.10 | 168.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 168.39 | 168.66 | 168.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 168.41 | 168.66 | 168.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 167.45 | 168.42 | 168.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 167.45 | 168.42 | 168.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 167.69 | 168.27 | 168.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 168.50 | 168.27 | 168.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 11:15:00 | 166.85 | 167.99 | 168.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 166.85 | 167.99 | 168.07 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 14:15:00 | 169.75 | 168.26 | 168.17 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 165.81 | 167.89 | 168.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 13:15:00 | 165.20 | 166.83 | 167.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 168.42 | 167.15 | 167.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 168.42 | 167.15 | 167.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 168.42 | 167.15 | 167.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 168.42 | 167.15 | 167.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 167.20 | 167.16 | 167.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 165.74 | 167.16 | 167.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 165.10 | 166.75 | 167.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:15:00 | 164.83 | 166.31 | 166.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:00:00 | 164.21 | 165.89 | 166.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 156.59 | 162.88 | 164.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 12:15:00 | 162.39 | 162.18 | 164.06 | SL hit (close>ema200) qty=0.50 sl=162.18 alert=retest2 |

### Cycle 63 — BUY (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 15:15:00 | 164.50 | 163.88 | 163.82 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 162.31 | 163.56 | 163.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 159.50 | 162.75 | 163.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 156.07 | 152.97 | 154.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 156.07 | 152.97 | 154.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 156.07 | 152.97 | 154.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 156.07 | 152.97 | 154.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 157.78 | 153.94 | 155.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 157.62 | 153.94 | 155.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 160.30 | 156.73 | 156.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 161.70 | 157.72 | 156.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 159.61 | 161.39 | 160.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 159.61 | 161.39 | 160.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 159.61 | 161.39 | 160.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:45:00 | 159.42 | 161.39 | 160.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 158.06 | 160.73 | 160.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:00:00 | 158.06 | 160.73 | 160.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 159.26 | 160.08 | 159.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 159.40 | 160.08 | 159.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 160.75 | 160.21 | 160.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 15:15:00 | 162.29 | 160.21 | 160.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 10:15:00 | 161.45 | 160.61 | 160.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 15:15:00 | 157.66 | 159.69 | 159.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 15:15:00 | 157.66 | 159.69 | 159.95 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 09:15:00 | 165.25 | 160.81 | 160.43 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 11:15:00 | 157.33 | 160.62 | 160.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 156.10 | 159.71 | 160.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 13:15:00 | 155.29 | 155.10 | 156.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 14:00:00 | 155.29 | 155.10 | 156.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 157.72 | 155.46 | 156.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 158.15 | 155.46 | 156.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 157.92 | 155.95 | 156.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 157.90 | 155.95 | 156.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 158.04 | 156.37 | 156.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:45:00 | 157.97 | 156.37 | 156.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 157.61 | 156.62 | 156.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 161.78 | 158.24 | 157.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 160.98 | 162.21 | 160.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 11:15:00 | 160.98 | 162.21 | 160.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 160.98 | 162.21 | 160.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 160.98 | 162.21 | 160.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 160.80 | 161.93 | 160.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 160.80 | 161.93 | 160.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 162.77 | 162.09 | 160.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:15:00 | 163.81 | 162.09 | 160.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 13:15:00 | 160.52 | 162.05 | 161.54 | SL hit (close<static) qty=1.00 sl=160.75 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 153.59 | 159.84 | 160.62 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 158.60 | 156.90 | 156.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 160.28 | 157.84 | 157.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 13:15:00 | 165.30 | 165.56 | 163.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 14:00:00 | 165.30 | 165.56 | 163.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 177.36 | 178.10 | 176.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 176.70 | 178.10 | 176.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 176.55 | 177.79 | 176.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:45:00 | 176.80 | 177.79 | 176.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 176.10 | 177.45 | 176.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 176.10 | 177.45 | 176.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 177.20 | 177.40 | 176.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:15:00 | 177.99 | 177.40 | 176.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 177.99 | 177.52 | 176.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 178.97 | 177.52 | 176.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:45:00 | 178.78 | 177.64 | 176.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 10:15:00 | 178.50 | 177.64 | 176.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 14:15:00 | 173.65 | 176.06 | 176.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 173.65 | 176.06 | 176.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 15:15:00 | 172.95 | 175.44 | 175.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 176.44 | 175.64 | 176.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 176.44 | 175.64 | 176.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 176.44 | 175.64 | 176.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 176.78 | 175.64 | 176.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 175.45 | 175.60 | 175.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 169.31 | 176.04 | 176.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 12:30:00 | 173.71 | 173.14 | 173.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 15:15:00 | 175.30 | 174.17 | 174.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 15:15:00 | 175.30 | 174.17 | 174.16 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 173.43 | 174.02 | 174.10 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 12:15:00 | 175.18 | 174.13 | 174.11 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 166.70 | 173.27 | 173.80 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 173.09 | 170.46 | 170.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 180.84 | 173.81 | 172.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 180.07 | 181.58 | 179.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 180.07 | 181.58 | 179.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 189.18 | 189.90 | 186.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 13:30:00 | 193.58 | 189.72 | 187.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:45:00 | 192.87 | 190.58 | 189.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 14:15:00 | 187.41 | 191.08 | 191.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 187.41 | 191.08 | 191.49 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 199.32 | 192.14 | 191.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 207.04 | 196.37 | 193.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 14:15:00 | 203.29 | 203.84 | 200.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 15:00:00 | 203.29 | 203.84 | 200.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 201.90 | 203.33 | 201.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 11:45:00 | 204.31 | 202.50 | 201.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 15:15:00 | 204.25 | 203.84 | 202.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 09:30:00 | 204.14 | 203.53 | 202.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 12:15:00 | 199.45 | 202.02 | 202.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 199.45 | 202.02 | 202.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 197.24 | 200.80 | 201.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 200.05 | 199.31 | 200.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 200.05 | 199.31 | 200.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 200.05 | 199.31 | 200.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:45:00 | 200.53 | 199.31 | 200.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 200.78 | 199.60 | 200.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 200.78 | 199.60 | 200.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 201.20 | 199.92 | 200.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 201.20 | 199.92 | 200.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 201.59 | 200.26 | 200.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 201.59 | 200.26 | 200.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 201.60 | 200.52 | 200.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 204.78 | 200.52 | 200.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 204.90 | 201.40 | 201.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 10:15:00 | 205.55 | 202.23 | 201.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 200.77 | 202.63 | 201.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 200.77 | 202.63 | 201.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 200.77 | 202.63 | 201.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 200.77 | 202.63 | 201.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 197.89 | 201.68 | 201.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 197.40 | 201.68 | 201.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 194.35 | 200.21 | 200.91 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 204.65 | 199.85 | 199.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 206.60 | 203.56 | 201.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 209.35 | 209.76 | 208.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 209.35 | 209.76 | 208.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 208.53 | 209.52 | 208.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:45:00 | 208.43 | 209.52 | 208.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 208.87 | 209.39 | 208.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:15:00 | 209.26 | 209.39 | 208.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 208.20 | 210.98 | 211.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 208.20 | 210.98 | 211.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 207.35 | 209.30 | 210.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 208.50 | 208.46 | 209.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 208.50 | 208.46 | 209.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 208.50 | 208.46 | 209.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 206.60 | 208.25 | 208.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 210.75 | 208.93 | 209.03 | SL hit (close>static) qty=1.00 sl=209.79 alert=retest2 |

### Cycle 85 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 209.85 | 209.23 | 209.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 14:15:00 | 210.28 | 209.44 | 209.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 15:15:00 | 208.50 | 209.25 | 209.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 15:15:00 | 208.50 | 209.25 | 209.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 208.50 | 209.25 | 209.19 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 207.24 | 208.88 | 209.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 14:15:00 | 204.88 | 207.02 | 208.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 207.75 | 206.75 | 207.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 207.75 | 206.75 | 207.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 207.75 | 206.75 | 207.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 207.75 | 206.75 | 207.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 211.64 | 207.73 | 208.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 211.64 | 207.73 | 208.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 209.90 | 208.16 | 208.22 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 211.23 | 208.78 | 208.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 212.00 | 210.24 | 209.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 11:15:00 | 206.75 | 209.63 | 209.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 11:15:00 | 206.75 | 209.63 | 209.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 206.75 | 209.63 | 209.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 206.75 | 209.63 | 209.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 209.22 | 209.55 | 209.23 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 207.73 | 208.94 | 208.99 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 213.10 | 209.51 | 209.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 213.98 | 210.80 | 209.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 213.01 | 213.26 | 211.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 213.01 | 213.26 | 211.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 211.79 | 212.97 | 211.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 211.36 | 212.97 | 211.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 211.90 | 212.76 | 211.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 211.90 | 212.76 | 211.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 214.26 | 213.06 | 211.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:15:00 | 215.22 | 213.06 | 211.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:30:00 | 214.95 | 213.38 | 212.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 14:15:00 | 211.45 | 212.54 | 212.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 211.45 | 212.54 | 212.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 210.51 | 212.13 | 212.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 212.59 | 212.01 | 212.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 212.59 | 212.01 | 212.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 212.59 | 212.01 | 212.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:45:00 | 212.70 | 212.01 | 212.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 212.59 | 212.13 | 212.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 213.15 | 212.13 | 212.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 212.50 | 212.20 | 212.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:45:00 | 212.26 | 212.20 | 212.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 213.63 | 212.49 | 212.43 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 09:15:00 | 212.20 | 212.37 | 212.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 10:15:00 | 211.34 | 212.16 | 212.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 10:15:00 | 212.73 | 209.76 | 210.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 10:15:00 | 212.73 | 209.76 | 210.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 212.73 | 209.76 | 210.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:30:00 | 211.92 | 209.76 | 210.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 218.55 | 211.52 | 211.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 228.09 | 217.58 | 214.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 225.00 | 226.04 | 221.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 10:00:00 | 225.00 | 226.04 | 221.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 221.95 | 224.46 | 221.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 221.90 | 224.46 | 221.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 221.59 | 223.88 | 221.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:45:00 | 221.71 | 223.88 | 221.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 222.26 | 223.56 | 221.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:15:00 | 221.56 | 223.56 | 221.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 221.56 | 223.16 | 221.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 220.65 | 223.16 | 221.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 222.59 | 223.04 | 221.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 220.84 | 223.04 | 221.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 222.80 | 223.00 | 221.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:45:00 | 222.80 | 223.00 | 221.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 222.28 | 222.86 | 222.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:45:00 | 222.40 | 222.86 | 222.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 219.01 | 222.09 | 221.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 218.70 | 222.09 | 221.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 220.20 | 221.71 | 221.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 222.15 | 221.89 | 221.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 222.15 | 222.47 | 222.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 10:15:00 | 219.01 | 221.78 | 221.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 219.01 | 221.78 | 221.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 215.30 | 219.52 | 220.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 11:15:00 | 218.15 | 217.78 | 218.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-17 12:00:00 | 218.15 | 217.78 | 218.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 219.00 | 218.03 | 218.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:30:00 | 219.32 | 218.03 | 218.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 219.90 | 218.40 | 219.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:00:00 | 219.90 | 218.40 | 219.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 218.88 | 218.50 | 219.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 217.25 | 218.50 | 219.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 222.15 | 219.03 | 219.14 | SL hit (close>static) qty=1.00 sl=219.97 alert=retest2 |

### Cycle 95 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 223.42 | 219.91 | 219.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 225.11 | 220.95 | 220.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 224.68 | 225.68 | 223.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:00:00 | 224.68 | 225.68 | 223.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 222.01 | 224.94 | 223.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 221.94 | 224.94 | 223.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 223.90 | 224.73 | 223.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:30:00 | 224.64 | 224.29 | 223.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-30 09:15:00 | 247.10 | 240.47 | 238.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 251.65 | 252.72 | 252.78 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 15:15:00 | 253.40 | 252.94 | 252.88 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 251.26 | 252.66 | 252.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 13:15:00 | 250.32 | 252.08 | 252.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 255.04 | 252.06 | 252.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 255.04 | 252.06 | 252.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 255.04 | 252.06 | 252.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 255.04 | 252.06 | 252.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 256.89 | 253.03 | 252.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 260.75 | 256.50 | 254.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 257.13 | 257.68 | 255.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 257.13 | 257.68 | 255.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 253.73 | 256.89 | 255.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 253.73 | 256.89 | 255.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 253.76 | 256.26 | 255.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 13:15:00 | 254.65 | 256.26 | 255.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 15:15:00 | 255.60 | 255.55 | 255.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 11:00:00 | 256.79 | 255.46 | 255.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 264.84 | 266.01 | 266.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 264.84 | 266.01 | 266.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 260.70 | 264.94 | 265.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 262.56 | 262.44 | 263.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 262.56 | 262.44 | 263.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 259.67 | 257.40 | 259.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 259.67 | 257.40 | 259.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 258.79 | 257.67 | 259.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:30:00 | 257.76 | 257.90 | 259.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:00:00 | 257.75 | 258.16 | 259.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 260.26 | 255.77 | 256.46 | SL hit (close>static) qty=1.00 sl=259.95 alert=retest2 |

### Cycle 101 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 261.45 | 256.09 | 255.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 261.91 | 257.89 | 256.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 263.95 | 265.50 | 262.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 263.95 | 265.50 | 262.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 263.95 | 265.50 | 262.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 263.95 | 265.50 | 262.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 264.80 | 265.39 | 263.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:30:00 | 263.55 | 265.39 | 263.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 262.40 | 264.79 | 263.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 262.40 | 264.79 | 263.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 258.50 | 263.53 | 262.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 258.50 | 263.53 | 262.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 259.80 | 262.01 | 262.20 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 263.85 | 262.51 | 262.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 14:15:00 | 265.50 | 263.11 | 262.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 12:15:00 | 263.55 | 263.74 | 263.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 13:00:00 | 263.55 | 263.74 | 263.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 263.45 | 263.68 | 263.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:30:00 | 263.55 | 263.68 | 263.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 266.00 | 264.14 | 263.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 15:15:00 | 266.80 | 264.14 | 263.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 261.15 | 263.97 | 263.54 | SL hit (close<static) qty=1.00 sl=263.10 alert=retest2 |

### Cycle 104 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 258.20 | 262.82 | 263.06 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 269.10 | 263.43 | 262.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 11:15:00 | 270.20 | 266.23 | 264.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 12:15:00 | 265.25 | 266.03 | 264.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 13:00:00 | 265.25 | 266.03 | 264.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 264.00 | 265.63 | 264.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 264.00 | 265.63 | 264.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 261.80 | 264.86 | 264.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 261.80 | 264.86 | 264.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 261.00 | 264.09 | 263.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 263.50 | 264.09 | 263.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 260.95 | 263.24 | 263.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 11:15:00 | 260.95 | 263.24 | 263.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 12:15:00 | 259.40 | 262.47 | 263.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 257.55 | 256.86 | 259.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:00:00 | 257.55 | 256.86 | 259.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 256.50 | 253.97 | 255.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 256.50 | 253.97 | 255.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 259.05 | 254.98 | 255.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 259.05 | 254.98 | 255.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 258.60 | 255.71 | 256.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:30:00 | 259.80 | 255.71 | 256.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 258.80 | 256.33 | 256.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 261.80 | 258.76 | 257.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 258.20 | 259.08 | 258.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 258.20 | 259.08 | 258.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 258.20 | 259.08 | 258.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 257.75 | 259.08 | 258.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 259.70 | 259.21 | 258.24 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 257.10 | 257.80 | 257.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 253.45 | 256.18 | 256.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 13:15:00 | 255.75 | 254.24 | 255.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 13:15:00 | 255.75 | 254.24 | 255.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 255.75 | 254.24 | 255.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 255.75 | 254.24 | 255.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 253.00 | 253.99 | 255.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:15:00 | 254.00 | 253.99 | 255.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 254.00 | 254.00 | 255.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 257.45 | 254.00 | 255.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 259.05 | 255.01 | 255.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 257.80 | 255.01 | 255.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 258.55 | 255.72 | 255.74 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 257.65 | 256.10 | 255.91 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 253.60 | 256.16 | 256.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 10:15:00 | 251.75 | 254.45 | 255.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 253.00 | 251.72 | 253.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 15:15:00 | 253.00 | 251.72 | 253.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 253.00 | 251.72 | 253.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 258.80 | 251.72 | 253.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 261.10 | 253.59 | 254.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 260.80 | 253.59 | 254.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 265.05 | 255.88 | 255.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 270.15 | 264.74 | 261.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 265.70 | 265.89 | 263.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 265.70 | 265.89 | 263.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 266.65 | 268.76 | 267.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:45:00 | 267.30 | 268.76 | 267.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 265.80 | 268.17 | 266.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 265.50 | 268.17 | 266.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 267.80 | 268.10 | 267.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:30:00 | 267.00 | 268.10 | 267.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 267.15 | 267.91 | 267.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:15:00 | 266.95 | 267.91 | 267.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 266.85 | 267.70 | 267.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:30:00 | 266.00 | 267.70 | 267.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 266.10 | 267.38 | 266.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 266.10 | 267.38 | 266.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 265.50 | 267.00 | 266.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 276.00 | 267.00 | 266.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 266.00 | 272.06 | 272.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 266.00 | 272.06 | 272.52 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 273.70 | 272.05 | 271.89 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 09:15:00 | 267.50 | 271.11 | 271.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 15:15:00 | 265.95 | 268.33 | 269.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 265.60 | 265.49 | 267.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:30:00 | 266.90 | 265.49 | 267.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 267.00 | 265.79 | 267.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 267.15 | 265.79 | 267.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 267.35 | 266.10 | 267.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 267.35 | 266.10 | 267.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 266.75 | 266.23 | 267.10 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 271.00 | 268.03 | 267.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 272.55 | 268.93 | 268.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 10:15:00 | 269.30 | 270.74 | 269.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 10:15:00 | 269.30 | 270.74 | 269.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 269.30 | 270.74 | 269.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 269.30 | 270.74 | 269.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 270.95 | 270.78 | 269.93 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 267.10 | 269.52 | 269.61 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 272.20 | 269.88 | 269.75 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 268.60 | 269.67 | 269.71 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 271.90 | 270.12 | 269.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 10:15:00 | 273.80 | 271.41 | 270.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 273.50 | 274.25 | 272.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 12:00:00 | 273.50 | 274.25 | 272.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 273.55 | 274.06 | 273.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:15:00 | 271.85 | 274.06 | 273.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 272.65 | 273.78 | 273.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 275.40 | 273.44 | 272.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 10:45:00 | 274.30 | 275.67 | 274.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 274.50 | 274.78 | 274.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 268.45 | 274.22 | 274.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 268.45 | 274.22 | 274.49 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 276.00 | 274.19 | 274.02 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 273.60 | 274.61 | 274.72 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 279.75 | 275.44 | 275.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 11:15:00 | 282.90 | 277.43 | 276.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 288.80 | 289.63 | 286.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:45:00 | 288.25 | 289.63 | 286.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 287.75 | 289.26 | 287.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:00:00 | 287.75 | 289.26 | 287.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 289.70 | 289.35 | 287.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 297.45 | 289.37 | 287.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 292.60 | 291.71 | 289.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-20 09:15:00 | 321.86 | 305.35 | 301.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 325.80 | 326.43 | 326.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 324.90 | 326.12 | 326.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 326.70 | 325.62 | 326.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 326.70 | 325.62 | 326.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 326.70 | 325.62 | 326.06 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 13:15:00 | 326.95 | 326.39 | 326.34 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 325.35 | 326.18 | 326.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 323.40 | 325.51 | 325.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 327.30 | 325.62 | 325.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 327.30 | 325.62 | 325.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 327.30 | 325.62 | 325.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 328.00 | 325.62 | 325.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 329.20 | 326.33 | 326.19 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 323.40 | 325.98 | 326.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 318.30 | 323.35 | 324.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 13:15:00 | 319.30 | 318.95 | 320.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 14:00:00 | 319.30 | 318.95 | 320.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 319.25 | 319.16 | 320.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 321.95 | 319.16 | 320.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 319.30 | 319.19 | 320.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 320.80 | 319.19 | 320.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 320.15 | 319.38 | 320.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 320.50 | 319.38 | 320.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 318.00 | 319.11 | 320.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 13:45:00 | 317.25 | 318.69 | 319.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:30:00 | 316.50 | 316.97 | 318.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 316.90 | 317.35 | 318.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:30:00 | 317.45 | 317.68 | 318.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 319.50 | 318.05 | 318.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 319.50 | 318.05 | 318.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 320.05 | 318.45 | 318.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 320.05 | 318.45 | 318.43 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 13:15:00 | 317.70 | 318.30 | 318.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 14:15:00 | 316.65 | 317.97 | 318.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 10:15:00 | 317.90 | 317.58 | 317.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 10:15:00 | 317.90 | 317.58 | 317.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 317.90 | 317.58 | 317.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:00:00 | 317.90 | 317.58 | 317.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 318.35 | 317.73 | 317.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 318.35 | 317.73 | 317.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 316.80 | 317.54 | 317.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 319.25 | 317.54 | 317.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 314.00 | 312.61 | 314.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 314.00 | 312.61 | 314.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 314.40 | 312.97 | 314.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 314.45 | 312.97 | 314.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 314.70 | 313.32 | 314.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 313.20 | 313.62 | 314.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 313.10 | 313.62 | 314.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:15:00 | 313.40 | 313.69 | 314.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:30:00 | 313.45 | 313.93 | 314.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 12:15:00 | 317.10 | 314.57 | 314.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 12:15:00 | 317.10 | 314.57 | 314.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 13:15:00 | 318.20 | 315.29 | 314.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 14:15:00 | 312.15 | 314.66 | 314.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 14:15:00 | 312.15 | 314.66 | 314.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 312.15 | 314.66 | 314.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 312.15 | 314.66 | 314.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 311.40 | 314.01 | 314.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 310.50 | 312.81 | 313.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 15:15:00 | 309.30 | 309.15 | 310.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:15:00 | 315.00 | 309.15 | 310.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 315.05 | 310.33 | 311.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 315.05 | 310.33 | 311.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 316.55 | 311.58 | 311.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 316.55 | 311.58 | 311.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 316.25 | 312.51 | 312.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 317.75 | 314.69 | 313.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 314.65 | 315.11 | 313.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:45:00 | 314.80 | 315.11 | 313.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 314.25 | 314.94 | 313.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:45:00 | 313.75 | 314.94 | 313.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 313.25 | 314.60 | 313.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 313.25 | 314.60 | 313.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 312.00 | 314.08 | 313.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 312.00 | 314.08 | 313.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 310.00 | 313.26 | 313.23 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 312.15 | 313.04 | 313.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 307.95 | 311.10 | 311.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 303.90 | 303.02 | 305.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 14:00:00 | 303.90 | 303.02 | 305.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 304.00 | 303.22 | 305.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 304.00 | 303.22 | 305.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 300.40 | 300.08 | 301.67 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 305.70 | 303.09 | 302.73 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 301.25 | 302.48 | 302.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 300.45 | 302.07 | 302.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 301.70 | 300.48 | 301.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 301.70 | 300.48 | 301.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 301.70 | 300.48 | 301.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 301.70 | 300.48 | 301.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 303.60 | 301.10 | 301.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 303.60 | 301.10 | 301.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 303.35 | 301.83 | 301.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 304.35 | 302.34 | 301.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 13:15:00 | 302.50 | 303.84 | 303.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 13:15:00 | 302.50 | 303.84 | 303.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 302.50 | 303.84 | 303.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:45:00 | 302.20 | 303.84 | 303.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 302.15 | 303.50 | 303.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 302.40 | 303.50 | 303.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 306.50 | 308.81 | 306.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 305.75 | 308.81 | 306.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 306.80 | 308.40 | 306.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 306.80 | 308.40 | 306.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 307.35 | 308.19 | 306.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:15:00 | 306.60 | 308.19 | 306.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 308.50 | 308.26 | 307.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:30:00 | 307.15 | 308.26 | 307.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 307.60 | 308.01 | 307.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 305.75 | 308.01 | 307.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 304.70 | 307.35 | 306.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 303.90 | 307.35 | 306.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 304.85 | 306.85 | 306.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 304.75 | 306.85 | 306.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 305.00 | 306.34 | 306.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 300.65 | 304.40 | 305.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 297.55 | 297.47 | 299.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 300.30 | 298.04 | 299.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 300.30 | 298.04 | 299.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 300.30 | 298.04 | 299.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 300.40 | 298.51 | 299.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 300.45 | 298.51 | 299.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 298.75 | 298.60 | 299.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 299.95 | 298.60 | 299.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 297.65 | 298.41 | 299.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 298.90 | 298.41 | 299.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 300.60 | 298.67 | 299.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 300.60 | 298.67 | 299.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 299.75 | 298.89 | 299.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 299.90 | 298.89 | 299.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 301.95 | 299.63 | 299.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 306.10 | 303.14 | 301.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 303.95 | 304.48 | 302.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 303.95 | 304.48 | 302.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 303.95 | 304.48 | 302.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 303.75 | 304.48 | 302.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 302.90 | 304.17 | 302.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 308.95 | 304.17 | 302.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 304.20 | 305.67 | 304.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 304.45 | 304.89 | 304.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 303.00 | 304.35 | 304.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 303.00 | 304.35 | 304.44 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 309.15 | 304.76 | 304.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 309.60 | 306.94 | 305.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 312.45 | 313.28 | 310.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 09:30:00 | 313.15 | 313.28 | 310.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 313.85 | 313.39 | 310.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 11:15:00 | 314.65 | 313.39 | 310.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 314.15 | 313.43 | 311.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 13:30:00 | 314.45 | 313.50 | 311.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 314.15 | 313.50 | 311.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 313.50 | 318.03 | 315.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 321.65 | 316.64 | 315.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:00:00 | 321.05 | 317.52 | 316.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:00:00 | 321.35 | 318.29 | 316.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 311.80 | 316.72 | 317.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 311.80 | 316.72 | 317.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 310.20 | 315.42 | 316.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 306.20 | 305.68 | 308.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 305.15 | 305.68 | 308.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 306.25 | 303.63 | 305.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 306.25 | 303.63 | 305.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 304.50 | 303.81 | 305.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 306.00 | 303.81 | 305.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 305.70 | 304.18 | 305.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 305.70 | 304.18 | 305.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 308.35 | 305.02 | 305.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 308.35 | 305.02 | 305.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 307.40 | 305.49 | 306.11 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 311.25 | 306.65 | 306.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 311.80 | 308.41 | 307.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 303.75 | 317.86 | 314.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 303.75 | 317.86 | 314.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 303.75 | 317.86 | 314.81 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 301.25 | 311.89 | 312.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 299.00 | 304.59 | 308.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 298.50 | 297.25 | 301.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 298.50 | 297.25 | 301.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 298.65 | 297.53 | 300.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 298.65 | 297.53 | 300.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 298.40 | 297.23 | 299.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 297.15 | 297.23 | 299.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:30:00 | 296.70 | 296.69 | 298.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 295.30 | 296.42 | 297.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 11:00:00 | 297.10 | 294.39 | 294.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 296.15 | 294.74 | 294.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 296.15 | 294.74 | 294.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 300.70 | 296.65 | 295.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 297.30 | 298.39 | 297.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 297.30 | 298.39 | 297.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 297.30 | 298.39 | 297.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 300.05 | 298.39 | 297.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 299.40 | 298.59 | 297.54 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 291.40 | 296.65 | 297.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 289.35 | 294.03 | 295.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 297.55 | 293.88 | 294.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 297.55 | 293.88 | 294.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 297.55 | 293.88 | 294.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 297.55 | 293.88 | 294.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 296.60 | 294.43 | 295.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 302.90 | 294.43 | 295.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 300.00 | 295.54 | 295.54 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 11:15:00 | 301.50 | 302.80 | 302.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 299.90 | 302.22 | 302.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 302.15 | 301.96 | 302.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 302.15 | 301.96 | 302.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 302.15 | 301.96 | 302.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 302.15 | 301.96 | 302.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 301.95 | 301.96 | 302.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 305.40 | 301.96 | 302.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 307.55 | 303.08 | 302.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 308.30 | 305.97 | 304.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 305.90 | 306.23 | 304.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 10:00:00 | 305.90 | 306.23 | 304.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 306.85 | 306.36 | 305.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 306.85 | 306.36 | 305.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 305.05 | 306.54 | 305.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 304.25 | 306.54 | 305.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 305.10 | 306.26 | 305.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 303.55 | 306.26 | 305.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 306.60 | 306.32 | 305.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 307.75 | 306.32 | 305.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:45:00 | 308.65 | 308.05 | 307.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-19 10:15:00 | 338.53 | 327.72 | 322.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 322.90 | 326.84 | 327.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 14:15:00 | 320.75 | 324.26 | 325.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 324.90 | 322.43 | 323.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 324.90 | 322.43 | 323.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 324.90 | 322.43 | 323.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 324.90 | 322.43 | 323.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 328.80 | 323.70 | 324.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 323.30 | 323.70 | 324.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 326.75 | 324.68 | 324.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 326.75 | 324.68 | 324.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 329.95 | 325.73 | 325.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 328.20 | 328.38 | 326.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:00:00 | 328.20 | 328.38 | 326.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 324.90 | 327.69 | 326.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 324.90 | 327.69 | 326.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 326.55 | 327.46 | 326.78 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 320.50 | 325.57 | 326.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 319.50 | 322.11 | 323.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 311.60 | 308.70 | 312.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 311.60 | 308.70 | 312.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 311.60 | 308.70 | 312.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:00:00 | 308.20 | 309.26 | 311.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 307.30 | 308.92 | 311.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:00:00 | 308.05 | 308.86 | 310.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 292.79 | 303.35 | 307.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 291.94 | 303.35 | 307.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 292.65 | 303.35 | 307.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 298.35 | 298.02 | 302.49 | SL hit (close>ema200) qty=0.50 sl=298.02 alert=retest2 |

### Cycle 153 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 307.70 | 303.50 | 303.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 309.30 | 304.66 | 303.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 303.75 | 304.86 | 304.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 303.75 | 304.86 | 304.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 303.75 | 304.86 | 304.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 303.75 | 304.86 | 304.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 302.00 | 304.29 | 303.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:45:00 | 301.20 | 304.29 | 303.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 300.25 | 303.48 | 303.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 299.35 | 302.66 | 303.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 301.35 | 300.27 | 301.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 301.35 | 300.27 | 301.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 301.35 | 300.27 | 301.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 301.35 | 300.27 | 301.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 301.20 | 300.46 | 301.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:15:00 | 302.50 | 300.46 | 301.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 301.10 | 300.59 | 301.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 302.75 | 300.59 | 301.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 299.90 | 300.45 | 301.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:30:00 | 300.40 | 300.45 | 301.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 296.00 | 294.65 | 296.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 297.50 | 294.65 | 296.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 295.90 | 294.90 | 296.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 293.70 | 294.90 | 296.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 294.30 | 294.78 | 296.14 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 297.65 | 296.49 | 296.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 299.70 | 297.14 | 296.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 293.95 | 298.59 | 298.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 293.95 | 298.59 | 298.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 293.95 | 298.59 | 298.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 293.75 | 298.59 | 298.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 294.60 | 297.33 | 297.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 293.60 | 296.58 | 297.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 299.25 | 295.49 | 296.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 299.25 | 295.49 | 296.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 299.25 | 295.49 | 296.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 300.60 | 295.49 | 296.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 300.30 | 296.45 | 296.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 300.30 | 296.45 | 296.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 299.75 | 297.11 | 296.93 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 290.75 | 296.27 | 296.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 289.60 | 294.93 | 296.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 291.95 | 291.63 | 293.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 291.95 | 291.63 | 293.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 291.95 | 291.63 | 293.56 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 296.90 | 294.25 | 294.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 301.70 | 296.05 | 295.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 297.85 | 301.80 | 299.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 297.85 | 301.80 | 299.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 297.85 | 301.80 | 299.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 297.85 | 301.80 | 299.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 296.25 | 300.69 | 299.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 296.80 | 300.69 | 299.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 295.95 | 298.38 | 298.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 291.75 | 296.77 | 297.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 299.30 | 294.39 | 295.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 299.30 | 294.39 | 295.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 299.30 | 294.39 | 295.59 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 301.15 | 296.30 | 296.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 303.05 | 297.65 | 296.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 296.10 | 298.76 | 297.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 296.10 | 298.76 | 297.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 296.10 | 298.76 | 297.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 298.30 | 298.04 | 297.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 298.50 | 297.93 | 297.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:00:00 | 299.65 | 298.27 | 297.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 316.70 | 318.77 | 319.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 316.70 | 318.77 | 319.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 316.35 | 318.29 | 318.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 321.05 | 318.49 | 318.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 321.05 | 318.49 | 318.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 321.05 | 318.49 | 318.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 10:30:00 | 319.60 | 318.52 | 318.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:00:00 | 319.80 | 318.80 | 318.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 15:15:00 | 319.25 | 318.89 | 318.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 15:15:00 | 319.25 | 318.89 | 318.85 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 317.00 | 318.54 | 318.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 15:15:00 | 315.50 | 317.09 | 317.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 14:15:00 | 316.30 | 315.58 | 316.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 14:15:00 | 316.30 | 315.58 | 316.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 316.30 | 315.58 | 316.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:45:00 | 315.70 | 315.58 | 316.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 316.55 | 315.78 | 316.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 317.45 | 315.78 | 316.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 315.60 | 315.74 | 316.51 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 317.65 | 316.71 | 316.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 322.10 | 318.03 | 317.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 13:15:00 | 319.50 | 320.40 | 319.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 319.50 | 320.40 | 319.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 319.50 | 320.40 | 319.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 319.50 | 320.40 | 319.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 317.50 | 319.82 | 319.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 15:00:00 | 317.50 | 319.82 | 319.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 318.20 | 319.49 | 319.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 315.40 | 319.49 | 319.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 314.20 | 318.43 | 318.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 312.80 | 316.32 | 317.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 11:15:00 | 319.95 | 314.83 | 315.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 11:15:00 | 319.95 | 314.83 | 315.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 319.95 | 314.83 | 315.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:00:00 | 319.95 | 314.83 | 315.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 320.65 | 315.99 | 316.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:30:00 | 321.00 | 315.99 | 316.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 13:15:00 | 321.70 | 317.13 | 316.88 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 10:15:00 | 310.70 | 316.11 | 316.62 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 323.70 | 317.19 | 316.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 337.00 | 323.16 | 319.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 335.70 | 336.41 | 329.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 335.70 | 336.41 | 329.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 332.70 | 335.93 | 333.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:00:00 | 332.70 | 335.93 | 333.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 332.00 | 335.14 | 333.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:45:00 | 331.90 | 335.14 | 333.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 332.35 | 332.76 | 332.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 331.35 | 332.76 | 332.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 334.85 | 333.18 | 332.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:30:00 | 334.10 | 333.18 | 332.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 332.70 | 333.08 | 332.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 332.70 | 333.08 | 332.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 332.00 | 332.87 | 332.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:30:00 | 332.20 | 332.87 | 332.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 332.65 | 332.82 | 332.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 338.45 | 332.84 | 332.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 337.20 | 334.41 | 333.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 12:15:00 | 333.65 | 333.74 | 333.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 12:15:00 | 333.65 | 333.74 | 333.75 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 13:15:00 | 336.80 | 334.36 | 334.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 14:15:00 | 344.00 | 336.28 | 334.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 341.55 | 342.64 | 339.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:45:00 | 341.25 | 342.64 | 339.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:15:00 | 250.65 | 2024-05-16 13:15:00 | 248.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest1 | 2024-05-16 10:00:00 | 250.50 | 2024-05-16 13:15:00 | 248.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-05-16 15:15:00 | 252.05 | 2024-05-22 12:15:00 | 250.20 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-05-17 12:30:00 | 251.75 | 2024-05-22 12:15:00 | 250.20 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-05-17 13:15:00 | 251.70 | 2024-05-22 12:15:00 | 250.20 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-05-18 09:15:00 | 252.50 | 2024-05-22 12:15:00 | 250.20 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-05-27 11:15:00 | 254.80 | 2024-05-27 15:15:00 | 253.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-05-27 12:15:00 | 255.00 | 2024-05-27 15:15:00 | 253.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-05-29 12:00:00 | 250.20 | 2024-06-03 09:15:00 | 257.15 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2024-05-29 13:45:00 | 250.05 | 2024-06-03 09:15:00 | 257.15 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-06-18 09:15:00 | 259.95 | 2024-06-24 09:15:00 | 254.12 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-06-19 10:45:00 | 258.65 | 2024-06-24 09:15:00 | 254.12 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-07-01 09:30:00 | 263.60 | 2024-07-02 11:15:00 | 260.25 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-07-04 12:45:00 | 265.05 | 2024-07-05 09:15:00 | 259.50 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-07-04 14:00:00 | 265.80 | 2024-07-05 09:15:00 | 259.50 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-07-19 10:45:00 | 240.35 | 2024-07-25 09:15:00 | 228.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 11:30:00 | 240.80 | 2024-07-25 09:15:00 | 228.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 15:00:00 | 240.85 | 2024-07-25 09:15:00 | 228.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 09:45:00 | 240.70 | 2024-07-25 09:15:00 | 228.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 09:15:00 | 240.60 | 2024-07-25 09:15:00 | 228.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 10:45:00 | 240.35 | 2024-07-26 09:15:00 | 232.60 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2024-07-19 11:30:00 | 240.80 | 2024-07-26 09:15:00 | 232.60 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2024-07-19 15:00:00 | 240.85 | 2024-07-26 09:15:00 | 232.60 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2024-07-22 09:45:00 | 240.70 | 2024-07-26 09:15:00 | 232.60 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2024-07-23 09:15:00 | 240.60 | 2024-07-26 09:15:00 | 232.60 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2024-08-19 12:15:00 | 208.75 | 2024-08-20 09:15:00 | 214.33 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-08-26 09:15:00 | 226.45 | 2024-08-29 10:15:00 | 226.34 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-08-26 12:15:00 | 225.25 | 2024-08-29 10:15:00 | 226.34 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-09-11 13:30:00 | 212.02 | 2024-09-13 09:15:00 | 215.99 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-09-17 14:30:00 | 216.10 | 2024-09-18 13:15:00 | 212.79 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-09-30 10:30:00 | 205.42 | 2024-10-07 09:15:00 | 195.32 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2024-09-30 14:15:00 | 205.60 | 2024-10-07 10:15:00 | 195.15 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2024-09-30 10:30:00 | 205.42 | 2024-10-08 10:15:00 | 194.69 | STOP_HIT | 0.50 | 5.22% |
| SELL | retest2 | 2024-09-30 14:15:00 | 205.60 | 2024-10-08 10:15:00 | 194.69 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2024-10-21 09:15:00 | 184.73 | 2024-10-21 13:15:00 | 175.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 184.73 | 2024-10-22 15:15:00 | 166.26 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-12-09 09:15:00 | 176.88 | 2024-12-09 13:15:00 | 172.89 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest1 | 2024-12-09 10:15:00 | 178.84 | 2024-12-09 13:15:00 | 172.89 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2024-12-10 09:30:00 | 172.39 | 2024-12-10 10:15:00 | 170.49 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-12-12 14:00:00 | 175.59 | 2024-12-12 14:15:00 | 173.14 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-12-16 15:15:00 | 172.10 | 2024-12-19 09:15:00 | 163.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 15:15:00 | 172.10 | 2024-12-19 15:15:00 | 165.10 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-01-02 09:15:00 | 155.80 | 2025-01-02 13:15:00 | 159.21 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-01-23 09:15:00 | 153.80 | 2025-01-23 10:15:00 | 156.55 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-01-28 10:15:00 | 151.05 | 2025-01-28 12:15:00 | 156.05 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-01-28 10:45:00 | 151.07 | 2025-01-28 12:15:00 | 156.05 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2025-02-01 15:00:00 | 166.94 | 2025-02-07 11:15:00 | 166.85 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-02-03 10:45:00 | 167.41 | 2025-02-07 11:15:00 | 166.85 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-02-03 13:00:00 | 166.77 | 2025-02-07 11:15:00 | 166.85 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-02-04 09:15:00 | 167.20 | 2025-02-07 11:15:00 | 166.85 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-02-07 11:15:00 | 168.50 | 2025-02-07 11:15:00 | 166.85 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-02-11 12:15:00 | 164.83 | 2025-02-12 09:15:00 | 156.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 12:15:00 | 164.83 | 2025-02-12 12:15:00 | 162.39 | STOP_HIT | 0.50 | 1.48% |
| SELL | retest2 | 2025-02-11 13:00:00 | 164.21 | 2025-02-13 15:15:00 | 164.50 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-02-13 12:00:00 | 164.89 | 2025-02-13 15:15:00 | 164.50 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-02-13 13:00:00 | 164.98 | 2025-02-13 15:15:00 | 164.50 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-02-24 15:15:00 | 162.29 | 2025-02-25 15:15:00 | 157.66 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-02-25 10:15:00 | 161.45 | 2025-02-25 15:15:00 | 157.66 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-03-07 14:15:00 | 163.81 | 2025-03-10 13:15:00 | 160.52 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-03-28 09:15:00 | 178.97 | 2025-03-28 14:15:00 | 173.65 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-03-28 09:45:00 | 178.78 | 2025-03-28 14:15:00 | 173.65 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-03-28 10:15:00 | 178.50 | 2025-03-28 14:15:00 | 173.65 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-04-02 09:15:00 | 169.31 | 2025-04-03 15:15:00 | 175.30 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-04-03 12:30:00 | 173.71 | 2025-04-03 15:15:00 | 175.30 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-04-22 13:30:00 | 193.58 | 2025-04-25 14:15:00 | 187.41 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-04-24 09:45:00 | 192.87 | 2025-04-25 14:15:00 | 187.41 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-05-05 11:45:00 | 204.31 | 2025-05-06 12:15:00 | 199.45 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-05-05 15:15:00 | 204.25 | 2025-05-06 12:15:00 | 199.45 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-05-06 09:30:00 | 204.14 | 2025-05-06 12:15:00 | 199.45 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-05-15 13:15:00 | 209.26 | 2025-05-20 13:15:00 | 208.20 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-05-22 14:15:00 | 206.60 | 2025-05-23 11:15:00 | 210.75 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-05-30 13:15:00 | 215.22 | 2025-06-03 14:15:00 | 211.45 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-06-03 10:30:00 | 214.95 | 2025-06-03 14:15:00 | 211.45 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-06-12 09:30:00 | 222.15 | 2025-06-13 10:15:00 | 219.01 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-06-13 10:15:00 | 222.15 | 2025-06-13 10:15:00 | 219.01 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-06-17 15:15:00 | 217.25 | 2025-06-18 09:15:00 | 222.15 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-06-20 09:30:00 | 224.64 | 2025-06-30 09:15:00 | 247.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-10 13:15:00 | 254.65 | 2025-07-18 14:15:00 | 264.84 | STOP_HIT | 1.00 | 4.00% |
| BUY | retest2 | 2025-07-10 15:15:00 | 255.60 | 2025-07-18 14:15:00 | 264.84 | STOP_HIT | 1.00 | 3.62% |
| BUY | retest2 | 2025-07-11 11:00:00 | 256.79 | 2025-07-18 14:15:00 | 264.84 | STOP_HIT | 1.00 | 3.13% |
| SELL | retest2 | 2025-07-23 12:30:00 | 257.76 | 2025-07-28 09:15:00 | 260.26 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-07-24 11:00:00 | 257.75 | 2025-07-28 09:15:00 | 260.26 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-07-28 12:45:00 | 257.63 | 2025-07-30 10:15:00 | 261.45 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-08-05 15:15:00 | 266.80 | 2025-08-06 09:15:00 | 261.15 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-08-11 09:15:00 | 263.50 | 2025-08-11 11:15:00 | 260.95 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-09-05 09:15:00 | 276.00 | 2025-09-10 12:15:00 | 266.00 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2025-09-29 09:15:00 | 275.40 | 2025-10-01 09:15:00 | 268.45 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-09-30 10:45:00 | 274.30 | 2025-10-01 09:15:00 | 268.45 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-09-30 13:00:00 | 274.50 | 2025-10-01 09:15:00 | 268.45 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-10-14 09:15:00 | 297.45 | 2025-10-20 09:15:00 | 321.86 | TARGET_HIT | 1.00 | 8.21% |
| BUY | retest2 | 2025-10-14 14:30:00 | 292.60 | 2025-10-20 12:15:00 | 327.19 | TARGET_HIT | 1.00 | 11.82% |
| SELL | retest2 | 2025-11-13 13:45:00 | 317.25 | 2025-11-17 12:15:00 | 320.05 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-11-14 09:30:00 | 316.50 | 2025-11-17 12:15:00 | 320.05 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-11-14 14:45:00 | 316.90 | 2025-11-17 12:15:00 | 320.05 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-11-17 10:30:00 | 317.45 | 2025-11-17 12:15:00 | 320.05 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-11-20 14:30:00 | 313.20 | 2025-11-21 12:15:00 | 317.10 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-11-20 15:15:00 | 313.10 | 2025-11-21 12:15:00 | 317.10 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-11-21 10:15:00 | 313.40 | 2025-11-21 12:15:00 | 317.10 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-11-21 11:30:00 | 313.45 | 2025-11-21 12:15:00 | 317.10 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-24 09:15:00 | 308.95 | 2025-12-26 14:15:00 | 303.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-12-26 10:15:00 | 304.20 | 2025-12-26 14:15:00 | 303.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-12-26 12:15:00 | 304.45 | 2025-12-26 14:15:00 | 303.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2026-01-01 11:15:00 | 314.65 | 2026-01-08 10:15:00 | 311.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-01 12:15:00 | 314.15 | 2026-01-08 10:15:00 | 311.80 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-01-01 13:30:00 | 314.45 | 2026-01-08 10:15:00 | 311.80 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-01 14:15:00 | 314.15 | 2026-01-08 10:15:00 | 311.80 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-01-06 09:45:00 | 321.65 | 2026-01-08 10:15:00 | 311.80 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2026-01-06 11:00:00 | 321.05 | 2026-01-08 10:15:00 | 311.80 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2026-01-06 12:00:00 | 321.35 | 2026-01-08 10:15:00 | 311.80 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2026-01-22 10:15:00 | 297.15 | 2026-01-28 11:15:00 | 296.15 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2026-01-22 12:30:00 | 296.70 | 2026-01-28 11:15:00 | 296.15 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2026-01-23 09:30:00 | 295.30 | 2026-01-28 11:15:00 | 296.15 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-01-28 11:00:00 | 297.10 | 2026-01-28 11:15:00 | 296.15 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-02-11 11:15:00 | 307.75 | 2026-02-19 10:15:00 | 338.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-12 09:45:00 | 308.65 | 2026-02-19 10:15:00 | 339.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-25 09:15:00 | 323.30 | 2026-02-25 10:15:00 | 326.75 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-03-05 13:00:00 | 308.20 | 2026-03-09 09:15:00 | 292.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:45:00 | 307.30 | 2026-03-09 09:15:00 | 291.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:00:00 | 308.05 | 2026-03-09 09:15:00 | 292.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:00:00 | 308.20 | 2026-03-09 14:15:00 | 298.35 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2026-03-05 13:45:00 | 307.30 | 2026-03-09 14:15:00 | 298.35 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2026-03-06 10:00:00 | 308.05 | 2026-03-09 14:15:00 | 298.35 | STOP_HIT | 0.50 | 3.15% |
| BUY | retest2 | 2026-04-02 11:30:00 | 298.30 | 2026-04-13 13:15:00 | 316.70 | STOP_HIT | 1.00 | 6.17% |
| BUY | retest2 | 2026-04-02 13:15:00 | 298.50 | 2026-04-13 13:15:00 | 316.70 | STOP_HIT | 1.00 | 6.10% |
| BUY | retest2 | 2026-04-02 14:00:00 | 299.65 | 2026-04-13 13:15:00 | 316.70 | STOP_HIT | 1.00 | 5.69% |
| SELL | retest2 | 2026-04-15 10:30:00 | 319.60 | 2026-04-15 15:15:00 | 319.25 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-04-15 15:00:00 | 319.80 | 2026-04-15 15:15:00 | 319.25 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2026-05-06 09:15:00 | 338.45 | 2026-05-07 12:15:00 | 333.65 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-05-06 14:15:00 | 337.20 | 2026-05-07 12:15:00 | 333.65 | STOP_HIT | 1.00 | -1.05% |
