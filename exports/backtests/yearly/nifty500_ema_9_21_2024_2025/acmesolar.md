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
| CROSSOVER | 106 |
| ALERT1 | 71 |
| ALERT2 | 68 |
| ALERT2_SKIP | 37 |
| ALERT3 | 193 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 76 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 62
- **Target hits / Stop hits / Partials:** 7 / 76 / 7
- **Avg / median % per leg:** 0.20% / -1.02%
- **Sum % (uncompounded):** 18.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 11 | 30.6% | 4 | 32 | 0 | 0.22% | 7.9% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.33% | -9.3% |
| BUY @ 3rd Alert (retest2) | 32 | 11 | 34.4% | 4 | 28 | 0 | 0.54% | 17.2% |
| SELL (all) | 54 | 17 | 31.5% | 3 | 44 | 7 | 0.19% | 10.3% |
| SELL @ 2nd Alert (retest1) | 5 | 5 | 100.0% | 0 | 3 | 2 | 4.31% | 21.5% |
| SELL @ 3rd Alert (retest2) | 49 | 12 | 24.5% | 3 | 41 | 5 | -0.23% | -11.3% |
| retest1 (combined) | 9 | 5 | 55.6% | 0 | 7 | 2 | 1.36% | 12.2% |
| retest2 (combined) | 81 | 23 | 28.4% | 7 | 69 | 5 | 0.07% | 5.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 275.15 | 254.48 | 253.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 275.95 | 258.77 | 255.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 263.35 | 264.16 | 259.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 14:30:00 | 266.15 | 264.16 | 259.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 249.85 | 261.26 | 259.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 244.95 | 261.26 | 259.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 246.30 | 258.27 | 257.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:30:00 | 244.15 | 258.27 | 257.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 244.75 | 255.57 | 256.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 12:15:00 | 241.90 | 252.83 | 255.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 13:15:00 | 243.25 | 243.02 | 247.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 13:30:00 | 246.05 | 243.02 | 247.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 245.25 | 243.46 | 247.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:30:00 | 245.80 | 243.46 | 247.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 262.75 | 247.57 | 248.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:00:00 | 262.75 | 247.57 | 248.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 261.95 | 250.44 | 249.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 11:15:00 | 267.85 | 253.92 | 251.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 247.00 | 254.83 | 253.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 247.00 | 254.83 | 253.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 247.00 | 254.83 | 253.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:45:00 | 240.85 | 254.83 | 253.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 250.25 | 252.88 | 252.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 12:30:00 | 251.65 | 252.78 | 252.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 14:15:00 | 250.95 | 252.25 | 252.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 14:15:00 | 250.95 | 252.25 | 252.27 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 15:15:00 | 256.60 | 253.12 | 252.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 261.90 | 254.88 | 253.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 267.50 | 268.76 | 263.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 267.50 | 268.76 | 263.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 263.30 | 269.36 | 266.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 263.30 | 269.36 | 266.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 262.15 | 267.91 | 266.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:00:00 | 262.15 | 267.91 | 266.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 265.50 | 266.22 | 265.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:30:00 | 263.55 | 266.22 | 265.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 261.55 | 265.29 | 265.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 261.55 | 265.29 | 265.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 15:15:00 | 260.40 | 264.31 | 264.79 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 266.35 | 265.20 | 265.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 13:15:00 | 270.80 | 266.72 | 265.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 15:15:00 | 276.90 | 277.08 | 273.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:15:00 | 281.75 | 277.08 | 273.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:45:00 | 279.80 | 277.31 | 274.03 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 10:15:00 | 279.10 | 277.31 | 274.03 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 11:15:00 | 279.85 | 277.48 | 274.41 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 273.60 | 277.30 | 275.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-05 14:15:00 | 273.60 | 277.30 | 275.38 | SL hit (close<ema400) qty=1.00 sl=275.38 alert=retest1 |

### Cycle 8 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 11:15:00 | 271.25 | 274.29 | 274.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 12:15:00 | 269.15 | 273.26 | 273.91 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 280.05 | 274.62 | 274.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 284.35 | 277.36 | 275.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 13:15:00 | 276.95 | 278.10 | 276.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 13:15:00 | 276.95 | 278.10 | 276.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 276.95 | 278.10 | 276.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:00:00 | 276.95 | 278.10 | 276.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 276.90 | 277.86 | 276.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:45:00 | 276.50 | 277.86 | 276.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 277.90 | 277.87 | 276.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:30:00 | 275.70 | 276.74 | 276.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 271.50 | 275.69 | 275.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 15:15:00 | 269.40 | 272.63 | 274.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 15:15:00 | 268.95 | 268.77 | 270.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-12 09:15:00 | 273.85 | 268.77 | 270.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 270.30 | 269.07 | 270.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 268.00 | 270.87 | 271.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 09:15:00 | 254.60 | 258.59 | 261.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-19 11:15:00 | 241.20 | 247.34 | 253.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 11 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 238.20 | 237.12 | 237.10 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 12:15:00 | 236.60 | 237.08 | 237.09 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 239.55 | 237.49 | 237.27 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 12:15:00 | 236.50 | 237.23 | 237.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 233.00 | 236.38 | 236.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 233.40 | 232.88 | 234.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 14:00:00 | 233.40 | 232.88 | 234.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 235.60 | 233.42 | 234.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:30:00 | 236.50 | 233.42 | 234.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 235.80 | 233.90 | 234.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 236.51 | 233.90 | 234.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 235.91 | 234.72 | 234.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 11:15:00 | 235.44 | 234.72 | 234.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 12:15:00 | 235.50 | 234.92 | 234.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 12:15:00 | 235.48 | 235.03 | 235.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 235.48 | 235.03 | 235.03 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 14:15:00 | 234.20 | 234.91 | 234.98 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 236.00 | 235.13 | 235.07 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 234.07 | 234.92 | 234.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 12:15:00 | 232.71 | 234.10 | 234.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 14:15:00 | 234.51 | 234.10 | 234.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 14:15:00 | 234.51 | 234.10 | 234.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 234.51 | 234.10 | 234.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:00:00 | 234.51 | 234.10 | 234.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 236.15 | 234.51 | 234.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 238.90 | 234.51 | 234.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 239.15 | 235.44 | 235.04 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 230.99 | 235.42 | 235.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 228.37 | 233.40 | 234.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 232.42 | 231.82 | 233.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 232.42 | 231.82 | 233.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 232.42 | 231.82 | 233.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 233.72 | 231.82 | 233.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 232.29 | 232.07 | 233.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 232.65 | 232.07 | 233.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 233.68 | 232.44 | 233.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 15:15:00 | 231.50 | 232.62 | 233.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 239.82 | 232.66 | 232.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 239.82 | 232.66 | 232.59 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 231.50 | 233.71 | 233.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 230.13 | 232.99 | 233.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 230.16 | 225.60 | 228.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 230.16 | 225.60 | 228.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 230.16 | 225.60 | 228.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 230.16 | 225.60 | 228.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 230.08 | 226.50 | 228.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:30:00 | 230.08 | 226.50 | 228.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 227.70 | 226.44 | 227.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 227.51 | 226.44 | 227.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 224.59 | 226.07 | 227.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:00:00 | 223.16 | 224.99 | 226.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:45:00 | 222.81 | 223.43 | 224.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 11:15:00 | 230.14 | 224.77 | 225.45 | SL hit (close>static) qty=1.00 sl=229.79 alert=retest2 |

### Cycle 23 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 230.58 | 225.93 | 225.92 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 223.42 | 226.18 | 226.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 11:15:00 | 223.02 | 225.55 | 225.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 09:15:00 | 214.64 | 213.48 | 215.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 10:00:00 | 214.64 | 213.48 | 215.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 186.58 | 178.65 | 184.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 186.58 | 178.65 | 184.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 191.23 | 181.17 | 185.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 191.23 | 181.17 | 185.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 191.23 | 187.93 | 187.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 210.35 | 192.42 | 189.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 13:15:00 | 218.53 | 220.10 | 213.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 13:15:00 | 218.53 | 220.10 | 213.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 218.53 | 220.10 | 213.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:30:00 | 216.08 | 220.10 | 213.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 209.08 | 217.30 | 213.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:45:00 | 207.55 | 217.30 | 213.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 207.40 | 215.32 | 213.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:00:00 | 207.40 | 215.32 | 213.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 206.67 | 211.31 | 211.65 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 222.52 | 210.56 | 210.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 225.11 | 213.47 | 211.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 13:15:00 | 234.20 | 235.19 | 230.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 13:45:00 | 234.33 | 235.19 | 230.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 214.91 | 230.81 | 229.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 214.91 | 230.81 | 229.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 213.00 | 227.25 | 227.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 211.00 | 216.82 | 221.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 178.05 | 177.57 | 183.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 14:45:00 | 177.77 | 177.57 | 183.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 178.61 | 178.00 | 182.42 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 190.76 | 183.81 | 183.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 196.40 | 190.07 | 186.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 192.00 | 196.37 | 192.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 192.00 | 196.37 | 192.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 192.00 | 196.37 | 192.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 192.00 | 196.37 | 192.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 194.27 | 195.95 | 192.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:45:00 | 197.07 | 195.98 | 193.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:15:00 | 196.51 | 195.98 | 193.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 14:15:00 | 193.52 | 193.58 | 193.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 14:15:00 | 193.52 | 193.58 | 193.59 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 09:15:00 | 194.19 | 193.69 | 193.63 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 11:15:00 | 193.22 | 193.76 | 193.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 183.60 | 191.38 | 192.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 15:15:00 | 188.75 | 188.42 | 190.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 09:15:00 | 191.50 | 188.42 | 190.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 196.75 | 190.08 | 190.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 196.75 | 190.08 | 190.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 192.70 | 190.61 | 190.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:30:00 | 196.75 | 190.61 | 190.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 192.23 | 190.92 | 191.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:00:00 | 192.23 | 190.92 | 191.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 193.17 | 191.37 | 191.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 198.54 | 193.01 | 192.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-11 09:15:00 | 209.58 | 210.94 | 208.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 209.58 | 210.94 | 208.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 209.58 | 210.94 | 208.18 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 14:15:00 | 200.59 | 206.44 | 206.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 199.71 | 204.16 | 205.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 202.78 | 200.86 | 202.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 202.78 | 200.86 | 202.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 202.78 | 200.86 | 202.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 202.78 | 200.86 | 202.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 202.30 | 201.15 | 202.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:15:00 | 200.67 | 201.56 | 202.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 200.53 | 201.36 | 202.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:15:00 | 200.01 | 201.28 | 202.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 14:15:00 | 199.88 | 198.39 | 198.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 200.00 | 198.71 | 199.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 204.42 | 200.06 | 199.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 204.42 | 200.06 | 199.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 11:15:00 | 207.80 | 202.43 | 200.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 203.45 | 204.06 | 202.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 203.45 | 204.06 | 202.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 203.45 | 204.06 | 202.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 203.45 | 204.06 | 202.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 203.47 | 203.89 | 202.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:30:00 | 203.01 | 203.89 | 202.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 201.83 | 203.48 | 202.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:00:00 | 201.83 | 203.48 | 202.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 201.75 | 203.13 | 202.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 14:15:00 | 201.72 | 203.13 | 202.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 15:15:00 | 200.00 | 202.00 | 202.05 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 10:15:00 | 203.31 | 202.30 | 202.18 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 11:15:00 | 201.72 | 202.25 | 202.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-24 13:15:00 | 201.15 | 201.93 | 202.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 195.50 | 191.85 | 193.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 195.50 | 191.85 | 193.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 195.50 | 191.85 | 193.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 195.50 | 191.85 | 193.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 194.95 | 192.47 | 193.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 199.92 | 192.47 | 193.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 193.35 | 194.07 | 194.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:30:00 | 194.13 | 194.07 | 194.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 196.00 | 193.66 | 193.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 196.00 | 193.66 | 193.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 194.91 | 193.91 | 193.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 193.39 | 193.91 | 193.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 12:15:00 | 195.24 | 193.73 | 193.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 195.24 | 193.73 | 193.61 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 14:15:00 | 192.00 | 193.25 | 193.41 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 201.70 | 194.89 | 194.12 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 192.83 | 196.24 | 196.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 191.61 | 195.31 | 195.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 191.73 | 189.57 | 191.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 14:15:00 | 191.73 | 189.57 | 191.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 191.73 | 189.57 | 191.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 191.73 | 189.57 | 191.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 191.00 | 189.86 | 191.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 192.75 | 189.86 | 191.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 195.07 | 190.90 | 192.03 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 196.45 | 193.00 | 192.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 198.10 | 195.68 | 194.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 196.89 | 197.04 | 195.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 13:45:00 | 197.30 | 197.04 | 195.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 196.96 | 197.14 | 196.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 205.31 | 197.14 | 196.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-24 10:15:00 | 225.84 | 219.96 | 215.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 213.66 | 217.11 | 217.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 211.90 | 215.10 | 216.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 212.97 | 211.82 | 213.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 212.97 | 211.82 | 213.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 212.97 | 211.82 | 213.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:15:00 | 211.03 | 211.97 | 213.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 209.62 | 212.25 | 213.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:15:00 | 211.03 | 212.18 | 212.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:45:00 | 209.90 | 211.68 | 212.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 211.99 | 210.98 | 211.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-02 12:15:00 | 214.90 | 212.30 | 212.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 12:15:00 | 214.90 | 212.30 | 212.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 221.44 | 215.14 | 213.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 10:15:00 | 217.71 | 219.76 | 217.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 10:15:00 | 217.71 | 219.76 | 217.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 217.71 | 219.76 | 217.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:15:00 | 216.99 | 219.76 | 217.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 216.25 | 219.06 | 217.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:30:00 | 215.73 | 219.06 | 217.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 216.33 | 218.52 | 217.39 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 211.13 | 216.50 | 216.63 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 220.46 | 216.84 | 216.42 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 212.72 | 216.33 | 216.71 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 220.63 | 215.76 | 215.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 225.40 | 220.71 | 218.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 221.75 | 222.15 | 220.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:30:00 | 221.86 | 222.15 | 220.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 231.60 | 233.95 | 232.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:45:00 | 231.31 | 233.95 | 232.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 232.77 | 233.71 | 232.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 14:30:00 | 233.82 | 233.61 | 232.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 15:15:00 | 233.80 | 233.61 | 232.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-22 09:15:00 | 257.20 | 243.95 | 241.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 15:15:00 | 246.50 | 248.49 | 248.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 243.82 | 247.55 | 248.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 15:15:00 | 246.26 | 246.07 | 246.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 15:15:00 | 246.26 | 246.07 | 246.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 246.26 | 246.07 | 246.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 252.94 | 246.07 | 246.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 254.50 | 247.75 | 247.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 255.01 | 249.21 | 248.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 250.94 | 252.10 | 250.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 250.94 | 252.10 | 250.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 250.94 | 252.10 | 250.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:00:00 | 257.08 | 252.75 | 251.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 252.90 | 256.90 | 256.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 252.90 | 256.90 | 256.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 13:15:00 | 251.70 | 255.13 | 256.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 253.45 | 253.19 | 254.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 253.45 | 253.19 | 254.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 253.45 | 253.19 | 254.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:45:00 | 253.55 | 253.19 | 254.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 254.00 | 253.07 | 254.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 253.05 | 253.07 | 254.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 253.00 | 253.06 | 253.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:15:00 | 250.60 | 252.81 | 253.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:00:00 | 252.05 | 252.66 | 253.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 252.15 | 253.08 | 253.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 12:45:00 | 252.05 | 252.71 | 253.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 253.15 | 252.80 | 253.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:30:00 | 253.35 | 252.80 | 253.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 251.50 | 252.54 | 253.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:30:00 | 252.35 | 252.54 | 253.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 248.80 | 251.31 | 252.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-10 12:15:00 | 255.80 | 250.87 | 251.18 | SL hit (close>static) qty=1.00 sl=254.75 alert=retest2 |

### Cycle 53 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 253.45 | 251.39 | 251.38 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 249.10 | 252.53 | 252.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 245.85 | 251.19 | 251.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 245.20 | 244.11 | 246.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 245.20 | 244.11 | 246.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 239.80 | 243.54 | 246.03 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 247.00 | 245.09 | 244.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 248.25 | 245.72 | 245.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 250.20 | 251.48 | 249.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 250.20 | 251.48 | 249.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 247.15 | 250.62 | 249.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 247.15 | 250.62 | 249.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 245.15 | 249.52 | 248.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 245.15 | 249.52 | 248.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 246.30 | 248.15 | 248.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 09:15:00 | 244.00 | 246.38 | 247.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 12:15:00 | 249.20 | 246.03 | 246.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 12:15:00 | 249.20 | 246.03 | 246.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 249.20 | 246.03 | 246.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 249.20 | 246.03 | 246.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 246.20 | 246.07 | 246.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 245.00 | 246.07 | 246.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 250.35 | 247.24 | 246.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 250.35 | 247.24 | 246.98 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 15:15:00 | 246.30 | 247.77 | 247.94 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 249.60 | 248.32 | 248.15 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 247.70 | 248.77 | 248.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 246.27 | 248.08 | 248.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 13:15:00 | 248.20 | 248.10 | 248.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 13:15:00 | 248.20 | 248.10 | 248.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 248.20 | 248.10 | 248.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 247.76 | 248.10 | 248.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 249.95 | 248.47 | 248.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 249.95 | 248.47 | 248.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 15:15:00 | 249.43 | 248.66 | 248.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 257.17 | 250.99 | 250.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 15:15:00 | 253.10 | 253.52 | 251.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:30:00 | 253.99 | 253.53 | 252.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 251.15 | 253.05 | 252.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 251.15 | 253.05 | 252.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 250.90 | 252.62 | 251.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:00:00 | 252.42 | 252.58 | 251.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-10 10:15:00 | 277.66 | 268.45 | 262.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 14:15:00 | 282.32 | 285.58 | 285.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 13:15:00 | 279.99 | 282.54 | 283.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 10:15:00 | 283.38 | 281.64 | 282.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 10:15:00 | 283.38 | 281.64 | 282.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 283.38 | 281.64 | 282.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 283.38 | 281.64 | 282.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 281.20 | 281.55 | 282.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 13:00:00 | 279.90 | 281.22 | 282.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:30:00 | 280.78 | 279.90 | 281.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 280.25 | 279.90 | 281.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:45:00 | 280.70 | 277.78 | 279.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 286.13 | 279.45 | 279.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-21 10:15:00 | 286.13 | 279.45 | 279.82 | SL hit (close>static) qty=1.00 sl=283.41 alert=retest2 |

### Cycle 63 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 11:15:00 | 282.58 | 280.07 | 280.07 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 281.01 | 282.67 | 282.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 280.51 | 282.24 | 282.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 296.80 | 278.79 | 279.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 296.80 | 278.79 | 279.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 296.80 | 278.79 | 279.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 296.80 | 278.79 | 279.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 291.54 | 281.34 | 280.52 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 10:15:00 | 279.00 | 284.80 | 285.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 277.31 | 280.12 | 281.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 282.15 | 280.17 | 281.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 282.15 | 280.17 | 281.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 282.15 | 280.17 | 281.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 282.10 | 280.17 | 281.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 283.00 | 280.74 | 281.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:30:00 | 282.80 | 280.74 | 281.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 278.50 | 276.55 | 278.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 279.20 | 276.55 | 278.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 275.65 | 276.37 | 278.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 275.40 | 276.37 | 278.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 275.25 | 276.09 | 277.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 278.50 | 272.77 | 272.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 278.50 | 272.77 | 272.14 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 269.15 | 271.53 | 271.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 267.45 | 270.33 | 270.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 271.45 | 270.56 | 271.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 271.45 | 270.56 | 271.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 271.45 | 270.56 | 271.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 271.95 | 270.56 | 271.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 272.25 | 270.90 | 271.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 272.25 | 270.90 | 271.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 271.60 | 271.04 | 271.17 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 275.00 | 271.96 | 271.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 277.00 | 273.47 | 272.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 283.55 | 283.65 | 279.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:00:00 | 283.55 | 283.65 | 279.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 278.60 | 281.77 | 280.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 279.10 | 281.77 | 280.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 279.20 | 281.26 | 279.99 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 276.70 | 279.09 | 279.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 276.00 | 278.12 | 278.72 | Break + close below crossover candle low |

### Cycle 71 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 284.15 | 279.32 | 279.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 287.30 | 283.70 | 282.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 288.45 | 290.06 | 287.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 288.45 | 290.06 | 287.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 288.45 | 290.06 | 287.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 287.75 | 290.06 | 287.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 287.45 | 289.54 | 287.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 290.35 | 288.56 | 287.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:30:00 | 289.70 | 289.17 | 287.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:45:00 | 289.75 | 298.71 | 295.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 287.10 | 293.36 | 293.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 287.10 | 293.36 | 293.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 284.80 | 290.64 | 292.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 289.85 | 289.58 | 291.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 289.85 | 289.58 | 291.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 289.85 | 289.58 | 291.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:30:00 | 290.70 | 289.58 | 291.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 289.80 | 289.79 | 291.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:30:00 | 289.70 | 289.79 | 291.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 284.25 | 287.54 | 289.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:30:00 | 282.65 | 284.95 | 287.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:15:00 | 282.75 | 283.76 | 286.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 14:15:00 | 287.45 | 286.25 | 286.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 14:15:00 | 287.45 | 286.25 | 286.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 296.55 | 288.37 | 287.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 295.80 | 295.99 | 292.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:30:00 | 295.65 | 295.99 | 292.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 294.85 | 295.50 | 293.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 294.70 | 295.50 | 293.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 294.50 | 294.90 | 293.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 309.15 | 295.22 | 294.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:45:00 | 300.05 | 299.11 | 298.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 307.20 | 311.86 | 312.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 11:15:00 | 307.20 | 311.86 | 312.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 12:15:00 | 303.25 | 310.14 | 311.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 310.50 | 310.21 | 311.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 13:15:00 | 310.50 | 310.21 | 311.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 310.50 | 310.21 | 311.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 310.50 | 310.21 | 311.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 310.65 | 310.30 | 311.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:30:00 | 311.45 | 310.30 | 311.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 311.35 | 310.51 | 311.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 311.40 | 310.51 | 311.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 306.15 | 309.64 | 310.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:15:00 | 304.40 | 307.58 | 309.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:45:00 | 304.60 | 306.51 | 307.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 14:15:00 | 289.18 | 293.85 | 296.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 14:15:00 | 289.37 | 293.85 | 296.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-26 09:15:00 | 273.96 | 280.68 | 285.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 75 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 283.95 | 276.51 | 276.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 11:15:00 | 286.65 | 279.72 | 277.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 12:15:00 | 286.00 | 286.13 | 283.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 282.95 | 285.55 | 283.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 282.95 | 285.55 | 283.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 282.95 | 285.55 | 283.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 281.85 | 284.81 | 283.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 281.85 | 284.81 | 283.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 285.90 | 285.03 | 283.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 282.35 | 285.03 | 283.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 286.20 | 286.59 | 285.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:45:00 | 291.15 | 287.47 | 285.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:15:00 | 289.75 | 287.87 | 286.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 290.10 | 289.43 | 287.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 284.00 | 288.24 | 287.29 | SL hit (close<static) qty=1.00 sl=284.20 alert=retest2 |

### Cycle 76 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 283.80 | 286.34 | 286.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 282.50 | 285.57 | 286.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 286.50 | 283.47 | 284.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 286.50 | 283.47 | 284.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 286.50 | 283.47 | 284.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 286.50 | 283.47 | 284.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 286.00 | 283.97 | 284.53 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 286.50 | 284.90 | 284.88 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 283.75 | 284.75 | 284.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 282.10 | 283.65 | 284.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 13:15:00 | 280.45 | 280.04 | 281.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 14:00:00 | 280.45 | 280.04 | 281.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 280.95 | 280.20 | 281.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 278.00 | 280.20 | 281.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 277.50 | 279.66 | 280.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:15:00 | 275.35 | 278.10 | 279.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:00:00 | 275.30 | 276.55 | 278.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:45:00 | 275.05 | 276.21 | 277.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 285.20 | 277.65 | 277.97 | SL hit (close>static) qty=1.00 sl=281.90 alert=retest2 |

### Cycle 79 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 289.05 | 279.93 | 278.98 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 281.70 | 282.42 | 282.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 281.35 | 282.13 | 282.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 282.20 | 282.14 | 282.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 282.20 | 282.14 | 282.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 282.10 | 282.13 | 282.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 283.65 | 282.44 | 282.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 285.90 | 283.13 | 282.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 287.35 | 284.92 | 283.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 283.60 | 285.08 | 284.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 283.60 | 285.08 | 284.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 283.60 | 285.08 | 284.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 283.15 | 285.08 | 284.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 284.95 | 285.06 | 284.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 284.05 | 285.06 | 284.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 288.95 | 285.84 | 284.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:45:00 | 283.80 | 285.84 | 284.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 282.35 | 286.13 | 285.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 282.35 | 286.13 | 285.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 282.85 | 285.47 | 285.28 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 282.85 | 284.95 | 285.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 281.30 | 283.92 | 284.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 15:15:00 | 269.50 | 269.39 | 272.54 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:15:00 | 263.95 | 269.39 | 272.54 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 12:45:00 | 268.35 | 268.07 | 270.78 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 09:15:00 | 254.93 | 258.53 | 262.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-12 10:15:00 | 255.65 | 254.54 | 257.97 | SL hit (close>ema200) qty=0.50 sl=254.54 alert=retest1 |

### Cycle 83 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 216.81 | 214.09 | 213.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 09:15:00 | 225.98 | 218.18 | 216.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 229.10 | 229.53 | 227.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 229.10 | 229.53 | 227.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 229.10 | 229.53 | 227.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:15:00 | 231.90 | 229.71 | 227.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 231.50 | 230.18 | 228.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:45:00 | 231.50 | 230.30 | 229.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 12:30:00 | 231.20 | 230.57 | 229.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 230.90 | 232.90 | 232.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 230.90 | 232.90 | 232.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 232.75 | 232.87 | 232.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 232.80 | 232.87 | 232.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 230.78 | 232.45 | 232.10 | SL hit (close<static) qty=1.00 sl=230.90 alert=retest2 |

### Cycle 84 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 231.70 | 235.34 | 235.67 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 238.18 | 235.04 | 234.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 238.88 | 237.33 | 236.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 238.35 | 239.32 | 238.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 238.35 | 239.32 | 238.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 238.35 | 239.32 | 238.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 237.27 | 239.32 | 238.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 237.40 | 238.94 | 238.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 237.40 | 238.94 | 238.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 237.76 | 238.70 | 238.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 236.92 | 238.70 | 238.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 237.79 | 238.43 | 238.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 237.79 | 238.43 | 238.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 239.00 | 238.54 | 238.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 238.72 | 238.54 | 238.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 238.43 | 238.52 | 238.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 234.95 | 238.52 | 238.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 235.60 | 237.94 | 237.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 231.68 | 236.69 | 237.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 15:15:00 | 231.75 | 231.16 | 232.84 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 09:15:00 | 229.50 | 231.16 | 232.84 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 225.97 | 226.46 | 228.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:45:00 | 229.19 | 226.46 | 228.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 224.06 | 223.09 | 224.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:30:00 | 224.00 | 223.09 | 224.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 225.03 | 223.48 | 224.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 225.03 | 223.48 | 224.88 | SL hit (close>ema400) qty=1.00 sl=224.88 alert=retest1 |

### Cycle 87 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 208.05 | 205.17 | 204.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 212.64 | 207.23 | 205.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 212.19 | 212.74 | 209.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:45:00 | 212.27 | 212.74 | 209.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 218.00 | 223.04 | 220.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 217.10 | 223.04 | 220.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 216.78 | 221.79 | 220.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 216.78 | 221.79 | 220.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 212.65 | 219.96 | 219.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 212.65 | 219.96 | 219.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 215.31 | 219.03 | 219.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-03 09:15:00 | 209.70 | 216.52 | 217.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 217.20 | 216.08 | 217.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 11:15:00 | 217.20 | 216.08 | 217.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 217.20 | 216.08 | 217.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:00:00 | 217.20 | 216.08 | 217.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 217.00 | 216.27 | 217.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 14:45:00 | 215.86 | 216.99 | 217.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 227.52 | 219.63 | 218.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 10:15:00 | 227.52 | 219.63 | 218.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 229.37 | 224.32 | 221.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 224.41 | 226.32 | 224.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 224.41 | 226.32 | 224.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 224.41 | 226.32 | 224.60 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 221.57 | 223.70 | 223.77 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 227.40 | 224.38 | 224.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 15:15:00 | 228.50 | 226.86 | 225.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 10:15:00 | 232.40 | 232.42 | 229.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 11:00:00 | 232.40 | 232.42 | 229.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 230.12 | 232.05 | 230.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 230.12 | 232.05 | 230.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 231.00 | 231.84 | 230.30 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 227.26 | 229.45 | 229.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 225.58 | 228.68 | 229.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 224.40 | 223.83 | 226.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 11:45:00 | 224.75 | 223.83 | 226.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 224.51 | 224.28 | 225.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 222.11 | 224.32 | 225.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 223.90 | 224.32 | 225.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 13:15:00 | 226.18 | 224.97 | 225.44 | SL hit (close>static) qty=1.00 sl=225.87 alert=retest2 |

### Cycle 93 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 228.00 | 225.88 | 225.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 234.38 | 227.70 | 226.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 10:15:00 | 230.76 | 231.21 | 229.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:45:00 | 230.50 | 231.21 | 229.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 229.69 | 230.92 | 229.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 229.69 | 230.92 | 229.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 229.51 | 230.64 | 229.89 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 226.39 | 229.18 | 229.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 225.50 | 228.53 | 229.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 229.06 | 228.28 | 228.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 229.06 | 228.28 | 228.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 229.06 | 228.28 | 228.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 229.38 | 228.28 | 228.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 226.56 | 227.94 | 228.62 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 15:15:00 | 231.50 | 229.34 | 229.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 233.25 | 230.12 | 229.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 14:15:00 | 230.62 | 231.79 | 230.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 230.62 | 231.79 | 230.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 230.62 | 231.79 | 230.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 230.62 | 231.79 | 230.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 232.29 | 231.89 | 230.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 230.05 | 231.89 | 230.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 231.00 | 231.71 | 230.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 229.37 | 231.71 | 230.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 231.30 | 231.90 | 231.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 231.30 | 231.90 | 231.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 232.01 | 231.92 | 231.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 232.20 | 231.92 | 231.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 232.89 | 232.11 | 231.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 13:15:00 | 233.75 | 232.09 | 231.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:30:00 | 233.59 | 232.67 | 232.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 15:00:00 | 233.99 | 232.67 | 232.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:45:00 | 234.24 | 233.49 | 232.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 234.28 | 234.35 | 233.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:00:00 | 235.16 | 234.53 | 233.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:45:00 | 235.14 | 235.08 | 234.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 231.44 | 234.34 | 233.98 | SL hit (close<static) qty=1.00 sl=231.47 alert=retest2 |

### Cycle 96 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 232.30 | 233.61 | 233.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 223.78 | 231.06 | 232.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 228.45 | 226.67 | 228.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 228.45 | 226.67 | 228.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 228.45 | 226.67 | 228.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 227.88 | 226.67 | 228.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 228.05 | 226.94 | 228.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:45:00 | 228.29 | 226.94 | 228.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 228.76 | 227.31 | 228.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 228.76 | 227.31 | 228.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 230.10 | 227.87 | 228.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 230.09 | 227.87 | 228.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 229.00 | 228.09 | 228.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 230.15 | 228.09 | 228.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 230.72 | 228.62 | 229.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 231.62 | 228.62 | 229.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 231.53 | 229.20 | 229.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 234.71 | 229.20 | 229.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 232.82 | 229.92 | 229.62 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 220.87 | 228.30 | 229.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 11:15:00 | 218.84 | 225.10 | 227.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 223.62 | 221.52 | 224.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:00:00 | 223.62 | 221.52 | 224.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 224.19 | 222.05 | 224.25 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 225.71 | 224.92 | 224.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 231.30 | 226.13 | 225.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 251.14 | 252.05 | 244.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 09:30:00 | 247.71 | 252.05 | 244.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 245.38 | 250.72 | 244.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 246.42 | 250.72 | 244.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 246.60 | 249.90 | 245.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 246.60 | 249.90 | 245.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 243.33 | 248.58 | 244.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 243.33 | 248.58 | 244.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 242.70 | 247.41 | 244.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:00:00 | 242.70 | 247.41 | 244.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 244.21 | 246.52 | 244.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 244.40 | 246.74 | 245.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 246.50 | 247.45 | 246.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 250.79 | 247.45 | 246.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 249.26 | 247.81 | 246.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 256.19 | 249.29 | 248.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 14:15:00 | 245.43 | 252.41 | 252.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 245.43 | 252.41 | 252.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 241.30 | 249.32 | 251.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 242.50 | 241.20 | 244.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 242.50 | 241.20 | 244.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 248.58 | 242.68 | 245.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 248.58 | 242.68 | 245.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 247.03 | 243.55 | 245.23 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 257.30 | 247.74 | 246.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 270.90 | 256.19 | 252.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 10:15:00 | 261.41 | 265.01 | 260.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 11:00:00 | 261.41 | 265.01 | 260.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 261.39 | 264.28 | 260.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:30:00 | 261.09 | 264.28 | 260.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 259.67 | 263.36 | 260.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:00:00 | 259.67 | 263.36 | 260.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 259.20 | 262.53 | 260.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:45:00 | 258.90 | 262.53 | 260.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 259.80 | 262.00 | 260.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 261.55 | 261.84 | 260.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 262.75 | 262.02 | 260.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 261.10 | 262.02 | 260.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 277.00 | 279.08 | 276.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 273.50 | 278.22 | 276.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 273.60 | 277.30 | 276.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 273.60 | 277.30 | 276.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 12:15:00 | 271.85 | 275.52 | 275.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-08 13:15:00 | 271.05 | 274.63 | 275.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 11:15:00 | 276.55 | 274.28 | 274.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 11:15:00 | 276.55 | 274.28 | 274.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 276.55 | 274.28 | 274.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:00:00 | 276.55 | 274.28 | 274.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 274.30 | 274.29 | 274.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:30:00 | 273.75 | 274.10 | 274.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 09:45:00 | 273.60 | 273.77 | 274.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 283.35 | 274.19 | 274.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 09:15:00 | 283.35 | 274.19 | 274.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 290.00 | 284.16 | 280.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 13:15:00 | 285.60 | 285.76 | 282.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:00:00 | 285.60 | 285.76 | 282.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 298.60 | 299.49 | 296.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 301.25 | 297.89 | 296.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:00:00 | 301.25 | 297.89 | 296.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 303.70 | 298.25 | 297.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 303.80 | 306.87 | 306.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 303.80 | 306.87 | 306.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 10:15:00 | 301.30 | 305.32 | 306.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 13:15:00 | 304.30 | 304.16 | 305.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 304.30 | 304.16 | 305.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 303.25 | 303.91 | 304.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 306.95 | 303.91 | 304.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 302.55 | 303.08 | 304.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 304.10 | 303.08 | 304.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 301.40 | 302.75 | 303.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 304.70 | 302.75 | 303.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 300.40 | 302.28 | 303.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 303.00 | 302.28 | 303.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 303.10 | 302.55 | 303.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 304.65 | 302.55 | 303.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 302.00 | 302.44 | 303.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 299.65 | 302.44 | 303.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:45:00 | 298.85 | 301.54 | 302.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:00:00 | 300.40 | 300.52 | 301.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 13:15:00 | 299.65 | 297.82 | 297.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 13:15:00 | 299.65 | 297.82 | 297.76 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 291.05 | 296.73 | 297.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 11:15:00 | 283.60 | 293.00 | 295.44 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-11-26 12:30:00 | 251.65 | 2024-11-26 14:15:00 | 250.95 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-05 09:15:00 | 281.75 | 2024-12-05 14:15:00 | 273.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2024-12-05 09:45:00 | 279.80 | 2024-12-05 14:15:00 | 273.60 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest1 | 2024-12-05 10:15:00 | 279.10 | 2024-12-05 14:15:00 | 273.60 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2024-12-05 11:15:00 | 279.85 | 2024-12-05 14:15:00 | 273.60 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-12-13 09:15:00 | 268.00 | 2024-12-18 09:15:00 | 254.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 268.00 | 2024-12-19 11:15:00 | 241.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-01 11:15:00 | 235.44 | 2025-01-01 12:15:00 | 235.48 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-01-01 12:15:00 | 235.50 | 2025-01-01 12:15:00 | 235.48 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-01-07 15:15:00 | 231.50 | 2025-01-09 09:15:00 | 239.82 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-01-15 13:00:00 | 223.16 | 2025-01-16 11:15:00 | 230.14 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-01-16 10:45:00 | 222.81 | 2025-01-16 11:15:00 | 230.14 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-02-21 11:45:00 | 197.07 | 2025-02-25 14:15:00 | 193.52 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-02-21 12:15:00 | 196.51 | 2025-02-25 14:15:00 | 193.52 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-03-13 13:15:00 | 200.67 | 2025-03-19 09:15:00 | 204.42 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-03-13 14:00:00 | 200.53 | 2025-03-19 09:15:00 | 204.42 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-03-13 15:15:00 | 200.01 | 2025-03-19 09:15:00 | 204.42 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-03-18 14:15:00 | 199.88 | 2025-03-19 09:15:00 | 204.42 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-04-01 11:15:00 | 193.39 | 2025-04-02 12:15:00 | 195.24 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-04-15 09:15:00 | 205.31 | 2025-04-24 10:15:00 | 225.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-29 12:15:00 | 211.03 | 2025-05-02 12:15:00 | 214.90 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-04-30 09:15:00 | 209.62 | 2025-05-02 12:15:00 | 214.90 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-04-30 13:15:00 | 211.03 | 2025-05-02 12:15:00 | 214.90 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-04-30 13:45:00 | 209.90 | 2025-05-02 12:15:00 | 214.90 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-05-16 14:30:00 | 233.82 | 2025-05-22 09:15:00 | 257.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-16 15:15:00 | 233.80 | 2025-05-22 09:15:00 | 257.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-29 15:00:00 | 257.08 | 2025-06-03 11:15:00 | 252.90 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-05 12:15:00 | 250.60 | 2025-06-10 12:15:00 | 255.80 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-05 13:00:00 | 252.05 | 2025-06-10 12:15:00 | 255.80 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-06-06 09:15:00 | 252.15 | 2025-06-10 12:15:00 | 255.80 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-06-06 12:45:00 | 252.05 | 2025-06-10 12:15:00 | 255.80 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-06-23 14:15:00 | 245.00 | 2025-06-24 10:15:00 | 250.35 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-07-08 13:00:00 | 252.42 | 2025-07-10 10:15:00 | 277.66 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-17 13:00:00 | 279.90 | 2025-07-21 10:15:00 | 286.13 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-07-18 09:30:00 | 280.78 | 2025-07-21 10:15:00 | 286.13 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-07-18 10:00:00 | 280.25 | 2025-07-21 10:15:00 | 286.13 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-07-21 09:45:00 | 280.70 | 2025-07-21 10:15:00 | 286.13 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-08-05 10:15:00 | 275.40 | 2025-08-07 15:15:00 | 278.50 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-08-05 10:45:00 | 275.25 | 2025-08-07 15:15:00 | 278.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-08-22 09:45:00 | 290.35 | 2025-08-26 12:15:00 | 287.10 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-08-22 10:30:00 | 289.70 | 2025-08-26 12:15:00 | 287.10 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-26 09:45:00 | 289.75 | 2025-08-26 12:15:00 | 287.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-01 10:30:00 | 282.65 | 2025-09-02 14:15:00 | 287.45 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-09-01 14:15:00 | 282.75 | 2025-09-02 14:15:00 | 287.45 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-09-08 09:15:00 | 309.15 | 2025-09-16 11:15:00 | 307.20 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-09-09 14:45:00 | 300.05 | 2025-09-16 11:15:00 | 307.20 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2025-09-17 14:15:00 | 304.40 | 2025-09-23 14:15:00 | 289.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 12:45:00 | 304.60 | 2025-09-23 14:15:00 | 289.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 14:15:00 | 304.40 | 2025-09-26 09:15:00 | 273.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 12:45:00 | 304.60 | 2025-09-26 09:15:00 | 274.14 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-10-07 10:45:00 | 291.15 | 2025-10-08 10:15:00 | 284.00 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-10-07 12:15:00 | 289.75 | 2025-10-08 10:15:00 | 284.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-10-08 09:15:00 | 290.10 | 2025-10-08 10:15:00 | 284.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-10-17 13:15:00 | 275.35 | 2025-10-21 13:15:00 | 285.20 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2025-10-20 13:00:00 | 275.30 | 2025-10-21 13:15:00 | 285.20 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-10-20 13:45:00 | 275.05 | 2025-10-21 13:15:00 | 285.20 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest1 | 2025-11-07 09:15:00 | 263.95 | 2025-11-11 09:15:00 | 254.93 | PARTIAL | 0.50 | 3.42% |
| SELL | retest1 | 2025-11-07 09:15:00 | 263.95 | 2025-11-12 10:15:00 | 255.65 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest1 | 2025-11-07 12:45:00 | 268.35 | 2025-11-13 11:15:00 | 250.75 | PARTIAL | 0.50 | 6.56% |
| SELL | retest1 | 2025-11-07 12:45:00 | 268.35 | 2025-11-17 09:15:00 | 251.00 | STOP_HIT | 0.50 | 6.47% |
| SELL | retest2 | 2025-11-17 14:15:00 | 251.60 | 2025-11-17 15:15:00 | 252.90 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-18 09:15:00 | 250.20 | 2025-11-21 10:15:00 | 237.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 250.20 | 2025-11-25 09:15:00 | 238.10 | STOP_HIT | 0.50 | 4.84% |
| BUY | retest2 | 2025-12-16 13:15:00 | 231.90 | 2025-12-22 09:15:00 | 230.78 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-12-17 09:45:00 | 231.50 | 2025-12-26 13:15:00 | 231.70 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-17 11:45:00 | 231.50 | 2025-12-26 13:15:00 | 231.70 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-17 12:30:00 | 231.20 | 2025-12-26 13:15:00 | 231.70 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-12-22 09:15:00 | 232.80 | 2025-12-26 13:15:00 | 231.70 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-22 11:45:00 | 233.87 | 2025-12-26 13:15:00 | 231.70 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest1 | 2026-01-08 09:15:00 | 229.50 | 2026-01-13 10:15:00 | 225.03 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest2 | 2026-01-16 10:30:00 | 220.16 | 2026-01-20 13:15:00 | 209.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 10:30:00 | 220.16 | 2026-01-22 14:15:00 | 207.00 | STOP_HIT | 0.50 | 5.98% |
| SELL | retest2 | 2026-02-03 14:45:00 | 215.86 | 2026-02-04 10:15:00 | 227.52 | STOP_HIT | 1.00 | -5.40% |
| SELL | retest2 | 2026-02-16 09:15:00 | 222.11 | 2026-02-16 13:15:00 | 226.18 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-16 10:15:00 | 223.90 | 2026-02-16 13:15:00 | 226.18 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-02-25 13:15:00 | 233.75 | 2026-03-02 09:15:00 | 231.44 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-02-25 14:30:00 | 233.59 | 2026-03-02 09:15:00 | 231.44 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-02-25 15:00:00 | 233.99 | 2026-03-02 11:15:00 | 232.30 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-02-26 10:45:00 | 234.24 | 2026-03-02 11:15:00 | 232.30 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-02-27 14:00:00 | 235.16 | 2026-03-02 11:15:00 | 232.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-02-27 14:45:00 | 235.14 | 2026-03-02 11:15:00 | 232.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-03-19 09:15:00 | 256.19 | 2026-03-20 14:15:00 | 245.43 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2026-04-09 14:30:00 | 273.75 | 2026-04-13 09:15:00 | 283.35 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2026-04-10 09:45:00 | 273.60 | 2026-04-13 09:15:00 | 283.35 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2026-04-22 09:30:00 | 301.25 | 2026-04-28 13:15:00 | 303.80 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2026-04-22 10:00:00 | 301.25 | 2026-04-28 13:15:00 | 303.80 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2026-04-22 11:15:00 | 303.70 | 2026-04-28 13:15:00 | 303.80 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2026-05-04 13:15:00 | 299.65 | 2026-05-07 13:15:00 | 299.65 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2026-05-04 13:45:00 | 298.85 | 2026-05-07 13:15:00 | 299.65 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-05-05 10:00:00 | 300.40 | 2026-05-07 13:15:00 | 299.65 | STOP_HIT | 1.00 | 0.25% |
