# Hindustan Zinc Ltd. (HINDZINC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 634.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 205 |
| ALERT1 | 139 |
| ALERT2 | 139 |
| ALERT2_SKIP | 80 |
| ALERT3 | 352 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 184 |
| PARTIAL | 14 |
| TARGET_HIT | 9 |
| STOP_HIT | 179 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 202 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 71 / 131
- **Target hits / Stop hits / Partials:** 9 / 179 / 14
- **Avg / median % per leg:** 0.18% / -0.60%
- **Sum % (uncompounded):** 35.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 80 | 19 | 23.8% | 6 | 74 | 0 | -0.24% | -19.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.01% | -2.0% |
| BUY @ 3rd Alert (retest2) | 79 | 19 | 24.1% | 6 | 73 | 0 | -0.22% | -17.3% |
| SELL (all) | 122 | 52 | 42.6% | 3 | 105 | 14 | 0.45% | 55.1% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 3 | 1 | 2.82% | 14.1% |
| SELL @ 3rd Alert (retest2) | 117 | 49 | 41.9% | 2 | 102 | 13 | 0.35% | 41.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 4 | 1 | 2.02% | 12.1% |
| retest2 (combined) | 196 | 68 | 34.7% | 8 | 175 | 13 | 0.12% | 23.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 12:15:00 | 312.30 | 311.32 | 311.32 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 11:15:00 | 310.85 | 311.32 | 311.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 12:15:00 | 310.50 | 311.16 | 311.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 309.90 | 309.41 | 310.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 309.90 | 309.41 | 310.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 309.90 | 309.41 | 310.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 14:15:00 | 308.30 | 309.16 | 309.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 11:00:00 | 308.30 | 308.86 | 309.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-29 13:15:00 | 306.90 | 306.47 | 306.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 13:15:00 | 306.90 | 306.47 | 306.43 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 12:15:00 | 306.10 | 306.47 | 306.47 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 15:15:00 | 307.00 | 306.51 | 306.48 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 09:15:00 | 305.50 | 306.31 | 306.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-01 09:15:00 | 305.00 | 305.93 | 306.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 12:15:00 | 305.70 | 305.62 | 305.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-01 12:45:00 | 305.45 | 305.62 | 305.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 14:15:00 | 305.50 | 305.60 | 305.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 14:45:00 | 306.05 | 305.60 | 305.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 306.00 | 305.68 | 305.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 09:15:00 | 307.00 | 305.68 | 305.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 306.30 | 305.80 | 305.91 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 10:15:00 | 306.70 | 305.98 | 305.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 307.95 | 306.69 | 306.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 14:15:00 | 306.60 | 307.08 | 306.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 14:15:00 | 306.60 | 307.08 | 306.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 14:15:00 | 306.60 | 307.08 | 306.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 14:45:00 | 306.50 | 307.08 | 306.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 307.35 | 307.13 | 306.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 09:15:00 | 309.40 | 307.13 | 306.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 10:30:00 | 307.80 | 307.26 | 306.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 09:15:00 | 307.90 | 307.06 | 306.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-07 10:15:00 | 305.90 | 306.75 | 306.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 10:15:00 | 305.90 | 306.75 | 306.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-07 13:15:00 | 305.20 | 306.14 | 306.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-08 15:15:00 | 304.85 | 304.76 | 305.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-09 09:15:00 | 304.20 | 304.76 | 305.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 302.60 | 304.33 | 305.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 11:00:00 | 301.95 | 303.85 | 304.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 12:30:00 | 301.90 | 303.10 | 304.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 09:30:00 | 301.30 | 300.04 | 301.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 14:15:00 | 303.85 | 301.81 | 301.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 14:15:00 | 303.85 | 301.81 | 301.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 15:15:00 | 304.20 | 302.29 | 302.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 12:15:00 | 302.10 | 302.77 | 302.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 12:15:00 | 302.10 | 302.77 | 302.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 12:15:00 | 302.10 | 302.77 | 302.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 12:45:00 | 302.10 | 302.77 | 302.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 13:15:00 | 301.60 | 302.54 | 302.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 14:00:00 | 301.60 | 302.54 | 302.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 14:15:00 | 302.00 | 302.43 | 302.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 14:30:00 | 301.20 | 302.43 | 302.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 15:15:00 | 302.85 | 302.51 | 302.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 09:15:00 | 300.50 | 302.51 | 302.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 301.50 | 302.31 | 302.27 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 10:15:00 | 301.95 | 302.24 | 302.24 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 14:15:00 | 303.90 | 302.48 | 302.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 10:15:00 | 308.50 | 304.23 | 303.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 14:15:00 | 304.35 | 304.93 | 303.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 15:00:00 | 304.35 | 304.93 | 303.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 304.85 | 304.91 | 304.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 09:15:00 | 306.25 | 304.91 | 304.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 12:15:00 | 306.00 | 305.16 | 304.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 09:15:00 | 304.75 | 308.14 | 308.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 09:15:00 | 304.75 | 308.14 | 308.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 10:15:00 | 304.25 | 307.36 | 307.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 306.70 | 306.63 | 307.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 14:00:00 | 306.70 | 306.63 | 307.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 307.50 | 306.80 | 307.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 15:00:00 | 307.50 | 306.80 | 307.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 306.45 | 306.73 | 307.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 306.80 | 306.73 | 307.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 306.95 | 306.78 | 307.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:15:00 | 307.35 | 306.78 | 307.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 305.20 | 306.46 | 307.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 12:30:00 | 305.05 | 306.07 | 306.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 14:45:00 | 305.05 | 305.80 | 306.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 15:15:00 | 305.10 | 305.80 | 306.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-30 10:15:00 | 306.95 | 306.62 | 306.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 306.95 | 306.62 | 306.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 11:15:00 | 307.10 | 306.72 | 306.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 13:15:00 | 306.45 | 306.74 | 306.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 13:15:00 | 306.45 | 306.74 | 306.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 13:15:00 | 306.45 | 306.74 | 306.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 13:45:00 | 306.35 | 306.74 | 306.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 14:15:00 | 307.65 | 306.92 | 306.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 14:30:00 | 306.00 | 306.92 | 306.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 327.50 | 335.78 | 329.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 09:15:00 | 332.80 | 331.10 | 330.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 10:30:00 | 332.45 | 331.45 | 330.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 12:30:00 | 332.25 | 331.70 | 330.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 13:15:00 | 332.35 | 331.70 | 330.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 329.00 | 331.87 | 331.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 329.00 | 331.87 | 331.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 331.50 | 331.80 | 331.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-13 15:15:00 | 329.60 | 331.36 | 331.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 15:15:00 | 329.60 | 331.36 | 331.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 09:15:00 | 327.65 | 330.62 | 331.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 15:15:00 | 328.95 | 328.83 | 329.75 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 09:15:00 | 323.70 | 328.83 | 329.75 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 321.00 | 321.76 | 323.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 11:45:00 | 321.80 | 321.76 | 323.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 14:15:00 | 323.50 | 321.97 | 323.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-19 14:15:00 | 323.50 | 321.97 | 323.13 | SL hit (close>ema400) qty=1.00 sl=323.13 alert=retest1 |

### Cycle 15 — BUY (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 11:15:00 | 319.40 | 318.02 | 317.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 14:15:00 | 320.00 | 318.56 | 318.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 09:15:00 | 318.60 | 318.83 | 318.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 318.60 | 318.83 | 318.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 318.60 | 318.83 | 318.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:30:00 | 318.20 | 318.83 | 318.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 318.35 | 318.73 | 318.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:45:00 | 316.30 | 318.73 | 318.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 11:15:00 | 319.00 | 318.79 | 318.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 11:30:00 | 319.00 | 318.79 | 318.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 320.30 | 319.44 | 318.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:45:00 | 320.35 | 319.44 | 318.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 317.65 | 319.31 | 318.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-31 09:30:00 | 322.60 | 320.26 | 319.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-31 14:45:00 | 321.40 | 320.66 | 320.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 15:00:00 | 322.25 | 322.29 | 321.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-03 09:15:00 | 318.10 | 321.25 | 321.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 318.10 | 321.25 | 321.39 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 13:15:00 | 320.25 | 318.76 | 318.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 15:15:00 | 321.00 | 319.85 | 319.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 12:15:00 | 319.85 | 320.19 | 319.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 12:15:00 | 319.85 | 320.19 | 319.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 12:15:00 | 319.85 | 320.19 | 319.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 12:45:00 | 319.60 | 320.19 | 319.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 319.70 | 320.09 | 319.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 13:30:00 | 319.70 | 320.09 | 319.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 319.55 | 319.98 | 319.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 15:00:00 | 319.55 | 319.98 | 319.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 319.50 | 319.89 | 319.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:15:00 | 317.65 | 319.89 | 319.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 09:15:00 | 316.55 | 319.22 | 319.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 12:15:00 | 316.35 | 317.93 | 318.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 11:15:00 | 316.90 | 316.69 | 317.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-14 11:45:00 | 316.85 | 316.69 | 317.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 13:15:00 | 317.60 | 316.86 | 317.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 14:00:00 | 317.60 | 316.86 | 317.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 316.35 | 316.76 | 317.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 09:15:00 | 315.50 | 316.84 | 317.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 09:45:00 | 315.80 | 315.10 | 316.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 11:15:00 | 315.50 | 314.57 | 314.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 315.50 | 314.57 | 314.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 13:15:00 | 316.60 | 315.13 | 314.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 10:15:00 | 314.70 | 315.43 | 315.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 10:15:00 | 314.70 | 315.43 | 315.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 314.70 | 315.43 | 315.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 11:00:00 | 314.70 | 315.43 | 315.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 11:15:00 | 314.65 | 315.28 | 315.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 12:00:00 | 314.65 | 315.28 | 315.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 315.00 | 315.22 | 315.01 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 15:15:00 | 314.15 | 314.88 | 314.90 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 11:15:00 | 315.25 | 314.96 | 314.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 12:15:00 | 316.35 | 315.24 | 315.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 10:15:00 | 315.10 | 315.72 | 315.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 10:15:00 | 315.10 | 315.72 | 315.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 315.10 | 315.72 | 315.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:30:00 | 315.05 | 315.72 | 315.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 11:15:00 | 316.50 | 315.88 | 315.52 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 15:15:00 | 313.95 | 315.42 | 315.43 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 10:15:00 | 315.80 | 315.48 | 315.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 15:15:00 | 317.10 | 315.95 | 315.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 09:15:00 | 315.40 | 315.84 | 315.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 315.40 | 315.84 | 315.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 315.40 | 315.84 | 315.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:30:00 | 315.90 | 315.84 | 315.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 315.55 | 315.78 | 315.66 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 13:15:00 | 315.15 | 315.56 | 315.58 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 15:15:00 | 315.80 | 315.60 | 315.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 316.20 | 315.72 | 315.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 13:15:00 | 317.65 | 318.51 | 317.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 13:15:00 | 317.65 | 318.51 | 317.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 13:15:00 | 317.65 | 318.51 | 317.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 14:00:00 | 317.65 | 318.51 | 317.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 318.05 | 318.42 | 317.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 09:15:00 | 319.90 | 318.39 | 317.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-07 09:15:00 | 321.60 | 321.92 | 321.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 09:15:00 | 321.60 | 321.92 | 321.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 10:15:00 | 320.70 | 321.68 | 321.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 14:15:00 | 321.15 | 320.96 | 321.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 14:15:00 | 321.15 | 320.96 | 321.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 14:15:00 | 321.15 | 320.96 | 321.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 15:00:00 | 321.15 | 320.96 | 321.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 322.65 | 321.31 | 321.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 09:45:00 | 322.30 | 321.31 | 321.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 10:15:00 | 321.25 | 321.30 | 321.44 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 12:15:00 | 321.80 | 321.56 | 321.55 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 13:15:00 | 321.20 | 321.49 | 321.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 14:15:00 | 320.10 | 321.21 | 321.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 15:15:00 | 316.00 | 314.88 | 316.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 15:15:00 | 316.00 | 314.88 | 316.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 316.00 | 314.88 | 316.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 318.05 | 314.88 | 316.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 317.45 | 315.40 | 316.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 318.50 | 315.40 | 316.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-09-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 11:15:00 | 321.00 | 317.04 | 317.03 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 14:15:00 | 318.55 | 319.32 | 319.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 15:15:00 | 317.50 | 318.96 | 319.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 11:15:00 | 314.00 | 313.68 | 314.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 12:00:00 | 314.00 | 313.68 | 314.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 15:15:00 | 314.00 | 313.60 | 314.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:15:00 | 311.00 | 313.60 | 314.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 311.25 | 313.13 | 314.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 10:15:00 | 310.25 | 313.13 | 314.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 11:00:00 | 310.00 | 312.50 | 313.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-26 13:15:00 | 315.00 | 313.89 | 313.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 13:15:00 | 315.00 | 313.89 | 313.74 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 14:15:00 | 311.35 | 313.38 | 313.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 309.35 | 311.06 | 311.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 11:15:00 | 304.95 | 304.83 | 307.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 12:15:00 | 312.90 | 306.45 | 307.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 12:15:00 | 312.90 | 306.45 | 307.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 13:00:00 | 312.90 | 306.45 | 307.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 310.10 | 307.18 | 308.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 13:45:00 | 311.25 | 307.18 | 308.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 308.40 | 307.49 | 308.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:15:00 | 307.75 | 307.49 | 308.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 305.00 | 306.99 | 307.86 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 15:15:00 | 308.00 | 307.23 | 307.20 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 09:15:00 | 305.70 | 306.93 | 307.06 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 09:15:00 | 311.80 | 306.40 | 306.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 09:15:00 | 315.80 | 311.70 | 309.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 12:15:00 | 315.35 | 316.38 | 313.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 12:15:00 | 315.35 | 316.38 | 313.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 315.35 | 316.38 | 313.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:00:00 | 315.35 | 316.38 | 313.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 318.90 | 317.08 | 315.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 09:30:00 | 324.25 | 319.37 | 318.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 13:00:00 | 322.55 | 320.79 | 319.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 14:30:00 | 322.50 | 321.22 | 319.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 15:00:00 | 322.65 | 321.22 | 319.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 15:15:00 | 320.50 | 321.63 | 320.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 09:30:00 | 322.15 | 321.90 | 320.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 11:15:00 | 317.80 | 320.85 | 320.65 | SL hit (close<static) qty=1.00 sl=320.10 alert=retest2 |

### Cycle 36 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 318.45 | 320.37 | 320.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 314.10 | 318.79 | 319.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 298.25 | 296.37 | 299.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 298.25 | 296.37 | 299.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 298.25 | 296.37 | 299.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 298.25 | 296.37 | 299.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 298.35 | 296.95 | 299.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:30:00 | 297.35 | 296.95 | 299.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 09:15:00 | 294.05 | 293.82 | 295.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 09:45:00 | 292.35 | 294.45 | 295.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 10:45:00 | 292.85 | 293.10 | 293.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 09:15:00 | 299.00 | 294.84 | 294.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 299.00 | 294.84 | 294.44 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 11:15:00 | 295.00 | 296.26 | 296.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 13:15:00 | 292.75 | 295.47 | 295.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 296.60 | 294.97 | 295.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 296.60 | 294.97 | 295.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 296.60 | 294.97 | 295.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 10:00:00 | 296.60 | 294.97 | 295.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 296.30 | 295.24 | 295.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 10:30:00 | 296.50 | 295.24 | 295.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 14:15:00 | 296.80 | 296.01 | 295.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 10:15:00 | 297.55 | 296.89 | 296.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 14:15:00 | 301.55 | 302.11 | 300.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 14:15:00 | 301.55 | 302.11 | 300.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 14:15:00 | 301.55 | 302.11 | 300.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 14:45:00 | 300.75 | 302.11 | 300.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 15:15:00 | 301.65 | 302.01 | 300.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 09:15:00 | 305.05 | 302.01 | 300.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 12:15:00 | 304.10 | 304.86 | 304.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 12:15:00 | 304.10 | 304.86 | 304.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 13:15:00 | 303.75 | 304.64 | 304.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 302.00 | 300.97 | 302.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 302.00 | 300.97 | 302.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 302.00 | 300.97 | 302.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:30:00 | 302.00 | 300.97 | 302.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 13:15:00 | 302.05 | 301.30 | 302.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 13:45:00 | 301.85 | 301.30 | 302.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 14:15:00 | 301.45 | 301.33 | 302.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 14:15:00 | 300.85 | 301.65 | 301.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 09:30:00 | 300.75 | 301.43 | 301.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 11:45:00 | 301.00 | 301.17 | 301.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 09:45:00 | 300.95 | 301.01 | 301.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 300.25 | 300.86 | 301.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 09:15:00 | 299.45 | 300.15 | 300.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 11:15:00 | 299.40 | 300.06 | 300.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 12:00:00 | 299.20 | 299.89 | 300.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 13:45:00 | 299.30 | 299.69 | 300.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 298.75 | 299.43 | 299.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-12-01 14:15:00 | 300.85 | 300.29 | 300.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 14:15:00 | 300.85 | 300.29 | 300.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 314.30 | 303.05 | 301.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 09:15:00 | 323.45 | 324.70 | 321.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-07 10:00:00 | 323.45 | 324.70 | 321.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 322.90 | 323.94 | 322.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:00:00 | 322.90 | 323.94 | 322.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 322.55 | 323.66 | 322.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 12:00:00 | 322.55 | 323.66 | 322.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 321.20 | 323.17 | 322.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 321.20 | 323.17 | 322.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 321.70 | 322.88 | 322.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:45:00 | 320.50 | 322.88 | 322.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 15:15:00 | 322.80 | 322.84 | 322.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 09:15:00 | 324.80 | 322.84 | 322.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-14 09:15:00 | 319.05 | 324.02 | 324.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 09:15:00 | 319.05 | 324.02 | 324.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 09:15:00 | 315.40 | 319.59 | 321.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 311.00 | 307.79 | 309.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 311.00 | 307.79 | 309.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 311.00 | 307.79 | 309.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:30:00 | 309.90 | 307.79 | 309.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 309.95 | 308.22 | 309.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:30:00 | 309.00 | 308.80 | 309.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 13:30:00 | 308.60 | 308.79 | 309.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 09:15:00 | 312.25 | 309.71 | 309.77 | SL hit (close>static) qty=1.00 sl=311.00 alert=retest2 |

### Cycle 43 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 312.15 | 310.20 | 309.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 10:15:00 | 315.45 | 312.39 | 311.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 14:15:00 | 312.05 | 313.38 | 312.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 14:15:00 | 312.05 | 313.38 | 312.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 312.05 | 313.38 | 312.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 15:00:00 | 312.05 | 313.38 | 312.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 313.20 | 313.34 | 312.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 318.00 | 313.34 | 312.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 13:15:00 | 317.30 | 318.01 | 318.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 13:15:00 | 317.30 | 318.01 | 318.02 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 14:15:00 | 318.55 | 318.12 | 318.07 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 315.85 | 317.68 | 317.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 14:15:00 | 314.60 | 315.35 | 316.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 13:15:00 | 313.40 | 313.35 | 314.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-10 14:00:00 | 313.40 | 313.35 | 314.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 318.65 | 314.41 | 314.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 15:00:00 | 318.65 | 314.41 | 314.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 15:15:00 | 316.35 | 314.80 | 315.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:15:00 | 318.25 | 314.80 | 315.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 317.90 | 315.42 | 315.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 11:15:00 | 322.05 | 318.07 | 317.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 318.35 | 320.84 | 319.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 12:15:00 | 318.35 | 320.84 | 319.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 318.35 | 320.84 | 319.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 318.35 | 320.84 | 319.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 320.55 | 320.79 | 319.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:30:00 | 318.05 | 320.79 | 319.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 320.85 | 320.80 | 319.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 14:30:00 | 318.75 | 320.80 | 319.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 316.50 | 319.96 | 319.60 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 10:15:00 | 315.00 | 318.97 | 319.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 309.85 | 315.39 | 317.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 15:15:00 | 313.25 | 313.07 | 314.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 09:15:00 | 315.55 | 313.07 | 314.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 315.95 | 313.65 | 315.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:45:00 | 316.75 | 313.65 | 315.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 317.25 | 314.37 | 315.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 11:00:00 | 317.25 | 314.37 | 315.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 13:15:00 | 317.15 | 315.78 | 315.75 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 15:15:00 | 314.00 | 315.43 | 315.60 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 13:15:00 | 316.00 | 315.69 | 315.65 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 14:15:00 | 315.15 | 315.58 | 315.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 313.25 | 315.02 | 315.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 310.50 | 310.41 | 311.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 14:00:00 | 310.50 | 310.41 | 311.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 311.80 | 310.80 | 311.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 312.60 | 310.80 | 311.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 313.35 | 311.31 | 312.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 13:15:00 | 311.45 | 311.77 | 312.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-25 15:15:00 | 313.50 | 312.48 | 312.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-01-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 15:15:00 | 313.50 | 312.48 | 312.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 313.70 | 312.72 | 312.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 09:15:00 | 317.10 | 317.14 | 315.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-31 09:45:00 | 317.85 | 317.14 | 315.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 317.15 | 317.60 | 316.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 317.15 | 317.60 | 316.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 317.10 | 317.50 | 316.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 319.05 | 317.30 | 316.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 11:45:00 | 317.90 | 317.57 | 317.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 13:30:00 | 319.60 | 317.75 | 317.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-05 09:15:00 | 316.00 | 317.51 | 317.31 | SL hit (close<static) qty=1.00 sl=316.10 alert=retest2 |

### Cycle 54 — SELL (started 2024-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 10:15:00 | 315.40 | 317.09 | 317.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 15:15:00 | 314.60 | 315.98 | 316.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 14:15:00 | 316.00 | 315.55 | 316.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 14:15:00 | 316.00 | 315.55 | 316.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 14:15:00 | 316.00 | 315.55 | 316.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 14:30:00 | 316.95 | 315.55 | 316.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 15:15:00 | 315.75 | 315.59 | 315.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 09:15:00 | 318.00 | 315.59 | 315.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 316.50 | 315.77 | 316.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 10:30:00 | 315.55 | 315.90 | 316.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:15:00 | 316.00 | 315.90 | 316.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 12:30:00 | 315.50 | 315.87 | 316.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 14:30:00 | 315.40 | 315.99 | 316.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 15:15:00 | 315.65 | 315.92 | 316.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:15:00 | 315.60 | 315.92 | 316.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 315.65 | 315.87 | 315.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 10:45:00 | 314.60 | 315.53 | 315.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 11:45:00 | 314.50 | 315.37 | 315.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 09:30:00 | 314.00 | 312.70 | 313.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 09:15:00 | 314.00 | 310.35 | 310.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 314.00 | 310.35 | 310.35 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 15:15:00 | 311.30 | 312.67 | 312.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 12:15:00 | 310.35 | 311.22 | 311.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 13:15:00 | 310.25 | 310.12 | 310.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 13:30:00 | 310.30 | 310.12 | 310.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 310.40 | 310.19 | 310.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 310.40 | 310.19 | 310.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 310.55 | 310.26 | 310.66 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 09:15:00 | 311.95 | 310.81 | 310.76 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 09:15:00 | 308.80 | 310.84 | 310.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 308.00 | 308.99 | 309.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 14:15:00 | 309.30 | 309.02 | 309.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 14:15:00 | 309.30 | 309.02 | 309.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 14:15:00 | 309.30 | 309.02 | 309.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 15:00:00 | 309.30 | 309.02 | 309.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 310.65 | 307.97 | 308.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:45:00 | 311.40 | 307.97 | 308.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 311.00 | 308.58 | 308.70 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 310.45 | 308.95 | 308.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 11:15:00 | 312.45 | 310.21 | 309.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 310.05 | 311.51 | 310.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 09:15:00 | 310.05 | 311.51 | 310.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 310.05 | 311.51 | 310.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 09:45:00 | 310.25 | 311.51 | 310.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 310.40 | 311.29 | 310.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 11:30:00 | 311.00 | 310.73 | 310.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-05 13:15:00 | 310.35 | 310.56 | 310.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 13:15:00 | 310.35 | 310.56 | 310.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 14:15:00 | 307.25 | 309.90 | 310.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 311.15 | 308.57 | 309.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 311.15 | 308.57 | 309.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 311.15 | 308.57 | 309.02 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 311.10 | 309.43 | 309.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 13:15:00 | 311.25 | 310.04 | 309.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 10:15:00 | 309.75 | 310.37 | 309.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 10:15:00 | 309.75 | 310.37 | 309.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 309.75 | 310.37 | 309.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 11:00:00 | 309.75 | 310.37 | 309.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 309.20 | 310.13 | 309.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 11:45:00 | 309.30 | 310.13 | 309.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 12:15:00 | 308.65 | 309.84 | 309.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 13:00:00 | 308.65 | 309.84 | 309.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 13:15:00 | 308.50 | 309.57 | 309.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 308.10 | 309.00 | 309.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 13:15:00 | 308.20 | 307.96 | 308.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-12 13:45:00 | 307.60 | 307.96 | 308.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 15:15:00 | 307.95 | 308.03 | 308.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 09:15:00 | 306.85 | 308.03 | 308.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 14:15:00 | 291.51 | 296.40 | 300.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-18 09:15:00 | 295.50 | 295.01 | 298.74 | SL hit (close>ema200) qty=0.50 sl=295.01 alert=retest2 |

### Cycle 63 — BUY (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 09:15:00 | 297.05 | 294.56 | 294.35 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 294.15 | 295.24 | 295.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 15:15:00 | 293.70 | 294.93 | 295.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 295.35 | 295.02 | 295.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 295.35 | 295.02 | 295.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 295.35 | 295.02 | 295.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 15:00:00 | 292.70 | 294.25 | 294.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 09:15:00 | 298.55 | 294.83 | 294.89 | SL hit (close>static) qty=1.00 sl=296.15 alert=retest2 |

### Cycle 65 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 299.60 | 295.78 | 295.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 300.55 | 296.74 | 295.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 414.35 | 416.57 | 401.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-15 09:30:00 | 408.90 | 416.57 | 401.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 405.85 | 413.99 | 405.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 15:00:00 | 405.85 | 413.99 | 405.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 15:15:00 | 408.60 | 412.92 | 406.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:15:00 | 407.65 | 412.92 | 406.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 414.30 | 413.19 | 406.88 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-04-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 11:15:00 | 404.85 | 406.36 | 406.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 13:15:00 | 397.80 | 404.51 | 405.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 09:15:00 | 411.25 | 404.00 | 404.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 09:15:00 | 411.25 | 404.00 | 404.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 411.25 | 404.00 | 404.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:45:00 | 412.75 | 404.00 | 404.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 10:15:00 | 410.40 | 405.28 | 405.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 10:45:00 | 412.75 | 405.28 | 405.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 11:15:00 | 411.70 | 406.57 | 406.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 09:15:00 | 416.30 | 410.86 | 409.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 11:15:00 | 410.75 | 411.35 | 409.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 11:45:00 | 410.95 | 411.35 | 409.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 412.35 | 411.55 | 410.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 417.70 | 411.62 | 410.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-03 09:15:00 | 459.47 | 437.53 | 431.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 14:15:00 | 444.35 | 449.61 | 450.01 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 09:15:00 | 469.00 | 453.31 | 451.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 13:15:00 | 474.50 | 463.96 | 457.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 10:15:00 | 468.25 | 468.56 | 462.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-09 11:00:00 | 468.25 | 468.56 | 462.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 463.05 | 466.70 | 462.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:00:00 | 463.05 | 466.70 | 462.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 457.10 | 464.78 | 462.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:30:00 | 457.10 | 464.78 | 462.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 458.00 | 463.43 | 462.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:15:00 | 469.85 | 463.43 | 462.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-10 13:15:00 | 516.84 | 485.81 | 474.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 14:15:00 | 705.00 | 723.13 | 724.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 683.50 | 708.10 | 714.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 680.90 | 680.26 | 693.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 11:15:00 | 686.30 | 680.26 | 693.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 690.10 | 683.86 | 691.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:45:00 | 692.15 | 683.86 | 691.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 687.25 | 684.54 | 691.33 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 699.00 | 694.46 | 694.26 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 678.00 | 691.90 | 693.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 642.10 | 681.94 | 688.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 663.00 | 651.32 | 666.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 663.00 | 651.32 | 666.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 663.00 | 651.32 | 666.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 663.00 | 651.32 | 666.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 675.85 | 656.22 | 667.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 676.40 | 656.22 | 667.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 682.00 | 661.38 | 668.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:45:00 | 683.85 | 661.38 | 668.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 698.30 | 674.89 | 673.02 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 11:15:00 | 669.10 | 680.50 | 681.26 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 10:15:00 | 697.00 | 682.71 | 681.16 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 10:15:00 | 674.20 | 685.56 | 686.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 09:15:00 | 669.60 | 676.13 | 680.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 659.80 | 644.40 | 651.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 659.80 | 644.40 | 651.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 659.80 | 644.40 | 651.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 659.80 | 644.40 | 651.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 668.55 | 649.23 | 653.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 668.55 | 649.23 | 653.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 648.05 | 650.54 | 652.66 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 11:15:00 | 676.05 | 658.26 | 655.97 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 11:15:00 | 656.40 | 660.15 | 660.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 13:15:00 | 652.50 | 657.83 | 659.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 11:15:00 | 658.50 | 655.81 | 657.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 11:15:00 | 658.50 | 655.81 | 657.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 658.50 | 655.81 | 657.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 658.50 | 655.81 | 657.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 653.00 | 655.25 | 657.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:45:00 | 650.40 | 654.20 | 656.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 674.90 | 658.34 | 658.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 674.90 | 658.34 | 658.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 15:15:00 | 676.10 | 661.89 | 659.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 09:15:00 | 655.25 | 664.89 | 663.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 655.25 | 664.89 | 663.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 655.25 | 664.89 | 663.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 655.70 | 664.89 | 663.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 654.00 | 662.71 | 662.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 654.00 | 662.71 | 662.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 11:15:00 | 653.80 | 660.93 | 661.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 651.60 | 656.14 | 658.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 12:15:00 | 668.90 | 656.79 | 657.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 12:15:00 | 668.90 | 656.79 | 657.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 668.90 | 656.79 | 657.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:00:00 | 668.90 | 656.79 | 657.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 13:15:00 | 683.45 | 662.12 | 659.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 686.00 | 669.88 | 663.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 12:15:00 | 686.40 | 693.62 | 687.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 12:15:00 | 686.40 | 693.62 | 687.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 686.40 | 693.62 | 687.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:30:00 | 683.50 | 693.62 | 687.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 686.60 | 692.21 | 686.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:15:00 | 686.00 | 692.21 | 686.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 685.20 | 690.81 | 686.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:45:00 | 683.00 | 690.81 | 686.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 695.00 | 691.65 | 687.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:45:00 | 695.75 | 691.43 | 687.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 12:15:00 | 678.85 | 687.16 | 686.64 | SL hit (close<static) qty=1.00 sl=683.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 13:15:00 | 679.30 | 685.59 | 685.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 14:15:00 | 675.00 | 683.47 | 684.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 672.40 | 666.85 | 672.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 672.40 | 666.85 | 672.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 672.40 | 666.85 | 672.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 673.35 | 666.85 | 672.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 673.35 | 668.15 | 672.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 15:15:00 | 669.00 | 670.52 | 672.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 678.40 | 671.85 | 672.95 | SL hit (close>static) qty=1.00 sl=676.85 alert=retest2 |

### Cycle 83 — BUY (started 2024-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 15:15:00 | 640.65 | 623.51 | 621.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 14:15:00 | 648.00 | 638.64 | 634.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 15:15:00 | 643.45 | 645.08 | 640.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 15:15:00 | 643.45 | 645.08 | 640.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 643.45 | 645.08 | 640.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 637.60 | 645.08 | 640.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 637.25 | 643.51 | 640.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:15:00 | 642.05 | 643.51 | 640.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 12:15:00 | 640.40 | 642.38 | 640.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 628.15 | 642.41 | 641.45 | SL hit (close<static) qty=1.00 sl=632.55 alert=retest2 |

### Cycle 84 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 615.00 | 636.93 | 639.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 613.70 | 632.28 | 636.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 624.05 | 616.29 | 622.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 624.05 | 616.29 | 622.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 624.05 | 616.29 | 622.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:00:00 | 624.05 | 616.29 | 622.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 620.00 | 617.03 | 622.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 625.85 | 617.03 | 622.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 602.50 | 598.59 | 604.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 598.85 | 598.59 | 604.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 620.60 | 601.32 | 602.49 | SL hit (close>static) qty=1.00 sl=610.40 alert=retest2 |

### Cycle 85 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 619.00 | 604.86 | 603.99 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 598.95 | 606.01 | 606.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 593.95 | 603.60 | 605.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 611.25 | 588.82 | 595.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 11:15:00 | 611.25 | 588.82 | 595.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 611.25 | 588.82 | 595.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:00:00 | 611.25 | 588.82 | 595.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 573.95 | 585.84 | 593.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 13:15:00 | 570.65 | 585.84 | 593.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 14:30:00 | 570.95 | 581.66 | 590.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 535.00 | 580.53 | 588.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 09:15:00 | 542.12 | 572.00 | 584.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 09:15:00 | 542.40 | 572.00 | 584.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-19 09:15:00 | 513.59 | 528.20 | 552.73 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 87 — BUY (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 15:15:00 | 518.90 | 516.32 | 516.25 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 514.35 | 515.94 | 516.09 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 525.00 | 516.72 | 516.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 14:15:00 | 527.70 | 522.79 | 519.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 511.95 | 527.08 | 525.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 511.95 | 527.08 | 525.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 511.95 | 527.08 | 525.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 512.60 | 527.08 | 525.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 511.80 | 521.70 | 522.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 13:15:00 | 511.50 | 518.07 | 520.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 14:15:00 | 500.70 | 498.78 | 503.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 15:00:00 | 500.70 | 498.78 | 503.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 492.80 | 487.93 | 490.70 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 14:15:00 | 495.00 | 492.30 | 492.07 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 487.80 | 491.95 | 491.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 477.35 | 487.02 | 489.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 487.95 | 483.38 | 485.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 487.95 | 483.38 | 485.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 487.95 | 483.38 | 485.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 489.00 | 483.38 | 485.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 488.45 | 484.40 | 485.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:45:00 | 489.00 | 484.40 | 485.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 487.70 | 485.11 | 485.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 487.70 | 485.11 | 485.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 486.00 | 485.29 | 485.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 483.30 | 485.29 | 485.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 10:30:00 | 483.35 | 480.84 | 482.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 13:15:00 | 487.30 | 483.18 | 483.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 487.30 | 483.18 | 483.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 489.45 | 484.44 | 483.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 496.05 | 496.13 | 492.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 11:45:00 | 495.15 | 496.13 | 492.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 492.35 | 495.59 | 493.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 492.35 | 495.59 | 493.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 490.85 | 494.64 | 493.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 490.85 | 494.64 | 493.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 491.30 | 493.97 | 493.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 13:15:00 | 492.25 | 493.49 | 493.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 14:15:00 | 491.35 | 492.70 | 492.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 14:15:00 | 491.35 | 492.70 | 492.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 487.85 | 491.42 | 492.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 490.50 | 488.02 | 489.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 490.50 | 488.02 | 489.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 490.50 | 488.02 | 489.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:30:00 | 486.50 | 487.09 | 489.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:45:00 | 488.10 | 487.09 | 488.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 13:30:00 | 488.25 | 487.25 | 488.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 496.00 | 489.75 | 489.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 496.00 | 489.75 | 489.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 505.25 | 499.64 | 495.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 12:15:00 | 500.25 | 500.32 | 496.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 13:00:00 | 500.25 | 500.32 | 496.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 506.00 | 508.78 | 506.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 505.75 | 508.78 | 506.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 508.35 | 508.69 | 506.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 506.25 | 508.69 | 506.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 517.95 | 522.33 | 518.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 517.95 | 522.33 | 518.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 523.45 | 522.55 | 519.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 529.40 | 520.12 | 519.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 15:00:00 | 524.00 | 523.78 | 522.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 12:00:00 | 523.80 | 523.32 | 522.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 13:15:00 | 517.55 | 521.23 | 521.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 517.55 | 521.23 | 521.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 507.50 | 517.48 | 519.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 503.20 | 502.66 | 509.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 506.35 | 502.66 | 509.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 512.60 | 505.22 | 508.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 512.60 | 505.22 | 508.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 509.40 | 506.06 | 508.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 514.60 | 506.86 | 508.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 508.35 | 507.16 | 508.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 507.00 | 507.09 | 508.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:30:00 | 507.25 | 507.08 | 508.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 507.05 | 507.08 | 508.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:45:00 | 507.40 | 507.07 | 507.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 508.50 | 506.76 | 507.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:30:00 | 511.00 | 506.76 | 507.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 509.00 | 507.21 | 507.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 10:45:00 | 509.35 | 507.21 | 507.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-10 12:15:00 | 511.00 | 508.55 | 508.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 12:15:00 | 511.00 | 508.55 | 508.26 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 506.40 | 508.31 | 508.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 505.70 | 507.52 | 507.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 508.00 | 506.44 | 507.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 508.00 | 506.44 | 507.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 508.00 | 506.44 | 507.26 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 514.90 | 509.15 | 508.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 14:15:00 | 521.10 | 512.55 | 510.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 12:15:00 | 512.00 | 514.57 | 512.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 12:15:00 | 512.00 | 514.57 | 512.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 512.00 | 514.57 | 512.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 512.00 | 514.57 | 512.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 514.00 | 514.46 | 512.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 517.10 | 514.27 | 512.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 14:15:00 | 510.45 | 513.89 | 513.34 | SL hit (close<static) qty=1.00 sl=511.80 alert=retest2 |

### Cycle 100 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 509.65 | 512.50 | 512.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 504.05 | 509.12 | 510.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 510.00 | 506.67 | 508.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 14:15:00 | 510.00 | 506.67 | 508.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 510.00 | 506.67 | 508.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:15:00 | 499.70 | 506.67 | 508.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:45:00 | 503.15 | 506.53 | 507.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 503.35 | 505.89 | 507.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 10:15:00 | 519.35 | 509.70 | 508.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-22 10:15:00 | 519.35 | 509.70 | 508.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 09:15:00 | 529.60 | 516.04 | 512.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 512.20 | 529.18 | 525.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 512.20 | 529.18 | 525.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 512.20 | 529.18 | 525.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 512.20 | 529.18 | 525.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 508.20 | 524.98 | 523.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 506.65 | 524.98 | 523.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 511.15 | 522.22 | 522.66 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 528.00 | 521.25 | 520.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 538.95 | 526.77 | 523.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 554.15 | 554.25 | 543.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 13:30:00 | 553.75 | 554.25 | 543.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 545.75 | 554.38 | 549.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:00:00 | 545.75 | 554.38 | 549.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 552.70 | 554.04 | 549.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:45:00 | 549.75 | 554.04 | 549.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 557.00 | 554.63 | 550.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 562.05 | 556.12 | 551.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 528.95 | 550.91 | 550.02 | SL hit (close<static) qty=1.00 sl=549.25 alert=retest2 |

### Cycle 104 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 525.75 | 545.87 | 547.82 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 10:15:00 | 563.85 | 549.52 | 547.86 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 09:15:00 | 518.90 | 549.25 | 549.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-06 14:15:00 | 513.35 | 526.37 | 536.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 511.40 | 511.17 | 520.61 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 15:00:00 | 506.20 | 509.33 | 516.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 507.90 | 503.29 | 509.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 15:00:00 | 507.90 | 503.29 | 509.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 507.60 | 504.15 | 508.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 511.45 | 504.15 | 508.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 509.15 | 505.15 | 508.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 509.15 | 505.15 | 508.92 | SL hit (close>ema400) qty=1.00 sl=508.92 alert=retest1 |

### Cycle 107 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 500.00 | 497.76 | 497.52 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 495.90 | 497.17 | 497.28 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 498.30 | 497.40 | 497.37 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 10:15:00 | 497.00 | 497.32 | 497.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 11:15:00 | 495.95 | 497.04 | 497.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 489.00 | 488.21 | 491.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 489.00 | 488.21 | 491.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 489.00 | 488.21 | 491.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 489.00 | 488.21 | 491.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 490.30 | 488.76 | 490.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 499.25 | 488.76 | 490.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 496.50 | 490.31 | 490.70 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 498.85 | 492.02 | 491.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 15:15:00 | 500.80 | 496.43 | 494.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 10:15:00 | 496.50 | 496.59 | 494.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 13:15:00 | 494.45 | 495.93 | 494.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 494.45 | 495.93 | 494.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:00:00 | 494.45 | 495.93 | 494.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 494.15 | 495.58 | 494.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 494.15 | 495.58 | 494.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 494.50 | 495.36 | 494.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 496.75 | 495.36 | 494.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 13:15:00 | 493.95 | 496.05 | 496.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 493.95 | 496.05 | 496.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 09:15:00 | 492.75 | 494.84 | 495.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 12:15:00 | 498.05 | 495.05 | 495.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 12:15:00 | 498.05 | 495.05 | 495.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 498.05 | 495.05 | 495.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:00:00 | 498.05 | 495.05 | 495.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 502.50 | 496.54 | 496.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 15:15:00 | 504.50 | 499.08 | 497.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 11:15:00 | 502.00 | 502.41 | 500.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 12:00:00 | 502.00 | 502.41 | 500.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 502.20 | 502.36 | 500.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 13:30:00 | 503.70 | 502.83 | 501.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 12:15:00 | 501.00 | 503.22 | 503.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 12:15:00 | 501.00 | 503.22 | 503.42 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 514.00 | 504.70 | 503.82 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 502.65 | 505.45 | 505.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 499.20 | 504.20 | 504.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 513.65 | 503.94 | 504.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 513.65 | 503.94 | 504.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 513.65 | 503.94 | 504.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 513.65 | 503.94 | 504.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 512.70 | 505.69 | 505.10 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 504.35 | 505.93 | 506.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 501.85 | 504.60 | 505.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 500.80 | 499.58 | 501.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 500.80 | 499.58 | 501.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 501.00 | 499.87 | 501.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 501.00 | 499.87 | 501.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 499.95 | 499.88 | 501.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 497.70 | 499.88 | 501.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:45:00 | 498.40 | 499.63 | 501.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:45:00 | 498.65 | 499.40 | 501.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:45:00 | 499.00 | 499.36 | 500.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 501.30 | 499.75 | 500.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 501.30 | 499.75 | 500.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 498.05 | 499.41 | 500.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 497.75 | 498.79 | 500.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 472.81 | 485.24 | 490.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 473.48 | 485.24 | 490.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 473.72 | 485.24 | 490.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 474.05 | 485.24 | 490.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 472.86 | 485.24 | 490.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 481.90 | 480.99 | 484.74 | SL hit (close>ema200) qty=0.50 sl=480.99 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 450.60 | 445.57 | 445.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 453.95 | 448.20 | 446.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 460.10 | 463.44 | 457.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 460.10 | 463.44 | 457.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 460.10 | 463.44 | 457.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 460.10 | 463.44 | 457.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 452.55 | 461.26 | 457.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 452.55 | 461.26 | 457.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 453.50 | 459.71 | 456.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 451.75 | 459.71 | 456.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 445.75 | 454.22 | 454.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 443.45 | 449.14 | 451.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 444.60 | 443.11 | 447.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 13:45:00 | 444.80 | 443.11 | 447.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 444.25 | 443.34 | 447.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 441.35 | 443.75 | 446.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 11:15:00 | 449.75 | 443.86 | 446.09 | SL hit (close>static) qty=1.00 sl=447.45 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 15:15:00 | 437.65 | 434.04 | 433.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 09:15:00 | 444.85 | 436.20 | 434.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 462.40 | 462.88 | 458.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 462.40 | 462.88 | 458.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 460.45 | 462.39 | 458.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 460.45 | 462.39 | 458.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 460.50 | 463.51 | 460.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 461.80 | 463.51 | 460.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 461.10 | 463.03 | 460.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 462.60 | 463.03 | 460.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 11:45:00 | 462.80 | 463.54 | 461.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 13:00:00 | 463.05 | 463.45 | 461.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:00:00 | 462.90 | 463.34 | 461.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 469.90 | 464.65 | 462.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:45:00 | 473.00 | 467.71 | 464.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 453.70 | 466.78 | 468.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 453.70 | 466.78 | 468.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 451.20 | 458.75 | 463.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 13:15:00 | 449.50 | 449.10 | 455.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 13:15:00 | 449.50 | 449.10 | 455.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 449.50 | 449.10 | 455.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:45:00 | 449.65 | 449.10 | 455.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 448.20 | 440.55 | 444.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 449.50 | 440.55 | 444.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 446.60 | 441.76 | 444.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 12:00:00 | 443.20 | 442.05 | 444.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 12:30:00 | 442.95 | 442.36 | 444.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 442.00 | 442.36 | 444.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 441.65 | 442.39 | 444.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 441.50 | 442.21 | 443.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 448.95 | 443.56 | 444.34 | SL hit (close>static) qty=1.00 sl=448.30 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 12:15:00 | 447.40 | 445.04 | 444.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 450.10 | 446.42 | 445.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 444.20 | 448.85 | 447.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 444.20 | 448.85 | 447.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 444.20 | 448.85 | 447.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 444.20 | 448.85 | 447.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 449.05 | 448.89 | 447.62 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 435.80 | 445.46 | 446.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 10:15:00 | 431.45 | 442.66 | 444.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 436.35 | 435.29 | 439.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 436.35 | 435.29 | 439.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 436.35 | 435.29 | 439.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:15:00 | 435.40 | 435.29 | 439.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 14:00:00 | 435.95 | 435.29 | 438.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 457.00 | 440.29 | 439.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 457.00 | 440.29 | 439.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 461.70 | 444.58 | 441.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 453.80 | 457.77 | 451.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 09:30:00 | 455.25 | 457.77 | 451.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 448.90 | 455.44 | 451.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:45:00 | 449.65 | 455.44 | 451.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 445.90 | 453.53 | 450.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 445.90 | 453.53 | 450.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 15:15:00 | 443.85 | 448.53 | 448.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 440.30 | 446.88 | 448.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 421.45 | 420.04 | 426.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 423.15 | 420.04 | 426.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 422.50 | 420.89 | 424.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 422.60 | 420.89 | 424.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 415.75 | 415.80 | 419.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:30:00 | 419.85 | 415.80 | 419.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 416.50 | 415.83 | 419.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:45:00 | 420.00 | 415.83 | 419.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 417.45 | 415.83 | 418.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:30:00 | 421.30 | 415.83 | 418.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 415.45 | 415.75 | 418.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:15:00 | 409.75 | 415.75 | 418.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 410.90 | 411.64 | 413.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:30:00 | 413.45 | 409.93 | 411.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 15:15:00 | 413.50 | 411.84 | 411.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 15:15:00 | 413.50 | 411.84 | 411.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 414.90 | 412.45 | 412.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 423.30 | 423.59 | 420.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 423.30 | 423.59 | 420.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 423.30 | 423.59 | 420.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 425.55 | 423.42 | 420.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 416.40 | 421.56 | 420.49 | SL hit (close<static) qty=1.00 sl=418.65 alert=retest2 |

### Cycle 128 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 415.80 | 419.44 | 419.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 411.60 | 417.87 | 418.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 13:15:00 | 410.00 | 409.43 | 412.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 13:15:00 | 410.00 | 409.43 | 412.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 410.00 | 409.43 | 412.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:30:00 | 412.50 | 409.43 | 412.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 411.65 | 409.87 | 412.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 411.65 | 409.87 | 412.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 410.70 | 410.04 | 412.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 401.45 | 410.04 | 412.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 381.38 | 393.06 | 400.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 390.15 | 388.96 | 396.61 | SL hit (close>ema200) qty=0.50 sl=388.96 alert=retest2 |

### Cycle 129 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 403.00 | 396.58 | 395.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 406.25 | 399.30 | 397.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 427.15 | 428.43 | 422.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 430.65 | 427.78 | 424.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 430.65 | 427.78 | 424.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 09:45:00 | 432.50 | 428.84 | 427.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 11:00:00 | 433.35 | 429.74 | 427.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 439.75 | 430.66 | 429.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-19 10:15:00 | 475.75 | 446.36 | 439.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 14:15:00 | 446.50 | 448.38 | 448.48 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 452.55 | 449.03 | 448.75 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 445.25 | 449.24 | 449.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 442.10 | 445.52 | 446.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 449.45 | 444.70 | 445.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 449.45 | 444.70 | 445.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 449.45 | 444.70 | 445.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 449.45 | 444.70 | 445.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 441.90 | 444.14 | 445.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 467.00 | 444.14 | 445.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 467.80 | 448.87 | 447.27 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 435.10 | 454.60 | 456.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 432.90 | 447.40 | 452.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 410.90 | 410.60 | 421.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 410.90 | 410.60 | 421.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 409.40 | 404.19 | 409.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:30:00 | 407.75 | 405.35 | 409.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 426.05 | 413.42 | 412.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 426.05 | 413.42 | 412.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 432.40 | 422.35 | 417.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 11:15:00 | 450.60 | 451.44 | 445.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 12:00:00 | 450.60 | 451.44 | 445.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 445.45 | 449.98 | 446.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 445.45 | 449.98 | 446.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 444.70 | 448.92 | 445.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 448.20 | 448.92 | 445.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 441.50 | 447.44 | 445.58 | SL hit (close<static) qty=1.00 sl=444.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 445.35 | 449.68 | 450.05 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 454.90 | 449.85 | 449.72 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 446.75 | 450.62 | 450.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 445.55 | 448.31 | 449.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 442.25 | 442.02 | 444.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 09:45:00 | 442.50 | 442.02 | 444.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 422.15 | 419.06 | 423.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:30:00 | 423.95 | 419.06 | 423.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 423.00 | 419.85 | 423.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 423.00 | 419.85 | 423.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 422.50 | 420.38 | 423.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:15:00 | 423.85 | 420.38 | 423.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 423.85 | 421.08 | 423.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 419.80 | 421.08 | 423.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 422.00 | 421.26 | 423.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:45:00 | 418.75 | 420.53 | 422.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:45:00 | 418.10 | 419.81 | 422.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 427.50 | 415.19 | 415.40 | SL hit (close>static) qty=1.00 sl=425.65 alert=retest2 |

### Cycle 139 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 426.80 | 417.51 | 416.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 433.30 | 420.67 | 417.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 428.75 | 429.24 | 424.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 428.75 | 429.24 | 424.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 442.65 | 447.72 | 445.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 442.65 | 447.72 | 445.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 439.40 | 446.05 | 444.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 439.40 | 446.05 | 444.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 435.80 | 444.00 | 444.12 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 445.85 | 442.64 | 442.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 446.85 | 443.48 | 442.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 449.65 | 449.69 | 447.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:00:00 | 449.65 | 449.69 | 447.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 457.20 | 459.80 | 457.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:45:00 | 456.25 | 459.80 | 457.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 454.95 | 458.83 | 456.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 454.95 | 458.83 | 456.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 459.45 | 458.95 | 457.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:15:00 | 460.65 | 458.95 | 457.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 451.70 | 455.98 | 456.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 451.70 | 455.98 | 456.38 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 468.95 | 457.27 | 456.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 471.05 | 460.02 | 457.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 457.55 | 464.26 | 461.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 457.55 | 464.26 | 461.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 457.55 | 464.26 | 461.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 456.65 | 464.26 | 461.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 461.00 | 463.60 | 461.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:15:00 | 462.80 | 463.60 | 461.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:30:00 | 463.40 | 463.61 | 461.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 12:15:00 | 509.08 | 492.40 | 480.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 517.20 | 521.88 | 522.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 507.15 | 514.80 | 517.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 15:15:00 | 513.70 | 512.82 | 515.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 09:15:00 | 497.55 | 512.82 | 515.23 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 09:15:00 | 472.67 | 488.63 | 500.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-06-19 09:15:00 | 447.80 | 460.26 | 477.65 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 145 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 448.30 | 444.52 | 444.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 453.00 | 446.22 | 444.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 448.85 | 449.93 | 448.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 13:15:00 | 448.85 | 449.93 | 448.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 448.85 | 449.93 | 448.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 449.25 | 449.93 | 448.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 451.10 | 450.16 | 448.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 452.60 | 450.06 | 449.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 453.05 | 450.05 | 449.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 14:15:00 | 447.10 | 449.19 | 449.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 447.10 | 449.19 | 449.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 446.00 | 448.00 | 448.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 427.20 | 424.44 | 427.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 427.20 | 424.44 | 427.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 427.20 | 424.44 | 427.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 430.15 | 424.44 | 427.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 426.70 | 425.22 | 427.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:45:00 | 427.50 | 425.22 | 427.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 427.00 | 425.57 | 427.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 427.20 | 425.57 | 427.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 434.15 | 427.00 | 427.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 434.15 | 427.00 | 427.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 434.60 | 428.52 | 428.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 435.80 | 430.81 | 429.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 433.55 | 436.27 | 434.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 433.55 | 436.27 | 434.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 433.55 | 436.27 | 434.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 433.55 | 436.27 | 434.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 433.50 | 435.72 | 434.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 433.60 | 435.72 | 434.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 435.60 | 435.69 | 434.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 437.45 | 435.66 | 434.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:15:00 | 437.70 | 435.91 | 435.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 438.10 | 436.35 | 435.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:30:00 | 437.20 | 436.33 | 435.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 433.70 | 437.19 | 436.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 434.90 | 437.19 | 436.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 435.70 | 436.89 | 436.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 438.05 | 436.59 | 436.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 439.60 | 443.70 | 443.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 439.60 | 443.70 | 443.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 438.05 | 441.22 | 442.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 434.80 | 434.74 | 437.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:30:00 | 434.50 | 434.74 | 437.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 438.00 | 435.73 | 437.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 438.00 | 435.73 | 437.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 437.20 | 436.02 | 437.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 435.70 | 436.02 | 437.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 433.50 | 435.52 | 436.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:30:00 | 432.20 | 435.11 | 436.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:00:00 | 432.45 | 434.57 | 436.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:15:00 | 432.35 | 434.26 | 435.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:45:00 | 432.20 | 433.60 | 435.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 420.55 | 420.55 | 422.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 419.00 | 420.58 | 421.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 419.10 | 420.22 | 421.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 10:00:00 | 419.25 | 420.11 | 420.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 12:30:00 | 419.25 | 419.40 | 420.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 424.30 | 420.24 | 420.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 424.30 | 420.24 | 420.49 | SL hit (close>static) qty=1.00 sl=423.60 alert=retest2 |

### Cycle 149 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 425.00 | 421.19 | 420.90 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 416.20 | 420.17 | 420.68 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 423.95 | 420.17 | 419.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 431.55 | 423.84 | 421.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 423.85 | 427.52 | 425.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 423.85 | 427.52 | 425.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 423.85 | 427.52 | 425.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 423.85 | 427.52 | 425.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 423.20 | 426.65 | 425.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 423.20 | 426.65 | 425.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 424.95 | 426.31 | 425.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:30:00 | 422.85 | 426.31 | 425.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 425.80 | 426.21 | 425.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:15:00 | 424.20 | 426.21 | 425.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 425.50 | 426.07 | 425.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:30:00 | 425.35 | 426.07 | 425.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 426.85 | 426.22 | 425.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:45:00 | 427.85 | 426.80 | 425.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 428.15 | 426.65 | 426.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 428.20 | 426.49 | 426.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:00:00 | 428.60 | 426.91 | 426.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 428.05 | 428.13 | 427.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 426.45 | 428.13 | 427.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 427.60 | 428.43 | 427.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:30:00 | 427.35 | 428.43 | 427.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 427.60 | 428.26 | 427.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 427.60 | 428.26 | 427.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 429.75 | 428.56 | 427.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 427.05 | 429.07 | 429.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 427.05 | 429.07 | 429.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 426.35 | 428.53 | 428.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 431.65 | 428.01 | 428.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 431.65 | 428.01 | 428.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 431.65 | 428.01 | 428.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 431.95 | 428.01 | 428.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 435.15 | 429.44 | 429.01 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 424.40 | 430.13 | 430.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 421.70 | 426.03 | 428.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 425.90 | 424.54 | 426.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 425.90 | 424.54 | 426.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 425.90 | 424.54 | 426.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 425.90 | 424.54 | 426.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 438.00 | 425.35 | 426.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 438.00 | 425.35 | 426.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 437.40 | 427.76 | 427.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 441.10 | 435.05 | 431.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 441.95 | 442.03 | 438.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 441.95 | 442.03 | 438.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 440.40 | 442.99 | 440.17 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 436.95 | 439.02 | 439.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 434.65 | 438.15 | 438.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 438.95 | 437.97 | 438.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 438.95 | 437.97 | 438.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 438.95 | 437.97 | 438.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 438.95 | 437.97 | 438.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 440.05 | 438.38 | 438.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 440.05 | 438.38 | 438.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 440.50 | 438.81 | 438.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 439.60 | 438.81 | 438.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 441.35 | 439.31 | 439.15 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 437.60 | 439.00 | 439.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 434.90 | 438.18 | 438.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 435.05 | 434.58 | 436.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 435.05 | 434.58 | 436.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 435.05 | 434.58 | 436.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:30:00 | 432.00 | 433.84 | 435.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 439.25 | 434.60 | 435.13 | SL hit (close>static) qty=1.00 sl=436.70 alert=retest2 |

### Cycle 159 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 439.85 | 435.65 | 435.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 445.95 | 439.79 | 437.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 14:15:00 | 460.55 | 460.74 | 454.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 15:00:00 | 460.55 | 460.74 | 454.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 457.50 | 460.75 | 459.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 457.50 | 460.75 | 459.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 459.65 | 460.53 | 459.15 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 454.10 | 457.88 | 458.19 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 465.95 | 456.24 | 456.07 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 11:15:00 | 454.35 | 457.19 | 457.38 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 459.15 | 457.58 | 457.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 14:15:00 | 461.40 | 458.61 | 458.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 459.20 | 459.27 | 458.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 459.20 | 459.27 | 458.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 459.20 | 459.27 | 458.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:30:00 | 458.55 | 459.27 | 458.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 457.30 | 458.87 | 458.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 457.30 | 458.87 | 458.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 456.25 | 458.35 | 458.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 456.25 | 458.35 | 458.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 456.60 | 458.00 | 458.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 452.70 | 456.49 | 457.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 459.30 | 456.42 | 457.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 459.30 | 456.42 | 457.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 459.30 | 456.42 | 457.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 459.30 | 456.42 | 457.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 457.95 | 456.73 | 457.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 458.25 | 456.73 | 457.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 12:15:00 | 460.25 | 457.70 | 457.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 466.90 | 459.54 | 458.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 454.25 | 460.71 | 459.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 454.25 | 460.71 | 459.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 454.25 | 460.71 | 459.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 454.25 | 460.71 | 459.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 453.95 | 459.35 | 458.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 453.95 | 459.35 | 458.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 454.00 | 458.28 | 458.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 447.10 | 455.25 | 457.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 461.55 | 454.78 | 456.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 461.55 | 454.78 | 456.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 461.55 | 454.78 | 456.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 461.55 | 454.78 | 456.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 461.75 | 456.17 | 456.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 461.85 | 456.17 | 456.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 461.65 | 457.27 | 457.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 13:15:00 | 465.25 | 459.58 | 458.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 490.40 | 490.97 | 486.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:30:00 | 490.95 | 490.97 | 486.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 486.70 | 490.07 | 487.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 486.70 | 490.07 | 487.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 487.80 | 489.62 | 487.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:00:00 | 488.35 | 489.36 | 487.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 505.30 | 507.25 | 507.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 505.30 | 507.25 | 507.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 501.05 | 506.01 | 506.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 12:15:00 | 485.75 | 485.49 | 490.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:00:00 | 485.75 | 485.49 | 490.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 495.10 | 486.34 | 489.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 494.80 | 486.34 | 489.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 490.90 | 487.26 | 489.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 492.00 | 487.26 | 489.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 486.90 | 488.13 | 489.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:45:00 | 485.95 | 487.90 | 489.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:30:00 | 485.50 | 487.83 | 488.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:30:00 | 484.20 | 486.92 | 488.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 15:15:00 | 485.00 | 481.94 | 481.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 485.00 | 481.94 | 481.54 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 478.50 | 481.25 | 481.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 12:15:00 | 476.00 | 479.15 | 480.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 479.00 | 478.39 | 479.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 479.00 | 478.39 | 479.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 479.00 | 478.39 | 479.42 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 480.80 | 479.34 | 479.16 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 475.30 | 478.70 | 478.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 472.90 | 477.03 | 478.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 470.00 | 468.26 | 471.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 470.00 | 468.26 | 471.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 472.50 | 469.10 | 471.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 473.05 | 469.10 | 471.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 473.95 | 470.07 | 471.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 473.95 | 470.07 | 471.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 474.95 | 472.00 | 472.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 479.10 | 472.00 | 472.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 482.40 | 474.08 | 473.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 485.60 | 480.55 | 476.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 485.30 | 485.38 | 482.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:45:00 | 486.30 | 485.38 | 482.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 483.50 | 485.00 | 482.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 483.50 | 485.00 | 482.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 481.90 | 484.38 | 482.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:00:00 | 481.90 | 484.38 | 482.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 483.00 | 484.11 | 482.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:30:00 | 481.95 | 484.11 | 482.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 481.75 | 483.64 | 482.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 481.75 | 483.64 | 482.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 482.25 | 483.36 | 482.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 481.60 | 483.36 | 482.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 481.85 | 483.06 | 482.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 493.50 | 483.06 | 482.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 484.80 | 487.99 | 488.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 484.80 | 487.99 | 488.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 10:15:00 | 482.95 | 486.98 | 487.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 477.60 | 477.00 | 480.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:30:00 | 477.95 | 477.00 | 480.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 475.40 | 476.28 | 478.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:00:00 | 473.30 | 475.35 | 477.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 472.40 | 475.07 | 476.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 467.90 | 463.30 | 463.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 467.90 | 463.30 | 463.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 470.40 | 466.53 | 464.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 495.70 | 495.73 | 488.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 09:45:00 | 495.85 | 495.73 | 488.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 498.05 | 501.95 | 499.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 497.70 | 501.95 | 499.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 495.85 | 500.73 | 498.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 495.85 | 500.73 | 498.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 497.50 | 500.09 | 498.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 495.50 | 500.09 | 498.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 492.95 | 498.01 | 498.04 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 498.30 | 498.06 | 498.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 499.55 | 498.36 | 498.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 497.70 | 498.42 | 498.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 497.70 | 498.42 | 498.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 497.70 | 498.42 | 498.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 497.70 | 498.42 | 498.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 497.95 | 498.33 | 498.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 497.00 | 498.33 | 498.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 15:15:00 | 497.50 | 498.16 | 498.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 496.80 | 497.89 | 498.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 492.20 | 491.88 | 494.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:30:00 | 490.15 | 491.88 | 494.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 493.90 | 492.35 | 493.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 493.50 | 492.35 | 493.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 491.65 | 492.21 | 493.70 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 511.00 | 495.83 | 495.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 13:15:00 | 514.40 | 505.97 | 500.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 582.55 | 586.52 | 579.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 582.55 | 586.52 | 579.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 582.55 | 586.52 | 579.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 580.50 | 586.52 | 579.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 620.05 | 634.94 | 629.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 620.05 | 634.94 | 629.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 619.85 | 631.92 | 628.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:45:00 | 620.20 | 631.92 | 628.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 622.30 | 625.84 | 626.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 09:15:00 | 618.00 | 624.11 | 625.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 613.20 | 612.34 | 616.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 15:00:00 | 613.20 | 612.34 | 616.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 619.50 | 613.47 | 615.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 621.70 | 613.47 | 615.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 619.70 | 614.71 | 616.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 619.70 | 614.71 | 616.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 622.30 | 617.45 | 617.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 624.40 | 618.84 | 618.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 623.80 | 624.88 | 622.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 12:45:00 | 623.90 | 624.88 | 622.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 624.85 | 624.87 | 622.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 624.85 | 624.87 | 622.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 635.45 | 638.51 | 634.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:15:00 | 633.10 | 638.51 | 634.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 633.25 | 637.46 | 634.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 633.25 | 637.46 | 634.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 633.55 | 636.68 | 634.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 632.70 | 636.68 | 634.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 630.50 | 635.44 | 633.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 632.60 | 635.44 | 633.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 628.60 | 634.07 | 633.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 604.30 | 634.07 | 633.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 601.20 | 627.50 | 630.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 590.80 | 620.16 | 626.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 606.95 | 602.06 | 612.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:00:00 | 606.95 | 602.06 | 612.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 620.20 | 608.43 | 611.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:15:00 | 625.10 | 608.43 | 611.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 11:15:00 | 625.15 | 614.26 | 613.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 12:15:00 | 629.95 | 617.39 | 614.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 625.40 | 626.08 | 621.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:00:00 | 625.40 | 626.08 | 621.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 647.05 | 649.66 | 640.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 655.95 | 641.11 | 639.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:45:00 | 654.45 | 643.07 | 640.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:45:00 | 655.30 | 645.88 | 642.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 14:45:00 | 658.75 | 653.22 | 647.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 664.20 | 684.31 | 676.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 664.20 | 684.31 | 676.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 665.25 | 680.50 | 675.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 665.25 | 680.50 | 675.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 677.65 | 678.86 | 675.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 664.40 | 673.93 | 674.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 15:15:00 | 664.40 | 673.93 | 674.04 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 702.05 | 679.55 | 676.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 705.25 | 684.69 | 679.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 09:15:00 | 714.80 | 715.73 | 704.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 10:15:00 | 723.55 | 715.73 | 704.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 709.00 | 718.26 | 710.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 709.00 | 718.26 | 710.76 | SL hit (close<ema400) qty=1.00 sl=710.76 alert=retest1 |

### Cycle 186 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 668.60 | 705.50 | 708.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 11:15:00 | 655.80 | 688.92 | 699.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 591.15 | 588.56 | 622.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 10:00:00 | 591.15 | 588.56 | 622.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 609.10 | 590.39 | 610.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 609.10 | 590.39 | 610.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 613.50 | 595.01 | 610.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 617.40 | 595.01 | 610.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 611.05 | 598.22 | 610.52 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 629.15 | 616.13 | 615.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 639.20 | 622.48 | 618.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 603.00 | 628.51 | 625.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 603.00 | 628.51 | 625.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 603.00 | 628.51 | 625.22 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 606.75 | 620.03 | 621.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 596.70 | 610.67 | 616.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 609.40 | 606.36 | 611.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 609.40 | 606.36 | 611.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 618.40 | 608.83 | 611.64 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 623.15 | 613.94 | 613.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 630.20 | 619.83 | 616.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 618.00 | 620.61 | 617.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 618.00 | 620.61 | 617.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 618.00 | 620.61 | 617.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 618.00 | 620.61 | 617.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 616.00 | 619.69 | 617.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:45:00 | 615.70 | 619.69 | 617.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 618.40 | 619.43 | 617.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:45:00 | 619.20 | 617.91 | 617.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 623.45 | 617.73 | 617.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 597.70 | 619.57 | 621.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 597.70 | 619.57 | 621.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 588.50 | 598.80 | 607.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 594.50 | 594.37 | 601.15 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 09:15:00 | 586.15 | 594.37 | 601.15 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 588.25 | 585.29 | 588.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 588.25 | 585.29 | 588.11 | SL hit (close>ema400) qty=1.00 sl=588.11 alert=retest1 |

### Cycle 191 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 588.45 | 587.65 | 587.60 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 586.70 | 587.46 | 587.52 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 597.45 | 589.46 | 588.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 612.50 | 598.20 | 594.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 13:15:00 | 611.00 | 613.88 | 608.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 14:00:00 | 611.00 | 613.88 | 608.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 610.00 | 612.34 | 608.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 614.90 | 612.34 | 608.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 612.20 | 612.31 | 608.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 09:30:00 | 620.05 | 611.71 | 609.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 11:30:00 | 617.65 | 612.40 | 610.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:30:00 | 619.20 | 614.70 | 612.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 15:00:00 | 617.95 | 614.70 | 612.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 598.15 | 611.76 | 611.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 598.15 | 611.76 | 611.32 | SL hit (close<static) qty=1.00 sl=608.10 alert=retest2 |

### Cycle 194 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 590.15 | 607.44 | 609.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 587.65 | 603.48 | 607.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 601.85 | 597.77 | 602.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 601.85 | 597.77 | 602.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 601.85 | 597.77 | 602.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 601.85 | 597.77 | 602.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 603.65 | 598.95 | 602.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 606.25 | 598.95 | 602.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 595.00 | 598.16 | 601.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 590.00 | 594.81 | 598.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:00:00 | 591.00 | 594.05 | 597.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:30:00 | 591.50 | 593.07 | 596.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 589.45 | 591.90 | 595.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 583.35 | 578.12 | 584.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 593.75 | 587.40 | 586.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 593.75 | 587.40 | 586.65 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 572.30 | 585.60 | 586.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 556.85 | 576.66 | 581.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 543.15 | 539.56 | 550.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 543.15 | 539.56 | 550.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 499.45 | 493.09 | 501.40 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 515.85 | 504.71 | 503.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 518.95 | 507.56 | 505.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 513.35 | 514.08 | 510.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 12:00:00 | 513.35 | 514.08 | 510.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 509.50 | 512.83 | 510.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 509.50 | 512.83 | 510.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 510.10 | 512.28 | 510.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 514.40 | 512.28 | 510.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 510.90 | 512.01 | 510.60 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 503.90 | 508.69 | 509.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 499.65 | 506.88 | 508.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 511.00 | 507.71 | 508.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 15:15:00 | 511.00 | 507.71 | 508.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 511.00 | 507.71 | 508.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 523.40 | 507.71 | 508.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 519.75 | 510.12 | 509.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 530.50 | 518.29 | 513.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 505.30 | 518.55 | 515.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 505.30 | 518.55 | 515.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 505.30 | 518.55 | 515.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 505.30 | 518.55 | 515.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 506.75 | 516.19 | 514.97 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 509.20 | 513.35 | 513.80 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 516.45 | 514.52 | 514.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 523.45 | 516.31 | 515.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 553.95 | 560.13 | 555.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 553.95 | 560.13 | 555.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 553.95 | 560.13 | 555.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 559.50 | 560.18 | 556.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 587.10 | 592.41 | 592.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 587.10 | 592.41 | 592.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 586.10 | 591.15 | 592.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 592.15 | 591.35 | 592.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 13:15:00 | 592.15 | 591.35 | 592.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 592.15 | 591.35 | 592.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 592.15 | 591.35 | 592.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 588.20 | 590.72 | 591.84 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 616.50 | 595.66 | 593.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 621.10 | 600.75 | 596.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 621.05 | 621.42 | 612.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:00:00 | 621.05 | 621.42 | 612.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 617.00 | 618.91 | 613.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 608.80 | 618.91 | 613.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 614.05 | 617.94 | 613.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 11:00:00 | 620.50 | 618.45 | 614.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 585.65 | 610.30 | 612.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 585.65 | 610.30 | 612.24 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 611.80 | 605.85 | 605.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 618.75 | 611.46 | 609.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 636.85 | 636.96 | 628.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 636.85 | 636.96 | 628.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 635.00 | 636.53 | 629.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:15:00 | 640.15 | 636.91 | 630.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 14:15:00 | 308.30 | 2023-05-29 13:15:00 | 306.90 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2023-05-23 11:00:00 | 308.30 | 2023-05-29 13:15:00 | 306.90 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2023-06-06 09:15:00 | 309.40 | 2023-06-07 10:15:00 | 305.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-06-06 10:30:00 | 307.80 | 2023-06-07 10:15:00 | 305.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-06-07 09:15:00 | 307.90 | 2023-06-07 10:15:00 | 305.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-06-09 11:00:00 | 301.95 | 2023-06-13 14:15:00 | 303.85 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2023-06-09 12:30:00 | 301.90 | 2023-06-13 14:15:00 | 303.85 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-06-13 09:30:00 | 301.30 | 2023-06-13 14:15:00 | 303.85 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2023-06-19 09:15:00 | 306.25 | 2023-06-26 09:15:00 | 304.75 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-06-19 12:15:00 | 306.00 | 2023-06-26 09:15:00 | 304.75 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-06-27 12:30:00 | 305.05 | 2023-06-30 10:15:00 | 306.95 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-06-27 14:45:00 | 305.05 | 2023-06-30 10:15:00 | 306.95 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-06-27 15:15:00 | 305.10 | 2023-06-30 10:15:00 | 306.95 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-07-12 09:15:00 | 332.80 | 2023-07-13 15:15:00 | 329.60 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2023-07-12 10:30:00 | 332.45 | 2023-07-13 15:15:00 | 329.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-07-12 12:30:00 | 332.25 | 2023-07-13 15:15:00 | 329.60 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2023-07-12 13:15:00 | 332.35 | 2023-07-13 15:15:00 | 329.60 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest1 | 2023-07-17 09:15:00 | 323.70 | 2023-07-19 14:15:00 | 323.50 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2023-07-21 09:15:00 | 320.55 | 2023-07-26 11:15:00 | 319.40 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2023-07-26 09:30:00 | 319.75 | 2023-07-26 11:15:00 | 319.40 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2023-07-31 09:30:00 | 322.60 | 2023-08-03 09:15:00 | 318.10 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-07-31 14:45:00 | 321.40 | 2023-08-03 09:15:00 | 318.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-08-02 15:00:00 | 322.25 | 2023-08-03 09:15:00 | 318.10 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-08-16 09:15:00 | 315.50 | 2023-08-22 11:15:00 | 315.50 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-08-17 09:45:00 | 315.80 | 2023-08-22 11:15:00 | 315.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2023-09-01 09:15:00 | 319.90 | 2023-09-07 09:15:00 | 321.60 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2023-09-25 10:15:00 | 310.25 | 2023-09-26 13:15:00 | 315.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2023-09-25 11:00:00 | 310.00 | 2023-09-26 13:15:00 | 315.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2023-10-16 09:30:00 | 324.25 | 2023-10-18 11:15:00 | 317.80 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2023-10-16 13:00:00 | 322.55 | 2023-10-18 12:15:00 | 318.45 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-10-16 14:30:00 | 322.50 | 2023-10-18 12:15:00 | 318.45 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-10-16 15:00:00 | 322.65 | 2023-10-18 12:15:00 | 318.45 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-10-18 09:30:00 | 322.15 | 2023-10-18 12:15:00 | 318.45 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2023-11-01 09:45:00 | 292.35 | 2023-11-03 09:15:00 | 299.00 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2023-11-02 10:45:00 | 292.85 | 2023-11-03 09:15:00 | 299.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2023-11-15 09:15:00 | 305.05 | 2023-11-21 12:15:00 | 304.10 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2023-11-24 14:15:00 | 300.85 | 2023-12-01 14:15:00 | 300.85 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-11-28 09:30:00 | 300.75 | 2023-12-01 14:15:00 | 300.85 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2023-11-28 11:45:00 | 301.00 | 2023-12-01 14:15:00 | 300.85 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2023-11-29 09:45:00 | 300.95 | 2023-12-01 14:15:00 | 300.85 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2023-11-30 09:15:00 | 299.45 | 2023-12-01 14:15:00 | 300.85 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-11-30 11:15:00 | 299.40 | 2023-12-01 14:15:00 | 300.85 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-11-30 12:00:00 | 299.20 | 2023-12-01 14:15:00 | 300.85 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-11-30 13:45:00 | 299.30 | 2023-12-01 14:15:00 | 300.85 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-12-11 09:15:00 | 324.80 | 2023-12-14 09:15:00 | 319.05 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2023-12-22 12:30:00 | 309.00 | 2023-12-26 09:15:00 | 312.25 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2023-12-22 13:30:00 | 308.60 | 2023-12-26 09:15:00 | 312.25 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-12-29 09:15:00 | 318.00 | 2024-01-05 13:15:00 | 317.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-01-25 13:15:00 | 311.45 | 2024-01-25 15:15:00 | 313.50 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-02-02 09:15:00 | 319.05 | 2024-02-05 09:15:00 | 316.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-02-02 11:45:00 | 317.90 | 2024-02-05 09:15:00 | 316.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-02-02 13:30:00 | 319.60 | 2024-02-05 09:15:00 | 316.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-02-07 10:30:00 | 315.55 | 2024-02-15 09:15:00 | 314.00 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-02-07 11:15:00 | 316.00 | 2024-02-15 09:15:00 | 314.00 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2024-02-07 12:30:00 | 315.50 | 2024-02-15 09:15:00 | 314.00 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-02-07 14:30:00 | 315.40 | 2024-02-15 09:15:00 | 314.00 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-02-08 10:45:00 | 314.60 | 2024-02-15 09:15:00 | 314.00 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2024-02-08 11:45:00 | 314.50 | 2024-02-15 09:15:00 | 314.00 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2024-02-12 09:30:00 | 314.00 | 2024-02-15 09:15:00 | 314.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-03-05 11:30:00 | 311.00 | 2024-03-05 13:15:00 | 310.35 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-03-13 09:15:00 | 306.85 | 2024-03-15 14:15:00 | 291.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-13 09:15:00 | 306.85 | 2024-03-18 09:15:00 | 295.50 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2024-03-28 15:00:00 | 292.70 | 2024-04-01 09:15:00 | 298.55 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-04-26 09:15:00 | 417.70 | 2024-05-03 09:15:00 | 459.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-10 09:15:00 | 469.85 | 2024-05-10 13:15:00 | 516.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-27 13:45:00 | 650.40 | 2024-06-27 14:15:00 | 674.90 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2024-07-09 09:45:00 | 695.75 | 2024-07-09 12:15:00 | 678.85 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-07-11 15:15:00 | 669.00 | 2024-07-12 09:15:00 | 678.40 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-07-12 14:45:00 | 669.85 | 2024-07-19 10:15:00 | 636.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 14:45:00 | 669.85 | 2024-07-19 13:15:00 | 648.95 | STOP_HIT | 0.50 | 3.12% |
| BUY | retest2 | 2024-08-02 10:15:00 | 642.05 | 2024-08-05 09:15:00 | 628.15 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-08-02 12:15:00 | 640.40 | 2024-08-05 09:15:00 | 628.15 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-08-09 10:15:00 | 598.85 | 2024-08-12 09:15:00 | 620.60 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2024-08-14 13:15:00 | 570.65 | 2024-08-16 09:15:00 | 542.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-14 14:30:00 | 570.95 | 2024-08-16 09:15:00 | 542.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-14 13:15:00 | 570.65 | 2024-08-19 09:15:00 | 513.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-14 14:30:00 | 570.95 | 2024-08-19 09:15:00 | 513.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-16 09:15:00 | 535.00 | 2024-08-19 09:15:00 | 508.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-16 09:15:00 | 535.00 | 2024-08-20 11:15:00 | 503.20 | STOP_HIT | 0.50 | 5.94% |
| SELL | retest2 | 2024-09-11 09:15:00 | 483.30 | 2024-09-12 13:15:00 | 487.30 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-09-12 10:30:00 | 483.35 | 2024-09-12 13:15:00 | 487.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-09-17 13:15:00 | 492.25 | 2024-09-17 14:15:00 | 491.35 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-09-19 10:30:00 | 486.50 | 2024-09-20 09:15:00 | 496.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-09-19 12:45:00 | 488.10 | 2024-09-20 09:15:00 | 496.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-09-19 13:30:00 | 488.25 | 2024-09-20 09:15:00 | 496.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-10-03 09:15:00 | 529.40 | 2024-10-04 13:15:00 | 517.55 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-10-03 15:00:00 | 524.00 | 2024-10-04 13:15:00 | 517.55 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-10-04 12:00:00 | 523.80 | 2024-10-04 13:15:00 | 517.55 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-10-09 11:45:00 | 507.00 | 2024-10-10 12:15:00 | 511.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-10-09 12:30:00 | 507.25 | 2024-10-10 12:15:00 | 511.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-10-09 13:00:00 | 507.05 | 2024-10-10 12:15:00 | 511.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-10-09 13:45:00 | 507.40 | 2024-10-10 12:15:00 | 511.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-10-16 09:15:00 | 517.10 | 2024-10-16 14:15:00 | 510.45 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-10-17 09:15:00 | 516.00 | 2024-10-17 09:15:00 | 509.65 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-10-18 15:15:00 | 499.70 | 2024-10-22 10:15:00 | 519.35 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2024-10-21 12:45:00 | 503.15 | 2024-10-22 10:15:00 | 519.35 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2024-10-21 14:00:00 | 503.35 | 2024-10-22 10:15:00 | 519.35 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2024-11-01 18:00:00 | 562.05 | 2024-11-04 09:15:00 | 528.95 | STOP_HIT | 1.00 | -5.89% |
| SELL | retest1 | 2024-11-08 15:00:00 | 506.20 | 2024-11-12 09:15:00 | 509.15 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-11-12 11:15:00 | 506.70 | 2024-11-18 13:15:00 | 500.00 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2024-11-12 12:00:00 | 506.45 | 2024-11-18 13:15:00 | 500.00 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2024-11-27 09:15:00 | 496.75 | 2024-11-28 13:15:00 | 493.95 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-12-03 13:30:00 | 503.70 | 2024-12-05 12:15:00 | 501.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-12-16 10:15:00 | 497.70 | 2024-12-19 09:15:00 | 472.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 498.40 | 2024-12-19 09:15:00 | 473.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 11:45:00 | 498.65 | 2024-12-19 09:15:00 | 473.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 12:45:00 | 499.00 | 2024-12-19 09:15:00 | 474.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:30:00 | 497.75 | 2024-12-19 09:15:00 | 472.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 497.70 | 2024-12-20 10:15:00 | 481.90 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2024-12-16 10:45:00 | 498.40 | 2024-12-20 10:15:00 | 481.90 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2024-12-16 11:45:00 | 498.65 | 2024-12-20 10:15:00 | 481.90 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2024-12-16 12:45:00 | 499.00 | 2024-12-20 10:15:00 | 481.90 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2024-12-17 09:30:00 | 497.75 | 2024-12-20 10:15:00 | 481.90 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-01-09 09:15:00 | 441.35 | 2025-01-09 11:15:00 | 449.75 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-01-09 12:30:00 | 439.25 | 2025-01-13 14:15:00 | 417.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 13:30:00 | 438.05 | 2025-01-13 14:15:00 | 416.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:30:00 | 439.25 | 2025-01-14 09:15:00 | 433.55 | STOP_HIT | 0.50 | 1.30% |
| SELL | retest2 | 2025-01-09 13:30:00 | 438.05 | 2025-01-14 09:15:00 | 433.55 | STOP_HIT | 0.50 | 1.03% |
| BUY | retest2 | 2025-01-22 09:15:00 | 462.60 | 2025-01-27 09:15:00 | 453.70 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-01-22 11:45:00 | 462.80 | 2025-01-27 09:15:00 | 453.70 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-01-22 13:00:00 | 463.05 | 2025-01-27 09:15:00 | 453.70 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-01-22 14:00:00 | 462.90 | 2025-01-27 09:15:00 | 453.70 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-01-23 09:45:00 | 473.00 | 2025-01-27 09:15:00 | 453.70 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2025-01-30 12:00:00 | 443.20 | 2025-01-31 10:15:00 | 448.95 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-01-30 12:30:00 | 442.95 | 2025-01-31 10:15:00 | 448.95 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-01-30 13:15:00 | 442.00 | 2025-01-31 10:15:00 | 448.95 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-01-31 09:15:00 | 441.65 | 2025-01-31 10:15:00 | 448.95 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-02-04 10:15:00 | 435.40 | 2025-02-05 09:15:00 | 457.00 | STOP_HIT | 1.00 | -4.96% |
| SELL | retest2 | 2025-02-04 14:00:00 | 435.95 | 2025-02-05 09:15:00 | 457.00 | STOP_HIT | 1.00 | -4.83% |
| SELL | retest2 | 2025-02-17 09:15:00 | 409.75 | 2025-02-19 15:15:00 | 413.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-02-18 09:15:00 | 410.90 | 2025-02-19 15:15:00 | 413.50 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-02-19 10:30:00 | 413.45 | 2025-02-19 15:15:00 | 413.50 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-02-24 11:30:00 | 425.55 | 2025-02-24 13:15:00 | 416.40 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-02-28 09:15:00 | 401.45 | 2025-03-03 09:15:00 | 381.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 401.45 | 2025-03-03 12:15:00 | 390.15 | STOP_HIT | 0.50 | 2.81% |
| BUY | retest2 | 2025-03-13 09:45:00 | 432.50 | 2025-03-19 10:15:00 | 475.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 11:00:00 | 433.35 | 2025-03-19 10:15:00 | 476.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 09:15:00 | 439.75 | 2025-03-21 14:15:00 | 446.50 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2025-04-11 10:30:00 | 407.75 | 2025-04-15 09:15:00 | 426.05 | STOP_HIT | 1.00 | -4.49% |
| BUY | retest2 | 2025-04-23 09:15:00 | 448.20 | 2025-04-23 09:15:00 | 441.50 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-04-23 14:15:00 | 447.20 | 2025-04-25 10:15:00 | 442.65 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-04-23 14:45:00 | 447.40 | 2025-04-25 10:15:00 | 442.65 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-05-08 11:45:00 | 418.75 | 2025-05-12 10:15:00 | 427.50 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-05-08 12:45:00 | 418.10 | 2025-05-12 10:15:00 | 427.50 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-05-30 13:15:00 | 460.65 | 2025-06-02 11:15:00 | 451.70 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-06-04 11:15:00 | 462.80 | 2025-06-06 12:15:00 | 509.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 12:30:00 | 463.40 | 2025-06-06 12:15:00 | 509.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-06-17 09:15:00 | 497.55 | 2025-06-18 09:15:00 | 472.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-06-17 09:15:00 | 497.55 | 2025-06-19 09:15:00 | 447.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-24 13:45:00 | 441.75 | 2025-06-26 14:15:00 | 449.45 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-06-25 11:30:00 | 441.95 | 2025-06-26 14:15:00 | 449.45 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-06-25 12:00:00 | 441.45 | 2025-06-26 14:15:00 | 449.45 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-06-25 13:15:00 | 441.80 | 2025-06-26 14:15:00 | 449.45 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-06-26 10:15:00 | 440.90 | 2025-06-26 14:15:00 | 449.45 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-07-02 09:15:00 | 452.60 | 2025-07-02 14:15:00 | 447.10 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-02 11:45:00 | 453.05 | 2025-07-02 14:15:00 | 447.10 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-07-17 09:15:00 | 437.45 | 2025-07-25 10:15:00 | 439.60 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-07-17 10:15:00 | 437.70 | 2025-07-25 10:15:00 | 439.60 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-07-17 11:30:00 | 438.10 | 2025-07-25 10:15:00 | 439.60 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-07-17 13:30:00 | 437.20 | 2025-07-25 10:15:00 | 439.60 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-07-21 09:15:00 | 438.05 | 2025-07-25 10:15:00 | 439.60 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-07-30 10:30:00 | 432.20 | 2025-08-07 14:15:00 | 424.30 | STOP_HIT | 1.00 | 1.83% |
| SELL | retest2 | 2025-07-30 12:00:00 | 432.45 | 2025-08-07 14:15:00 | 424.30 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2025-07-30 13:15:00 | 432.35 | 2025-08-07 14:15:00 | 424.30 | STOP_HIT | 1.00 | 1.86% |
| SELL | retest2 | 2025-07-30 14:45:00 | 432.20 | 2025-08-07 14:15:00 | 424.30 | STOP_HIT | 1.00 | 1.83% |
| SELL | retest2 | 2025-08-06 10:00:00 | 419.00 | 2025-08-07 15:15:00 | 425.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-08-06 10:30:00 | 419.10 | 2025-08-07 15:15:00 | 425.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-08-07 10:00:00 | 419.25 | 2025-08-07 15:15:00 | 425.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-08-07 12:30:00 | 419.25 | 2025-08-07 15:15:00 | 425.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-08-18 09:45:00 | 427.85 | 2025-08-22 11:15:00 | 427.05 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-08-18 14:45:00 | 428.15 | 2025-08-22 11:15:00 | 427.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-08-19 09:15:00 | 428.20 | 2025-08-22 11:15:00 | 427.05 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-08-19 10:00:00 | 428.60 | 2025-08-22 11:15:00 | 427.05 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-09-10 12:30:00 | 432.00 | 2025-09-11 09:15:00 | 439.25 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-10-07 13:00:00 | 488.35 | 2025-10-17 11:15:00 | 505.30 | STOP_HIT | 1.00 | 3.47% |
| SELL | retest2 | 2025-10-24 14:45:00 | 485.95 | 2025-10-29 15:15:00 | 485.00 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-10-27 09:30:00 | 485.50 | 2025-10-29 15:15:00 | 485.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-10-27 10:30:00 | 484.20 | 2025-10-29 15:15:00 | 485.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-11-13 09:15:00 | 493.50 | 2025-11-17 09:15:00 | 484.80 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-11-20 13:00:00 | 473.30 | 2025-11-26 10:15:00 | 467.90 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2025-11-20 14:15:00 | 472.40 | 2025-11-26 10:15:00 | 467.90 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2026-01-19 09:15:00 | 655.95 | 2026-01-22 15:15:00 | 664.40 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2026-01-19 09:45:00 | 654.45 | 2026-01-22 15:15:00 | 664.40 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2026-01-19 10:45:00 | 655.30 | 2026-01-22 15:15:00 | 664.40 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2026-01-19 14:45:00 | 658.75 | 2026-01-22 15:15:00 | 664.40 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest1 | 2026-01-28 10:15:00 | 723.55 | 2026-01-28 14:15:00 | 709.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-02-10 14:45:00 | 619.20 | 2026-02-13 09:15:00 | 597.70 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2026-02-11 09:15:00 | 623.45 | 2026-02-13 09:15:00 | 597.70 | STOP_HIT | 1.00 | -4.13% |
| SELL | retest1 | 2026-02-17 09:15:00 | 586.15 | 2026-02-19 09:15:00 | 588.25 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-02-19 10:15:00 | 586.00 | 2026-02-20 14:15:00 | 588.45 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-02-19 14:15:00 | 587.45 | 2026-02-20 14:15:00 | 588.45 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2026-02-20 13:00:00 | 587.30 | 2026-02-20 14:15:00 | 588.45 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-03-02 09:30:00 | 620.05 | 2026-03-04 09:15:00 | 598.15 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2026-03-02 11:30:00 | 617.65 | 2026-03-04 09:15:00 | 598.15 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2026-03-02 14:30:00 | 619.20 | 2026-03-04 09:15:00 | 598.15 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-03-02 15:00:00 | 617.95 | 2026-03-04 09:15:00 | 598.15 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2026-03-06 10:45:00 | 590.00 | 2026-03-11 09:15:00 | 593.75 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-03-06 12:00:00 | 591.00 | 2026-03-11 09:15:00 | 593.75 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-03-06 13:30:00 | 591.50 | 2026-03-11 09:15:00 | 593.75 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-03-06 14:30:00 | 589.45 | 2026-03-11 09:15:00 | 593.75 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-04-13 10:45:00 | 559.50 | 2026-04-24 11:15:00 | 587.10 | STOP_HIT | 1.00 | 4.93% |
| BUY | retest2 | 2026-04-29 11:00:00 | 620.50 | 2026-04-30 09:15:00 | 585.65 | STOP_HIT | 1.00 | -5.62% |
