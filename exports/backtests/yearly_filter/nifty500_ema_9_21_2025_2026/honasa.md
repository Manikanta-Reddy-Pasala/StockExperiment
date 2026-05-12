# Honasa Consumer Ltd. (HONASA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 354.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 43 |
| ALERT2 | 44 |
| ALERT2_SKIP | 26 |
| ALERT3 | 119 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 23
- **Target hits / Stop hits / Partials:** 4 / 41 / 11
- **Avg / median % per leg:** 2.24% / 2.66%
- **Sum % (uncompounded):** 125.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 10 | 58.8% | 3 | 14 | 0 | 2.11% | 35.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 10 | 58.8% | 3 | 14 | 0 | 2.11% | 35.9% |
| SELL (all) | 39 | 23 | 59.0% | 1 | 27 | 11 | 2.30% | 89.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 39 | 23 | 59.0% | 1 | 27 | 11 | 2.30% | 89.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 56 | 33 | 58.9% | 4 | 41 | 11 | 2.24% | 125.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 251.60 | 248.03 | 247.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 255.46 | 249.51 | 248.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 255.10 | 256.45 | 254.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:15:00 | 258.95 | 256.45 | 254.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 258.75 | 256.91 | 254.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 261.00 | 256.16 | 255.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-23 09:15:00 | 287.10 | 280.35 | 274.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 312.35 | 313.72 | 313.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 311.75 | 313.32 | 313.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 312.10 | 311.99 | 312.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 312.10 | 311.99 | 312.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 312.10 | 311.99 | 312.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 312.10 | 311.99 | 312.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 312.00 | 311.99 | 312.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 316.20 | 311.99 | 312.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 316.80 | 312.95 | 313.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:15:00 | 323.80 | 312.95 | 313.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 318.10 | 313.98 | 313.61 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 313.75 | 316.26 | 316.27 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 318.95 | 316.80 | 316.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 320.05 | 317.45 | 316.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 320.80 | 321.50 | 320.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 10:15:00 | 320.80 | 321.50 | 320.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 320.80 | 321.50 | 320.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 320.80 | 321.50 | 320.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 320.00 | 321.20 | 320.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 320.00 | 321.20 | 320.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 316.15 | 320.19 | 319.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:45:00 | 316.15 | 320.19 | 319.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 318.00 | 319.75 | 319.58 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 318.20 | 319.29 | 319.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 316.40 | 318.61 | 319.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 308.35 | 308.00 | 310.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:30:00 | 308.20 | 308.00 | 310.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 310.25 | 308.45 | 310.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 310.20 | 308.45 | 310.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 310.00 | 308.76 | 310.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 307.25 | 309.45 | 310.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 306.70 | 309.12 | 310.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:45:00 | 308.70 | 306.82 | 306.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 12:15:00 | 308.20 | 307.10 | 306.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 308.20 | 307.10 | 306.98 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 13:15:00 | 306.00 | 306.88 | 306.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 297.45 | 304.99 | 306.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 14:15:00 | 304.45 | 303.05 | 304.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 14:15:00 | 304.45 | 303.05 | 304.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 304.45 | 303.05 | 304.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:45:00 | 304.65 | 303.05 | 304.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 303.15 | 303.07 | 304.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 306.25 | 303.07 | 304.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 304.70 | 303.40 | 304.15 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 311.60 | 305.81 | 305.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 15:15:00 | 313.00 | 308.22 | 306.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 308.50 | 308.76 | 307.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 11:00:00 | 308.50 | 308.76 | 307.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 307.75 | 310.12 | 308.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 307.75 | 310.12 | 308.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 305.90 | 309.27 | 308.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 305.35 | 309.27 | 308.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 308.90 | 309.13 | 308.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 308.45 | 309.13 | 308.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 308.65 | 312.63 | 311.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 308.65 | 312.63 | 311.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 310.30 | 312.16 | 311.33 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 308.25 | 310.81 | 310.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 305.85 | 309.40 | 310.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 298.40 | 296.00 | 298.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 13:15:00 | 298.40 | 296.00 | 298.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 298.40 | 296.00 | 298.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:45:00 | 299.30 | 296.00 | 298.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 299.05 | 296.61 | 298.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:15:00 | 298.05 | 296.61 | 298.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 297.45 | 295.78 | 297.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 300.40 | 296.97 | 297.01 | SL hit (close>static) qty=1.00 sl=300.15 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 10:15:00 | 298.30 | 297.24 | 297.13 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 295.85 | 296.90 | 296.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 13:15:00 | 295.35 | 296.59 | 296.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 297.15 | 296.70 | 296.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 297.15 | 296.70 | 296.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 297.15 | 296.70 | 296.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 297.15 | 296.70 | 296.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 297.95 | 296.95 | 296.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 299.05 | 296.95 | 296.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 299.40 | 297.44 | 297.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 300.45 | 298.04 | 297.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 300.35 | 300.69 | 299.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 300.35 | 300.69 | 299.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 299.80 | 300.44 | 299.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 300.55 | 300.44 | 299.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 300.85 | 300.52 | 299.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 299.00 | 300.52 | 299.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 296.30 | 300.00 | 299.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 296.30 | 300.00 | 299.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 296.85 | 299.37 | 299.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 291.10 | 295.41 | 297.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 292.35 | 291.05 | 293.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 292.35 | 291.05 | 293.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 292.35 | 291.05 | 293.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 292.10 | 291.05 | 293.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 292.90 | 291.42 | 293.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:30:00 | 293.55 | 291.42 | 293.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 295.65 | 292.27 | 293.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 295.60 | 292.27 | 293.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 298.90 | 293.59 | 294.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 298.90 | 293.59 | 294.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 297.15 | 294.30 | 294.35 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 299.45 | 295.33 | 294.82 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 293.95 | 295.96 | 296.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 291.75 | 295.12 | 295.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 13:15:00 | 283.95 | 282.99 | 285.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 14:00:00 | 283.95 | 282.99 | 285.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 284.20 | 283.23 | 285.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:30:00 | 284.70 | 283.23 | 285.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 280.90 | 282.95 | 284.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 279.35 | 281.80 | 284.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 265.38 | 273.16 | 276.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 266.25 | 264.80 | 268.65 | SL hit (close>ema200) qty=0.50 sl=264.80 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 268.45 | 266.65 | 266.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 271.85 | 267.69 | 267.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 271.45 | 273.25 | 271.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 271.45 | 273.25 | 271.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 271.45 | 273.25 | 271.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 271.45 | 273.25 | 271.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 272.25 | 273.05 | 271.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 268.85 | 273.05 | 271.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 269.35 | 272.31 | 271.29 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 11:15:00 | 265.15 | 270.16 | 270.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 10:15:00 | 264.10 | 266.61 | 268.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 13:15:00 | 262.00 | 261.63 | 263.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 14:00:00 | 262.00 | 261.63 | 263.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 262.80 | 261.86 | 263.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:45:00 | 263.30 | 261.86 | 263.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 266.10 | 262.87 | 263.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 268.45 | 262.87 | 263.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 264.85 | 263.26 | 263.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:30:00 | 262.30 | 262.80 | 263.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:30:00 | 263.25 | 262.45 | 263.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 266.35 | 262.39 | 262.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 266.35 | 262.39 | 262.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 269.15 | 264.59 | 263.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 15:15:00 | 284.40 | 284.67 | 277.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 09:15:00 | 282.65 | 284.67 | 277.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 284.45 | 283.84 | 278.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:00:00 | 286.75 | 281.47 | 279.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:45:00 | 286.75 | 284.19 | 282.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 295.90 | 298.06 | 298.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 295.90 | 298.06 | 298.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 295.05 | 297.05 | 297.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 295.45 | 294.87 | 295.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 15:15:00 | 295.45 | 294.87 | 295.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 295.45 | 294.87 | 295.89 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 296.00 | 295.70 | 295.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 299.10 | 296.46 | 296.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 297.45 | 298.47 | 297.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 297.45 | 298.47 | 297.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 297.45 | 298.47 | 297.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 297.45 | 298.47 | 297.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 297.70 | 298.32 | 297.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 297.70 | 298.32 | 297.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 296.65 | 297.98 | 297.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 304.30 | 297.98 | 297.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 299.55 | 300.40 | 300.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 299.55 | 300.40 | 300.43 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 301.80 | 300.68 | 300.55 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 300.10 | 300.47 | 300.49 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 305.50 | 301.38 | 300.86 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 300.45 | 301.68 | 301.71 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 305.80 | 301.31 | 300.76 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 301.35 | 303.32 | 303.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 299.95 | 301.55 | 302.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 300.75 | 300.64 | 301.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 300.75 | 300.64 | 301.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 300.25 | 300.65 | 301.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 301.35 | 300.85 | 301.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 299.40 | 300.56 | 301.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 296.85 | 299.76 | 300.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:15:00 | 297.50 | 298.68 | 299.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 12:15:00 | 282.01 | 289.54 | 294.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 12:15:00 | 282.62 | 289.54 | 294.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 280.75 | 279.62 | 284.35 | SL hit (close>ema200) qty=0.50 sl=279.62 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 286.50 | 283.01 | 282.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 292.15 | 285.73 | 284.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 290.05 | 293.68 | 292.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 290.05 | 293.68 | 292.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 290.05 | 293.68 | 292.03 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 285.00 | 290.78 | 290.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 284.10 | 287.36 | 289.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 287.70 | 285.69 | 287.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 287.70 | 285.69 | 287.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 287.70 | 285.69 | 287.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 287.70 | 285.69 | 287.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 287.10 | 285.97 | 287.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 289.20 | 286.56 | 287.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 289.15 | 287.08 | 287.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 289.70 | 287.08 | 287.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 291.35 | 287.93 | 287.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 292.00 | 288.74 | 288.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 288.35 | 289.68 | 288.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 288.35 | 289.68 | 288.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 288.35 | 289.68 | 288.98 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 287.45 | 288.63 | 288.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 282.95 | 287.49 | 288.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 285.05 | 284.09 | 285.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:45:00 | 285.15 | 284.09 | 285.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 285.15 | 284.30 | 285.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 282.65 | 284.30 | 285.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:45:00 | 284.50 | 284.76 | 285.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 284.80 | 284.87 | 285.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:45:00 | 284.50 | 284.70 | 285.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 284.80 | 284.72 | 285.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 284.50 | 284.72 | 285.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 284.35 | 284.65 | 285.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 284.80 | 284.65 | 285.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 282.80 | 284.28 | 284.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 282.95 | 284.28 | 284.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 279.80 | 277.23 | 278.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 280.05 | 277.23 | 278.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 278.80 | 277.54 | 278.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 278.80 | 277.54 | 278.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 276.55 | 274.77 | 276.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 276.55 | 274.77 | 276.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 276.35 | 275.09 | 276.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 277.35 | 275.09 | 276.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 275.85 | 275.24 | 276.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 273.30 | 275.37 | 276.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:15:00 | 274.30 | 274.69 | 275.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 14:15:00 | 270.27 | 273.00 | 274.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 14:15:00 | 270.56 | 273.00 | 274.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 14:15:00 | 270.27 | 273.00 | 274.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 12:15:00 | 268.52 | 271.27 | 272.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 271.30 | 271.27 | 272.89 | SL hit (close>static) qty=0.50 sl=271.27 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 278.20 | 273.46 | 273.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 283.75 | 278.34 | 276.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 15:15:00 | 283.75 | 283.89 | 280.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 09:15:00 | 286.25 | 283.89 | 280.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 280.80 | 283.05 | 281.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:00:00 | 280.80 | 283.05 | 281.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 280.60 | 282.56 | 281.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 281.50 | 282.56 | 281.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 279.65 | 281.98 | 281.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 279.65 | 281.98 | 281.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 280.00 | 281.58 | 280.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 279.15 | 281.58 | 280.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 279.30 | 280.63 | 280.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:15:00 | 278.85 | 280.63 | 280.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 278.50 | 280.21 | 280.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 277.35 | 279.61 | 280.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 281.20 | 279.29 | 279.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 281.20 | 279.29 | 279.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 281.20 | 279.29 | 279.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 281.10 | 279.29 | 279.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 280.90 | 279.61 | 279.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:30:00 | 281.85 | 279.61 | 279.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 277.00 | 273.38 | 275.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 275.55 | 273.38 | 275.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 277.55 | 274.21 | 275.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 277.80 | 274.21 | 275.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 274.20 | 274.62 | 275.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:45:00 | 273.85 | 274.66 | 275.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:15:00 | 273.80 | 274.66 | 275.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 277.00 | 274.28 | 274.69 | SL hit (close>static) qty=1.00 sl=276.90 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 278.10 | 275.04 | 275.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 293.30 | 282.59 | 279.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 288.75 | 289.51 | 284.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 288.75 | 289.51 | 284.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 285.15 | 287.79 | 284.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 284.70 | 287.79 | 284.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 284.95 | 287.22 | 284.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 284.40 | 287.22 | 284.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 291.30 | 288.04 | 285.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:15:00 | 292.85 | 288.04 | 285.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:15:00 | 291.75 | 288.73 | 286.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 292.05 | 289.39 | 286.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:30:00 | 296.50 | 290.94 | 288.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 290.30 | 293.27 | 290.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:00:00 | 290.30 | 293.27 | 290.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 291.25 | 292.87 | 290.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:15:00 | 291.95 | 292.87 | 290.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 14:15:00 | 288.40 | 291.47 | 290.73 | SL hit (close<static) qty=1.00 sl=290.20 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 286.70 | 289.91 | 290.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 284.75 | 288.88 | 289.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 290.05 | 288.72 | 289.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 12:15:00 | 290.05 | 288.72 | 289.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 290.05 | 288.72 | 289.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:45:00 | 289.90 | 288.72 | 289.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 294.40 | 289.85 | 289.85 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 289.45 | 290.31 | 290.42 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 10:15:00 | 291.80 | 290.61 | 290.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 11:15:00 | 293.10 | 291.11 | 290.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 13:15:00 | 289.00 | 290.77 | 290.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 13:15:00 | 289.00 | 290.77 | 290.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 289.00 | 290.77 | 290.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 289.00 | 290.77 | 290.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 287.15 | 290.04 | 290.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 285.80 | 289.19 | 289.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 13:15:00 | 288.60 | 287.32 | 288.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 13:15:00 | 288.60 | 287.32 | 288.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 288.60 | 287.32 | 288.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 288.60 | 287.32 | 288.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 290.05 | 287.86 | 288.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:45:00 | 290.00 | 287.86 | 288.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 290.00 | 288.29 | 288.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 286.75 | 288.29 | 288.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 291.00 | 289.23 | 289.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 291.00 | 289.23 | 289.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 12:15:00 | 293.00 | 289.98 | 289.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 10:15:00 | 292.40 | 292.56 | 291.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:30:00 | 292.05 | 292.56 | 291.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 293.20 | 292.69 | 291.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 289.85 | 292.69 | 291.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 294.05 | 293.87 | 292.55 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 289.55 | 291.66 | 291.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 287.70 | 290.42 | 291.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 290.40 | 289.42 | 290.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 290.40 | 289.42 | 290.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 290.40 | 289.42 | 290.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 290.40 | 289.42 | 290.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 290.60 | 289.65 | 290.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 288.65 | 289.65 | 290.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 11:15:00 | 274.22 | 277.40 | 280.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 10:15:00 | 259.78 | 270.38 | 275.54 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 264.00 | 259.94 | 259.51 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 256.40 | 259.60 | 259.73 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 13:15:00 | 261.70 | 258.91 | 258.77 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 258.50 | 258.77 | 258.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 252.05 | 256.80 | 257.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 257.70 | 256.13 | 257.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 12:15:00 | 257.70 | 256.13 | 257.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 257.70 | 256.13 | 257.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 257.70 | 256.13 | 257.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 256.80 | 256.26 | 257.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 257.55 | 256.26 | 257.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 258.05 | 256.62 | 257.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 258.05 | 256.62 | 257.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 257.90 | 256.88 | 257.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 261.70 | 256.88 | 257.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 264.50 | 258.40 | 257.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 269.80 | 260.68 | 259.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 267.10 | 268.13 | 265.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 13:00:00 | 267.10 | 268.13 | 265.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 268.05 | 267.60 | 265.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 12:00:00 | 270.70 | 268.16 | 266.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 267.30 | 270.69 | 270.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 267.30 | 270.69 | 270.75 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 10:15:00 | 272.50 | 271.11 | 270.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 11:15:00 | 272.60 | 271.41 | 271.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 12:15:00 | 271.20 | 271.37 | 271.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 12:15:00 | 271.20 | 271.37 | 271.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 271.20 | 271.37 | 271.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:30:00 | 271.15 | 271.37 | 271.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 272.05 | 271.50 | 271.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 272.05 | 271.50 | 271.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 283.90 | 275.28 | 273.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 286.85 | 275.28 | 273.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 286.55 | 286.00 | 281.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 14:15:00 | 286.45 | 286.06 | 282.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 286.25 | 286.10 | 282.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 284.00 | 285.66 | 283.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 287.20 | 284.49 | 283.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 293.85 | 295.49 | 295.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 293.85 | 295.49 | 295.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 292.55 | 294.71 | 295.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 288.25 | 288.13 | 290.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:00:00 | 288.25 | 288.13 | 290.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 290.60 | 288.62 | 290.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 290.60 | 288.62 | 290.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 287.60 | 288.42 | 290.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 286.10 | 288.49 | 289.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:45:00 | 285.05 | 287.89 | 288.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:00:00 | 286.20 | 286.70 | 287.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 271.80 | 276.37 | 280.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 270.80 | 276.37 | 280.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 271.89 | 276.37 | 280.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 274.00 | 273.83 | 277.00 | SL hit (close>ema200) qty=0.50 sl=273.83 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 274.95 | 268.65 | 268.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 276.90 | 271.93 | 269.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 270.75 | 273.49 | 271.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 270.75 | 273.49 | 271.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 270.75 | 273.49 | 271.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:00:00 | 270.75 | 273.49 | 271.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 280.55 | 274.90 | 272.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 284.10 | 281.59 | 279.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-13 09:15:00 | 312.51 | 299.92 | 295.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 296.00 | 300.87 | 301.35 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 302.20 | 301.29 | 301.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 304.35 | 301.90 | 301.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 300.40 | 303.49 | 302.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 300.40 | 303.49 | 302.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 300.40 | 303.49 | 302.83 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 301.20 | 302.50 | 302.57 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 306.40 | 303.28 | 302.92 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 303.00 | 304.25 | 304.31 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 305.70 | 304.54 | 304.43 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 304.00 | 304.33 | 304.35 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 308.60 | 305.18 | 304.73 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 301.85 | 305.27 | 305.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 294.05 | 302.43 | 304.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 294.50 | 290.54 | 293.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 294.50 | 290.54 | 293.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 294.50 | 290.54 | 293.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 295.05 | 290.54 | 293.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 289.70 | 290.37 | 293.40 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 299.00 | 294.58 | 294.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 300.20 | 295.70 | 294.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 13:15:00 | 294.55 | 295.90 | 295.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 13:15:00 | 294.55 | 295.90 | 295.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 294.55 | 295.90 | 295.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:30:00 | 294.80 | 295.90 | 295.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 291.40 | 295.00 | 294.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 291.40 | 295.00 | 294.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 288.20 | 293.64 | 294.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 280.35 | 290.98 | 293.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 287.40 | 286.31 | 289.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 15:00:00 | 287.40 | 286.31 | 289.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 290.10 | 287.31 | 289.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:30:00 | 287.75 | 287.97 | 289.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 294.35 | 289.82 | 289.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 294.35 | 289.82 | 289.76 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 285.25 | 289.47 | 290.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 280.00 | 283.89 | 286.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 15:15:00 | 265.55 | 265.00 | 268.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 09:15:00 | 268.95 | 265.00 | 268.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 268.00 | 265.60 | 268.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 268.00 | 265.60 | 268.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 268.15 | 266.11 | 268.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 268.55 | 266.11 | 268.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 272.90 | 267.47 | 268.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:00:00 | 272.90 | 267.47 | 268.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 278.60 | 269.70 | 269.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 286.75 | 273.11 | 271.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 275.40 | 275.99 | 273.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:45:00 | 274.20 | 275.99 | 273.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 280.40 | 284.12 | 280.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:30:00 | 282.35 | 284.12 | 280.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 281.55 | 283.61 | 280.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 279.50 | 283.61 | 280.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 281.30 | 283.15 | 280.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:00:00 | 281.30 | 283.15 | 280.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 281.00 | 282.72 | 280.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:30:00 | 281.85 | 282.72 | 280.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 282.10 | 282.59 | 280.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 283.80 | 281.69 | 280.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 312.18 | 304.71 | 303.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 344.10 | 346.96 | 347.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 342.15 | 345.85 | 346.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 345.90 | 345.71 | 346.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 13:15:00 | 345.90 | 345.71 | 346.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 345.90 | 345.71 | 346.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 345.90 | 345.71 | 346.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 348.55 | 346.28 | 346.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 348.55 | 346.28 | 346.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 347.10 | 346.44 | 346.57 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 347.55 | 346.79 | 346.71 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 12:15:00 | 344.75 | 346.33 | 346.51 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 349.85 | 347.00 | 346.75 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 346.20 | 346.59 | 346.62 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 347.00 | 346.67 | 346.66 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 345.25 | 346.64 | 346.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 343.95 | 345.93 | 346.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 341.50 | 340.76 | 342.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 15:00:00 | 341.50 | 340.76 | 342.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 339.20 | 340.45 | 342.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 338.45 | 340.45 | 342.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:45:00 | 337.90 | 339.70 | 341.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 344.00 | 340.66 | 341.27 | SL hit (close>static) qty=1.00 sl=343.75 alert=retest2 |

### Cycle 73 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 347.25 | 342.46 | 341.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 353.50 | 345.76 | 343.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-11 09:15:00 | 350.95 | 355.26 | 352.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-11 09:15:00 | 350.95 | 355.26 | 352.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 350.95 | 355.26 | 352.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 10:00:00 | 350.95 | 355.26 | 352.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 10:15:00 | 350.30 | 354.27 | 352.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 10:30:00 | 349.40 | 354.27 | 352.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 14:15:00 | 352.15 | 353.55 | 352.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 15:00:00 | 352.15 | 353.55 | 352.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 15:15:00 | 354.00 | 353.64 | 352.46 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:30:00 | 248.34 | 2025-05-13 09:15:00 | 251.60 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-05-16 10:15:00 | 261.00 | 2025-05-23 09:15:00 | 287.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 307.25 | 2025-06-20 12:15:00 | 308.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-06-17 11:15:00 | 306.70 | 2025-06-20 12:15:00 | 308.20 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-06-20 11:45:00 | 308.70 | 2025-06-20 12:15:00 | 308.20 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-07-03 15:15:00 | 298.05 | 2025-07-08 09:15:00 | 300.40 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-07-07 09:45:00 | 297.45 | 2025-07-08 09:15:00 | 300.40 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-23 10:30:00 | 279.35 | 2025-07-25 10:15:00 | 265.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 10:30:00 | 279.35 | 2025-07-28 13:15:00 | 266.25 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2025-08-07 11:30:00 | 262.30 | 2025-08-11 14:15:00 | 266.35 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-08-08 09:30:00 | 263.25 | 2025-08-11 14:15:00 | 266.35 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-08-18 14:00:00 | 286.75 | 2025-08-26 12:15:00 | 295.90 | STOP_HIT | 1.00 | 3.19% |
| BUY | retest2 | 2025-08-20 09:45:00 | 286.75 | 2025-08-26 12:15:00 | 295.90 | STOP_HIT | 1.00 | 3.19% |
| BUY | retest2 | 2025-09-03 09:15:00 | 304.30 | 2025-09-05 14:15:00 | 299.55 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-09-25 09:30:00 | 296.85 | 2025-09-26 12:15:00 | 282.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 14:15:00 | 297.50 | 2025-09-26 12:15:00 | 282.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:30:00 | 296.85 | 2025-09-30 09:15:00 | 280.75 | STOP_HIT | 0.50 | 5.42% |
| SELL | retest2 | 2025-09-25 14:15:00 | 297.50 | 2025-09-30 09:15:00 | 280.75 | STOP_HIT | 0.50 | 5.63% |
| SELL | retest2 | 2025-10-15 09:15:00 | 282.65 | 2025-10-27 14:15:00 | 270.27 | PARTIAL | 0.50 | 4.38% |
| SELL | retest2 | 2025-10-15 14:45:00 | 284.50 | 2025-10-27 14:15:00 | 270.56 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-10-16 09:15:00 | 284.80 | 2025-10-27 14:15:00 | 270.27 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-10-16 09:45:00 | 284.50 | 2025-10-28 12:15:00 | 268.52 | PARTIAL | 0.50 | 5.62% |
| SELL | retest2 | 2025-10-15 09:15:00 | 282.65 | 2025-10-28 12:15:00 | 271.30 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2025-10-15 14:45:00 | 284.50 | 2025-10-28 12:15:00 | 271.30 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2025-10-16 09:15:00 | 284.80 | 2025-10-28 12:15:00 | 271.30 | STOP_HIT | 0.50 | 4.74% |
| SELL | retest2 | 2025-10-16 09:45:00 | 284.50 | 2025-10-28 12:15:00 | 271.30 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2025-10-27 09:15:00 | 273.30 | 2025-10-29 12:15:00 | 278.20 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-10-27 11:15:00 | 274.30 | 2025-10-29 12:15:00 | 278.20 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-11 09:45:00 | 273.85 | 2025-11-11 15:15:00 | 277.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-11 11:15:00 | 273.80 | 2025-11-11 15:15:00 | 277.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-14 13:15:00 | 292.85 | 2025-11-18 14:15:00 | 288.40 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-11-14 14:15:00 | 291.75 | 2025-11-19 09:15:00 | 286.70 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-11-14 15:00:00 | 292.05 | 2025-11-19 09:15:00 | 286.70 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-11-17 12:30:00 | 296.50 | 2025-11-19 09:15:00 | 286.70 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2025-11-18 12:15:00 | 291.95 | 2025-11-19 09:15:00 | 286.70 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-11-25 09:15:00 | 286.75 | 2025-11-25 11:15:00 | 291.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-01 09:15:00 | 288.65 | 2025-12-05 11:15:00 | 274.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:15:00 | 288.65 | 2025-12-08 10:15:00 | 259.78 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-23 12:00:00 | 270.70 | 2025-12-26 15:15:00 | 267.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-30 10:15:00 | 286.85 | 2026-01-09 10:15:00 | 293.85 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2025-12-31 12:00:00 | 286.55 | 2026-01-09 10:15:00 | 293.85 | STOP_HIT | 1.00 | 2.55% |
| BUY | retest2 | 2025-12-31 14:15:00 | 286.45 | 2026-01-09 10:15:00 | 293.85 | STOP_HIT | 1.00 | 2.58% |
| BUY | retest2 | 2025-12-31 15:00:00 | 286.25 | 2026-01-09 10:15:00 | 293.85 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2026-01-02 12:15:00 | 287.20 | 2026-01-09 10:15:00 | 293.85 | STOP_HIT | 1.00 | 2.32% |
| SELL | retest2 | 2026-01-16 09:15:00 | 286.10 | 2026-01-21 10:15:00 | 271.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:45:00 | 285.05 | 2026-01-21 10:15:00 | 270.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 14:00:00 | 286.20 | 2026-01-21 10:15:00 | 271.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 286.10 | 2026-01-22 09:15:00 | 274.00 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2026-01-16 09:45:00 | 285.05 | 2026-01-22 09:15:00 | 274.00 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2026-01-16 14:00:00 | 286.20 | 2026-01-22 09:15:00 | 274.00 | STOP_HIT | 0.50 | 4.26% |
| BUY | retest2 | 2026-02-06 14:30:00 | 284.10 | 2026-02-13 09:15:00 | 312.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-10 12:30:00 | 287.75 | 2026-03-11 09:15:00 | 294.35 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-03-24 09:15:00 | 283.80 | 2026-04-08 09:15:00 | 312.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-05 09:15:00 | 338.45 | 2026-05-05 15:15:00 | 344.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-05-05 09:45:00 | 337.90 | 2026-05-05 15:15:00 | 344.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-05-06 10:00:00 | 338.30 | 2026-05-06 12:15:00 | 345.60 | STOP_HIT | 1.00 | -2.16% |
