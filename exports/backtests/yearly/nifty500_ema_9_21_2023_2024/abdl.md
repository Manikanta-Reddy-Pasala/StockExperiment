# Allied Blenders and Distillers Ltd. (ABDL)

## Backtest Summary

- **Window:** 2024-07-02 09:15:00 → 2026-05-08 15:15:00 (3203 bars)
- **Last close:** 594.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 134 |
| ALERT1 | 92 |
| ALERT2 | 91 |
| ALERT2_SKIP | 43 |
| ALERT3 | 217 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 78 |
| PARTIAL | 12 |
| TARGET_HIT | 15 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 94 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 52
- **Target hits / Stop hits / Partials:** 15 / 67 / 12
- **Avg / median % per leg:** 1.95% / -0.76%
- **Sum % (uncompounded):** 182.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 13 | 26.0% | 12 | 38 | 0 | 1.35% | 67.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.96% | -3.9% |
| BUY @ 3rd Alert (retest2) | 48 | 13 | 27.1% | 12 | 36 | 0 | 1.48% | 71.3% |
| SELL (all) | 44 | 29 | 65.9% | 3 | 29 | 12 | 2.63% | 115.6% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 3rd Alert (retest2) | 40 | 25 | 62.5% | 1 | 29 | 10 | 2.14% | 85.6% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.35% | 26.1% |
| retest2 (combined) | 88 | 38 | 43.2% | 13 | 65 | 10 | 1.78% | 156.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 15:15:00 | 335.00 | 335.58 | 335.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 321.90 | 332.84 | 334.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 331.05 | 328.55 | 331.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 331.05 | 328.55 | 331.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 331.05 | 328.55 | 331.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 331.05 | 328.55 | 331.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 331.50 | 329.14 | 331.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 329.70 | 329.14 | 331.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 331.30 | 329.57 | 331.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 335.55 | 329.57 | 331.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 332.25 | 330.11 | 331.28 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 335.30 | 332.04 | 332.01 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 329.50 | 332.74 | 332.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 327.25 | 331.65 | 332.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 12:15:00 | 309.50 | 308.61 | 313.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 13:00:00 | 309.50 | 308.61 | 313.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 300.40 | 297.72 | 302.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:45:00 | 300.10 | 297.72 | 302.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 301.05 | 298.39 | 302.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:30:00 | 302.75 | 298.39 | 302.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 309.20 | 299.09 | 301.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 310.75 | 299.09 | 301.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 307.10 | 300.69 | 301.69 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 313.40 | 303.23 | 302.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 317.00 | 310.64 | 307.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 15:15:00 | 312.00 | 314.22 | 311.34 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:15:00 | 317.65 | 314.22 | 311.34 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 11:00:00 | 320.85 | 315.75 | 312.55 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 313.00 | 315.88 | 313.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-29 14:15:00 | 313.00 | 315.88 | 313.76 | SL hit (close<ema400) qty=1.00 sl=313.76 alert=retest1 |

### Cycle 5 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 308.05 | 312.25 | 312.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 15:15:00 | 306.55 | 309.77 | 311.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 10:15:00 | 311.10 | 309.91 | 311.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 10:15:00 | 311.10 | 309.91 | 311.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 311.10 | 309.91 | 311.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 311.10 | 309.91 | 311.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 310.00 | 309.93 | 310.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 12:45:00 | 307.90 | 309.56 | 310.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 292.50 | 300.53 | 303.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 300.00 | 292.35 | 296.45 | SL hit (close>ema200) qty=0.50 sl=292.35 alert=retest2 |

### Cycle 6 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 301.75 | 296.55 | 296.17 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 293.50 | 296.57 | 296.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 290.00 | 295.26 | 296.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 14:15:00 | 292.15 | 291.73 | 293.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 14:15:00 | 292.15 | 291.73 | 293.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 292.15 | 291.73 | 293.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 292.15 | 291.73 | 293.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 305.80 | 294.59 | 294.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 10:15:00 | 312.20 | 298.11 | 295.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 10:15:00 | 306.95 | 307.61 | 303.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 11:00:00 | 306.95 | 307.61 | 303.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 304.95 | 307.08 | 303.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 304.95 | 307.08 | 303.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 304.40 | 306.54 | 303.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 303.35 | 306.54 | 303.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 303.00 | 305.83 | 303.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:45:00 | 302.25 | 305.83 | 303.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 302.00 | 305.07 | 303.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 302.00 | 305.07 | 303.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 305.00 | 305.05 | 303.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 312.00 | 305.05 | 303.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 10:00:00 | 306.00 | 305.24 | 303.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:00:00 | 306.95 | 305.37 | 303.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 15:15:00 | 299.85 | 303.12 | 303.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 15:15:00 | 299.85 | 303.12 | 303.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 09:15:00 | 299.40 | 302.37 | 302.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 12:15:00 | 305.65 | 302.92 | 303.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 12:15:00 | 305.65 | 302.92 | 303.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 305.65 | 302.92 | 303.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:00:00 | 305.65 | 302.92 | 303.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 302.50 | 302.84 | 302.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 14:30:00 | 302.25 | 302.49 | 302.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:45:00 | 301.10 | 301.63 | 302.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 311.05 | 303.51 | 302.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 311.05 | 303.51 | 302.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 11:15:00 | 315.60 | 305.93 | 303.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 15:15:00 | 309.95 | 310.89 | 308.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 15:15:00 | 309.95 | 310.89 | 308.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 309.95 | 310.89 | 308.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 320.45 | 310.89 | 308.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-27 14:15:00 | 352.50 | 342.21 | 335.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 15:15:00 | 350.00 | 351.52 | 351.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 346.95 | 350.61 | 351.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 15:15:00 | 351.20 | 349.32 | 350.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 15:15:00 | 351.20 | 349.32 | 350.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 351.20 | 349.32 | 350.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:15:00 | 356.30 | 349.32 | 350.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 353.50 | 350.16 | 350.40 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 10:15:00 | 354.15 | 350.96 | 350.74 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 344.35 | 350.73 | 351.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 342.75 | 347.09 | 349.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 12:15:00 | 347.10 | 344.40 | 346.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 12:15:00 | 347.10 | 344.40 | 346.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 347.10 | 344.40 | 346.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:00:00 | 347.10 | 344.40 | 346.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 345.70 | 344.66 | 346.61 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 352.80 | 348.14 | 347.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 359.65 | 353.61 | 351.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 352.75 | 354.17 | 352.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 352.75 | 354.17 | 352.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 352.75 | 354.17 | 352.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 352.75 | 354.17 | 352.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 350.80 | 353.49 | 351.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 350.80 | 353.49 | 351.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 351.00 | 352.99 | 351.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 360.05 | 352.99 | 351.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 10:15:00 | 350.75 | 356.78 | 356.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 350.75 | 356.78 | 356.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 11:15:00 | 348.65 | 355.15 | 356.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 13:15:00 | 347.25 | 347.25 | 350.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-18 14:00:00 | 347.25 | 347.25 | 350.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 346.95 | 346.66 | 349.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 344.05 | 346.66 | 349.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 345.55 | 343.00 | 344.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 348.95 | 345.64 | 345.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 348.95 | 345.64 | 345.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 350.30 | 347.12 | 346.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 349.95 | 353.99 | 351.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 349.95 | 353.99 | 351.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 349.95 | 353.99 | 351.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 349.95 | 353.99 | 351.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 351.00 | 353.39 | 351.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 351.20 | 353.39 | 351.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 348.00 | 352.31 | 351.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:45:00 | 348.00 | 352.31 | 351.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 347.00 | 351.25 | 350.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:30:00 | 346.65 | 351.25 | 350.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 348.70 | 350.41 | 350.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 13:15:00 | 345.45 | 348.35 | 349.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 12:15:00 | 341.60 | 338.73 | 340.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 12:15:00 | 341.60 | 338.73 | 340.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 341.60 | 338.73 | 340.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 339.60 | 338.73 | 340.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 343.60 | 339.70 | 341.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:45:00 | 342.40 | 339.70 | 341.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 342.00 | 340.16 | 341.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 342.00 | 340.16 | 341.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 340.50 | 340.23 | 341.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 338.25 | 340.23 | 341.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 321.34 | 331.83 | 335.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 314.45 | 313.37 | 319.51 | SL hit (close>ema200) qty=0.50 sl=313.37 alert=retest2 |

### Cycle 18 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 328.60 | 322.94 | 322.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 10:15:00 | 330.65 | 327.45 | 325.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 326.10 | 327.53 | 325.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:00:00 | 326.10 | 327.53 | 325.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 329.65 | 327.96 | 326.31 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 323.10 | 326.21 | 326.43 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 11:15:00 | 338.55 | 328.67 | 327.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 13:15:00 | 340.30 | 332.17 | 329.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 14:15:00 | 335.15 | 336.97 | 334.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 15:00:00 | 335.15 | 336.97 | 334.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 334.50 | 336.47 | 334.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:15:00 | 334.15 | 336.47 | 334.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 336.15 | 336.41 | 334.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 12:30:00 | 338.25 | 337.03 | 335.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 338.20 | 337.42 | 335.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 15:15:00 | 333.00 | 334.71 | 334.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 333.00 | 334.71 | 334.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 328.50 | 333.47 | 334.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 334.10 | 333.12 | 333.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 334.10 | 333.12 | 333.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 334.10 | 333.12 | 333.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 334.10 | 333.12 | 333.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 331.65 | 332.83 | 333.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 334.40 | 332.83 | 333.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 326.80 | 331.33 | 332.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 323.25 | 329.11 | 331.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 322.10 | 327.52 | 330.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:15:00 | 307.09 | 315.31 | 322.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 306.00 | 313.58 | 320.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 313.05 | 310.72 | 317.39 | SL hit (close>ema200) qty=0.50 sl=310.72 alert=retest2 |

### Cycle 22 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 324.00 | 311.50 | 310.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 331.00 | 321.36 | 317.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 12:15:00 | 322.45 | 326.12 | 321.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 12:15:00 | 322.45 | 326.12 | 321.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 322.45 | 326.12 | 321.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:00:00 | 322.45 | 326.12 | 321.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 321.00 | 325.10 | 321.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:30:00 | 322.00 | 325.10 | 321.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 327.70 | 325.62 | 322.05 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 14:15:00 | 322.00 | 323.65 | 323.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 15:15:00 | 320.50 | 323.02 | 323.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 324.35 | 323.29 | 323.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 324.35 | 323.29 | 323.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 324.35 | 323.29 | 323.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 324.35 | 323.29 | 323.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 323.70 | 323.37 | 323.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 323.70 | 323.37 | 323.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 322.50 | 323.20 | 323.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:00:00 | 322.50 | 323.20 | 323.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 323.40 | 323.24 | 323.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:45:00 | 323.45 | 323.24 | 323.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 328.35 | 324.26 | 323.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 14:15:00 | 332.50 | 325.91 | 324.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 13:15:00 | 331.70 | 332.01 | 328.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-06 14:00:00 | 331.70 | 332.01 | 328.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 331.00 | 331.49 | 329.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 331.50 | 331.49 | 329.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 330.30 | 331.25 | 329.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 329.60 | 331.25 | 329.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 330.40 | 330.88 | 329.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:15:00 | 330.70 | 330.88 | 329.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 330.40 | 330.79 | 329.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 334.10 | 330.74 | 329.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 12:30:00 | 331.30 | 331.28 | 330.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 13:00:00 | 331.50 | 331.28 | 330.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 14:15:00 | 328.40 | 330.51 | 330.26 | SL hit (close<static) qty=1.00 sl=329.10 alert=retest2 |

### Cycle 25 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 326.00 | 329.61 | 329.87 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 330.35 | 329.70 | 329.64 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 326.00 | 328.90 | 329.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 314.70 | 326.06 | 327.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 315.40 | 314.89 | 319.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 09:45:00 | 317.00 | 314.89 | 319.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 314.25 | 315.06 | 317.48 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 317.75 | 313.87 | 313.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 327.75 | 322.04 | 319.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 322.30 | 322.87 | 320.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 322.30 | 322.87 | 320.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 342.60 | 344.44 | 341.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 342.60 | 344.44 | 341.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 347.55 | 345.06 | 341.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:45:00 | 347.20 | 345.06 | 341.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 374.25 | 375.43 | 373.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 374.25 | 375.43 | 373.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 374.90 | 375.32 | 373.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 377.00 | 375.32 | 373.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 370.85 | 374.43 | 372.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 370.85 | 374.43 | 372.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 374.85 | 374.51 | 373.14 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 366.85 | 372.06 | 372.52 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 373.65 | 372.08 | 372.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 13:15:00 | 374.60 | 372.58 | 372.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 12:15:00 | 392.00 | 392.72 | 387.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 12:45:00 | 391.65 | 392.72 | 387.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 397.00 | 393.46 | 389.52 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 386.00 | 390.12 | 390.33 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 09:15:00 | 393.90 | 390.88 | 390.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 12:15:00 | 402.00 | 394.45 | 392.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 11:15:00 | 421.00 | 421.63 | 416.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 12:00:00 | 421.00 | 421.63 | 416.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 425.00 | 422.51 | 418.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 10:45:00 | 430.00 | 425.00 | 420.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:45:00 | 429.30 | 426.41 | 423.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:30:00 | 433.00 | 436.29 | 434.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 11:15:00 | 426.75 | 433.06 | 433.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 426.75 | 433.06 | 433.16 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 433.25 | 431.45 | 431.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 09:15:00 | 437.60 | 432.77 | 431.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 12:15:00 | 430.00 | 432.62 | 432.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 12:15:00 | 430.00 | 432.62 | 432.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 430.00 | 432.62 | 432.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:00:00 | 430.00 | 432.62 | 432.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 431.15 | 432.33 | 432.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:30:00 | 432.75 | 432.88 | 432.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 10:15:00 | 433.20 | 432.58 | 432.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 11:45:00 | 431.95 | 432.23 | 432.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 427.10 | 431.20 | 431.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 427.10 | 431.20 | 431.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 415.25 | 426.66 | 429.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 396.10 | 395.67 | 405.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 396.10 | 395.67 | 405.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 400.70 | 395.29 | 400.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:45:00 | 402.90 | 395.29 | 400.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 404.95 | 397.22 | 401.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 407.00 | 397.22 | 401.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 404.90 | 398.76 | 401.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:15:00 | 402.20 | 400.01 | 401.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 14:15:00 | 406.50 | 401.48 | 402.29 | SL hit (close>static) qty=1.00 sl=405.90 alert=retest2 |

### Cycle 36 — BUY (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 15:15:00 | 408.55 | 402.89 | 402.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 10:15:00 | 412.70 | 405.90 | 404.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 15:15:00 | 408.00 | 408.31 | 406.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 09:15:00 | 406.55 | 408.31 | 406.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 407.40 | 408.13 | 406.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 10:15:00 | 409.00 | 408.13 | 406.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 13:15:00 | 402.75 | 405.72 | 405.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 13:15:00 | 402.75 | 405.72 | 405.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 400.80 | 404.31 | 405.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 10:15:00 | 406.35 | 404.71 | 405.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 10:15:00 | 406.35 | 404.71 | 405.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 406.35 | 404.71 | 405.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:45:00 | 407.85 | 404.71 | 405.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 409.35 | 405.64 | 405.55 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 400.35 | 405.65 | 405.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 396.00 | 402.99 | 404.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 15:15:00 | 402.20 | 402.11 | 403.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 15:15:00 | 402.20 | 402.11 | 403.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 402.20 | 402.11 | 403.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 10:30:00 | 393.85 | 399.62 | 402.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 14:15:00 | 405.00 | 400.52 | 399.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 405.00 | 400.52 | 399.94 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 394.65 | 399.51 | 399.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 377.30 | 394.33 | 397.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 373.25 | 372.43 | 381.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 373.25 | 372.43 | 381.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 383.55 | 374.54 | 379.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 383.55 | 374.54 | 379.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 386.15 | 376.86 | 379.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 386.15 | 376.86 | 379.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 388.65 | 381.95 | 381.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 389.70 | 383.50 | 382.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 14:15:00 | 394.50 | 394.83 | 390.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:30:00 | 398.65 | 394.83 | 390.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 395.35 | 395.01 | 391.03 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 09:15:00 | 385.40 | 389.48 | 389.98 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 13:15:00 | 409.55 | 392.80 | 391.15 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 10:15:00 | 398.20 | 399.55 | 399.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 391.90 | 397.06 | 398.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 391.50 | 391.15 | 393.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 15:00:00 | 391.50 | 391.15 | 393.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 355.25 | 356.09 | 361.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:30:00 | 350.50 | 354.41 | 359.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:15:00 | 351.00 | 354.41 | 359.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 350.05 | 353.53 | 358.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:45:00 | 350.25 | 353.11 | 358.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 332.97 | 345.49 | 352.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 333.45 | 345.49 | 352.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 332.55 | 345.49 | 352.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 332.74 | 345.49 | 352.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 329.85 | 327.16 | 335.66 | SL hit (close>ema200) qty=0.50 sl=327.16 alert=retest2 |

### Cycle 46 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 339.40 | 332.82 | 332.76 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 333.20 | 335.25 | 335.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 327.55 | 329.91 | 331.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 311.20 | 310.66 | 314.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:15:00 | 311.80 | 310.66 | 314.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 315.00 | 311.45 | 314.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 316.45 | 312.30 | 314.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 313.85 | 312.61 | 314.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 312.65 | 312.28 | 313.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 320.80 | 314.51 | 314.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 320.80 | 314.51 | 314.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 323.85 | 316.38 | 315.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 334.60 | 335.85 | 330.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 334.60 | 335.85 | 330.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 328.90 | 334.24 | 330.69 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 319.90 | 327.59 | 328.35 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 10:15:00 | 330.40 | 327.33 | 326.93 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 09:15:00 | 322.45 | 326.09 | 326.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 10:15:00 | 317.40 | 321.69 | 323.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 315.50 | 314.61 | 318.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 12:00:00 | 315.50 | 314.61 | 318.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 317.70 | 315.70 | 317.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 10:00:00 | 315.00 | 316.25 | 316.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 13:15:00 | 320.20 | 317.61 | 317.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 13:15:00 | 320.20 | 317.61 | 317.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 10:15:00 | 322.50 | 319.24 | 318.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 319.95 | 320.20 | 319.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 13:15:00 | 319.95 | 320.20 | 319.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 319.95 | 320.20 | 319.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:00:00 | 319.95 | 320.20 | 319.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 320.50 | 320.26 | 319.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:45:00 | 319.85 | 320.26 | 319.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 321.00 | 320.41 | 319.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 326.50 | 320.41 | 319.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 318.60 | 324.44 | 322.99 | SL hit (close<static) qty=1.00 sl=318.85 alert=retest2 |

### Cycle 53 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 316.85 | 321.75 | 321.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 10:15:00 | 314.85 | 317.54 | 318.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 306.35 | 306.29 | 309.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 308.80 | 306.79 | 309.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 308.80 | 306.79 | 309.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 308.80 | 306.79 | 309.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 313.60 | 308.15 | 309.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 313.60 | 308.15 | 309.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 313.35 | 309.19 | 309.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 313.35 | 309.19 | 309.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 314.15 | 311.02 | 310.65 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 308.35 | 311.62 | 311.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 297.65 | 308.77 | 310.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 306.30 | 304.39 | 307.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 14:15:00 | 306.30 | 304.39 | 307.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 306.30 | 304.39 | 307.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 306.30 | 304.39 | 307.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 306.90 | 304.89 | 307.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 308.20 | 304.89 | 307.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 305.90 | 305.09 | 307.10 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 310.95 | 308.11 | 308.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 14:15:00 | 313.90 | 309.27 | 308.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 11:15:00 | 296.50 | 315.65 | 314.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 11:15:00 | 296.50 | 315.65 | 314.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 296.50 | 315.65 | 314.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 12:00:00 | 296.50 | 315.65 | 314.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 12:15:00 | 295.55 | 311.63 | 312.72 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 09:15:00 | 320.05 | 312.32 | 311.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 324.80 | 319.46 | 316.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 11:15:00 | 330.25 | 330.82 | 328.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 11:30:00 | 330.25 | 330.82 | 328.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 328.50 | 330.36 | 328.49 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 322.80 | 327.27 | 327.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 315.05 | 324.82 | 326.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 15:15:00 | 318.00 | 317.61 | 320.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 09:15:00 | 320.05 | 317.61 | 320.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 320.00 | 318.08 | 320.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:30:00 | 316.25 | 318.10 | 319.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 14:15:00 | 316.25 | 312.20 | 311.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 316.25 | 312.20 | 311.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 317.30 | 313.22 | 312.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 312.70 | 313.12 | 312.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 312.70 | 313.12 | 312.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 312.70 | 313.12 | 312.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 312.70 | 313.12 | 312.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 316.50 | 313.79 | 312.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 11:45:00 | 316.95 | 314.50 | 313.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 13:15:00 | 310.10 | 313.54 | 313.03 | SL hit (close<static) qty=1.00 sl=312.50 alert=retest2 |

### Cycle 61 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 308.25 | 312.48 | 312.59 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 09:15:00 | 320.60 | 313.39 | 312.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 15:15:00 | 324.00 | 319.79 | 316.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 320.05 | 322.33 | 319.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 320.05 | 322.33 | 319.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 320.05 | 322.33 | 319.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:45:00 | 320.25 | 322.33 | 319.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 319.50 | 321.76 | 319.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 318.95 | 321.76 | 319.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 316.00 | 320.61 | 319.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 310.90 | 320.61 | 319.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 315.50 | 319.59 | 318.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 10:30:00 | 318.00 | 320.27 | 319.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-12 09:15:00 | 349.80 | 331.41 | 325.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 392.35 | 395.78 | 395.82 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 399.70 | 395.77 | 395.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 400.95 | 396.81 | 396.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 394.60 | 397.45 | 396.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 394.60 | 397.45 | 396.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 394.60 | 397.45 | 396.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 393.40 | 397.45 | 396.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 393.90 | 396.74 | 396.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 393.90 | 396.74 | 396.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 11:15:00 | 394.00 | 396.19 | 396.43 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 398.65 | 396.70 | 396.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 407.35 | 400.21 | 398.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 402.95 | 403.10 | 400.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:30:00 | 403.40 | 403.10 | 400.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 399.45 | 402.31 | 400.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 399.45 | 402.31 | 400.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 398.75 | 401.60 | 400.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 398.75 | 401.60 | 400.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 398.60 | 401.00 | 400.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:30:00 | 398.30 | 401.00 | 400.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 396.25 | 399.40 | 399.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 393.25 | 396.02 | 397.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 397.45 | 395.74 | 397.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 397.45 | 395.74 | 397.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 397.45 | 395.74 | 397.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 399.30 | 395.74 | 397.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 395.35 | 395.66 | 396.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 396.40 | 395.66 | 396.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 397.20 | 396.05 | 396.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:00:00 | 397.20 | 396.05 | 396.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 394.65 | 395.77 | 396.72 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 398.75 | 397.49 | 397.36 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 396.25 | 397.50 | 397.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 393.25 | 396.65 | 397.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 401.05 | 395.89 | 396.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 401.05 | 395.89 | 396.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 401.05 | 395.89 | 396.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:45:00 | 399.75 | 395.89 | 396.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 403.50 | 397.41 | 396.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 419.05 | 405.34 | 401.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 435.95 | 439.22 | 432.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 435.95 | 439.22 | 432.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 435.95 | 439.22 | 432.62 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 424.75 | 429.20 | 429.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 421.40 | 427.64 | 428.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 426.10 | 420.30 | 423.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 10:15:00 | 426.10 | 420.30 | 423.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 426.10 | 420.30 | 423.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 426.10 | 420.30 | 423.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 420.30 | 420.30 | 422.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:45:00 | 419.60 | 420.40 | 422.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 431.05 | 424.31 | 423.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 431.05 | 424.31 | 423.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 434.45 | 427.73 | 425.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 433.50 | 434.22 | 431.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 433.50 | 434.22 | 431.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 433.50 | 434.22 | 431.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:00:00 | 433.50 | 434.22 | 431.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 430.75 | 433.52 | 431.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:00:00 | 430.75 | 433.52 | 431.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 431.00 | 433.02 | 431.41 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 425.25 | 429.71 | 430.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 422.80 | 428.33 | 429.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 421.35 | 420.97 | 424.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:30:00 | 421.00 | 420.97 | 424.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 423.55 | 421.48 | 424.60 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 425.95 | 424.70 | 424.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 431.00 | 425.96 | 425.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 429.55 | 429.67 | 427.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:00:00 | 429.55 | 429.67 | 427.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 428.65 | 429.47 | 427.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 428.65 | 429.47 | 427.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 427.50 | 429.07 | 427.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 427.50 | 429.07 | 427.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 428.00 | 428.86 | 427.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 432.30 | 428.86 | 427.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 428.45 | 431.06 | 431.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 13:15:00 | 428.45 | 431.06 | 431.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 426.00 | 430.04 | 430.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 13:15:00 | 427.35 | 426.78 | 428.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 14:00:00 | 427.35 | 426.78 | 428.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 430.65 | 427.56 | 428.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 430.65 | 427.56 | 428.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 429.90 | 428.03 | 428.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 428.05 | 428.03 | 428.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 432.05 | 428.48 | 428.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 432.05 | 428.48 | 428.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 11:15:00 | 433.55 | 429.50 | 429.28 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 424.10 | 428.32 | 428.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 422.00 | 425.39 | 427.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 425.75 | 424.77 | 426.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 11:00:00 | 425.75 | 424.77 | 426.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 422.60 | 423.89 | 425.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:15:00 | 422.00 | 423.89 | 425.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 434.00 | 426.04 | 426.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 434.00 | 426.04 | 426.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 436.00 | 430.79 | 428.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 440.00 | 443.36 | 439.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 440.00 | 443.36 | 439.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 440.00 | 443.36 | 439.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 440.00 | 443.36 | 439.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 440.30 | 442.75 | 439.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 445.20 | 442.75 | 439.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:30:00 | 440.35 | 442.71 | 441.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 15:00:00 | 444.30 | 442.71 | 441.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 438.45 | 441.91 | 441.04 | SL hit (close<static) qty=1.00 sl=439.10 alert=retest2 |

### Cycle 79 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 478.60 | 484.53 | 484.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 10:15:00 | 476.45 | 480.41 | 482.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 10:15:00 | 475.50 | 474.51 | 477.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:30:00 | 474.55 | 474.51 | 477.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 481.35 | 475.88 | 478.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 483.80 | 475.88 | 478.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 479.90 | 476.68 | 478.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:00:00 | 479.50 | 477.25 | 478.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 15:15:00 | 484.00 | 479.32 | 479.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 484.00 | 479.32 | 479.17 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 476.10 | 478.82 | 479.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 14:15:00 | 474.10 | 477.87 | 478.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 475.00 | 466.41 | 469.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 475.00 | 466.41 | 469.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 475.00 | 466.41 | 469.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 475.00 | 466.41 | 469.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 475.45 | 468.22 | 470.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:15:00 | 477.90 | 468.22 | 470.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 13:15:00 | 474.30 | 471.42 | 471.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 14:15:00 | 477.60 | 472.66 | 471.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 510.50 | 511.89 | 505.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 14:45:00 | 513.45 | 511.89 | 505.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 512.00 | 511.91 | 506.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 518.15 | 511.91 | 506.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:45:00 | 516.50 | 512.76 | 507.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 13:45:00 | 515.40 | 519.50 | 516.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 504.90 | 513.73 | 514.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 504.90 | 513.73 | 514.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 498.05 | 510.60 | 512.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 507.80 | 496.53 | 501.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 507.80 | 496.53 | 501.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 507.80 | 496.53 | 501.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 507.80 | 496.53 | 501.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 504.00 | 498.03 | 501.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 499.00 | 498.03 | 501.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 504.40 | 493.95 | 493.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 504.40 | 493.95 | 493.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 519.40 | 500.91 | 497.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 521.35 | 521.37 | 514.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 521.35 | 521.37 | 514.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 526.35 | 524.87 | 520.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 520.85 | 524.87 | 520.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 522.20 | 523.36 | 521.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 522.20 | 523.36 | 521.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 523.00 | 523.29 | 521.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 523.45 | 522.93 | 521.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 519.55 | 522.25 | 521.52 | SL hit (close<static) qty=1.00 sl=521.20 alert=retest2 |

### Cycle 85 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 515.75 | 520.27 | 520.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 13:15:00 | 512.85 | 518.79 | 519.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 12:15:00 | 514.80 | 514.40 | 516.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 12:30:00 | 514.95 | 514.40 | 516.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 514.05 | 514.33 | 516.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:30:00 | 514.10 | 514.33 | 516.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 510.90 | 511.52 | 514.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 512.50 | 511.52 | 514.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 508.35 | 507.53 | 510.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 510.45 | 507.53 | 510.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 511.35 | 508.29 | 510.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 511.35 | 508.29 | 510.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 511.90 | 509.02 | 511.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 511.60 | 509.02 | 511.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 511.60 | 509.53 | 511.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:30:00 | 515.00 | 509.53 | 511.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 507.95 | 509.22 | 510.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:30:00 | 505.75 | 508.37 | 510.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 513.00 | 509.08 | 510.23 | SL hit (close>static) qty=1.00 sl=511.60 alert=retest2 |

### Cycle 86 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 520.50 | 512.49 | 511.58 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 505.40 | 511.87 | 512.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 10:15:00 | 498.35 | 507.65 | 510.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 498.65 | 498.46 | 503.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-04 09:30:00 | 502.10 | 498.46 | 503.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 496.00 | 496.19 | 499.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:45:00 | 498.35 | 496.19 | 499.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 497.00 | 496.08 | 498.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:45:00 | 497.70 | 496.08 | 498.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 499.10 | 496.68 | 498.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:45:00 | 499.50 | 496.68 | 498.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 502.60 | 497.87 | 498.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 499.60 | 497.87 | 498.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 507.60 | 499.81 | 499.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 10:15:00 | 513.75 | 502.60 | 501.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 10:15:00 | 506.20 | 506.80 | 504.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:15:00 | 505.90 | 506.80 | 504.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 506.50 | 506.42 | 504.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:45:00 | 505.45 | 506.42 | 504.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 507.00 | 506.54 | 505.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 508.80 | 506.54 | 505.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 507.30 | 506.73 | 505.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-15 09:15:00 | 559.68 | 540.71 | 529.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 543.30 | 547.46 | 547.61 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 548.55 | 547.65 | 547.64 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 546.45 | 547.41 | 547.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 544.50 | 546.83 | 547.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 550.35 | 547.40 | 547.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 550.35 | 547.40 | 547.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 550.35 | 547.40 | 547.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 550.35 | 547.40 | 547.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 551.00 | 548.12 | 547.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 554.60 | 549.42 | 548.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 13:15:00 | 550.00 | 551.09 | 549.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 13:15:00 | 550.00 | 551.09 | 549.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 550.00 | 551.09 | 549.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 550.00 | 551.09 | 549.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 549.90 | 550.86 | 549.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:45:00 | 551.05 | 550.86 | 549.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 551.65 | 551.01 | 549.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 553.50 | 551.01 | 549.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 554.80 | 551.77 | 550.32 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 545.00 | 549.21 | 549.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 536.25 | 542.85 | 545.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 535.00 | 532.80 | 537.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:00:00 | 535.00 | 532.80 | 537.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 537.00 | 533.64 | 537.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 537.00 | 533.64 | 537.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 532.95 | 533.50 | 536.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 530.35 | 533.50 | 536.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:15:00 | 530.65 | 533.51 | 536.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 528.80 | 516.78 | 515.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 528.80 | 516.78 | 515.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 532.95 | 520.01 | 517.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 546.85 | 552.02 | 545.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:00:00 | 546.85 | 552.02 | 545.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 551.00 | 551.81 | 546.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 551.55 | 551.81 | 546.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:45:00 | 551.35 | 551.53 | 546.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:45:00 | 553.20 | 548.46 | 546.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 14:15:00 | 544.75 | 546.80 | 546.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 14:15:00 | 544.75 | 546.80 | 546.83 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 553.00 | 547.75 | 547.24 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 543.40 | 547.70 | 548.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 537.40 | 544.29 | 546.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 534.05 | 531.16 | 535.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 534.05 | 531.16 | 535.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 535.40 | 532.01 | 535.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 535.40 | 532.01 | 535.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 536.00 | 532.81 | 535.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 535.95 | 532.81 | 535.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 534.25 | 533.10 | 535.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 536.15 | 533.10 | 535.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 541.55 | 534.79 | 535.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 541.55 | 534.79 | 535.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 543.95 | 536.62 | 536.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 544.90 | 541.61 | 539.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 15:15:00 | 538.30 | 540.95 | 539.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 15:15:00 | 538.30 | 540.95 | 539.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 538.30 | 540.95 | 539.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 552.75 | 540.95 | 539.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-21 13:15:00 | 608.03 | 579.19 | 565.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 655.20 | 663.32 | 663.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 640.90 | 656.18 | 660.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 13:15:00 | 633.60 | 620.61 | 625.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 13:15:00 | 633.60 | 620.61 | 625.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 633.60 | 620.61 | 625.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 633.60 | 620.61 | 625.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 637.15 | 623.92 | 626.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 637.15 | 623.92 | 626.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 632.50 | 627.81 | 627.72 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 625.85 | 628.14 | 628.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 622.10 | 626.93 | 627.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 633.40 | 627.93 | 627.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 11:15:00 | 633.40 | 627.93 | 627.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 633.40 | 627.93 | 627.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:00:00 | 633.40 | 627.93 | 627.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 640.10 | 630.37 | 629.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 13:15:00 | 645.60 | 633.41 | 630.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 664.60 | 665.25 | 657.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 15:00:00 | 664.60 | 665.25 | 657.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 664.25 | 665.89 | 660.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:45:00 | 664.20 | 665.89 | 660.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 661.00 | 664.91 | 660.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:45:00 | 661.05 | 664.91 | 660.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 661.05 | 664.14 | 660.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:30:00 | 661.70 | 664.14 | 660.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 659.90 | 663.29 | 660.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 665.65 | 663.29 | 660.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 13:30:00 | 662.70 | 663.66 | 662.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 15:15:00 | 650.00 | 660.02 | 660.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 650.00 | 660.02 | 660.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 11:15:00 | 641.70 | 655.60 | 658.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 652.00 | 651.45 | 654.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:00:00 | 652.00 | 651.45 | 654.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 650.40 | 651.24 | 654.46 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 12:15:00 | 661.65 | 654.55 | 654.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 13:15:00 | 665.20 | 656.68 | 655.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 09:15:00 | 656.40 | 657.78 | 656.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 656.40 | 657.78 | 656.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 656.40 | 657.78 | 656.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 655.10 | 657.78 | 656.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 651.00 | 656.42 | 655.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 651.00 | 656.42 | 655.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 650.80 | 655.30 | 655.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:45:00 | 650.80 | 655.30 | 655.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 13:15:00 | 653.70 | 654.91 | 655.02 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 655.75 | 655.03 | 654.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 668.70 | 658.30 | 656.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 12:15:00 | 658.80 | 658.81 | 657.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 13:00:00 | 658.80 | 658.81 | 657.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 656.50 | 658.59 | 657.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 656.50 | 658.59 | 657.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 649.05 | 656.68 | 656.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 633.00 | 651.95 | 654.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 621.35 | 610.26 | 621.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 621.35 | 610.26 | 621.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 621.35 | 610.26 | 621.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 621.80 | 610.26 | 621.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 618.35 | 611.88 | 621.65 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 628.95 | 621.72 | 621.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 630.75 | 624.69 | 622.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 627.40 | 627.41 | 624.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 627.40 | 627.41 | 624.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 627.40 | 627.41 | 624.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 627.40 | 627.41 | 624.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 627.55 | 627.44 | 625.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 623.40 | 627.44 | 625.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 622.85 | 626.93 | 625.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 622.85 | 626.93 | 625.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 619.00 | 625.34 | 624.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 619.00 | 625.34 | 624.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 618.95 | 624.06 | 624.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 607.35 | 620.72 | 622.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 621.80 | 614.17 | 617.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 621.80 | 614.17 | 617.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 621.80 | 614.17 | 617.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:45:00 | 621.90 | 614.17 | 617.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 616.35 | 614.61 | 617.54 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 622.70 | 619.01 | 618.71 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 614.85 | 618.50 | 618.67 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 622.00 | 619.22 | 618.98 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 15:15:00 | 615.00 | 618.89 | 619.05 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 624.20 | 619.95 | 619.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 626.40 | 622.09 | 620.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 625.05 | 625.13 | 623.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 12:45:00 | 624.00 | 625.13 | 623.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 621.90 | 624.75 | 623.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 621.90 | 624.75 | 623.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 620.55 | 623.91 | 623.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 620.55 | 623.91 | 623.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 619.95 | 623.12 | 623.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 619.95 | 623.12 | 623.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 623.05 | 623.08 | 623.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 623.05 | 623.08 | 623.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 623.00 | 623.06 | 623.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 625.75 | 623.06 | 623.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 622.85 | 623.02 | 623.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 623.25 | 623.02 | 623.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 624.65 | 623.35 | 623.16 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 619.05 | 622.49 | 622.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 613.55 | 620.70 | 621.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 613.95 | 611.80 | 615.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 613.95 | 611.80 | 615.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 616.35 | 612.71 | 615.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 616.15 | 612.71 | 615.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 613.25 | 612.82 | 615.67 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 622.25 | 617.28 | 616.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 625.50 | 621.28 | 619.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 626.55 | 627.10 | 623.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:00:00 | 626.55 | 627.10 | 623.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 623.80 | 626.44 | 623.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 623.80 | 626.44 | 623.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 624.40 | 626.03 | 623.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 627.15 | 626.27 | 624.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:30:00 | 627.30 | 626.53 | 624.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 14:15:00 | 621.60 | 626.61 | 625.57 | SL hit (close<static) qty=1.00 sl=623.50 alert=retest2 |

### Cycle 117 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 618.00 | 623.92 | 624.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 12:15:00 | 611.40 | 619.23 | 622.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 598.00 | 594.95 | 600.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:00:00 | 598.00 | 594.95 | 600.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 603.05 | 596.57 | 600.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 603.05 | 596.57 | 600.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 614.70 | 600.20 | 601.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 614.70 | 600.20 | 601.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 613.55 | 602.87 | 602.85 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 601.30 | 603.44 | 603.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 589.50 | 599.87 | 601.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 586.35 | 585.03 | 591.36 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 13:45:00 | 575.00 | 580.67 | 587.13 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 09:15:00 | 546.25 | 557.14 | 568.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-01-08 11:15:00 | 517.50 | 529.96 | 545.20 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 120 — BUY (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 13:15:00 | 452.00 | 445.53 | 445.41 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 441.10 | 445.23 | 445.36 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 450.35 | 445.99 | 445.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 454.95 | 448.94 | 447.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 455.65 | 459.86 | 456.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 14:15:00 | 455.65 | 459.86 | 456.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 455.65 | 459.86 | 456.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 455.65 | 459.86 | 456.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 454.80 | 458.85 | 456.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 444.15 | 458.85 | 456.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 476.30 | 484.48 | 477.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:00:00 | 495.15 | 485.19 | 480.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 15:15:00 | 523.30 | 526.43 | 526.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 15:15:00 | 523.30 | 526.43 | 526.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 10:15:00 | 520.10 | 525.25 | 526.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 507.00 | 505.71 | 511.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 507.00 | 505.71 | 511.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 498.75 | 496.95 | 500.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:45:00 | 494.45 | 496.10 | 499.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-02 09:15:00 | 445.00 | 476.50 | 482.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 14:15:00 | 473.65 | 461.22 | 459.92 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 455.30 | 459.23 | 459.64 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 467.35 | 460.82 | 460.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 10:15:00 | 473.20 | 463.30 | 461.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 475.40 | 476.13 | 470.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 475.40 | 476.13 | 470.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 471.90 | 474.57 | 471.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 463.40 | 474.57 | 471.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 463.50 | 472.36 | 470.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:00:00 | 467.85 | 471.46 | 470.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 463.85 | 469.03 | 469.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 463.85 | 469.03 | 469.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 457.70 | 466.04 | 468.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 15:15:00 | 443.00 | 442.86 | 449.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 09:15:00 | 434.95 | 442.86 | 449.63 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:15:00 | 413.20 | 419.16 | 425.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-03-23 11:15:00 | 391.45 | 402.54 | 411.38 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 128 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 411.85 | 402.13 | 401.72 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 390.45 | 402.04 | 402.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 387.80 | 399.19 | 400.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 13:15:00 | 398.70 | 398.21 | 399.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 398.70 | 398.21 | 399.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 396.10 | 397.79 | 399.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 396.10 | 397.79 | 399.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 397.00 | 397.63 | 399.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 406.10 | 397.63 | 399.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 404.55 | 399.02 | 399.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:30:00 | 406.40 | 399.02 | 399.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 399.70 | 399.15 | 399.78 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 415.00 | 402.46 | 401.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 418.60 | 407.84 | 403.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 414.95 | 416.15 | 410.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 414.95 | 416.15 | 410.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 414.95 | 416.15 | 410.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:30:00 | 423.50 | 417.82 | 412.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:00:00 | 423.50 | 417.82 | 412.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 15:00:00 | 424.35 | 419.13 | 413.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:00:00 | 423.80 | 420.46 | 416.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 419.70 | 422.19 | 419.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 419.70 | 422.19 | 419.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 420.00 | 421.52 | 419.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:15:00 | 421.25 | 421.52 | 419.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 420.75 | 421.36 | 419.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:15:00 | 420.25 | 421.36 | 419.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 420.25 | 421.14 | 419.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 430.50 | 421.14 | 419.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 465.85 | 447.63 | 436.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 552.15 | 558.65 | 559.53 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 562.00 | 559.83 | 559.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 12:15:00 | 567.45 | 561.35 | 560.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 560.95 | 561.82 | 560.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 560.95 | 561.82 | 560.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 560.95 | 561.82 | 560.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 560.95 | 561.82 | 560.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 563.00 | 562.06 | 561.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 564.85 | 562.06 | 561.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 555.45 | 560.78 | 560.67 | SL hit (close<static) qty=1.00 sl=558.50 alert=retest2 |

### Cycle 133 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 554.00 | 559.42 | 560.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 11:15:00 | 547.00 | 552.60 | 555.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 15:15:00 | 549.00 | 548.96 | 552.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 09:15:00 | 557.00 | 548.96 | 552.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 550.00 | 549.17 | 552.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 545.35 | 549.17 | 552.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:45:00 | 547.35 | 546.86 | 549.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 542.00 | 539.29 | 539.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 542.00 | 539.29 | 539.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 542.10 | 539.85 | 539.31 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-29 09:15:00 | 317.65 | 2024-07-29 14:15:00 | 313.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest1 | 2024-07-29 11:00:00 | 320.85 | 2024-07-29 14:15:00 | 313.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-07-31 12:45:00 | 307.90 | 2024-08-05 09:15:00 | 292.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 12:45:00 | 307.90 | 2024-08-06 09:15:00 | 300.00 | STOP_HIT | 0.50 | 2.57% |
| BUY | retest2 | 2024-08-14 09:15:00 | 312.00 | 2024-08-14 15:15:00 | 299.85 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2024-08-14 10:00:00 | 306.00 | 2024-08-14 15:15:00 | 299.85 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-08-14 12:00:00 | 306.95 | 2024-08-14 15:15:00 | 299.85 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-08-16 14:30:00 | 302.25 | 2024-08-20 10:15:00 | 311.05 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-08-19 10:45:00 | 301.10 | 2024-08-20 10:15:00 | 311.05 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-08-22 09:15:00 | 320.45 | 2024-08-27 14:15:00 | 352.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-12 09:15:00 | 360.05 | 2024-09-17 10:15:00 | 350.75 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-09-19 10:15:00 | 344.05 | 2024-09-23 10:15:00 | 348.95 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-09-20 13:30:00 | 345.55 | 2024-09-23 10:15:00 | 348.95 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-10-03 09:15:00 | 338.25 | 2024-10-04 09:15:00 | 321.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 338.25 | 2024-10-08 10:15:00 | 314.45 | STOP_HIT | 0.50 | 7.04% |
| BUY | retest2 | 2024-10-16 12:30:00 | 338.25 | 2024-10-17 15:15:00 | 333.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-10-16 15:00:00 | 338.20 | 2024-10-17 15:15:00 | 333.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-10-21 12:00:00 | 323.25 | 2024-10-22 12:15:00 | 307.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 322.10 | 2024-10-22 13:15:00 | 306.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 323.25 | 2024-10-23 09:15:00 | 313.05 | STOP_HIT | 0.50 | 3.16% |
| SELL | retest2 | 2024-10-21 12:30:00 | 322.10 | 2024-10-23 09:15:00 | 313.05 | STOP_HIT | 0.50 | 2.81% |
| BUY | retest2 | 2024-11-08 09:15:00 | 334.10 | 2024-11-08 14:15:00 | 328.40 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-11-08 12:30:00 | 331.30 | 2024-11-08 14:15:00 | 328.40 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-11-08 13:00:00 | 331.50 | 2024-11-08 14:15:00 | 328.40 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-12-30 10:45:00 | 430.00 | 2025-01-06 11:15:00 | 426.75 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-12-31 14:45:00 | 429.30 | 2025-01-06 11:15:00 | 426.75 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-01-06 09:30:00 | 433.00 | 2025-01-06 11:15:00 | 426.75 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-01-08 14:30:00 | 432.75 | 2025-01-09 12:15:00 | 427.10 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-01-09 10:15:00 | 433.20 | 2025-01-09 12:15:00 | 427.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-01-09 11:45:00 | 431.95 | 2025-01-09 12:15:00 | 427.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-01-15 13:15:00 | 402.20 | 2025-01-15 14:15:00 | 406.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-01-17 10:15:00 | 409.00 | 2025-01-17 13:15:00 | 402.75 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-01-22 10:30:00 | 393.85 | 2025-01-23 14:15:00 | 405.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-02-13 11:30:00 | 350.50 | 2025-02-14 10:15:00 | 332.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:15:00 | 351.00 | 2025-02-14 10:15:00 | 333.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:00:00 | 350.05 | 2025-02-14 10:15:00 | 332.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:45:00 | 350.25 | 2025-02-14 10:15:00 | 332.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:30:00 | 350.50 | 2025-02-17 13:15:00 | 329.85 | STOP_HIT | 0.50 | 5.89% |
| SELL | retest2 | 2025-02-13 12:15:00 | 351.00 | 2025-02-17 13:15:00 | 329.85 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2025-02-13 13:00:00 | 350.05 | 2025-02-17 13:15:00 | 329.85 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2025-02-13 13:45:00 | 350.25 | 2025-02-17 13:15:00 | 329.85 | STOP_HIT | 0.50 | 5.82% |
| SELL | retest2 | 2025-03-04 11:30:00 | 312.65 | 2025-03-05 09:15:00 | 320.80 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-03-20 10:00:00 | 315.00 | 2025-03-20 13:15:00 | 320.20 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-03-24 09:15:00 | 326.50 | 2025-03-25 09:15:00 | 318.60 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-04-29 13:30:00 | 316.25 | 2025-05-05 14:15:00 | 316.25 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-05-06 11:45:00 | 316.95 | 2025-05-06 13:15:00 | 310.10 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-05-09 10:30:00 | 318.00 | 2025-05-12 09:15:00 | 349.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-13 12:45:00 | 419.60 | 2025-06-16 11:15:00 | 431.05 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-06-25 09:15:00 | 432.30 | 2025-06-27 13:15:00 | 428.45 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-03 14:15:00 | 422.00 | 2025-07-04 09:15:00 | 434.00 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-07-10 09:15:00 | 445.20 | 2025-07-11 09:15:00 | 438.45 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-10 14:30:00 | 440.35 | 2025-07-11 09:15:00 | 438.45 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-07-10 15:00:00 | 444.30 | 2025-07-11 09:15:00 | 438.45 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-11 10:15:00 | 440.55 | 2025-07-14 14:15:00 | 484.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-11 12:15:00 | 446.30 | 2025-07-14 15:15:00 | 490.93 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-22 14:00:00 | 479.50 | 2025-07-22 15:15:00 | 484.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-08-04 09:15:00 | 518.15 | 2025-08-06 09:15:00 | 504.90 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-08-04 09:45:00 | 516.50 | 2025-08-06 09:15:00 | 504.90 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-08-05 13:45:00 | 515.40 | 2025-08-06 09:15:00 | 504.90 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-08-08 09:15:00 | 499.00 | 2025-08-18 09:15:00 | 504.40 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-08-25 09:30:00 | 523.45 | 2025-08-25 10:15:00 | 519.55 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-08-29 14:30:00 | 505.75 | 2025-09-01 09:15:00 | 513.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-09-10 09:15:00 | 508.80 | 2025-09-15 09:15:00 | 559.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-10 14:15:00 | 507.30 | 2025-09-15 09:15:00 | 558.03 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 530.35 | 2025-10-01 14:15:00 | 528.80 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-09-25 11:15:00 | 530.65 | 2025-10-01 14:15:00 | 528.80 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-10-07 13:15:00 | 551.55 | 2025-10-09 14:15:00 | 544.75 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-10-07 13:45:00 | 551.35 | 2025-10-09 14:15:00 | 544.75 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-10-08 14:45:00 | 553.20 | 2025-10-09 14:15:00 | 544.75 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-10-17 09:15:00 | 552.75 | 2025-10-21 13:15:00 | 608.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-20 09:15:00 | 665.65 | 2025-11-20 15:15:00 | 650.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-11-20 13:30:00 | 662.70 | 2025-11-20 15:15:00 | 650.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-12-23 14:45:00 | 627.15 | 2025-12-24 14:15:00 | 621.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-12-24 09:30:00 | 627.30 | 2025-12-24 14:15:00 | 621.60 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest1 | 2026-01-05 13:45:00 | 575.00 | 2026-01-07 09:15:00 | 546.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-05 13:45:00 | 575.00 | 2026-01-08 11:15:00 | 517.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 13:45:00 | 472.50 | 2026-01-20 09:15:00 | 448.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 15:15:00 | 471.85 | 2026-01-20 09:15:00 | 448.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:45:00 | 472.50 | 2026-01-22 09:15:00 | 444.00 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2026-01-16 15:15:00 | 471.85 | 2026-01-22 09:15:00 | 444.00 | STOP_HIT | 0.50 | 5.90% |
| BUY | retest2 | 2026-02-02 15:00:00 | 495.15 | 2026-02-17 15:15:00 | 523.30 | STOP_HIT | 1.00 | 5.69% |
| SELL | retest2 | 2026-02-25 10:45:00 | 494.45 | 2026-03-02 09:15:00 | 445.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-12 11:00:00 | 467.85 | 2026-03-12 14:15:00 | 463.85 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest1 | 2026-03-17 09:15:00 | 434.95 | 2026-03-19 13:15:00 | 413.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-17 09:15:00 | 434.95 | 2026-03-23 11:15:00 | 391.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 15:00:00 | 396.00 | 2026-03-25 10:15:00 | 403.40 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-04-02 13:30:00 | 423.50 | 2026-04-09 09:15:00 | 465.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 14:00:00 | 423.50 | 2026-04-09 09:15:00 | 465.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 15:00:00 | 424.35 | 2026-04-09 09:15:00 | 466.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 14:00:00 | 423.80 | 2026-04-09 09:15:00 | 466.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 430.50 | 2026-04-09 09:15:00 | 473.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-24 09:15:00 | 564.85 | 2026-04-24 10:15:00 | 555.45 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-04-29 10:15:00 | 545.35 | 2026-05-05 15:15:00 | 542.00 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2026-04-30 09:45:00 | 547.35 | 2026-05-05 15:15:00 | 542.00 | STOP_HIT | 1.00 | 0.98% |
