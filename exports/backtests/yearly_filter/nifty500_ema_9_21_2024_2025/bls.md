# BLS International Services Ltd. (BLS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 290.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 133 |
| ALERT1 | 97 |
| ALERT2 | 95 |
| ALERT2_SKIP | 53 |
| ALERT3 | 232 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 92 |
| PARTIAL | 22 |
| TARGET_HIT | 12 |
| STOP_HIT | 91 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 58
- **Target hits / Stop hits / Partials:** 12 / 87 / 22
- **Avg / median % per leg:** 1.32% / 0.37%
- **Sum % (uncompounded):** 159.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 14 | 30.4% | 7 | 37 | 2 | 0.86% | 39.7% |
| BUY @ 2nd Alert (retest1) | 5 | 5 | 100.0% | 0 | 3 | 2 | 3.55% | 17.7% |
| BUY @ 3rd Alert (retest2) | 41 | 9 | 22.0% | 7 | 34 | 0 | 0.54% | 22.0% |
| SELL (all) | 75 | 49 | 65.3% | 5 | 50 | 20 | 1.60% | 120.0% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 4 | 0 | 0.44% | 1.7% |
| SELL @ 3rd Alert (retest2) | 71 | 45 | 63.4% | 5 | 46 | 20 | 1.67% | 118.3% |
| retest1 (combined) | 9 | 9 | 100.0% | 0 | 7 | 2 | 2.16% | 19.5% |
| retest2 (combined) | 112 | 54 | 48.2% | 12 | 80 | 20 | 1.25% | 140.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 325.00 | 320.53 | 320.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 326.65 | 321.75 | 320.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 320.00 | 321.61 | 321.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 320.00 | 321.61 | 321.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 320.00 | 321.61 | 321.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 320.00 | 321.61 | 321.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 328.60 | 323.01 | 321.79 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 11:15:00 | 321.15 | 321.85 | 321.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 14:15:00 | 317.50 | 320.61 | 321.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 320.15 | 317.92 | 319.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 320.15 | 317.92 | 319.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 320.15 | 317.92 | 319.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 319.45 | 318.03 | 318.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 318.40 | 318.10 | 318.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 317.45 | 318.10 | 318.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 321.00 | 318.68 | 319.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 321.00 | 318.68 | 319.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 321.15 | 319.18 | 319.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 321.25 | 319.18 | 319.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 11:15:00 | 323.50 | 320.04 | 319.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 12:15:00 | 324.00 | 320.83 | 320.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 319.65 | 320.63 | 320.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 14:15:00 | 319.65 | 320.63 | 320.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 319.65 | 320.63 | 320.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:45:00 | 319.50 | 320.63 | 320.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 317.00 | 319.90 | 319.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 323.05 | 319.90 | 319.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 320.90 | 320.05 | 319.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 15:15:00 | 320.20 | 323.59 | 323.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 320.20 | 323.59 | 323.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 319.25 | 322.72 | 323.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 14:15:00 | 319.75 | 318.06 | 319.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 14:15:00 | 319.75 | 318.06 | 319.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 319.75 | 318.06 | 319.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 319.75 | 318.06 | 319.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 320.00 | 318.45 | 319.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 318.50 | 318.45 | 319.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 317.00 | 318.16 | 319.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:15:00 | 315.75 | 318.16 | 319.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 10:15:00 | 319.00 | 314.85 | 314.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 319.00 | 314.85 | 314.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 321.80 | 316.24 | 315.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 313.00 | 316.76 | 316.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 313.00 | 316.76 | 316.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 313.00 | 316.76 | 316.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 309.00 | 316.76 | 316.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 295.10 | 312.43 | 314.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 290.10 | 307.96 | 312.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 299.15 | 296.21 | 301.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 13:15:00 | 299.15 | 296.21 | 301.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 299.15 | 296.21 | 301.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 299.15 | 296.21 | 301.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 303.05 | 297.58 | 301.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 302.15 | 297.58 | 301.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 302.50 | 298.56 | 301.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 312.80 | 298.56 | 301.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 312.20 | 301.29 | 302.87 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 305.40 | 304.06 | 303.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 308.65 | 304.98 | 304.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 09:15:00 | 346.40 | 346.58 | 340.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 15:15:00 | 353.00 | 346.41 | 342.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:45:00 | 353.95 | 348.52 | 344.37 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 14:00:00 | 352.50 | 350.13 | 346.53 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 09:15:00 | 370.12 | 355.19 | 349.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 11:15:00 | 370.65 | 359.96 | 353.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-20 11:15:00 | 364.35 | 364.68 | 359.45 | SL hit (close<ema200) qty=0.50 sl=364.68 alert=retest1 |

### Cycle 8 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 355.50 | 359.57 | 359.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 15:15:00 | 351.60 | 356.48 | 358.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 11:15:00 | 358.50 | 353.76 | 355.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 11:15:00 | 358.50 | 353.76 | 355.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 358.50 | 353.76 | 355.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:00:00 | 358.50 | 353.76 | 355.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 357.35 | 354.48 | 355.41 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 363.05 | 356.64 | 356.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 368.55 | 360.12 | 357.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 358.20 | 360.07 | 358.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 358.20 | 360.07 | 358.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 358.20 | 360.07 | 358.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 358.20 | 360.07 | 358.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 356.55 | 359.37 | 358.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 356.55 | 359.37 | 358.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 356.00 | 358.69 | 358.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 356.00 | 358.69 | 358.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 356.00 | 358.15 | 357.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 354.80 | 358.15 | 357.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 09:15:00 | 356.00 | 357.72 | 357.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 11:15:00 | 352.65 | 356.33 | 357.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 369.10 | 356.63 | 356.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 369.10 | 356.63 | 356.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 369.10 | 356.63 | 356.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 365.60 | 356.63 | 356.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 368.30 | 358.96 | 357.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 11:15:00 | 374.40 | 362.05 | 359.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 15:15:00 | 371.55 | 372.64 | 368.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 09:15:00 | 371.10 | 372.64 | 368.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 368.25 | 371.76 | 368.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:00:00 | 368.25 | 371.76 | 368.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 368.80 | 371.17 | 368.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 13:00:00 | 371.65 | 370.98 | 368.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:45:00 | 371.75 | 374.16 | 372.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:45:00 | 372.20 | 372.51 | 371.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 14:15:00 | 365.75 | 372.61 | 373.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 14:15:00 | 365.75 | 372.61 | 373.03 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 375.65 | 373.44 | 373.35 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 14:15:00 | 372.50 | 375.45 | 375.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 11:15:00 | 371.45 | 374.16 | 375.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 13:15:00 | 375.20 | 373.79 | 374.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 13:15:00 | 375.20 | 373.79 | 374.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 375.20 | 373.79 | 374.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:45:00 | 375.55 | 373.79 | 374.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 373.00 | 373.63 | 374.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:30:00 | 375.50 | 373.63 | 374.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 370.50 | 372.79 | 373.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 15:15:00 | 365.00 | 369.78 | 371.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 12:15:00 | 366.35 | 368.01 | 370.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 346.75 | 355.96 | 361.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 348.03 | 355.96 | 361.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-07-23 12:15:00 | 328.50 | 339.81 | 344.80 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 14:15:00 | 351.80 | 341.90 | 340.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 359.35 | 346.37 | 342.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 356.30 | 359.06 | 354.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 356.30 | 359.06 | 354.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 358.20 | 359.89 | 357.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 366.50 | 359.89 | 357.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 11:30:00 | 361.20 | 361.61 | 360.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 358.00 | 360.14 | 360.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 14:15:00 | 358.00 | 360.14 | 360.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 15:15:00 | 356.80 | 359.47 | 359.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 09:15:00 | 361.40 | 359.86 | 360.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 361.40 | 359.86 | 360.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 361.40 | 359.86 | 360.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 10:00:00 | 361.40 | 359.86 | 360.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 351.55 | 358.20 | 359.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 11:15:00 | 350.65 | 358.20 | 359.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 12:45:00 | 349.00 | 356.02 | 358.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 15:15:00 | 347.50 | 354.67 | 357.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 370.00 | 356.59 | 357.45 | SL hit (close>static) qty=1.00 sl=361.50 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 10:15:00 | 376.50 | 360.57 | 359.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 11:15:00 | 379.50 | 364.36 | 361.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 09:15:00 | 394.00 | 394.45 | 386.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 10:00:00 | 394.00 | 394.45 | 386.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 389.50 | 391.62 | 388.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 386.40 | 391.62 | 388.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 384.30 | 390.16 | 388.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 384.30 | 390.16 | 388.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 386.00 | 389.32 | 387.96 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 384.05 | 386.86 | 387.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 379.15 | 385.32 | 386.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 10:15:00 | 370.95 | 367.73 | 371.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 10:15:00 | 370.95 | 367.73 | 371.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 370.95 | 367.73 | 371.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 370.95 | 367.73 | 371.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 374.60 | 369.10 | 371.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 374.60 | 369.10 | 371.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 371.40 | 369.56 | 371.92 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 390.20 | 375.78 | 374.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 403.45 | 389.40 | 382.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 397.45 | 398.72 | 392.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 10:00:00 | 397.45 | 398.72 | 392.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 394.10 | 396.12 | 392.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:45:00 | 393.50 | 396.12 | 392.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 392.00 | 395.08 | 392.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 395.65 | 395.08 | 392.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:00:00 | 395.65 | 395.19 | 393.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:30:00 | 396.30 | 395.62 | 393.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 401.80 | 393.47 | 393.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-26 11:15:00 | 435.22 | 409.86 | 401.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 12:15:00 | 423.30 | 426.57 | 426.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 14:15:00 | 422.40 | 425.26 | 426.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 11:15:00 | 427.90 | 424.30 | 425.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 427.90 | 424.30 | 425.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 427.90 | 424.30 | 425.14 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 13:15:00 | 431.50 | 425.86 | 425.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 446.25 | 431.85 | 428.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 447.30 | 450.31 | 444.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 10:00:00 | 447.30 | 450.31 | 444.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 447.75 | 449.56 | 446.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 447.30 | 449.56 | 446.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 446.00 | 448.85 | 446.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 440.50 | 448.85 | 446.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 442.20 | 447.52 | 446.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 444.70 | 447.52 | 446.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 442.40 | 446.50 | 445.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 11:15:00 | 444.60 | 446.50 | 445.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 12:15:00 | 442.00 | 444.65 | 444.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 442.00 | 444.65 | 444.99 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 448.70 | 445.66 | 445.37 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 10:15:00 | 442.55 | 445.45 | 445.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 437.95 | 443.44 | 444.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 10:15:00 | 446.80 | 442.52 | 443.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 10:15:00 | 446.80 | 442.52 | 443.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 446.80 | 442.52 | 443.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 446.80 | 442.52 | 443.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 457.20 | 445.45 | 444.87 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 12:15:00 | 441.50 | 446.23 | 446.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 14:15:00 | 437.45 | 443.35 | 445.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 12:15:00 | 436.75 | 436.23 | 440.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 12:15:00 | 436.75 | 436.23 | 440.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 436.75 | 436.23 | 440.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:00:00 | 436.75 | 436.23 | 440.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 388.00 | 391.60 | 395.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 10:30:00 | 386.20 | 390.55 | 394.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 15:15:00 | 386.80 | 389.75 | 392.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:00:00 | 386.00 | 388.52 | 391.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 14:15:00 | 386.90 | 385.92 | 389.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 385.35 | 384.96 | 386.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 385.45 | 384.96 | 386.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 366.89 | 376.26 | 380.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 367.46 | 376.26 | 380.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 366.70 | 376.26 | 380.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 367.55 | 376.26 | 380.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 376.60 | 371.73 | 375.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-01 09:15:00 | 376.60 | 371.73 | 375.71 | SL hit (close>ema200) qty=0.50 sl=371.73 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 361.50 | 355.35 | 354.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 367.80 | 359.75 | 357.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 14:15:00 | 363.30 | 363.47 | 360.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:30:00 | 363.00 | 363.47 | 360.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 364.40 | 363.65 | 361.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 377.45 | 362.74 | 361.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 11:15:00 | 374.15 | 377.29 | 377.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 374.15 | 377.29 | 377.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 373.40 | 376.51 | 377.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 371.10 | 358.49 | 363.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 10:15:00 | 371.10 | 358.49 | 363.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 371.10 | 358.49 | 363.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:45:00 | 365.95 | 358.49 | 363.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 385.05 | 363.80 | 365.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 385.05 | 363.80 | 365.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 381.40 | 367.32 | 366.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 14:15:00 | 391.75 | 384.73 | 378.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 374.25 | 383.72 | 378.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 374.25 | 383.72 | 378.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 374.25 | 383.72 | 378.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 374.25 | 383.72 | 378.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 372.80 | 381.53 | 378.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 371.85 | 381.53 | 378.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 13:15:00 | 371.05 | 378.22 | 377.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:00:00 | 371.05 | 378.22 | 377.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 15:15:00 | 375.00 | 376.68 | 376.89 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 379.45 | 377.41 | 377.16 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 373.00 | 376.40 | 376.84 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 381.60 | 376.92 | 376.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 392.00 | 381.60 | 378.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 393.60 | 397.02 | 390.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 393.60 | 397.02 | 390.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 392.70 | 396.16 | 390.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 392.70 | 396.16 | 390.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 402.45 | 399.13 | 394.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:00:00 | 411.75 | 403.66 | 398.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:45:00 | 411.75 | 405.70 | 399.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 417.75 | 417.10 | 415.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 10:15:00 | 409.00 | 416.30 | 417.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 409.00 | 416.30 | 417.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 396.45 | 407.74 | 412.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 394.00 | 392.56 | 399.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 394.00 | 392.56 | 399.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 397.40 | 389.03 | 391.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 396.50 | 389.03 | 391.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 391.80 | 389.59 | 391.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 393.65 | 389.59 | 391.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 390.75 | 389.82 | 391.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:30:00 | 392.80 | 389.82 | 391.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 390.30 | 389.92 | 391.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 385.20 | 388.92 | 390.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 11:30:00 | 387.30 | 385.42 | 385.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 12:15:00 | 387.95 | 385.92 | 385.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 12:15:00 | 387.95 | 385.92 | 385.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 395.40 | 388.39 | 386.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 09:15:00 | 394.35 | 394.59 | 392.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 391.25 | 393.92 | 392.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 391.25 | 393.92 | 392.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 391.65 | 393.92 | 392.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 390.90 | 393.32 | 392.36 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 15:15:00 | 389.90 | 391.52 | 391.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 09:15:00 | 385.65 | 390.34 | 391.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 389.10 | 388.40 | 389.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 13:15:00 | 389.10 | 388.40 | 389.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 389.10 | 388.40 | 389.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:30:00 | 389.25 | 388.40 | 389.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 388.85 | 388.49 | 389.71 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 393.40 | 390.64 | 390.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 410.65 | 396.30 | 393.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 412.05 | 413.28 | 406.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 12:00:00 | 412.05 | 413.28 | 406.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 411.75 | 413.20 | 409.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 412.60 | 413.20 | 409.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 447.35 | 450.85 | 443.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:45:00 | 446.00 | 450.85 | 443.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 447.15 | 450.12 | 447.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 447.15 | 450.12 | 447.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 444.00 | 448.90 | 446.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 444.00 | 448.90 | 446.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 445.60 | 448.24 | 446.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 13:30:00 | 446.75 | 453.82 | 449.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 14:00:00 | 476.15 | 453.82 | 449.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-18 09:15:00 | 491.43 | 478.81 | 473.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 15:15:00 | 482.40 | 484.54 | 484.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 09:15:00 | 478.55 | 483.34 | 484.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 11:15:00 | 475.60 | 473.90 | 476.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 11:15:00 | 475.60 | 473.90 | 476.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 475.60 | 473.90 | 476.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 477.70 | 473.90 | 476.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 478.40 | 474.80 | 476.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 478.40 | 474.80 | 476.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 478.80 | 475.60 | 476.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:45:00 | 478.60 | 475.60 | 476.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 473.95 | 476.57 | 477.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 472.20 | 475.75 | 476.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:00:00 | 471.80 | 474.96 | 476.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 12:15:00 | 481.00 | 474.19 | 474.56 | SL hit (close>static) qty=1.00 sl=479.15 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 483.65 | 476.09 | 475.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 15:15:00 | 484.65 | 478.88 | 476.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 481.40 | 482.28 | 480.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 10:00:00 | 481.40 | 482.28 | 480.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 485.35 | 482.90 | 480.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:30:00 | 489.10 | 484.55 | 481.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 14:15:00 | 478.80 | 490.84 | 492.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 478.80 | 490.84 | 492.26 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 499.05 | 493.06 | 492.64 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 485.40 | 492.58 | 492.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 482.80 | 489.33 | 491.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 436.20 | 435.79 | 446.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 10:45:00 | 437.00 | 435.79 | 446.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 443.00 | 434.01 | 436.94 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 447.95 | 439.07 | 438.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 14:15:00 | 452.05 | 444.47 | 441.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 460.40 | 476.91 | 463.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 460.40 | 476.91 | 463.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 460.40 | 476.91 | 463.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 460.40 | 476.91 | 463.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 454.20 | 472.37 | 462.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 454.20 | 472.37 | 462.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 456.75 | 469.24 | 462.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:30:00 | 454.25 | 469.24 | 462.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 464.55 | 467.17 | 462.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:30:00 | 462.00 | 467.17 | 462.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 469.00 | 467.54 | 463.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:30:00 | 462.55 | 467.54 | 463.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 471.00 | 473.24 | 468.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:00:00 | 471.00 | 473.24 | 468.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 467.05 | 472.00 | 468.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:30:00 | 466.55 | 472.00 | 468.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 466.00 | 470.80 | 467.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:30:00 | 460.60 | 470.80 | 467.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 461.00 | 468.84 | 467.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 458.60 | 468.84 | 467.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 454.55 | 465.98 | 466.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 448.75 | 460.32 | 463.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 460.00 | 458.12 | 460.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 460.00 | 458.12 | 460.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 460.00 | 458.12 | 460.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 462.00 | 458.12 | 460.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 460.60 | 458.62 | 460.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 461.00 | 458.62 | 460.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 455.95 | 458.08 | 460.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 453.30 | 457.55 | 459.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 430.63 | 438.85 | 447.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 407.97 | 418.98 | 431.39 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 428.20 | 424.81 | 424.65 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 421.65 | 424.18 | 424.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 15:15:00 | 420.60 | 422.99 | 423.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 423.10 | 423.02 | 423.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 423.10 | 423.02 | 423.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 423.10 | 423.02 | 423.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:00:00 | 423.10 | 423.02 | 423.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 428.25 | 424.06 | 424.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 428.25 | 424.06 | 424.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 431.50 | 425.55 | 424.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 439.10 | 428.26 | 426.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 435.40 | 436.41 | 431.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 435.40 | 436.41 | 431.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 435.40 | 436.41 | 431.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 435.90 | 436.41 | 431.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 431.15 | 435.36 | 431.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 433.65 | 435.36 | 431.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 442.30 | 436.75 | 432.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:30:00 | 447.75 | 440.66 | 438.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 10:00:00 | 446.45 | 440.66 | 438.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 11:15:00 | 446.00 | 441.70 | 438.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:30:00 | 448.65 | 444.75 | 441.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 447.40 | 447.95 | 444.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 447.40 | 447.95 | 444.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 445.00 | 447.36 | 444.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 442.70 | 447.36 | 444.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 443.95 | 446.68 | 444.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 441.20 | 446.68 | 444.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 449.00 | 447.14 | 445.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-07 14:15:00 | 440.20 | 443.47 | 443.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 14:15:00 | 440.20 | 443.47 | 443.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 426.40 | 439.31 | 441.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 09:15:00 | 401.40 | 398.04 | 403.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 401.40 | 398.04 | 403.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 401.40 | 398.04 | 403.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:30:00 | 401.55 | 398.04 | 403.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 371.75 | 369.68 | 375.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 13:45:00 | 373.20 | 369.68 | 375.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 378.40 | 371.42 | 376.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:45:00 | 379.95 | 371.42 | 376.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 377.45 | 372.63 | 376.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 383.90 | 372.63 | 376.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 390.45 | 378.61 | 378.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 395.10 | 387.46 | 383.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 385.05 | 390.71 | 387.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 385.05 | 390.71 | 387.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 385.05 | 390.71 | 387.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 385.05 | 390.71 | 387.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 381.30 | 388.83 | 387.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 381.30 | 388.83 | 387.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 382.55 | 386.00 | 386.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 370.65 | 381.32 | 383.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 12:15:00 | 373.60 | 371.74 | 375.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 13:00:00 | 373.60 | 371.74 | 375.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 336.70 | 331.14 | 338.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 340.95 | 331.14 | 338.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 343.85 | 335.60 | 338.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 343.85 | 335.60 | 338.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 341.20 | 336.72 | 338.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 345.15 | 336.72 | 338.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 345.85 | 340.02 | 339.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 10:15:00 | 347.80 | 342.59 | 341.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 341.15 | 344.28 | 343.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 341.15 | 344.28 | 343.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 341.15 | 344.28 | 343.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 341.15 | 344.28 | 343.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 339.35 | 343.30 | 342.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 339.35 | 343.30 | 342.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 337.15 | 341.48 | 342.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 332.40 | 339.66 | 341.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 12:15:00 | 328.45 | 328.39 | 331.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 12:45:00 | 327.60 | 328.39 | 331.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 326.85 | 327.83 | 330.44 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 335.50 | 329.80 | 329.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 346.15 | 336.26 | 333.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 408.00 | 410.22 | 393.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 408.00 | 410.22 | 393.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 398.35 | 404.44 | 400.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:00:00 | 398.35 | 404.44 | 400.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 396.15 | 402.78 | 400.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:30:00 | 396.30 | 402.78 | 400.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 10:15:00 | 392.40 | 398.42 | 398.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 11:15:00 | 391.75 | 397.08 | 398.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 397.15 | 396.60 | 397.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 397.15 | 396.60 | 397.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 397.15 | 396.60 | 397.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 399.15 | 396.60 | 397.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 399.85 | 397.25 | 397.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:30:00 | 402.45 | 397.25 | 397.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 401.55 | 398.11 | 398.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 404.15 | 398.11 | 398.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 401.85 | 398.86 | 398.51 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 397.00 | 398.38 | 398.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 11:15:00 | 394.30 | 397.05 | 397.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 395.60 | 394.79 | 396.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 395.60 | 394.79 | 396.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 395.60 | 394.79 | 396.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 396.50 | 394.79 | 396.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 392.40 | 394.31 | 395.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:30:00 | 393.85 | 394.31 | 395.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 402.60 | 395.97 | 396.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 402.60 | 395.97 | 396.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 405.60 | 397.90 | 397.31 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 389.25 | 397.23 | 398.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 14:15:00 | 384.50 | 390.72 | 394.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 364.50 | 359.10 | 369.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 364.50 | 359.10 | 369.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 368.70 | 362.41 | 369.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 365.35 | 363.00 | 368.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 361.20 | 364.03 | 368.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 10:15:00 | 347.08 | 360.02 | 366.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 15:15:00 | 343.14 | 352.26 | 359.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 356.55 | 353.12 | 359.37 | SL hit (close>ema200) qty=0.50 sl=353.12 alert=retest2 |

### Cycle 59 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 375.15 | 361.88 | 361.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 385.65 | 366.64 | 363.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 383.00 | 383.36 | 377.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 383.00 | 383.36 | 377.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 396.50 | 399.62 | 395.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:30:00 | 394.40 | 397.39 | 395.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 386.05 | 395.12 | 394.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 384.90 | 395.12 | 394.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 387.00 | 393.49 | 393.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 14:15:00 | 384.20 | 389.85 | 391.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 12:15:00 | 389.50 | 387.83 | 389.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 12:15:00 | 389.50 | 387.83 | 389.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 389.50 | 387.83 | 389.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:30:00 | 388.60 | 387.83 | 389.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 386.25 | 387.51 | 389.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 381.15 | 386.62 | 388.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 11:15:00 | 362.09 | 377.36 | 383.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-29 11:15:00 | 371.45 | 370.77 | 374.26 | SL hit (close>ema200) qty=0.50 sl=370.77 alert=retest2 |

### Cycle 61 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 360.15 | 344.70 | 344.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 364.80 | 358.58 | 352.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 388.20 | 388.44 | 380.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 388.20 | 388.44 | 380.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 392.35 | 397.54 | 395.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 392.35 | 397.54 | 395.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 394.00 | 396.84 | 394.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 395.65 | 396.84 | 394.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 12:30:00 | 396.20 | 397.32 | 396.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 15:15:00 | 395.00 | 396.76 | 396.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 15:15:00 | 395.00 | 396.41 | 396.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 395.00 | 396.41 | 396.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 393.95 | 395.92 | 396.20 | Break + close below crossover candle low |

### Cycle 63 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 399.05 | 396.33 | 396.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 401.70 | 397.40 | 396.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 409.85 | 411.10 | 406.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:30:00 | 411.05 | 411.10 | 406.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 419.40 | 412.49 | 409.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 414.90 | 412.49 | 409.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 413.20 | 413.89 | 411.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 412.70 | 413.89 | 411.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 411.75 | 413.46 | 411.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 412.25 | 413.46 | 411.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 410.70 | 412.91 | 411.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 410.70 | 412.91 | 411.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 411.75 | 412.68 | 411.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 411.75 | 412.68 | 411.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 409.80 | 412.10 | 411.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 409.80 | 412.10 | 411.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 408.80 | 411.44 | 411.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:45:00 | 408.45 | 411.44 | 411.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 15:15:00 | 408.90 | 410.93 | 411.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 406.15 | 408.34 | 409.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 413.20 | 409.07 | 409.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 413.20 | 409.07 | 409.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 413.20 | 409.07 | 409.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 413.20 | 409.07 | 409.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 412.00 | 409.65 | 409.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:15:00 | 413.70 | 409.65 | 409.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 416.40 | 411.00 | 410.47 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 406.50 | 410.17 | 410.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 14:15:00 | 406.20 | 409.37 | 410.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 409.25 | 408.35 | 409.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 409.25 | 408.35 | 409.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 409.25 | 408.35 | 409.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 409.25 | 408.35 | 409.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 410.50 | 408.78 | 409.56 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 411.65 | 410.08 | 409.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 412.60 | 410.58 | 410.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 14:15:00 | 409.35 | 410.42 | 410.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 14:15:00 | 409.35 | 410.42 | 410.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 409.35 | 410.42 | 410.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 409.35 | 410.42 | 410.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 407.70 | 409.88 | 409.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 406.60 | 409.22 | 409.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 374.45 | 373.96 | 378.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:00:00 | 374.45 | 373.96 | 378.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 370.00 | 368.38 | 371.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 371.45 | 368.80 | 371.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 368.80 | 368.80 | 371.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 367.25 | 368.80 | 371.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 365.90 | 359.65 | 359.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 365.90 | 359.65 | 359.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 378.75 | 370.00 | 367.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 370.55 | 374.83 | 371.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 370.55 | 374.83 | 371.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 370.55 | 374.83 | 371.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 370.55 | 374.83 | 371.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 370.60 | 373.98 | 371.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 369.35 | 373.98 | 371.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 371.90 | 373.57 | 371.59 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 363.75 | 369.71 | 370.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 363.35 | 365.68 | 367.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 363.25 | 363.16 | 365.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 15:00:00 | 363.25 | 363.16 | 365.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 365.70 | 363.68 | 365.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 365.70 | 363.68 | 365.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 365.10 | 363.97 | 365.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:00:00 | 363.95 | 364.11 | 364.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:00:00 | 362.85 | 363.85 | 364.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:00:00 | 363.30 | 363.74 | 364.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 363.45 | 364.17 | 364.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 365.45 | 363.72 | 364.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 371.20 | 363.72 | 364.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 382.30 | 367.44 | 365.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 382.30 | 367.44 | 365.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 388.50 | 377.91 | 375.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 382.30 | 382.82 | 380.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 11:00:00 | 382.30 | 382.82 | 380.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 381.45 | 382.55 | 380.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 381.45 | 382.55 | 380.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 380.75 | 382.64 | 381.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 378.80 | 382.64 | 381.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 379.60 | 382.03 | 381.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 380.25 | 382.03 | 381.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 379.65 | 381.55 | 380.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:45:00 | 379.25 | 381.55 | 380.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 380.00 | 380.70 | 380.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:15:00 | 380.45 | 380.70 | 380.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 380.45 | 380.65 | 380.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 383.95 | 380.65 | 380.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 383.05 | 385.16 | 385.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 383.05 | 385.16 | 385.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 382.65 | 384.33 | 384.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 380.40 | 380.10 | 381.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 380.40 | 380.10 | 381.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 380.40 | 380.10 | 381.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 377.80 | 379.67 | 380.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 376.05 | 379.36 | 380.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 14:15:00 | 358.91 | 362.64 | 365.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 15:15:00 | 357.25 | 362.00 | 364.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 365.00 | 362.60 | 364.87 | SL hit (close>ema200) qty=0.50 sl=362.60 alert=retest2 |

### Cycle 73 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 371.90 | 366.55 | 365.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 373.50 | 367.94 | 366.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 373.00 | 374.90 | 371.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 373.00 | 374.90 | 371.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 373.00 | 374.90 | 371.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 376.00 | 374.53 | 372.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:15:00 | 375.05 | 374.53 | 372.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:00:00 | 377.45 | 379.61 | 378.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-07 09:15:00 | 413.60 | 401.50 | 394.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 388.85 | 396.71 | 396.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 12:15:00 | 385.45 | 391.15 | 393.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 375.25 | 374.63 | 378.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 11:15:00 | 373.50 | 374.55 | 378.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 15:15:00 | 371.70 | 373.90 | 376.85 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 09:30:00 | 372.40 | 373.68 | 376.21 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:30:00 | 372.70 | 373.39 | 375.85 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 374.80 | 373.81 | 374.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:30:00 | 370.70 | 372.37 | 373.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 369.65 | 372.33 | 373.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 370.95 | 369.05 | 369.99 | SL hit (close>ema400) qty=1.00 sl=369.99 alert=retest1 |

### Cycle 75 — BUY (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 09:15:00 | 377.20 | 370.90 | 370.50 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 370.50 | 371.88 | 371.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 366.65 | 370.40 | 371.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 11:15:00 | 368.75 | 367.94 | 369.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 11:15:00 | 368.75 | 367.94 | 369.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 368.75 | 367.94 | 369.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 368.75 | 367.94 | 369.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 368.50 | 368.05 | 369.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:15:00 | 370.50 | 368.05 | 369.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 370.80 | 368.60 | 369.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:15:00 | 371.15 | 368.60 | 369.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 371.50 | 369.18 | 369.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 372.65 | 369.18 | 369.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 372.50 | 370.05 | 369.77 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 368.35 | 369.98 | 370.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 366.40 | 368.34 | 369.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 367.00 | 366.91 | 367.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 11:30:00 | 367.35 | 366.91 | 367.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 367.20 | 366.97 | 367.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 367.20 | 366.97 | 367.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 364.30 | 366.44 | 367.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 363.95 | 366.01 | 367.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 363.25 | 365.24 | 366.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 368.20 | 364.02 | 364.95 | SL hit (close>static) qty=1.00 sl=367.70 alert=retest2 |

### Cycle 79 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 366.55 | 365.55 | 365.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 368.55 | 366.81 | 366.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 367.95 | 368.98 | 367.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 367.95 | 368.98 | 367.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 367.95 | 368.98 | 367.98 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 367.20 | 368.15 | 368.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 12:15:00 | 366.05 | 367.29 | 367.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 367.95 | 366.94 | 367.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 367.95 | 366.94 | 367.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 367.95 | 366.94 | 367.39 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 13:15:00 | 369.95 | 367.95 | 367.75 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 366.85 | 368.01 | 368.04 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 368.70 | 368.15 | 368.10 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 366.80 | 367.94 | 368.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 362.00 | 365.71 | 366.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 358.90 | 358.89 | 361.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:45:00 | 360.00 | 358.89 | 361.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 359.05 | 358.90 | 361.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 359.70 | 358.90 | 361.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 360.50 | 359.22 | 361.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 356.00 | 359.22 | 361.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 338.20 | 347.23 | 352.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-30 13:15:00 | 320.40 | 325.75 | 332.23 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 85 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 333.15 | 330.60 | 330.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 339.50 | 332.74 | 331.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 348.45 | 348.61 | 343.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:00:00 | 348.45 | 348.61 | 343.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 342.85 | 346.72 | 343.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 342.85 | 346.72 | 343.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 339.65 | 345.30 | 343.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 339.65 | 345.30 | 343.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 338.00 | 343.84 | 342.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 338.75 | 343.84 | 342.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 337.05 | 341.34 | 341.88 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 345.55 | 342.04 | 341.91 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 340.05 | 341.64 | 341.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 338.80 | 341.07 | 341.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 285.00 | 284.21 | 293.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 11:15:00 | 295.90 | 286.31 | 292.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 295.90 | 286.31 | 292.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 295.90 | 286.31 | 292.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 308.45 | 290.74 | 294.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 308.45 | 290.74 | 294.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 14:15:00 | 323.35 | 301.50 | 298.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 335.90 | 313.94 | 305.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 15:15:00 | 321.80 | 322.80 | 314.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:15:00 | 329.70 | 322.80 | 314.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 322.45 | 324.22 | 322.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 322.45 | 324.22 | 322.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 321.20 | 323.61 | 321.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 320.75 | 323.61 | 321.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 321.65 | 323.22 | 321.93 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 317.70 | 320.79 | 321.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 313.70 | 319.37 | 320.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 309.15 | 308.99 | 311.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 309.15 | 308.99 | 311.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 322.20 | 311.55 | 312.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 324.65 | 311.55 | 312.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 320.20 | 313.28 | 312.91 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 313.20 | 314.38 | 314.48 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 320.20 | 315.54 | 315.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 15:15:00 | 323.00 | 319.28 | 317.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 11:15:00 | 320.45 | 320.56 | 318.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 12:00:00 | 320.45 | 320.56 | 318.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 317.35 | 319.78 | 318.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 314.45 | 319.78 | 318.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 313.85 | 318.60 | 318.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:15:00 | 313.40 | 318.60 | 318.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 313.85 | 317.65 | 318.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 311.60 | 314.78 | 316.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 315.10 | 313.84 | 315.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 14:15:00 | 315.10 | 313.84 | 315.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 315.10 | 313.84 | 315.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:45:00 | 315.45 | 313.84 | 315.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 313.30 | 313.73 | 314.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 313.70 | 313.73 | 314.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 313.80 | 313.75 | 314.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:00:00 | 309.80 | 313.62 | 314.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 330.70 | 314.71 | 314.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 330.70 | 314.71 | 314.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 12:15:00 | 337.50 | 322.46 | 318.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 329.25 | 329.82 | 324.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 11:00:00 | 329.25 | 329.82 | 324.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 323.75 | 327.75 | 324.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 323.75 | 327.75 | 324.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 323.25 | 326.85 | 324.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 323.25 | 326.85 | 324.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 322.10 | 325.90 | 324.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 320.25 | 324.92 | 323.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 329.00 | 325.73 | 324.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:15:00 | 331.20 | 325.73 | 324.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 14:15:00 | 324.90 | 325.77 | 325.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 324.90 | 325.77 | 325.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 319.40 | 324.42 | 325.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 14:15:00 | 324.75 | 322.68 | 323.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 324.75 | 322.68 | 323.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 324.75 | 322.68 | 323.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:45:00 | 324.15 | 322.68 | 323.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 324.80 | 323.11 | 323.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 324.00 | 323.11 | 323.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 322.25 | 322.93 | 323.76 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 326.55 | 324.37 | 324.14 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 321.35 | 324.10 | 324.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 318.90 | 322.47 | 323.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 316.45 | 314.41 | 316.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 316.45 | 314.41 | 316.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 316.45 | 314.41 | 316.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 316.45 | 314.41 | 316.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 314.50 | 314.43 | 316.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 315.15 | 314.43 | 316.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 314.00 | 313.87 | 315.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 15:15:00 | 313.00 | 313.87 | 315.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 317.30 | 314.42 | 315.22 | SL hit (close>static) qty=1.00 sl=315.85 alert=retest2 |

### Cycle 99 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 320.45 | 316.65 | 316.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 331.65 | 321.81 | 319.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 327.50 | 327.79 | 324.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:30:00 | 326.60 | 327.79 | 324.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 325.45 | 326.38 | 324.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 326.25 | 326.38 | 324.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 326.35 | 327.50 | 326.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 322.50 | 325.45 | 325.45 | SL hit (close<static) qty=1.00 sl=323.15 alert=retest2 |

### Cycle 100 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 322.05 | 324.77 | 325.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 320.60 | 323.94 | 324.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 326.00 | 323.71 | 324.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 14:15:00 | 326.00 | 323.71 | 324.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 326.00 | 323.71 | 324.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 326.00 | 323.71 | 324.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 326.35 | 324.24 | 324.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 324.90 | 324.24 | 324.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 337.35 | 326.86 | 325.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 337.35 | 326.86 | 325.65 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 324.60 | 329.43 | 329.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 321.30 | 327.81 | 328.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 319.50 | 317.39 | 321.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:00:00 | 319.50 | 317.39 | 321.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 320.15 | 318.34 | 320.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 320.90 | 318.34 | 320.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 322.50 | 319.17 | 321.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 319.80 | 319.32 | 320.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 320.05 | 319.35 | 320.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:00:00 | 320.20 | 319.10 | 319.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 320.60 | 320.16 | 320.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 320.60 | 320.16 | 320.11 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 317.30 | 319.64 | 319.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 12:15:00 | 314.95 | 318.25 | 319.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 15:15:00 | 317.65 | 317.34 | 318.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 15:15:00 | 317.65 | 317.34 | 318.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 317.65 | 317.34 | 318.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 315.80 | 317.34 | 318.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 315.50 | 314.42 | 314.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 334.50 | 318.61 | 316.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 334.50 | 318.61 | 316.65 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 322.50 | 323.78 | 323.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 320.25 | 322.57 | 323.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 318.85 | 316.66 | 318.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 318.85 | 316.66 | 318.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 318.85 | 316.66 | 318.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 318.85 | 316.66 | 318.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 316.55 | 316.64 | 318.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 320.00 | 316.64 | 318.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 317.20 | 316.76 | 318.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 318.25 | 316.76 | 318.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 318.85 | 317.18 | 318.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 319.25 | 317.18 | 318.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 318.65 | 317.47 | 318.22 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 320.85 | 318.85 | 318.73 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 317.00 | 318.58 | 318.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 316.60 | 318.04 | 318.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 316.70 | 316.49 | 317.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 14:15:00 | 316.70 | 316.49 | 317.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 316.70 | 316.49 | 317.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 317.05 | 316.49 | 317.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 315.30 | 316.34 | 317.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:15:00 | 314.05 | 315.98 | 316.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:00:00 | 314.10 | 315.17 | 316.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 313.25 | 314.79 | 315.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 313.00 | 313.80 | 314.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 312.15 | 313.47 | 314.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 311.15 | 312.94 | 314.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 298.35 | 304.29 | 307.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 298.39 | 304.29 | 307.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 297.59 | 304.29 | 307.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 297.35 | 304.29 | 307.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 295.59 | 302.66 | 306.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 301.70 | 298.41 | 301.42 | SL hit (close>ema200) qty=0.50 sl=298.41 alert=retest2 |

### Cycle 109 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 309.05 | 302.23 | 301.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 311.75 | 306.34 | 304.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 14:15:00 | 305.60 | 306.19 | 304.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 15:00:00 | 305.60 | 306.19 | 304.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 303.50 | 305.65 | 304.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:45:00 | 306.05 | 305.95 | 304.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 14:15:00 | 302.00 | 304.28 | 304.13 | SL hit (close<static) qty=1.00 sl=302.60 alert=retest2 |

### Cycle 110 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 302.60 | 303.94 | 303.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 299.75 | 303.10 | 303.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 288.40 | 287.88 | 291.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 293.85 | 287.88 | 291.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 292.45 | 288.79 | 291.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 293.70 | 288.79 | 291.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 289.65 | 288.96 | 291.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 289.30 | 288.96 | 291.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 15:15:00 | 274.83 | 279.52 | 283.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-28 14:15:00 | 260.37 | 266.33 | 271.95 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 111 — BUY (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 15:15:00 | 261.85 | 261.32 | 261.32 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 254.45 | 259.95 | 260.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 249.85 | 257.93 | 259.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 254.75 | 254.67 | 257.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 254.75 | 254.67 | 257.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 272.65 | 258.38 | 258.56 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 273.60 | 261.42 | 259.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 280.15 | 264.52 | 261.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 293.40 | 295.09 | 285.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:45:00 | 292.50 | 295.09 | 285.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 294.95 | 293.76 | 289.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:45:00 | 297.50 | 294.24 | 290.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:00:00 | 297.50 | 295.28 | 291.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 299.00 | 295.37 | 292.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 284.80 | 292.80 | 293.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 284.80 | 292.80 | 293.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 10:15:00 | 284.00 | 291.04 | 292.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 273.10 | 272.59 | 276.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:30:00 | 273.00 | 272.59 | 276.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 279.55 | 273.98 | 276.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 279.55 | 273.98 | 276.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 285.15 | 276.22 | 277.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 285.15 | 276.22 | 277.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 285.80 | 279.58 | 278.81 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 13:15:00 | 278.65 | 279.98 | 280.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 277.40 | 278.80 | 279.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 277.15 | 272.67 | 274.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 277.15 | 272.67 | 274.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 277.15 | 272.67 | 274.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:15:00 | 291.40 | 272.67 | 274.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 293.85 | 276.90 | 276.49 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 279.10 | 283.31 | 283.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 278.15 | 280.83 | 281.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 258.20 | 257.80 | 262.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 258.20 | 257.80 | 262.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 259.80 | 258.26 | 261.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 262.90 | 258.26 | 261.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 249.00 | 248.32 | 252.40 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 259.70 | 253.11 | 253.04 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 248.15 | 253.21 | 253.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 246.50 | 251.41 | 252.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 251.95 | 251.24 | 252.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 251.95 | 251.24 | 252.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 251.95 | 251.24 | 252.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 252.75 | 251.24 | 252.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 251.85 | 251.36 | 252.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:15:00 | 250.70 | 251.45 | 252.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 247.85 | 251.48 | 252.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 15:15:00 | 246.00 | 244.78 | 244.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 15:15:00 | 246.00 | 244.78 | 244.72 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 241.55 | 244.13 | 244.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 240.75 | 242.56 | 243.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 247.30 | 231.57 | 234.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 247.30 | 231.57 | 234.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 247.30 | 231.57 | 234.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:00:00 | 247.30 | 231.57 | 234.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 246.75 | 234.60 | 235.81 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 249.60 | 237.60 | 237.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 254.60 | 241.00 | 238.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 261.00 | 263.08 | 255.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 254.80 | 263.08 | 255.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 256.00 | 261.66 | 255.30 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 242.25 | 251.32 | 252.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 237.80 | 244.29 | 248.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 253.50 | 243.75 | 246.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 253.50 | 243.75 | 246.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 253.50 | 243.75 | 246.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 253.50 | 243.75 | 246.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 260.96 | 247.19 | 248.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 260.96 | 247.19 | 248.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 260.30 | 249.81 | 249.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 268.50 | 255.43 | 252.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 289.94 | 293.44 | 285.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 289.32 | 293.44 | 285.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 289.10 | 291.76 | 286.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 291.04 | 291.76 | 286.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 292.83 | 291.14 | 286.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 289.50 | 291.02 | 288.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 282.24 | 286.95 | 287.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 282.24 | 286.95 | 287.03 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 291.84 | 287.26 | 286.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 292.70 | 288.35 | 287.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 10:15:00 | 296.99 | 297.61 | 294.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:30:00 | 298.00 | 297.61 | 294.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 293.05 | 296.48 | 295.33 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 292.50 | 294.80 | 294.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 291.75 | 293.89 | 294.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 295.66 | 294.24 | 294.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 295.66 | 294.24 | 294.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 295.66 | 294.24 | 294.50 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 296.50 | 294.69 | 294.68 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 292.73 | 294.46 | 294.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 290.26 | 293.36 | 294.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 15:15:00 | 292.82 | 292.04 | 292.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 15:15:00 | 292.82 | 292.04 | 292.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 292.82 | 292.04 | 292.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 286.00 | 292.04 | 292.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 281.85 | 279.08 | 278.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 281.85 | 279.08 | 278.83 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 278.25 | 278.65 | 278.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 276.35 | 277.88 | 278.28 | Break + close below crossover candle low |

### Cycle 133 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 286.75 | 278.87 | 278.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 288.40 | 280.78 | 279.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 289.50 | 290.52 | 287.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:00:00 | 289.50 | 290.52 | 287.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:30:00 | 316.90 | 2024-05-14 11:15:00 | 324.05 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-05-13 14:45:00 | 317.85 | 2024-05-14 11:15:00 | 324.05 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-05-22 09:15:00 | 323.05 | 2024-05-24 15:15:00 | 320.20 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-05-22 09:45:00 | 320.90 | 2024-05-24 15:15:00 | 320.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-05-29 10:15:00 | 315.75 | 2024-06-03 10:15:00 | 319.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest1 | 2024-06-14 15:15:00 | 353.00 | 2024-06-19 09:15:00 | 370.12 | PARTIAL | 0.50 | 4.85% |
| BUY | retest1 | 2024-06-18 09:45:00 | 353.95 | 2024-06-19 11:15:00 | 370.65 | PARTIAL | 0.50 | 4.72% |
| BUY | retest1 | 2024-06-14 15:15:00 | 353.00 | 2024-06-20 11:15:00 | 364.35 | STOP_HIT | 0.50 | 3.22% |
| BUY | retest1 | 2024-06-18 09:45:00 | 353.95 | 2024-06-20 11:15:00 | 364.35 | STOP_HIT | 0.50 | 2.94% |
| BUY | retest1 | 2024-06-18 14:00:00 | 352.50 | 2024-06-21 11:15:00 | 359.60 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2024-07-03 13:00:00 | 371.65 | 2024-07-08 14:15:00 | 365.75 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-07-04 12:45:00 | 371.75 | 2024-07-08 14:15:00 | 365.75 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-07-05 09:45:00 | 372.20 | 2024-07-08 14:15:00 | 365.75 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-07-15 15:15:00 | 365.00 | 2024-07-19 09:15:00 | 346.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 12:15:00 | 366.35 | 2024-07-19 09:15:00 | 348.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 15:15:00 | 365.00 | 2024-07-23 12:15:00 | 328.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-16 12:15:00 | 366.35 | 2024-07-23 12:15:00 | 329.72 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-01 09:15:00 | 366.50 | 2024-08-02 14:15:00 | 358.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-08-02 11:30:00 | 361.20 | 2024-08-02 14:15:00 | 358.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-08-05 11:15:00 | 350.65 | 2024-08-06 09:15:00 | 370.00 | STOP_HIT | 1.00 | -5.52% |
| SELL | retest2 | 2024-08-05 12:45:00 | 349.00 | 2024-08-06 09:15:00 | 370.00 | STOP_HIT | 1.00 | -6.02% |
| SELL | retest2 | 2024-08-05 15:15:00 | 347.50 | 2024-08-06 09:15:00 | 370.00 | STOP_HIT | 1.00 | -6.47% |
| BUY | retest2 | 2024-08-22 09:15:00 | 395.65 | 2024-08-26 11:15:00 | 435.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-22 10:00:00 | 395.65 | 2024-08-26 11:15:00 | 435.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-22 10:30:00 | 396.30 | 2024-08-26 12:15:00 | 435.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-23 09:15:00 | 401.80 | 2024-08-28 09:15:00 | 441.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-30 09:15:00 | 429.35 | 2024-08-30 11:15:00 | 424.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-09-09 11:15:00 | 444.60 | 2024-09-09 12:15:00 | 442.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-09-24 10:30:00 | 386.20 | 2024-09-30 09:15:00 | 366.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-24 15:15:00 | 386.80 | 2024-09-30 09:15:00 | 367.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 10:00:00 | 386.00 | 2024-09-30 09:15:00 | 366.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 14:15:00 | 386.90 | 2024-09-30 09:15:00 | 367.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-24 10:30:00 | 386.20 | 2024-10-01 09:15:00 | 376.60 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2024-09-24 15:15:00 | 386.80 | 2024-10-01 09:15:00 | 376.60 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2024-09-25 10:00:00 | 386.00 | 2024-10-01 09:15:00 | 376.60 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2024-09-25 14:15:00 | 386.90 | 2024-10-01 09:15:00 | 376.60 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2024-10-03 09:45:00 | 367.35 | 2024-10-04 09:15:00 | 348.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:45:00 | 367.35 | 2024-10-04 15:15:00 | 359.00 | STOP_HIT | 0.50 | 2.27% |
| BUY | retest2 | 2024-10-14 09:15:00 | 377.45 | 2024-10-21 11:15:00 | 374.15 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-11-05 14:00:00 | 411.75 | 2024-11-12 10:15:00 | 409.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-11-05 14:45:00 | 411.75 | 2024-11-12 10:15:00 | 409.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-11-11 09:15:00 | 417.75 | 2024-11-12 10:15:00 | 409.00 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-11-19 15:00:00 | 385.20 | 2024-11-25 12:15:00 | 387.95 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-11-25 11:30:00 | 387.30 | 2024-11-25 12:15:00 | 387.95 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-12-12 13:30:00 | 446.75 | 2024-12-18 09:15:00 | 491.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-12 14:00:00 | 476.15 | 2024-12-23 15:15:00 | 482.40 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2024-12-30 12:15:00 | 472.20 | 2024-12-31 12:15:00 | 481.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-12-30 13:00:00 | 471.80 | 2024-12-31 12:15:00 | 481.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-01-02 11:30:00 | 489.10 | 2025-01-06 14:15:00 | 478.80 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-01-23 15:15:00 | 453.30 | 2025-01-27 09:15:00 | 430.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 453.30 | 2025-01-28 09:15:00 | 407.97 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-05 09:30:00 | 447.75 | 2025-02-07 14:15:00 | 440.20 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-02-05 10:00:00 | 446.45 | 2025-02-07 14:15:00 | 440.20 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-02-05 11:15:00 | 446.00 | 2025-02-07 14:15:00 | 440.20 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-02-06 09:30:00 | 448.65 | 2025-02-07 14:15:00 | 440.20 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-04-08 15:00:00 | 365.35 | 2025-04-09 10:15:00 | 347.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 361.20 | 2025-04-09 15:15:00 | 343.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-08 15:00:00 | 365.35 | 2025-04-11 09:15:00 | 356.55 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2025-04-09 09:15:00 | 361.20 | 2025-04-11 09:15:00 | 356.55 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2025-04-25 09:15:00 | 381.15 | 2025-04-25 11:15:00 | 362.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:15:00 | 381.15 | 2025-04-29 11:15:00 | 371.45 | STOP_HIT | 0.50 | 2.54% |
| BUY | retest2 | 2025-05-21 09:15:00 | 395.65 | 2025-05-22 15:15:00 | 395.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-05-22 12:30:00 | 396.20 | 2025-05-22 15:15:00 | 395.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-05-22 15:15:00 | 395.00 | 2025-05-22 15:15:00 | 395.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 367.25 | 2025-06-24 10:15:00 | 365.90 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-07-03 13:00:00 | 363.95 | 2025-07-07 09:15:00 | 382.30 | STOP_HIT | 1.00 | -5.04% |
| SELL | retest2 | 2025-07-03 14:00:00 | 362.85 | 2025-07-07 09:15:00 | 382.30 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2025-07-03 15:00:00 | 363.30 | 2025-07-07 09:15:00 | 382.30 | STOP_HIT | 1.00 | -5.23% |
| SELL | retest2 | 2025-07-04 11:30:00 | 363.45 | 2025-07-07 09:15:00 | 382.30 | STOP_HIT | 1.00 | -5.19% |
| BUY | retest2 | 2025-07-15 09:15:00 | 383.95 | 2025-07-18 11:15:00 | 383.05 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-07-22 14:15:00 | 377.80 | 2025-07-28 14:15:00 | 358.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 376.05 | 2025-07-28 15:15:00 | 357.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 14:15:00 | 377.80 | 2025-07-29 09:15:00 | 365.00 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2025-07-23 09:15:00 | 376.05 | 2025-07-29 09:15:00 | 365.00 | STOP_HIT | 0.50 | 2.94% |
| BUY | retest2 | 2025-07-31 11:45:00 | 376.00 | 2025-08-07 09:15:00 | 413.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 12:15:00 | 375.05 | 2025-08-07 09:15:00 | 412.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 11:00:00 | 377.45 | 2025-08-11 09:15:00 | 388.85 | STOP_HIT | 1.00 | 3.02% |
| SELL | retest1 | 2025-08-18 11:15:00 | 373.50 | 2025-08-25 10:15:00 | 370.95 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest1 | 2025-08-18 15:15:00 | 371.70 | 2025-08-25 10:15:00 | 370.95 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest1 | 2025-08-19 09:30:00 | 372.40 | 2025-08-25 10:15:00 | 370.95 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest1 | 2025-08-19 10:30:00 | 372.70 | 2025-08-25 10:15:00 | 370.95 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-08-20 14:30:00 | 370.70 | 2025-08-26 09:15:00 | 377.20 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-08-21 12:15:00 | 369.65 | 2025-08-26 09:15:00 | 377.20 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-09-09 09:15:00 | 363.95 | 2025-09-10 09:15:00 | 368.20 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-09 09:45:00 | 363.25 | 2025-09-10 09:15:00 | 368.20 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-09-24 09:15:00 | 356.00 | 2025-09-26 09:15:00 | 338.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 356.00 | 2025-09-30 13:15:00 | 320.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-11 13:00:00 | 309.80 | 2025-11-12 09:15:00 | 330.70 | STOP_HIT | 1.00 | -6.75% |
| BUY | retest2 | 2025-11-14 11:15:00 | 331.20 | 2025-11-17 14:15:00 | 324.90 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-11-26 15:15:00 | 313.00 | 2025-11-27 09:15:00 | 317.30 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-01 14:15:00 | 326.25 | 2025-12-02 15:15:00 | 322.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-02 12:15:00 | 326.35 | 2025-12-02 15:15:00 | 322.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-04 09:15:00 | 324.90 | 2025-12-04 09:15:00 | 337.35 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-12-10 10:45:00 | 319.80 | 2025-12-12 10:15:00 | 320.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-12-10 13:30:00 | 320.05 | 2025-12-12 10:15:00 | 320.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-12-11 14:00:00 | 320.20 | 2025-12-12 10:15:00 | 320.60 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-12-16 09:15:00 | 315.80 | 2025-12-19 09:15:00 | 334.50 | STOP_HIT | 1.00 | -5.92% |
| SELL | retest2 | 2025-12-18 15:15:00 | 315.50 | 2025-12-19 09:15:00 | 334.50 | STOP_HIT | 1.00 | -6.02% |
| SELL | retest2 | 2026-01-06 11:15:00 | 314.05 | 2026-01-09 15:15:00 | 298.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:00:00 | 314.10 | 2026-01-09 15:15:00 | 298.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 15:00:00 | 313.25 | 2026-01-09 15:15:00 | 297.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 313.00 | 2026-01-09 15:15:00 | 297.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 311.15 | 2026-01-12 09:15:00 | 295.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 11:15:00 | 314.05 | 2026-01-13 10:15:00 | 301.70 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2026-01-06 14:00:00 | 314.10 | 2026-01-13 10:15:00 | 301.70 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2026-01-06 15:00:00 | 313.25 | 2026-01-13 10:15:00 | 301.70 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2026-01-08 09:15:00 | 313.00 | 2026-01-13 10:15:00 | 301.70 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2026-01-08 10:45:00 | 311.15 | 2026-01-13 10:15:00 | 301.70 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2026-01-14 10:00:00 | 309.35 | 2026-01-14 10:15:00 | 309.05 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2026-01-16 09:45:00 | 306.05 | 2026-01-16 14:15:00 | 302.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-22 11:15:00 | 289.30 | 2026-01-23 15:15:00 | 274.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 11:15:00 | 289.30 | 2026-01-28 14:15:00 | 260.37 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-09 10:45:00 | 297.50 | 2026-02-11 09:15:00 | 284.80 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2026-02-09 13:00:00 | 297.50 | 2026-02-11 09:15:00 | 284.80 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2026-02-10 09:15:00 | 299.00 | 2026-02-11 09:15:00 | 284.80 | STOP_HIT | 1.00 | -4.75% |
| SELL | retest2 | 2026-03-12 14:15:00 | 250.70 | 2026-03-18 15:15:00 | 246.00 | STOP_HIT | 1.00 | 1.87% |
| SELL | retest2 | 2026-03-13 09:15:00 | 247.85 | 2026-03-18 15:15:00 | 246.00 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2026-04-10 09:15:00 | 291.04 | 2026-04-13 11:15:00 | 282.24 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-04-10 10:15:00 | 292.83 | 2026-04-13 11:15:00 | 282.24 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2026-04-10 15:15:00 | 289.50 | 2026-04-13 11:15:00 | 282.24 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-04-23 09:15:00 | 286.00 | 2026-05-05 09:15:00 | 281.85 | STOP_HIT | 1.00 | 1.45% |
