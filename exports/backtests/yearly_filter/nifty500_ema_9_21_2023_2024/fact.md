# Fertilisers and Chemicals Travancore Ltd. (FACT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 902.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 216 |
| ALERT1 | 139 |
| ALERT2 | 137 |
| ALERT2_SKIP | 74 |
| ALERT3 | 337 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 12 |
| ENTRY2 | 172 |
| PARTIAL | 46 |
| TARGET_HIT | 41 |
| STOP_HIT | 145 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 230 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 122 / 108
- **Target hits / Stop hits / Partials:** 41 / 143 / 46
- **Avg / median % per leg:** 2.19% / 0.15%
- **Sum % (uncompounded):** 504.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 88 | 35 | 39.8% | 19 | 64 | 5 | 1.25% | 110.1% |
| BUY @ 2nd Alert (retest1) | 16 | 11 | 68.8% | 3 | 8 | 5 | 3.25% | 52.0% |
| BUY @ 3rd Alert (retest2) | 72 | 24 | 33.3% | 16 | 56 | 0 | 0.81% | 58.1% |
| SELL (all) | 142 | 87 | 61.3% | 22 | 79 | 41 | 2.78% | 394.2% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.24% | 8.5% |
| SELL @ 3rd Alert (retest2) | 140 | 85 | 60.7% | 22 | 78 | 40 | 2.75% | 385.7% |
| retest1 (combined) | 18 | 13 | 72.2% | 3 | 9 | 6 | 3.36% | 60.5% |
| retest2 (combined) | 212 | 109 | 51.4% | 38 | 134 | 40 | 2.09% | 443.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 13:15:00 | 313.35 | 310.33 | 310.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 09:15:00 | 317.40 | 311.80 | 310.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 10:15:00 | 314.65 | 315.10 | 313.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 10:15:00 | 314.65 | 315.10 | 313.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 314.65 | 315.10 | 313.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 10:30:00 | 314.40 | 315.10 | 313.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 12:15:00 | 313.05 | 314.59 | 313.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 13:00:00 | 313.05 | 314.59 | 313.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 13:15:00 | 315.60 | 314.79 | 313.79 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 305.55 | 313.15 | 313.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 09:15:00 | 296.80 | 302.95 | 304.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 09:15:00 | 299.40 | 298.52 | 301.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 299.40 | 298.52 | 301.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 299.40 | 298.52 | 301.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 10:30:00 | 297.10 | 298.22 | 299.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-29 11:15:00 | 306.15 | 299.81 | 300.26 | SL hit (close>static) qty=1.00 sl=304.75 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 12:15:00 | 304.55 | 300.76 | 300.65 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 09:15:00 | 298.30 | 301.03 | 301.36 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 10:15:00 | 303.20 | 301.09 | 300.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 12:15:00 | 303.55 | 301.82 | 301.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 15:15:00 | 302.00 | 302.04 | 301.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 15:15:00 | 302.00 | 302.04 | 301.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 302.00 | 302.04 | 301.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 302.95 | 302.04 | 301.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-06 14:15:00 | 333.25 | 316.32 | 311.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 12:15:00 | 322.00 | 325.22 | 325.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 14:15:00 | 319.70 | 323.56 | 324.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 10:15:00 | 323.65 | 322.36 | 323.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 10:15:00 | 323.65 | 322.36 | 323.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 323.65 | 322.36 | 323.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 11:00:00 | 323.65 | 322.36 | 323.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 323.25 | 322.54 | 323.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 12:15:00 | 321.45 | 322.54 | 323.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 14:00:00 | 320.35 | 322.03 | 323.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 09:15:00 | 329.15 | 323.10 | 323.31 | SL hit (close>static) qty=1.00 sl=326.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 329.75 | 324.43 | 323.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 10:15:00 | 358.80 | 334.17 | 329.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 09:15:00 | 380.40 | 385.65 | 371.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 09:45:00 | 379.20 | 385.65 | 371.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 375.00 | 378.54 | 373.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 09:15:00 | 379.00 | 378.54 | 373.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-20 10:15:00 | 416.90 | 388.51 | 381.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 13:15:00 | 434.30 | 444.62 | 444.78 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 11:15:00 | 448.50 | 444.21 | 444.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 13:15:00 | 458.00 | 447.53 | 445.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 14:15:00 | 477.30 | 479.26 | 473.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 14:15:00 | 477.30 | 479.26 | 473.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 477.30 | 479.26 | 473.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 15:00:00 | 477.30 | 479.26 | 473.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 472.55 | 477.23 | 473.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:00:00 | 472.55 | 477.23 | 473.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 472.95 | 476.38 | 473.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 469.90 | 476.38 | 473.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 469.40 | 474.98 | 473.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:00:00 | 469.40 | 474.98 | 473.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 468.25 | 473.64 | 472.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:30:00 | 467.90 | 473.64 | 472.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 15:15:00 | 467.25 | 471.55 | 471.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 450.20 | 467.28 | 469.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 480.85 | 462.43 | 464.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 480.85 | 462.43 | 464.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 480.85 | 462.43 | 464.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:45:00 | 487.00 | 462.43 | 464.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 479.55 | 465.85 | 465.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:45:00 | 480.90 | 465.85 | 465.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 11:15:00 | 476.40 | 467.96 | 466.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 12:15:00 | 481.40 | 470.65 | 468.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 10:15:00 | 472.15 | 472.89 | 470.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 10:15:00 | 472.15 | 472.89 | 470.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 472.15 | 472.89 | 470.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 09:15:00 | 504.65 | 470.62 | 470.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 12:15:00 | 484.60 | 488.44 | 488.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 484.60 | 488.44 | 488.44 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 498.00 | 488.24 | 488.10 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 11:15:00 | 486.80 | 488.96 | 489.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 13:15:00 | 485.15 | 486.55 | 487.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 477.80 | 473.64 | 477.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 477.80 | 473.64 | 477.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 477.80 | 473.64 | 477.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:30:00 | 481.30 | 473.64 | 477.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 478.45 | 474.60 | 477.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 13:15:00 | 475.15 | 475.34 | 477.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 10:00:00 | 475.00 | 476.15 | 477.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 11:15:00 | 475.55 | 476.21 | 477.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 15:15:00 | 475.00 | 475.98 | 476.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 15:15:00 | 475.00 | 475.79 | 476.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:15:00 | 476.40 | 475.79 | 476.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 477.65 | 476.16 | 476.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:45:00 | 479.10 | 476.16 | 476.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 476.55 | 476.24 | 476.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:30:00 | 479.40 | 476.24 | 476.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 476.00 | 476.19 | 476.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 12:00:00 | 476.00 | 476.19 | 476.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-01 09:15:00 | 497.00 | 479.53 | 477.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 09:15:00 | 497.00 | 479.53 | 477.84 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 14:15:00 | 479.50 | 483.38 | 483.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 09:15:00 | 466.80 | 479.21 | 481.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 473.40 | 472.69 | 476.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 15:00:00 | 473.40 | 472.69 | 476.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 478.55 | 473.67 | 476.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:30:00 | 477.65 | 473.67 | 476.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 477.90 | 474.52 | 476.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 11:00:00 | 477.90 | 474.52 | 476.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 476.85 | 474.98 | 476.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 12:00:00 | 476.85 | 474.98 | 476.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 12:15:00 | 476.85 | 475.36 | 476.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:30:00 | 474.40 | 475.35 | 476.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 14:30:00 | 475.55 | 475.42 | 476.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 15:00:00 | 475.70 | 475.42 | 476.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 09:30:00 | 475.00 | 474.67 | 476.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 479.35 | 475.20 | 476.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-07 11:15:00 | 479.35 | 475.20 | 476.00 | SL hit (close>static) qty=1.00 sl=477.90 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 459.85 | 445.52 | 443.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 10:15:00 | 461.55 | 451.05 | 446.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 09:15:00 | 447.50 | 456.10 | 451.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 09:15:00 | 447.50 | 456.10 | 451.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 447.50 | 456.10 | 451.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:00:00 | 447.50 | 456.10 | 451.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 451.75 | 455.23 | 451.86 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 09:15:00 | 443.35 | 450.67 | 450.80 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 10:15:00 | 454.50 | 449.75 | 449.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 09:15:00 | 462.85 | 454.57 | 452.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 15:15:00 | 465.05 | 466.77 | 462.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-25 09:15:00 | 463.75 | 466.77 | 462.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 461.00 | 465.62 | 462.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:30:00 | 462.50 | 465.62 | 462.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 457.60 | 464.01 | 462.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:30:00 | 458.00 | 464.01 | 462.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 460.00 | 461.48 | 461.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 14:45:00 | 458.90 | 461.48 | 461.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2023-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 15:15:00 | 458.90 | 460.97 | 461.17 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 11:15:00 | 467.50 | 461.24 | 461.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 09:15:00 | 471.95 | 464.40 | 463.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 15:15:00 | 466.45 | 468.10 | 466.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 15:15:00 | 466.45 | 468.10 | 466.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 466.45 | 468.10 | 466.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 11:30:00 | 470.70 | 468.19 | 466.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 09:15:00 | 479.80 | 468.14 | 467.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 12:15:00 | 457.45 | 466.76 | 466.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 12:15:00 | 457.45 | 466.76 | 466.92 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 10:15:00 | 485.85 | 468.10 | 466.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 11:15:00 | 495.50 | 473.58 | 469.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 12:15:00 | 518.10 | 520.50 | 501.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-07 13:00:00 | 518.10 | 520.50 | 501.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 511.75 | 516.98 | 511.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 09:45:00 | 510.55 | 516.98 | 511.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 10:15:00 | 514.70 | 516.53 | 511.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 11:45:00 | 549.15 | 522.02 | 514.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 505.90 | 527.32 | 521.31 | SL hit (close<static) qty=1.00 sl=511.75 alert=retest2 |

### Cycle 24 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 503.20 | 517.49 | 517.92 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 521.60 | 514.06 | 513.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 528.20 | 518.63 | 516.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 13:15:00 | 546.50 | 548.79 | 538.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-18 13:45:00 | 546.30 | 548.79 | 538.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 529.85 | 544.15 | 539.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:00:00 | 529.85 | 544.15 | 539.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 529.00 | 541.12 | 538.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:30:00 | 531.60 | 541.12 | 538.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 13:15:00 | 522.80 | 534.17 | 535.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 517.25 | 525.32 | 529.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 10:15:00 | 526.15 | 525.49 | 529.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 10:15:00 | 526.15 | 525.49 | 529.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 526.15 | 525.49 | 529.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 11:00:00 | 526.15 | 525.49 | 529.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 534.60 | 527.70 | 529.48 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 15:15:00 | 535.75 | 531.30 | 530.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 10:15:00 | 544.00 | 537.47 | 534.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 13:15:00 | 536.50 | 537.88 | 535.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 13:15:00 | 536.50 | 537.88 | 535.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 13:15:00 | 536.50 | 537.88 | 535.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 14:00:00 | 536.50 | 537.88 | 535.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 538.85 | 538.07 | 535.96 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 11:15:00 | 530.15 | 534.33 | 534.70 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 547.65 | 535.53 | 534.89 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 529.60 | 537.38 | 537.75 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 14:15:00 | 536.70 | 533.35 | 533.20 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 15:15:00 | 531.60 | 533.29 | 533.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 521.00 | 530.83 | 532.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 535.75 | 521.19 | 525.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 535.75 | 521.19 | 525.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 535.75 | 521.19 | 525.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:45:00 | 537.05 | 521.19 | 525.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 535.70 | 524.09 | 526.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:30:00 | 537.05 | 524.09 | 526.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 533.50 | 527.38 | 527.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 539.00 | 531.14 | 529.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 12:15:00 | 536.05 | 537.19 | 534.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 12:15:00 | 536.05 | 537.19 | 534.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 12:15:00 | 536.05 | 537.19 | 534.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 12:45:00 | 535.05 | 537.19 | 534.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 13:15:00 | 535.95 | 536.94 | 534.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 14:15:00 | 537.70 | 536.94 | 534.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 09:15:00 | 539.00 | 536.57 | 534.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 11:45:00 | 544.20 | 537.44 | 535.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-10-16 10:15:00 | 591.47 | 559.17 | 547.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 11:15:00 | 682.60 | 700.18 | 701.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 667.50 | 693.64 | 698.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 13:15:00 | 664.40 | 664.39 | 677.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 14:00:00 | 664.40 | 664.39 | 677.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 14:15:00 | 672.75 | 666.06 | 677.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 14:30:00 | 683.25 | 666.06 | 677.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 655.60 | 653.47 | 663.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 14:45:00 | 655.00 | 653.47 | 663.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 665.95 | 656.25 | 663.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:30:00 | 672.60 | 656.25 | 663.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 669.60 | 658.92 | 663.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 669.60 | 658.92 | 663.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 673.55 | 666.57 | 666.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 13:15:00 | 687.80 | 675.89 | 671.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 12:15:00 | 746.25 | 748.42 | 726.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 12:45:00 | 749.95 | 748.42 | 726.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 726.25 | 748.38 | 738.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:00:00 | 726.25 | 748.38 | 738.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 724.25 | 743.55 | 737.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 15:00:00 | 724.25 | 743.55 | 737.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 10:15:00 | 732.05 | 737.15 | 735.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 12:15:00 | 740.40 | 735.57 | 735.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 12:15:00 | 731.20 | 734.70 | 734.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 12:15:00 | 731.20 | 734.70 | 734.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 13:15:00 | 727.15 | 733.19 | 734.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 11:15:00 | 730.20 | 729.85 | 731.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 11:15:00 | 730.20 | 729.85 | 731.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 11:15:00 | 730.20 | 729.85 | 731.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 12:00:00 | 730.20 | 729.85 | 731.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 729.35 | 729.70 | 731.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 13:45:00 | 733.25 | 729.70 | 731.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 715.80 | 721.39 | 726.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-07 10:45:00 | 714.60 | 721.39 | 726.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 727.25 | 716.21 | 720.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 09:30:00 | 730.25 | 716.21 | 720.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 728.55 | 718.68 | 721.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:15:00 | 730.50 | 718.68 | 721.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 13:15:00 | 739.85 | 726.11 | 724.46 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 721.10 | 727.30 | 727.58 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 14:15:00 | 729.15 | 726.57 | 726.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 737.00 | 728.81 | 727.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 15:15:00 | 729.90 | 732.27 | 730.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 15:15:00 | 729.90 | 732.27 | 730.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 15:15:00 | 729.90 | 732.27 | 730.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 09:15:00 | 727.60 | 732.27 | 730.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 728.00 | 731.42 | 729.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 09:30:00 | 730.50 | 731.42 | 729.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 728.60 | 730.85 | 729.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 12:15:00 | 741.70 | 730.63 | 729.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 14:00:00 | 730.90 | 730.38 | 729.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 14:45:00 | 731.25 | 730.11 | 729.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 09:15:00 | 736.75 | 729.89 | 729.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 727.90 | 731.19 | 730.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 11:00:00 | 727.90 | 731.19 | 730.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 11:15:00 | 727.60 | 730.47 | 730.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-17 12:15:00 | 716.65 | 727.71 | 728.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 12:15:00 | 716.65 | 727.71 | 728.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 10:15:00 | 711.00 | 720.32 | 724.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 717.00 | 716.52 | 720.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 717.00 | 716.52 | 720.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 717.00 | 716.52 | 720.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 10:15:00 | 717.40 | 716.52 | 720.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 715.85 | 716.39 | 720.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 12:15:00 | 713.80 | 716.05 | 719.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 09:45:00 | 712.75 | 703.15 | 708.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 10:00:00 | 712.70 | 704.71 | 705.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 10:15:00 | 722.00 | 708.17 | 707.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 10:15:00 | 722.00 | 708.17 | 707.22 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 13:15:00 | 709.30 | 712.21 | 712.51 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 12:15:00 | 714.60 | 712.61 | 712.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 13:15:00 | 734.45 | 716.98 | 714.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 10:15:00 | 743.15 | 743.65 | 734.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-05 11:00:00 | 743.15 | 743.65 | 734.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 741.25 | 743.17 | 734.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 738.55 | 743.17 | 734.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 15:15:00 | 737.10 | 740.76 | 736.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 09:15:00 | 745.45 | 740.76 | 736.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-07 09:15:00 | 820.00 | 789.83 | 768.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 14:15:00 | 758.80 | 770.64 | 772.15 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 12:15:00 | 789.50 | 775.18 | 773.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 13:15:00 | 807.00 | 783.78 | 779.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 14:15:00 | 794.00 | 795.26 | 789.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-13 15:00:00 | 794.00 | 795.26 | 789.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 787.35 | 792.84 | 789.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:00:00 | 787.35 | 792.84 | 789.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 784.90 | 791.25 | 788.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 11:15:00 | 791.50 | 791.25 | 788.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-14 14:15:00 | 779.00 | 786.96 | 787.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 14:15:00 | 779.00 | 786.96 | 787.44 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 846.90 | 797.83 | 792.23 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 780.55 | 801.52 | 803.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 765.50 | 794.32 | 800.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 13:15:00 | 777.45 | 777.09 | 786.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 13:15:00 | 777.45 | 777.09 | 786.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 13:15:00 | 777.45 | 777.09 | 786.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 13:30:00 | 775.35 | 777.09 | 786.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 786.50 | 778.97 | 786.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 14:45:00 | 787.95 | 778.97 | 786.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 787.20 | 780.62 | 786.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 799.15 | 780.62 | 786.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 811.25 | 786.75 | 789.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 811.25 | 786.75 | 789.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 807.00 | 790.80 | 790.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 11:15:00 | 831.15 | 807.66 | 803.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 13:15:00 | 808.80 | 810.32 | 805.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 14:00:00 | 808.80 | 810.32 | 805.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 811.05 | 810.46 | 805.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 14:30:00 | 806.80 | 810.46 | 805.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 808.50 | 810.07 | 806.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 814.15 | 810.07 | 806.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:45:00 | 812.30 | 809.14 | 806.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:15:00 | 828.05 | 806.41 | 805.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 10:15:00 | 814.25 | 819.60 | 819.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 10:15:00 | 814.25 | 819.60 | 819.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 13:15:00 | 807.25 | 815.47 | 817.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 814.55 | 810.77 | 813.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 814.55 | 810.77 | 813.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 814.55 | 810.77 | 813.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 11:45:00 | 810.15 | 812.38 | 813.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 12:45:00 | 809.75 | 811.37 | 812.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 11:15:00 | 813.55 | 812.27 | 812.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 11:15:00 | 813.55 | 812.27 | 812.22 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 12:15:00 | 811.45 | 812.10 | 812.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 13:15:00 | 810.05 | 811.69 | 811.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 11:15:00 | 808.85 | 805.62 | 808.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 11:15:00 | 808.85 | 805.62 | 808.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 11:15:00 | 808.85 | 805.62 | 808.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 11:45:00 | 810.25 | 805.62 | 808.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 12:15:00 | 808.00 | 806.10 | 808.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 12:45:00 | 807.35 | 806.10 | 808.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 13:15:00 | 804.25 | 805.73 | 808.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 13:30:00 | 806.40 | 805.73 | 808.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 806.40 | 804.10 | 806.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:45:00 | 808.95 | 804.10 | 806.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 805.60 | 804.40 | 806.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 11:15:00 | 803.35 | 804.40 | 806.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 12:45:00 | 803.40 | 802.93 | 805.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 15:15:00 | 803.00 | 804.01 | 805.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 11:15:00 | 804.05 | 804.23 | 805.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 805.50 | 804.48 | 805.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-17 11:30:00 | 803.25 | 804.48 | 805.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 12:15:00 | 803.00 | 804.19 | 805.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 14:15:00 | 797.80 | 804.51 | 805.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 818.50 | 807.57 | 806.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 09:15:00 | 818.50 | 807.57 | 806.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 10:15:00 | 863.45 | 828.35 | 820.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 832.00 | 844.08 | 832.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 14:15:00 | 832.00 | 844.08 | 832.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 832.00 | 844.08 | 832.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 15:15:00 | 834.00 | 844.08 | 832.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 834.00 | 842.06 | 832.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 09:15:00 | 835.60 | 842.06 | 832.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 10:45:00 | 837.75 | 840.58 | 833.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 11:45:00 | 836.40 | 838.85 | 833.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 12:15:00 | 835.95 | 838.85 | 833.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 833.70 | 837.82 | 833.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:15:00 | 832.70 | 837.82 | 833.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 827.35 | 835.73 | 832.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-23 13:15:00 | 827.35 | 835.73 | 832.67 | SL hit (close<static) qty=1.00 sl=828.10 alert=retest2 |

### Cycle 54 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 811.80 | 827.43 | 829.21 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 843.30 | 826.69 | 826.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 10:15:00 | 867.00 | 834.75 | 830.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 10:15:00 | 863.40 | 867.53 | 853.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-29 10:45:00 | 865.85 | 867.53 | 853.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 12:15:00 | 849.75 | 862.43 | 853.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 12:45:00 | 853.95 | 862.43 | 853.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 13:15:00 | 858.75 | 861.69 | 854.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 14:30:00 | 872.80 | 861.41 | 854.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 15:15:00 | 864.90 | 861.41 | 854.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-30 14:15:00 | 848.65 | 856.39 | 855.40 | SL hit (close<static) qty=1.00 sl=849.05 alert=retest2 |

### Cycle 56 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 845.00 | 854.11 | 854.46 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 860.00 | 855.29 | 854.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-01 09:15:00 | 873.50 | 862.13 | 858.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 11:15:00 | 853.65 | 861.86 | 859.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 11:15:00 | 853.65 | 861.86 | 859.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 853.65 | 861.86 | 859.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 853.65 | 861.86 | 859.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 849.85 | 859.46 | 858.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:30:00 | 849.20 | 859.46 | 858.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 13:15:00 | 841.55 | 855.87 | 857.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 09:15:00 | 833.95 | 846.23 | 851.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 10:15:00 | 805.25 | 803.26 | 816.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 802.85 | 803.89 | 811.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 802.85 | 803.89 | 811.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:15:00 | 795.60 | 803.69 | 810.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:15:00 | 755.82 | 782.16 | 791.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-12 12:15:00 | 768.05 | 764.34 | 773.52 | SL hit (close>ema200) qty=0.50 sl=764.34 alert=retest2 |

### Cycle 59 — BUY (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 12:15:00 | 778.15 | 766.83 | 765.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 12:15:00 | 783.50 | 776.55 | 771.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 13:15:00 | 840.05 | 842.21 | 828.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-21 13:45:00 | 843.40 | 842.21 | 828.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 827.80 | 838.78 | 830.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:30:00 | 827.15 | 838.78 | 830.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 827.20 | 836.46 | 830.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 12:00:00 | 832.00 | 835.57 | 830.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 13:00:00 | 829.75 | 834.41 | 830.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 13:45:00 | 829.75 | 833.48 | 830.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-23 09:15:00 | 801.90 | 825.74 | 827.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 09:15:00 | 801.90 | 825.74 | 827.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 13:15:00 | 796.80 | 809.79 | 818.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 14:15:00 | 720.70 | 717.95 | 730.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-01 15:00:00 | 720.70 | 717.95 | 730.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 740.00 | 722.91 | 730.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:30:00 | 744.60 | 722.91 | 730.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 737.50 | 725.83 | 731.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 738.65 | 725.83 | 731.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 720.80 | 720.57 | 724.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 11:00:00 | 718.60 | 720.18 | 724.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 11:45:00 | 718.00 | 719.77 | 723.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 13:00:00 | 718.05 | 719.43 | 723.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 14:45:00 | 718.20 | 718.62 | 722.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 718.95 | 705.05 | 710.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:30:00 | 719.70 | 705.05 | 710.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 715.40 | 707.12 | 711.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:30:00 | 715.80 | 707.12 | 711.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 716.55 | 711.70 | 712.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 13:45:00 | 717.40 | 711.70 | 712.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 15:15:00 | 708.00 | 711.45 | 712.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 09:15:00 | 702.85 | 711.45 | 712.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 14:15:00 | 682.67 | 696.72 | 703.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 15:15:00 | 682.10 | 694.30 | 701.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 15:15:00 | 682.15 | 694.30 | 701.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 15:15:00 | 682.29 | 694.30 | 701.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 667.71 | 689.37 | 698.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 09:15:00 | 646.74 | 657.92 | 675.82 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 09:15:00 | 669.50 | 652.93 | 650.98 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 638.60 | 652.04 | 652.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 10:15:00 | 624.10 | 646.45 | 649.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 15:15:00 | 643.00 | 642.56 | 646.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-19 09:15:00 | 654.30 | 642.56 | 646.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 646.90 | 643.43 | 646.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 15:15:00 | 637.95 | 644.26 | 645.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 09:30:00 | 639.35 | 636.29 | 639.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 12:00:00 | 640.10 | 638.49 | 639.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 15:15:00 | 643.30 | 640.85 | 640.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 15:15:00 | 643.30 | 640.85 | 640.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 654.45 | 643.57 | 641.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 651.80 | 652.81 | 648.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 651.80 | 652.81 | 648.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 651.80 | 652.81 | 648.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 648.00 | 652.81 | 648.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 11:15:00 | 643.90 | 650.68 | 648.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 12:00:00 | 643.90 | 650.68 | 648.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 12:15:00 | 646.50 | 649.84 | 648.08 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 14:15:00 | 639.00 | 646.87 | 646.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 12:15:00 | 634.45 | 641.19 | 643.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 645.15 | 637.32 | 640.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 645.15 | 637.32 | 640.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 645.15 | 637.32 | 640.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 09:45:00 | 640.40 | 637.32 | 640.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 10:15:00 | 648.95 | 639.65 | 641.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 11:00:00 | 648.95 | 639.65 | 641.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 11:15:00 | 649.15 | 641.55 | 641.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 12:00:00 | 649.15 | 641.55 | 641.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 12:15:00 | 652.30 | 643.70 | 642.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 13:15:00 | 661.60 | 647.28 | 644.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 15:15:00 | 696.05 | 696.94 | 688.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-05 09:15:00 | 689.80 | 696.94 | 688.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 693.50 | 696.25 | 688.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:30:00 | 690.50 | 696.25 | 688.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 688.60 | 696.60 | 692.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:00:00 | 688.60 | 696.60 | 692.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 690.00 | 695.28 | 692.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 11:15:00 | 691.70 | 695.28 | 692.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 12:00:00 | 692.90 | 694.80 | 692.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 12:45:00 | 692.00 | 693.89 | 692.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-08 15:15:00 | 683.90 | 690.43 | 691.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 15:15:00 | 683.90 | 690.43 | 691.10 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 12:15:00 | 694.65 | 691.19 | 691.12 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 685.10 | 689.97 | 690.58 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 11:15:00 | 702.00 | 691.39 | 690.90 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 09:15:00 | 682.00 | 691.00 | 691.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 656.90 | 676.84 | 683.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 662.70 | 660.55 | 669.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 09:30:00 | 661.00 | 660.55 | 669.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 670.05 | 662.45 | 669.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:30:00 | 656.35 | 662.26 | 665.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 14:15:00 | 664.00 | 657.01 | 656.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 664.00 | 657.01 | 656.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 679.95 | 662.80 | 659.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 09:15:00 | 713.95 | 715.13 | 704.20 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 11:00:00 | 732.40 | 718.58 | 706.76 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 09:15:00 | 726.80 | 719.20 | 711.74 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 717.90 | 724.75 | 719.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-29 15:15:00 | 717.90 | 724.75 | 719.37 | SL hit (close<ema400) qty=1.00 sl=719.37 alert=retest1 |

### Cycle 72 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 713.95 | 718.16 | 718.41 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 10:15:00 | 720.55 | 718.64 | 718.61 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 11:15:00 | 716.60 | 718.23 | 718.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 14:15:00 | 714.20 | 717.08 | 717.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 09:15:00 | 717.25 | 716.94 | 717.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 09:15:00 | 717.25 | 716.94 | 717.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 717.25 | 716.94 | 717.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 11:45:00 | 714.60 | 716.58 | 717.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 12:45:00 | 714.90 | 716.26 | 717.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 09:15:00 | 678.87 | 696.01 | 703.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 09:15:00 | 679.15 | 696.01 | 703.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 09:15:00 | 689.50 | 684.05 | 692.06 | SL hit (close>ema200) qty=0.50 sl=684.05 alert=retest2 |

### Cycle 75 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 685.75 | 661.89 | 658.71 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 15:15:00 | 690.50 | 694.70 | 695.13 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 732.50 | 702.26 | 698.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 10:15:00 | 736.70 | 709.15 | 702.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 15:15:00 | 715.00 | 717.58 | 709.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 716.15 | 717.29 | 710.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 716.15 | 717.29 | 710.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 716.15 | 717.29 | 710.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 711.95 | 715.59 | 712.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 15:00:00 | 711.95 | 715.59 | 712.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 713.00 | 715.07 | 712.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:15:00 | 709.00 | 715.07 | 712.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 698.30 | 711.72 | 710.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:30:00 | 703.20 | 711.72 | 710.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 10:15:00 | 703.30 | 710.03 | 710.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 15:15:00 | 696.45 | 703.58 | 706.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 698.70 | 697.95 | 702.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 12:15:00 | 698.70 | 697.95 | 702.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 698.70 | 697.95 | 702.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:45:00 | 697.65 | 697.95 | 702.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 698.20 | 698.00 | 702.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:30:00 | 701.70 | 698.00 | 702.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 692.85 | 694.12 | 698.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:45:00 | 695.05 | 694.12 | 698.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 677.10 | 686.00 | 692.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 685.00 | 686.00 | 692.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 692.80 | 681.41 | 686.02 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 712.00 | 690.82 | 689.45 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 658.90 | 686.34 | 688.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 638.95 | 676.86 | 684.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 659.05 | 655.79 | 668.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:45:00 | 657.65 | 655.79 | 668.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 682.95 | 662.20 | 666.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 682.95 | 662.20 | 666.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 699.70 | 669.70 | 669.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 726.40 | 702.69 | 692.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 15:15:00 | 774.00 | 776.90 | 755.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:15:00 | 782.00 | 776.90 | 755.78 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 12:00:00 | 779.05 | 779.13 | 762.22 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 786.30 | 777.39 | 767.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:30:00 | 768.35 | 777.39 | 767.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 771.95 | 774.90 | 770.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 772.10 | 774.90 | 770.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 788.50 | 777.62 | 772.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:15:00 | 822.50 | 783.99 | 778.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 11:15:00 | 821.10 | 791.27 | 782.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 11:15:00 | 818.00 | 791.27 | 782.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-06-18 14:15:00 | 860.20 | 818.61 | 798.24 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 82 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 1006.05 | 1030.51 | 1030.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 999.00 | 1018.27 | 1024.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 1018.00 | 1011.29 | 1018.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 10:15:00 | 1018.00 | 1011.29 | 1018.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1018.00 | 1011.29 | 1018.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:45:00 | 1011.95 | 1011.29 | 1018.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 1027.20 | 1014.48 | 1019.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 13:00:00 | 1014.35 | 1014.45 | 1018.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 14:00:00 | 1014.00 | 1014.36 | 1018.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 13:15:00 | 963.63 | 989.44 | 1003.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 13:15:00 | 963.30 | 989.44 | 1003.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 14:15:00 | 984.95 | 974.25 | 985.15 | SL hit (close>ema200) qty=0.50 sl=974.25 alert=retest2 |

### Cycle 83 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 1030.65 | 995.65 | 992.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 1034.55 | 1011.99 | 1007.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 15:15:00 | 1011.15 | 1014.15 | 1010.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 15:15:00 | 1011.15 | 1014.15 | 1010.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 1011.15 | 1014.15 | 1010.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 1012.90 | 1014.15 | 1010.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1012.00 | 1013.72 | 1010.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 1025.70 | 1014.58 | 1011.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:30:00 | 1021.15 | 1021.45 | 1016.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:00:00 | 1025.65 | 1021.10 | 1017.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:30:00 | 1023.50 | 1019.81 | 1017.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1023.95 | 1020.64 | 1017.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 1032.60 | 1019.39 | 1017.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:45:00 | 1032.45 | 1027.11 | 1021.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-09 09:15:00 | 1128.27 | 1083.71 | 1056.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 1042.40 | 1063.63 | 1065.55 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 1105.20 | 1063.07 | 1061.43 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 1079.00 | 1090.69 | 1091.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 1074.50 | 1085.94 | 1088.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1065.90 | 1039.28 | 1055.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1065.90 | 1039.28 | 1055.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1065.90 | 1039.28 | 1055.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 1065.90 | 1039.28 | 1055.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1079.10 | 1047.24 | 1057.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 1083.95 | 1047.24 | 1057.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1082.40 | 1054.28 | 1059.92 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 13:15:00 | 1085.55 | 1065.94 | 1064.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 1104.65 | 1078.89 | 1071.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 11:15:00 | 1054.60 | 1074.03 | 1070.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 11:15:00 | 1054.60 | 1074.03 | 1070.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 1054.60 | 1074.03 | 1070.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:30:00 | 1047.50 | 1074.03 | 1070.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 1014.40 | 1062.10 | 1065.02 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 13:15:00 | 1032.45 | 1023.04 | 1022.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 1045.35 | 1030.50 | 1026.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 13:15:00 | 1033.05 | 1038.39 | 1032.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 14:00:00 | 1033.05 | 1038.39 | 1032.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1025.70 | 1035.85 | 1031.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 1025.70 | 1035.85 | 1031.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 1027.00 | 1034.08 | 1031.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 1013.00 | 1034.08 | 1031.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 1013.00 | 1026.82 | 1028.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 999.90 | 1014.15 | 1020.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 964.60 | 956.99 | 977.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 964.60 | 956.99 | 977.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 937.80 | 946.94 | 962.40 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 979.00 | 963.29 | 963.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 987.20 | 976.05 | 970.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 984.50 | 989.70 | 983.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 09:45:00 | 986.00 | 989.70 | 983.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 986.60 | 989.08 | 983.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 983.50 | 989.08 | 983.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 982.50 | 987.77 | 983.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:45:00 | 985.15 | 987.77 | 983.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 978.65 | 985.94 | 982.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 979.00 | 985.94 | 982.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 932.90 | 975.33 | 978.31 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 954.20 | 938.76 | 937.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 1036.15 | 967.30 | 953.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1000.95 | 1006.75 | 985.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:30:00 | 998.50 | 1006.75 | 985.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 980.25 | 994.24 | 988.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:30:00 | 979.85 | 994.24 | 988.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 978.60 | 991.12 | 987.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 978.60 | 991.12 | 987.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 977.00 | 986.71 | 986.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 977.00 | 986.71 | 986.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 13:15:00 | 976.55 | 984.67 | 985.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 14:15:00 | 973.85 | 982.51 | 984.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 1000.35 | 984.40 | 984.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1000.35 | 984.40 | 984.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1000.35 | 984.40 | 984.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 1011.95 | 984.40 | 984.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 1002.50 | 988.02 | 986.37 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 979.00 | 986.51 | 986.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 975.25 | 980.58 | 983.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 992.80 | 974.47 | 977.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 992.80 | 974.47 | 977.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 992.80 | 974.47 | 977.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 1002.00 | 974.47 | 977.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 999.95 | 979.56 | 979.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 1016.00 | 993.04 | 988.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 14:15:00 | 1003.60 | 1004.87 | 997.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 15:15:00 | 998.05 | 1004.87 | 997.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 998.05 | 1003.51 | 997.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 1014.05 | 1003.51 | 997.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 11:30:00 | 1004.10 | 1004.90 | 999.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 13:15:00 | 1006.50 | 1004.52 | 999.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 979.25 | 999.95 | 999.39 | SL hit (close<static) qty=1.00 sl=992.40 alert=retest2 |

### Cycle 98 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 981.85 | 996.33 | 997.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 974.70 | 986.58 | 992.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 977.00 | 972.31 | 979.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 15:15:00 | 977.00 | 972.31 | 979.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 977.00 | 972.31 | 979.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 994.85 | 972.31 | 979.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1002.95 | 978.44 | 981.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 998.90 | 978.44 | 981.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 998.55 | 982.46 | 983.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:45:00 | 1001.00 | 982.46 | 983.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 1000.20 | 986.01 | 984.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 1013.30 | 997.50 | 991.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 991.00 | 1001.31 | 995.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 991.00 | 1001.31 | 995.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 991.00 | 1001.31 | 995.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 991.00 | 1001.31 | 995.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 972.40 | 995.53 | 993.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 972.40 | 995.53 | 993.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 15:15:00 | 973.50 | 991.12 | 991.86 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 997.20 | 992.40 | 992.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1007.35 | 995.56 | 993.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 13:15:00 | 998.50 | 998.67 | 996.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 13:30:00 | 999.00 | 998.67 | 996.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 996.00 | 998.14 | 996.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 996.00 | 998.14 | 996.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 998.90 | 998.29 | 996.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 989.10 | 998.29 | 996.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 982.00 | 995.03 | 995.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 982.00 | 995.03 | 995.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 986.50 | 993.32 | 994.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 12:15:00 | 980.50 | 984.99 | 988.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 12:15:00 | 994.65 | 982.91 | 985.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 12:15:00 | 994.65 | 982.91 | 985.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 994.65 | 982.91 | 985.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:30:00 | 997.25 | 982.91 | 985.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 989.05 | 984.13 | 985.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 990.05 | 984.13 | 985.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 1016.45 | 990.60 | 988.25 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 973.80 | 987.06 | 987.60 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 983.95 | 981.16 | 981.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 987.70 | 982.90 | 981.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 13:15:00 | 983.60 | 984.29 | 983.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 14:00:00 | 983.60 | 984.29 | 983.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 985.40 | 984.51 | 983.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:30:00 | 986.00 | 984.51 | 983.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 983.00 | 984.21 | 983.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 986.75 | 984.21 | 983.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 978.25 | 983.02 | 982.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 978.25 | 983.02 | 982.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 979.95 | 982.40 | 982.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 12:15:00 | 973.20 | 979.59 | 981.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 980.05 | 972.57 | 975.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 980.05 | 972.57 | 975.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 980.05 | 972.57 | 975.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 978.40 | 972.57 | 975.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 977.95 | 973.64 | 975.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:45:00 | 973.70 | 973.71 | 975.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 09:15:00 | 996.00 | 968.09 | 967.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 996.00 | 968.09 | 967.95 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 961.75 | 970.27 | 970.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 958.65 | 967.95 | 969.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 869.85 | 868.47 | 890.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 869.85 | 868.47 | 890.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 895.75 | 876.81 | 887.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 895.75 | 876.81 | 887.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 894.90 | 880.43 | 888.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 896.00 | 880.43 | 888.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 888.00 | 885.71 | 888.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 15:15:00 | 882.90 | 885.71 | 888.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 938.90 | 895.90 | 892.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 938.90 | 895.90 | 892.89 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 914.95 | 920.56 | 921.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 910.50 | 917.65 | 919.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 908.45 | 906.11 | 910.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 908.45 | 906.11 | 910.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 908.45 | 906.11 | 910.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 893.20 | 902.68 | 908.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 892.30 | 899.01 | 905.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 848.54 | 862.54 | 880.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 847.68 | 862.54 | 880.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-25 10:15:00 | 803.88 | 828.93 | 844.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 111 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 850.25 | 828.90 | 826.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 860.80 | 843.16 | 833.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 13:15:00 | 856.00 | 857.56 | 848.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 14:00:00 | 856.00 | 857.56 | 848.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 841.00 | 856.66 | 851.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 841.00 | 856.66 | 851.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 840.40 | 853.41 | 850.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 840.40 | 853.41 | 850.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 839.70 | 849.07 | 849.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 14:15:00 | 839.20 | 845.63 | 847.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 852.15 | 845.99 | 847.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 852.15 | 845.99 | 847.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 852.15 | 845.99 | 847.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 852.15 | 845.99 | 847.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 844.90 | 845.77 | 847.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:15:00 | 843.95 | 845.77 | 847.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 12:15:00 | 856.90 | 849.22 | 848.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 12:15:00 | 856.90 | 849.22 | 848.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 874.45 | 856.89 | 852.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 893.05 | 899.90 | 885.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 893.05 | 899.90 | 885.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 890.05 | 896.44 | 886.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 880.00 | 896.44 | 886.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 886.70 | 894.50 | 886.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:15:00 | 883.45 | 894.50 | 886.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 879.05 | 891.41 | 885.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 879.05 | 891.41 | 885.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 883.70 | 889.86 | 885.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 15:00:00 | 886.85 | 886.32 | 884.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 871.40 | 882.64 | 883.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 871.40 | 882.64 | 883.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 15:15:00 | 857.00 | 868.17 | 873.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 12:15:00 | 822.75 | 820.97 | 831.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 12:30:00 | 822.00 | 820.97 | 831.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 829.60 | 820.20 | 827.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:15:00 | 831.45 | 820.20 | 827.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 829.35 | 822.03 | 827.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:15:00 | 826.30 | 823.25 | 827.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:00:00 | 824.00 | 824.39 | 827.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 12:15:00 | 826.65 | 819.33 | 820.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 825.40 | 821.02 | 820.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 825.40 | 821.02 | 820.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 828.25 | 822.46 | 821.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 836.00 | 837.56 | 831.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 836.00 | 837.56 | 831.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 833.25 | 836.70 | 831.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 845.95 | 836.70 | 831.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-29 11:15:00 | 930.55 | 906.92 | 887.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 997.90 | 1012.28 | 1013.53 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 1058.65 | 1018.47 | 1015.43 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 1007.00 | 1019.23 | 1020.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 1001.00 | 1009.59 | 1014.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 992.10 | 991.48 | 1001.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:30:00 | 993.15 | 991.48 | 1001.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1006.40 | 994.23 | 1001.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 1008.90 | 994.23 | 1001.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1009.30 | 997.24 | 1001.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 1010.15 | 997.24 | 1001.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 1009.70 | 1004.22 | 1004.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 1077.05 | 1019.31 | 1011.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 14:15:00 | 1015.20 | 1030.76 | 1021.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 14:15:00 | 1015.20 | 1030.76 | 1021.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 1015.20 | 1030.76 | 1021.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 1016.25 | 1030.76 | 1021.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 1018.90 | 1028.39 | 1021.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 1029.60 | 1028.39 | 1021.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 11:15:00 | 1008.10 | 1021.06 | 1019.57 | SL hit (close<static) qty=1.00 sl=1010.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 12:15:00 | 1003.25 | 1017.49 | 1018.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 989.00 | 1004.82 | 1011.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 1006.75 | 1005.20 | 1010.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 10:15:00 | 1006.75 | 1005.20 | 1010.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 1006.75 | 1005.20 | 1010.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 10:30:00 | 1014.25 | 1005.20 | 1010.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 1006.80 | 1005.52 | 1010.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 1006.45 | 1005.52 | 1010.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 995.80 | 1002.67 | 1007.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:30:00 | 990.10 | 997.04 | 1003.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 940.60 | 966.21 | 984.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 956.95 | 941.47 | 959.76 | SL hit (close>ema200) qty=0.50 sl=941.47 alert=retest2 |

### Cycle 121 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 967.90 | 957.08 | 955.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 09:15:00 | 980.30 | 963.01 | 959.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 986.50 | 996.13 | 989.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 986.50 | 996.13 | 989.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 986.50 | 996.13 | 989.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:45:00 | 993.25 | 996.13 | 989.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 977.20 | 992.34 | 987.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:45:00 | 976.00 | 992.34 | 987.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 974.35 | 984.38 | 985.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 961.90 | 977.90 | 982.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 978.00 | 950.30 | 961.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 10:15:00 | 978.00 | 950.30 | 961.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 978.00 | 950.30 | 961.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 978.00 | 950.30 | 961.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 972.00 | 954.64 | 962.02 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 973.25 | 966.95 | 966.50 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 960.40 | 965.78 | 966.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 955.00 | 963.62 | 965.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 14:15:00 | 964.30 | 961.98 | 963.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 14:15:00 | 964.30 | 961.98 | 963.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 964.30 | 961.98 | 963.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 964.30 | 961.98 | 963.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 959.05 | 961.40 | 963.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 982.65 | 961.40 | 963.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 993.15 | 967.75 | 966.10 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 938.65 | 961.97 | 965.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 912.00 | 937.65 | 949.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 894.30 | 893.67 | 915.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 894.30 | 893.67 | 915.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 892.85 | 893.72 | 905.69 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 912.00 | 906.01 | 905.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 962.00 | 917.84 | 911.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 14:15:00 | 945.35 | 949.14 | 932.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 15:00:00 | 945.35 | 949.14 | 932.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 981.30 | 956.23 | 945.50 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 929.05 | 945.31 | 945.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 922.90 | 940.83 | 943.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 939.00 | 937.28 | 940.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 939.00 | 937.28 | 940.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 939.00 | 937.28 | 940.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:45:00 | 939.20 | 937.28 | 940.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 937.10 | 937.25 | 940.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 937.85 | 937.25 | 940.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 968.90 | 943.58 | 943.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 979.00 | 950.66 | 946.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 954.70 | 961.94 | 955.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 954.70 | 961.94 | 955.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 954.70 | 961.94 | 955.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:45:00 | 950.00 | 961.94 | 955.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 958.20 | 961.19 | 955.64 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 913.80 | 947.53 | 951.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 902.20 | 938.46 | 946.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 13:15:00 | 885.25 | 883.18 | 905.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 14:00:00 | 885.25 | 883.18 | 905.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 902.55 | 883.66 | 899.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 910.05 | 883.66 | 899.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 899.65 | 886.86 | 899.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 898.50 | 886.86 | 899.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:00:00 | 895.75 | 888.64 | 899.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:30:00 | 898.10 | 891.24 | 898.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 924.95 | 902.02 | 902.13 | SL hit (close>static) qty=1.00 sl=906.50 alert=retest2 |

### Cycle 131 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 918.35 | 905.29 | 903.60 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 14:15:00 | 889.65 | 902.38 | 902.97 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 919.90 | 904.82 | 903.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 933.00 | 918.47 | 911.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 913.75 | 929.34 | 919.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 913.75 | 929.34 | 919.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 913.75 | 929.34 | 919.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 913.75 | 929.34 | 919.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 921.10 | 927.69 | 919.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 927.30 | 927.62 | 920.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 899.35 | 917.05 | 917.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 899.35 | 917.05 | 917.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 15:15:00 | 894.00 | 903.36 | 909.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 903.55 | 903.40 | 908.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 903.55 | 903.40 | 908.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 903.55 | 903.40 | 908.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 11:00:00 | 901.00 | 902.92 | 908.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 15:00:00 | 897.30 | 899.05 | 904.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 924.00 | 908.19 | 906.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 924.00 | 908.19 | 906.53 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 15:15:00 | 900.20 | 904.94 | 905.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 898.00 | 903.56 | 904.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 904.55 | 903.75 | 904.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 11:00:00 | 904.55 | 903.75 | 904.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 900.15 | 903.03 | 904.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 14:00:00 | 895.00 | 901.39 | 903.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 15:00:00 | 899.90 | 901.09 | 903.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 850.25 | 865.55 | 881.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 854.90 | 865.55 | 881.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 805.50 | 822.06 | 848.32 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 137 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 735.35 | 717.84 | 716.26 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 710.95 | 716.31 | 716.77 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 12:15:00 | 728.60 | 719.09 | 717.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 14:15:00 | 751.70 | 727.41 | 722.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 712.15 | 727.65 | 723.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 712.15 | 727.65 | 723.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 712.15 | 727.65 | 723.28 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 11:15:00 | 703.25 | 719.43 | 720.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 698.00 | 715.14 | 718.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 658.50 | 649.17 | 670.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 10:00:00 | 658.50 | 649.17 | 670.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 662.40 | 652.31 | 661.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 668.95 | 652.31 | 661.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 661.30 | 654.10 | 661.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 663.75 | 654.10 | 661.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 661.00 | 655.48 | 661.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:30:00 | 663.10 | 655.48 | 661.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 659.45 | 656.28 | 661.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:30:00 | 660.60 | 656.28 | 661.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 657.40 | 656.50 | 660.88 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 10:15:00 | 674.70 | 663.66 | 662.48 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 12:15:00 | 656.95 | 661.57 | 661.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 09:15:00 | 641.80 | 656.68 | 659.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 626.95 | 626.05 | 635.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 624.70 | 626.05 | 635.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 615.00 | 623.84 | 633.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 614.50 | 621.92 | 631.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:30:00 | 613.50 | 618.01 | 627.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 14:30:00 | 613.85 | 617.56 | 626.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 15:15:00 | 611.45 | 617.56 | 626.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 613.90 | 603.06 | 608.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 13:15:00 | 621.25 | 611.40 | 610.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 621.25 | 611.40 | 610.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 626.65 | 614.45 | 612.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 635.60 | 635.67 | 627.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 635.60 | 635.67 | 627.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 660.85 | 671.67 | 665.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 660.85 | 671.67 | 665.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 659.70 | 669.28 | 664.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 656.00 | 669.28 | 664.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 655.55 | 661.23 | 661.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 642.00 | 651.31 | 656.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 647.00 | 645.82 | 651.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:00:00 | 647.00 | 645.82 | 651.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 652.40 | 647.14 | 651.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 655.00 | 647.14 | 651.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 652.30 | 648.17 | 651.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:30:00 | 659.25 | 648.17 | 651.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 646.20 | 647.78 | 650.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 668.85 | 647.78 | 650.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 672.75 | 652.77 | 652.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 672.75 | 652.77 | 652.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 663.15 | 654.85 | 653.84 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 643.45 | 652.34 | 652.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 14:15:00 | 640.45 | 649.97 | 651.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 650.75 | 648.25 | 650.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 650.75 | 648.25 | 650.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 650.75 | 648.25 | 650.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 650.75 | 648.25 | 650.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 642.75 | 647.15 | 649.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 640.10 | 646.47 | 649.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 633.10 | 646.46 | 648.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:15:00 | 642.05 | 644.17 | 647.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 15:15:00 | 652.50 | 648.44 | 648.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 652.50 | 648.44 | 648.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 663.45 | 651.48 | 649.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 14:15:00 | 654.35 | 654.80 | 652.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 15:00:00 | 654.35 | 654.80 | 652.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 148 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 630.75 | 650.79 | 650.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 624.15 | 642.30 | 646.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 594.00 | 590.82 | 609.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 601.60 | 590.82 | 609.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 597.25 | 592.10 | 608.24 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 633.95 | 610.86 | 608.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 643.85 | 625.15 | 617.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 09:15:00 | 763.00 | 764.10 | 736.72 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:15:00 | 779.70 | 762.78 | 748.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 748.80 | 759.99 | 748.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 748.80 | 759.99 | 748.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 746.25 | 757.24 | 747.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 746.25 | 757.24 | 747.96 | SL hit (close<ema400) qty=1.00 sl=747.96 alert=retest1 |

### Cycle 150 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 728.05 | 748.26 | 750.33 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 756.60 | 745.31 | 745.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 777.20 | 754.98 | 750.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 759.00 | 762.90 | 757.30 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 09:15:00 | 779.65 | 762.90 | 757.30 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 787.95 | 767.91 | 760.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 763.00 | 768.06 | 765.02 | SL hit (close<ema400) qty=1.00 sl=765.02 alert=retest1 |

### Cycle 152 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 775.25 | 787.37 | 787.43 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 809.90 | 788.25 | 787.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 820.85 | 794.77 | 790.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 820.00 | 821.54 | 812.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:15:00 | 827.65 | 821.54 | 812.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 13:00:00 | 826.60 | 825.49 | 817.56 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 14:15:00 | 827.55 | 825.63 | 818.35 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 09:15:00 | 869.03 | 840.41 | 827.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 09:15:00 | 867.93 | 840.41 | 827.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 09:15:00 | 868.93 | 840.41 | 827.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 874.30 | 874.64 | 865.53 | SL hit (close<ema200) qty=0.50 sl=874.64 alert=retest1 |

### Cycle 154 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 897.20 | 909.57 | 911.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 882.80 | 898.04 | 904.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 891.20 | 890.76 | 897.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:45:00 | 891.15 | 890.76 | 897.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 896.90 | 891.99 | 897.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:15:00 | 894.05 | 891.99 | 897.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 890.00 | 891.59 | 896.91 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 973.10 | 907.72 | 903.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 984.90 | 923.16 | 910.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 12:15:00 | 1020.65 | 1022.93 | 1001.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 13:00:00 | 1020.65 | 1022.93 | 1001.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1065.00 | 1058.88 | 1051.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 1055.70 | 1058.88 | 1051.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1052.15 | 1057.39 | 1052.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 1052.60 | 1057.39 | 1052.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1051.50 | 1056.22 | 1052.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1051.50 | 1056.22 | 1052.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1026.55 | 1050.28 | 1049.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1026.55 | 1050.28 | 1049.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1035.35 | 1047.30 | 1048.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1018.00 | 1032.23 | 1039.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 1028.50 | 1024.41 | 1031.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:30:00 | 1027.20 | 1024.41 | 1031.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1038.85 | 1027.30 | 1032.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1038.85 | 1027.30 | 1032.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1045.20 | 1030.88 | 1033.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 1041.10 | 1030.88 | 1033.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1037.70 | 1033.23 | 1034.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1042.00 | 1033.23 | 1034.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 1048.00 | 1036.19 | 1035.34 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 1030.70 | 1035.03 | 1035.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 15:15:00 | 1022.60 | 1032.46 | 1033.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 1037.60 | 1033.48 | 1034.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 1037.60 | 1033.48 | 1034.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1037.60 | 1033.48 | 1034.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 1037.60 | 1033.48 | 1034.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1028.65 | 1032.52 | 1033.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 1025.20 | 1031.89 | 1033.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:15:00 | 1023.55 | 1030.85 | 1032.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 1023.05 | 1029.20 | 1031.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 1021.00 | 1027.56 | 1030.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1022.55 | 1025.35 | 1029.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 1023.00 | 1025.35 | 1029.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1016.55 | 1007.60 | 1015.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1016.55 | 1007.60 | 1015.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1017.00 | 1009.48 | 1015.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1017.00 | 1009.48 | 1015.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1008.45 | 1009.27 | 1014.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1022.95 | 1017.31 | 1017.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 1022.95 | 1017.31 | 1017.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 1029.00 | 1020.51 | 1018.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 1026.40 | 1028.54 | 1024.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 1026.40 | 1028.54 | 1024.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1026.40 | 1028.54 | 1024.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 1026.70 | 1028.54 | 1024.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1024.95 | 1027.82 | 1024.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:15:00 | 1018.90 | 1027.82 | 1024.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1018.90 | 1026.04 | 1024.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1034.00 | 1026.04 | 1024.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 991.80 | 1027.20 | 1027.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 991.80 | 1027.20 | 1027.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 10:15:00 | 975.35 | 1016.83 | 1022.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 974.30 | 973.88 | 986.22 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 10:30:00 | 968.00 | 973.11 | 984.75 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 973.00 | 973.59 | 979.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 11:30:00 | 971.50 | 973.10 | 978.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 970.25 | 972.95 | 978.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 968.50 | 971.42 | 976.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 12:15:00 | 922.92 | 929.28 | 936.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 12:15:00 | 921.74 | 929.28 | 936.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 931.05 | 928.86 | 933.89 | SL hit (close>ema200) qty=0.50 sl=928.86 alert=retest2 |

### Cycle 161 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 978.50 | 922.38 | 921.31 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 952.90 | 959.85 | 960.15 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 964.85 | 960.59 | 960.18 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 15:15:00 | 957.80 | 959.85 | 959.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 954.60 | 958.57 | 959.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 977.00 | 952.42 | 953.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 977.00 | 952.42 | 953.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 977.00 | 952.42 | 953.20 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 973.10 | 956.55 | 955.00 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 09:15:00 | 948.55 | 958.91 | 960.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 935.00 | 949.35 | 954.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 942.20 | 939.77 | 947.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 942.20 | 939.77 | 947.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 942.20 | 939.77 | 947.10 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 963.40 | 950.27 | 948.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 967.50 | 955.79 | 951.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 951.70 | 958.10 | 954.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 951.70 | 958.10 | 954.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 951.70 | 958.10 | 954.59 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 13:15:00 | 947.40 | 952.76 | 952.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 943.55 | 950.91 | 951.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 950.30 | 948.60 | 950.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 950.30 | 948.60 | 950.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 950.30 | 948.60 | 950.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 12:15:00 | 939.00 | 947.13 | 949.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 12:45:00 | 936.50 | 945.07 | 948.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 939.65 | 939.26 | 943.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 11:30:00 | 938.80 | 939.29 | 943.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 968.15 | 944.09 | 943.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 968.15 | 944.09 | 943.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 973.85 | 950.04 | 946.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 965.10 | 965.11 | 957.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 965.10 | 965.11 | 957.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 965.10 | 965.11 | 957.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 963.00 | 965.11 | 957.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 942.50 | 960.09 | 956.35 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 941.20 | 954.10 | 954.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 938.60 | 951.00 | 952.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 933.00 | 932.52 | 939.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 933.00 | 932.52 | 939.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 933.00 | 932.52 | 939.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 925.50 | 930.83 | 936.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 944.00 | 930.44 | 928.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 14:15:00 | 944.00 | 930.44 | 928.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 985.95 | 950.11 | 940.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 963.25 | 967.94 | 956.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 14:30:00 | 966.15 | 967.94 | 956.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 967.00 | 966.59 | 957.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 966.00 | 966.59 | 957.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 962.85 | 964.96 | 959.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 963.40 | 964.96 | 959.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:00:00 | 964.95 | 964.96 | 959.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 982.70 | 1007.66 | 1008.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 982.70 | 1007.66 | 1008.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 965.50 | 983.93 | 994.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 934.70 | 931.69 | 945.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 937.15 | 931.69 | 945.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 947.90 | 936.15 | 943.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 948.00 | 936.15 | 943.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 948.80 | 938.68 | 943.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 950.60 | 938.68 | 943.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 977.00 | 948.79 | 947.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 1011.35 | 961.31 | 953.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 999.00 | 1000.86 | 988.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:30:00 | 997.30 | 1000.86 | 988.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 986.60 | 1000.71 | 997.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 986.60 | 1000.71 | 997.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 990.00 | 998.57 | 996.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 986.90 | 998.57 | 996.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 988.00 | 995.72 | 995.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 988.00 | 995.72 | 995.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 986.20 | 993.82 | 994.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 12:15:00 | 983.30 | 991.71 | 993.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 981.15 | 974.51 | 980.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 981.15 | 974.51 | 980.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 981.15 | 974.51 | 980.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 980.00 | 974.51 | 980.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 976.70 | 974.95 | 979.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:00:00 | 971.45 | 974.25 | 979.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 972.10 | 973.82 | 978.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 986.85 | 975.79 | 977.75 | SL hit (close>static) qty=1.00 sl=981.15 alert=retest2 |

### Cycle 175 — BUY (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 15:15:00 | 985.90 | 979.57 | 979.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 1001.75 | 984.80 | 982.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 1003.70 | 1004.07 | 998.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 12:00:00 | 1003.70 | 1004.07 | 998.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 992.10 | 1002.17 | 999.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 993.25 | 1002.17 | 999.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 993.20 | 1000.38 | 999.37 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 987.70 | 996.77 | 997.84 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 999.00 | 996.82 | 996.80 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 992.35 | 996.28 | 996.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 990.40 | 994.61 | 995.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 979.80 | 976.01 | 982.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 979.80 | 976.01 | 982.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 979.80 | 976.01 | 982.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 979.80 | 976.01 | 982.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 974.40 | 975.69 | 981.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 972.15 | 975.69 | 981.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 991.10 | 977.85 | 981.57 | SL hit (close>static) qty=1.00 sl=983.90 alert=retest2 |

### Cycle 179 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 908.15 | 882.31 | 880.43 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 882.60 | 890.98 | 891.46 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 902.00 | 889.04 | 888.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 914.25 | 894.08 | 890.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 14:15:00 | 900.25 | 900.58 | 895.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 14:30:00 | 899.50 | 900.58 | 895.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 896.65 | 899.79 | 895.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 910.40 | 899.79 | 895.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:15:00 | 900.90 | 903.34 | 901.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:45:00 | 900.95 | 903.07 | 901.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 911.25 | 903.15 | 902.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 901.80 | 905.25 | 904.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 901.80 | 905.25 | 904.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 900.30 | 904.26 | 904.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 900.30 | 904.26 | 904.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 902.15 | 903.83 | 903.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 902.15 | 903.83 | 903.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 897.10 | 902.22 | 903.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 888.45 | 885.80 | 891.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 888.45 | 885.80 | 891.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 888.45 | 885.80 | 891.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 889.10 | 885.80 | 891.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 882.45 | 885.20 | 889.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 888.50 | 885.20 | 889.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 882.80 | 884.81 | 888.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 879.00 | 883.40 | 886.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 892.00 | 881.26 | 882.82 | SL hit (close>static) qty=1.00 sl=890.90 alert=retest2 |

### Cycle 183 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 896.80 | 886.40 | 885.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 14:15:00 | 903.00 | 893.92 | 890.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 13:15:00 | 903.85 | 904.24 | 898.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 14:00:00 | 903.85 | 904.24 | 898.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 908.10 | 911.36 | 908.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 908.10 | 911.36 | 908.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 909.40 | 910.97 | 908.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 907.60 | 910.97 | 908.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 903.25 | 909.43 | 908.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 903.25 | 909.43 | 908.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 903.05 | 908.15 | 907.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 903.55 | 908.15 | 907.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 902.05 | 906.93 | 907.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 886.80 | 900.48 | 903.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 872.85 | 871.65 | 880.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 11:30:00 | 871.90 | 871.65 | 880.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 873.75 | 872.64 | 876.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:45:00 | 875.35 | 872.64 | 876.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 875.95 | 873.41 | 875.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 875.95 | 873.41 | 875.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 875.95 | 873.92 | 875.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 884.25 | 873.92 | 875.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 888.00 | 876.73 | 876.92 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 884.65 | 878.32 | 877.62 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 876.10 | 878.24 | 878.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 874.50 | 876.74 | 877.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 13:15:00 | 884.75 | 876.97 | 877.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 13:15:00 | 884.75 | 876.97 | 877.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 884.75 | 876.97 | 877.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:45:00 | 883.70 | 876.97 | 877.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 873.95 | 876.37 | 877.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 871.55 | 876.37 | 877.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 827.97 | 839.29 | 847.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 845.00 | 837.02 | 843.03 | SL hit (close>ema200) qty=0.50 sl=837.02 alert=retest2 |

### Cycle 187 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 820.35 | 815.81 | 815.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 828.10 | 818.27 | 816.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 870.20 | 885.59 | 867.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 09:30:00 | 871.55 | 885.59 | 867.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 867.10 | 877.45 | 867.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:30:00 | 865.15 | 877.45 | 867.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 866.00 | 875.16 | 867.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:30:00 | 866.50 | 875.16 | 867.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 863.00 | 872.73 | 867.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 863.00 | 872.73 | 867.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 861.95 | 870.57 | 866.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 849.00 | 870.57 | 866.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 850.65 | 862.73 | 863.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 10:15:00 | 845.30 | 851.18 | 856.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 853.50 | 851.64 | 855.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 853.50 | 851.64 | 855.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 852.30 | 851.77 | 855.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 854.80 | 851.77 | 855.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 854.85 | 852.39 | 855.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:30:00 | 854.20 | 852.39 | 855.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 860.70 | 854.05 | 856.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 860.70 | 854.05 | 856.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 869.00 | 857.04 | 857.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 866.85 | 857.04 | 857.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 872.05 | 860.04 | 858.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 884.95 | 871.37 | 865.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 878.65 | 884.51 | 878.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 11:15:00 | 878.65 | 884.51 | 878.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 878.65 | 884.51 | 878.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 879.00 | 884.51 | 878.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 877.75 | 882.32 | 878.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 877.25 | 882.32 | 878.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 875.65 | 880.98 | 878.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 875.65 | 880.98 | 878.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 894.75 | 883.02 | 879.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 903.00 | 883.02 | 879.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 914.05 | 887.99 | 884.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:15:00 | 900.60 | 900.03 | 892.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:30:00 | 906.20 | 905.73 | 897.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 912.50 | 913.75 | 908.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 915.30 | 913.80 | 909.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 904.10 | 911.86 | 908.95 | SL hit (close<static) qty=1.00 sl=906.80 alert=retest2 |

### Cycle 190 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 904.45 | 907.17 | 907.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 15:15:00 | 901.00 | 905.94 | 906.87 | Break + close below crossover candle low |

### Cycle 191 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 914.25 | 907.60 | 907.54 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 906.55 | 907.49 | 907.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 897.10 | 905.13 | 906.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 890.75 | 878.80 | 883.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 890.75 | 878.80 | 883.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 890.75 | 878.80 | 883.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 895.65 | 878.80 | 883.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 885.00 | 880.04 | 883.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:45:00 | 892.05 | 880.04 | 883.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 879.60 | 880.11 | 882.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 873.55 | 883.68 | 883.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 877.15 | 881.53 | 882.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 15:15:00 | 833.29 | 843.89 | 853.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 829.87 | 842.49 | 851.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 789.43 | 807.85 | 822.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 193 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 823.20 | 799.78 | 798.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 825.60 | 816.23 | 812.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 805.85 | 816.28 | 812.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 805.85 | 816.28 | 812.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 805.85 | 816.28 | 812.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 805.85 | 816.28 | 812.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 798.95 | 812.82 | 811.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 801.00 | 812.82 | 811.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 790.80 | 808.41 | 809.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 782.85 | 803.30 | 807.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 777.50 | 772.94 | 786.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 777.50 | 772.94 | 786.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 787.30 | 775.81 | 786.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 792.55 | 775.81 | 786.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 789.95 | 778.64 | 786.84 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 796.70 | 791.47 | 790.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 810.40 | 796.03 | 793.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 795.85 | 798.79 | 796.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 15:15:00 | 795.85 | 798.79 | 796.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 795.85 | 798.79 | 796.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 796.10 | 798.79 | 796.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 788.90 | 796.81 | 795.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 789.15 | 796.81 | 795.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 780.35 | 793.52 | 794.11 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 793.50 | 790.52 | 790.52 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 786.00 | 789.62 | 790.11 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 797.80 | 791.25 | 790.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 811.90 | 795.38 | 792.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 814.15 | 814.46 | 808.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 814.15 | 814.46 | 808.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 802.85 | 812.05 | 808.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 802.85 | 812.05 | 808.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 807.90 | 811.22 | 808.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 811.80 | 808.77 | 807.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 800.05 | 807.51 | 807.40 | SL hit (close<static) qty=1.00 sl=802.85 alert=retest2 |

### Cycle 200 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 800.30 | 806.07 | 806.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 778.20 | 797.74 | 802.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 789.35 | 786.59 | 793.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 789.35 | 786.59 | 793.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 789.35 | 786.59 | 793.02 | EMA400 retest candle locked (from downside) |

### Cycle 201 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 800.30 | 793.57 | 793.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 808.55 | 796.57 | 794.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 802.60 | 803.09 | 798.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 802.60 | 803.09 | 798.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 801.90 | 803.01 | 800.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 801.90 | 803.01 | 800.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 797.25 | 801.70 | 800.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 797.25 | 801.70 | 800.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 796.65 | 800.69 | 799.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 797.00 | 800.69 | 799.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 791.00 | 798.75 | 799.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 790.40 | 797.08 | 798.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 767.25 | 762.40 | 770.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 767.25 | 762.40 | 770.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 767.25 | 762.40 | 770.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:30:00 | 758.60 | 761.71 | 768.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:45:00 | 757.65 | 758.61 | 763.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 720.67 | 738.96 | 745.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 719.77 | 738.96 | 745.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 682.74 | 711.81 | 726.61 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 203 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 744.95 | 687.48 | 682.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 771.00 | 704.18 | 690.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 846.90 | 849.54 | 799.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 09:15:00 | 912.10 | 849.82 | 821.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 836.75 | 849.08 | 830.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:30:00 | 839.75 | 849.08 | 830.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 816.90 | 843.39 | 834.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 816.90 | 843.39 | 834.04 | SL hit (close<ema400) qty=1.00 sl=834.04 alert=retest1 |

### Cycle 204 — SELL (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 12:15:00 | 808.45 | 827.94 | 828.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 13:15:00 | 804.50 | 823.25 | 826.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 826.20 | 816.57 | 821.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 10:15:00 | 826.20 | 816.57 | 821.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 826.20 | 816.57 | 821.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:30:00 | 802.50 | 813.21 | 817.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 11:00:00 | 802.50 | 811.07 | 815.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 13:00:00 | 802.20 | 808.05 | 813.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 09:30:00 | 803.55 | 800.75 | 807.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 803.50 | 801.30 | 807.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 11:30:00 | 798.90 | 801.44 | 806.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 798.00 | 801.44 | 806.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 762.38 | 788.82 | 798.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 762.38 | 788.82 | 798.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 762.09 | 788.82 | 798.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 763.37 | 788.82 | 798.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 758.95 | 788.82 | 798.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 758.10 | 782.67 | 794.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 772.00 | 767.16 | 779.84 | SL hit (close>ema200) qty=0.50 sl=767.16 alert=retest2 |

### Cycle 205 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 794.00 | 783.48 | 783.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 809.35 | 788.65 | 785.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 790.85 | 795.03 | 790.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 790.85 | 795.03 | 790.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 790.85 | 795.03 | 790.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 790.85 | 795.03 | 790.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 794.00 | 794.83 | 791.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 813.35 | 794.83 | 791.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 790.35 | 793.93 | 790.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 784.80 | 793.93 | 790.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 781.45 | 791.43 | 790.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 781.75 | 791.43 | 790.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 782.60 | 789.67 | 789.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:45:00 | 782.00 | 789.67 | 789.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 781.25 | 787.98 | 788.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 763.00 | 782.31 | 785.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 777.90 | 766.00 | 773.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 777.90 | 766.00 | 773.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 777.90 | 766.00 | 773.95 | EMA400 retest candle locked (from downside) |

### Cycle 207 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 786.05 | 778.57 | 778.09 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 755.00 | 775.88 | 777.17 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 782.85 | 776.02 | 775.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 791.95 | 781.47 | 778.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 786.00 | 786.61 | 783.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 804.85 | 786.61 | 783.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 885.34 | 820.92 | 805.43 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 210 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 902.95 | 913.89 | 915.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 899.05 | 910.92 | 913.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 904.30 | 901.72 | 907.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 904.30 | 901.72 | 907.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 904.30 | 901.72 | 907.78 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 913.00 | 908.43 | 908.18 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 890.20 | 905.47 | 907.06 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 905.85 | 903.84 | 903.76 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 902.70 | 903.61 | 903.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 11:15:00 | 900.75 | 903.04 | 903.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 12:15:00 | 905.35 | 903.50 | 903.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 12:15:00 | 905.35 | 903.50 | 903.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 905.35 | 903.50 | 903.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 905.35 | 903.50 | 903.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 900.35 | 902.87 | 903.28 | EMA400 retest candle locked (from downside) |

### Cycle 215 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 910.85 | 904.50 | 903.92 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 905.10 | 908.71 | 908.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 902.80 | 906.80 | 907.93 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-15 15:15:00 | 308.00 | 2023-05-16 13:15:00 | 313.35 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2023-05-29 10:30:00 | 297.10 | 2023-05-29 11:15:00 | 306.15 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2023-06-02 09:15:00 | 302.95 | 2023-06-06 14:15:00 | 333.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-06-12 12:15:00 | 321.45 | 2023-06-13 09:15:00 | 329.15 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2023-06-12 14:00:00 | 320.35 | 2023-06-13 09:15:00 | 329.15 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2023-06-19 09:15:00 | 379.00 | 2023-06-20 10:15:00 | 416.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-13 09:15:00 | 504.65 | 2023-07-18 12:15:00 | 484.60 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2023-07-27 13:15:00 | 475.15 | 2023-08-01 09:15:00 | 497.00 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2023-07-28 10:00:00 | 475.00 | 2023-08-01 09:15:00 | 497.00 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2023-07-28 11:15:00 | 475.55 | 2023-08-01 09:15:00 | 497.00 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2023-07-28 15:15:00 | 475.00 | 2023-08-01 09:15:00 | 497.00 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2023-08-04 13:30:00 | 474.40 | 2023-08-07 11:15:00 | 479.35 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2023-08-04 14:30:00 | 475.55 | 2023-08-07 11:15:00 | 479.35 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2023-08-04 15:00:00 | 475.70 | 2023-08-07 11:15:00 | 479.35 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-08-07 09:30:00 | 475.00 | 2023-08-07 11:15:00 | 479.35 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-08-07 13:45:00 | 473.20 | 2023-08-10 13:15:00 | 449.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-07 14:30:00 | 472.25 | 2023-08-10 13:15:00 | 448.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-08 09:30:00 | 473.80 | 2023-08-10 13:15:00 | 450.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-08 11:00:00 | 472.55 | 2023-08-10 13:15:00 | 448.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-08 14:30:00 | 472.45 | 2023-08-10 13:15:00 | 448.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-09 09:15:00 | 472.05 | 2023-08-10 13:15:00 | 448.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-09 10:00:00 | 471.00 | 2023-08-10 13:15:00 | 447.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-07 13:45:00 | 473.20 | 2023-08-14 09:15:00 | 425.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-08-07 14:30:00 | 472.25 | 2023-08-14 09:15:00 | 425.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-08-08 09:30:00 | 473.80 | 2023-08-14 09:15:00 | 426.42 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-08-08 11:00:00 | 472.55 | 2023-08-14 09:15:00 | 425.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-08-08 14:30:00 | 472.45 | 2023-08-14 09:15:00 | 425.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-08-09 09:15:00 | 472.05 | 2023-08-14 09:15:00 | 424.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-08-09 10:00:00 | 471.00 | 2023-08-14 09:15:00 | 423.90 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-09-04 11:30:00 | 470.70 | 2023-09-05 12:15:00 | 457.45 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2023-09-05 09:15:00 | 479.80 | 2023-09-05 12:15:00 | 457.45 | STOP_HIT | 1.00 | -4.66% |
| BUY | retest2 | 2023-09-11 11:45:00 | 549.15 | 2023-09-12 09:15:00 | 505.90 | STOP_HIT | 1.00 | -7.88% |
| BUY | retest2 | 2023-10-12 14:15:00 | 537.70 | 2023-10-16 10:15:00 | 591.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-13 09:15:00 | 539.00 | 2023-10-16 10:15:00 | 592.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-13 11:45:00 | 544.20 | 2023-10-16 10:15:00 | 598.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-03 12:15:00 | 740.40 | 2023-11-03 12:15:00 | 731.20 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-11-16 12:15:00 | 741.70 | 2023-11-17 12:15:00 | 716.65 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2023-11-16 14:00:00 | 730.90 | 2023-11-17 12:15:00 | 716.65 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2023-11-16 14:45:00 | 731.25 | 2023-11-17 12:15:00 | 716.65 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-11-17 09:15:00 | 736.75 | 2023-11-17 12:15:00 | 716.65 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2023-11-21 12:15:00 | 713.80 | 2023-11-28 10:15:00 | 722.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2023-11-23 09:45:00 | 712.75 | 2023-11-28 10:15:00 | 722.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2023-11-28 10:00:00 | 712.70 | 2023-11-28 10:15:00 | 722.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-12-06 09:15:00 | 745.45 | 2023-12-07 09:15:00 | 820.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-14 11:15:00 | 791.50 | 2023-12-14 14:15:00 | 779.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2023-12-29 09:15:00 | 814.15 | 2024-01-05 10:15:00 | 814.25 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2023-12-29 09:45:00 | 812.30 | 2024-01-05 10:15:00 | 814.25 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-01-01 09:15:00 | 828.05 | 2024-01-05 10:15:00 | 814.25 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-01-10 11:45:00 | 810.15 | 2024-01-12 11:15:00 | 813.55 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-01-11 12:45:00 | 809.75 | 2024-01-12 11:15:00 | 813.55 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-01-16 11:15:00 | 803.35 | 2024-01-18 09:15:00 | 818.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-01-16 12:45:00 | 803.40 | 2024-01-18 09:15:00 | 818.50 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-01-16 15:15:00 | 803.00 | 2024-01-18 09:15:00 | 818.50 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-01-17 11:15:00 | 804.05 | 2024-01-18 09:15:00 | 818.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-01-17 14:15:00 | 797.80 | 2024-01-18 09:15:00 | 818.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-01-23 09:15:00 | 835.60 | 2024-01-23 13:15:00 | 827.35 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-01-23 10:45:00 | 837.75 | 2024-01-23 13:15:00 | 827.35 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-01-23 11:45:00 | 836.40 | 2024-01-23 13:15:00 | 827.35 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-01-23 12:15:00 | 835.95 | 2024-01-23 13:15:00 | 827.35 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-01-29 14:30:00 | 872.80 | 2024-01-30 14:15:00 | 848.65 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-01-29 15:15:00 | 864.90 | 2024-01-30 14:15:00 | 848.65 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-02-07 11:15:00 | 795.60 | 2024-02-09 09:15:00 | 755.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 11:15:00 | 795.60 | 2024-02-12 12:15:00 | 768.05 | STOP_HIT | 0.50 | 3.46% |
| BUY | retest2 | 2024-02-22 12:00:00 | 832.00 | 2024-02-23 09:15:00 | 801.90 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2024-02-22 13:00:00 | 829.75 | 2024-02-23 09:15:00 | 801.90 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2024-02-22 13:45:00 | 829.75 | 2024-02-23 09:15:00 | 801.90 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-03-05 11:00:00 | 718.60 | 2024-03-11 14:15:00 | 682.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 11:45:00 | 718.00 | 2024-03-11 15:15:00 | 682.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 13:00:00 | 718.05 | 2024-03-11 15:15:00 | 682.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 14:45:00 | 718.20 | 2024-03-11 15:15:00 | 682.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 09:15:00 | 702.85 | 2024-03-12 09:15:00 | 667.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 11:00:00 | 718.60 | 2024-03-13 09:15:00 | 646.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-05 11:45:00 | 718.00 | 2024-03-13 09:15:00 | 646.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-05 13:00:00 | 718.05 | 2024-03-13 09:15:00 | 646.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-05 14:45:00 | 718.20 | 2024-03-13 09:15:00 | 646.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-11 09:15:00 | 702.85 | 2024-03-13 09:15:00 | 632.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-19 15:15:00 | 637.95 | 2024-03-21 15:15:00 | 643.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-03-21 09:30:00 | 639.35 | 2024-03-21 15:15:00 | 643.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-03-21 12:00:00 | 640.10 | 2024-03-21 15:15:00 | 643.30 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-04-08 11:15:00 | 691.70 | 2024-04-08 15:15:00 | 683.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-04-08 12:00:00 | 692.90 | 2024-04-08 15:15:00 | 683.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-04-08 12:45:00 | 692.00 | 2024-04-08 15:15:00 | 683.90 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-04-18 14:30:00 | 656.35 | 2024-04-22 14:15:00 | 664.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2024-04-26 11:00:00 | 732.40 | 2024-04-29 15:15:00 | 717.90 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest1 | 2024-04-29 09:15:00 | 726.80 | 2024-04-29 15:15:00 | 717.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-05-03 11:45:00 | 714.60 | 2024-05-07 09:15:00 | 678.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 12:45:00 | 714.90 | 2024-05-07 09:15:00 | 679.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 11:45:00 | 714.60 | 2024-05-08 09:15:00 | 689.50 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2024-05-03 12:45:00 | 714.90 | 2024-05-08 09:15:00 | 689.50 | STOP_HIT | 0.50 | 3.55% |
| BUY | retest1 | 2024-06-12 09:15:00 | 782.00 | 2024-06-18 11:15:00 | 821.10 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-12 12:00:00 | 779.05 | 2024-06-18 11:15:00 | 818.00 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-12 09:15:00 | 782.00 | 2024-06-18 14:15:00 | 860.20 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-06-12 12:00:00 | 779.05 | 2024-06-18 14:15:00 | 856.96 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-18 11:15:00 | 822.50 | 2024-06-19 09:15:00 | 904.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-26 13:00:00 | 1014.35 | 2024-06-27 13:15:00 | 963.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-26 14:00:00 | 1014.00 | 2024-06-27 13:15:00 | 963.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-26 13:00:00 | 1014.35 | 2024-06-28 14:15:00 | 984.95 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2024-06-26 14:00:00 | 1014.00 | 2024-06-28 14:15:00 | 984.95 | STOP_HIT | 0.50 | 2.86% |
| BUY | retest2 | 2024-07-04 11:15:00 | 1025.70 | 2024-07-09 09:15:00 | 1128.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-04 14:30:00 | 1021.15 | 2024-07-09 09:15:00 | 1123.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-05 10:00:00 | 1025.65 | 2024-07-09 09:15:00 | 1128.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-05 10:30:00 | 1023.50 | 2024-07-09 09:15:00 | 1125.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-08 09:15:00 | 1032.60 | 2024-07-09 09:15:00 | 1135.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-08 09:45:00 | 1032.45 | 2024-07-09 09:15:00 | 1135.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-05 09:15:00 | 1014.05 | 2024-09-06 09:15:00 | 979.25 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2024-09-05 11:30:00 | 1004.10 | 2024-09-06 09:15:00 | 979.25 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-09-05 13:15:00 | 1006.50 | 2024-09-06 09:15:00 | 979.25 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-09-27 11:45:00 | 973.70 | 2024-10-01 09:15:00 | 996.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-10-09 15:15:00 | 882.90 | 2024-10-10 09:15:00 | 938.90 | STOP_HIT | 1.00 | -6.34% |
| SELL | retest2 | 2024-10-21 12:00:00 | 893.20 | 2024-10-22 14:15:00 | 848.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 892.30 | 2024-10-22 14:15:00 | 847.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 893.20 | 2024-10-25 10:15:00 | 803.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 892.30 | 2024-10-25 10:15:00 | 803.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-05 11:15:00 | 843.95 | 2024-11-05 12:15:00 | 856.90 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-11-08 15:00:00 | 886.85 | 2024-11-11 09:15:00 | 871.40 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-11-19 12:15:00 | 826.30 | 2024-11-22 13:15:00 | 825.40 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2024-11-19 14:00:00 | 824.00 | 2024-11-22 13:15:00 | 825.40 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-11-22 12:15:00 | 826.65 | 2024-11-22 13:15:00 | 825.40 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-11-26 09:15:00 | 845.95 | 2024-11-29 11:15:00 | 930.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-18 09:15:00 | 1029.60 | 2024-12-18 11:15:00 | 1008.10 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-12-20 12:30:00 | 990.10 | 2024-12-23 09:15:00 | 940.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:30:00 | 990.10 | 2024-12-24 09:15:00 | 956.95 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2025-01-29 11:15:00 | 898.50 | 2025-01-30 09:15:00 | 924.95 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-01-29 12:00:00 | 895.75 | 2025-01-30 09:15:00 | 924.95 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-01-29 13:30:00 | 898.10 | 2025-01-30 09:15:00 | 924.95 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-02-01 15:00:00 | 927.30 | 2025-02-03 10:15:00 | 899.35 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-02-04 11:00:00 | 901.00 | 2025-02-06 09:15:00 | 924.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-02-04 15:00:00 | 897.30 | 2025-02-06 09:15:00 | 924.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-02-07 14:00:00 | 895.00 | 2025-02-11 09:15:00 | 850.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 15:00:00 | 899.90 | 2025-02-11 09:15:00 | 854.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 14:00:00 | 895.00 | 2025-02-12 09:15:00 | 805.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 15:00:00 | 899.90 | 2025-02-12 09:15:00 | 809.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-12 10:45:00 | 614.50 | 2025-03-18 13:15:00 | 621.25 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-03-12 13:30:00 | 613.50 | 2025-03-18 13:15:00 | 621.25 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-03-12 14:30:00 | 613.85 | 2025-03-18 13:15:00 | 621.25 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-03-12 15:15:00 | 611.45 | 2025-03-18 13:15:00 | 621.25 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-04-01 11:30:00 | 640.10 | 2025-04-02 15:15:00 | 652.50 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-04-02 09:15:00 | 633.10 | 2025-04-02 15:15:00 | 652.50 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-04-02 11:15:00 | 642.05 | 2025-04-02 15:15:00 | 652.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest1 | 2025-04-23 09:15:00 | 779.70 | 2025-04-23 10:15:00 | 746.25 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2025-04-23 13:15:00 | 753.65 | 2025-04-25 09:15:00 | 734.60 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-04-23 14:45:00 | 751.60 | 2025-04-25 09:15:00 | 734.60 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest1 | 2025-05-02 09:15:00 | 779.65 | 2025-05-05 10:15:00 | 763.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-05-05 14:15:00 | 797.25 | 2025-05-09 10:15:00 | 775.25 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-05-06 09:15:00 | 802.55 | 2025-05-09 10:15:00 | 775.25 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-05-06 12:00:00 | 796.35 | 2025-05-09 10:15:00 | 775.25 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-05-08 09:15:00 | 810.80 | 2025-05-09 10:15:00 | 775.25 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest1 | 2025-05-14 09:15:00 | 827.65 | 2025-05-15 09:15:00 | 869.03 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-14 13:00:00 | 826.60 | 2025-05-15 09:15:00 | 867.93 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-14 14:15:00 | 827.55 | 2025-05-15 09:15:00 | 868.93 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-14 09:15:00 | 827.65 | 2025-05-19 13:15:00 | 874.30 | STOP_HIT | 0.50 | 5.64% |
| BUY | retest1 | 2025-05-14 13:00:00 | 826.60 | 2025-05-19 13:15:00 | 874.30 | STOP_HIT | 0.50 | 5.77% |
| BUY | retest1 | 2025-05-14 14:15:00 | 827.55 | 2025-05-19 13:15:00 | 874.30 | STOP_HIT | 0.50 | 5.65% |
| BUY | retest2 | 2025-05-22 09:15:00 | 910.00 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-05-22 09:45:00 | 902.00 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-05-22 11:30:00 | 900.10 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-05-22 13:15:00 | 900.35 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-05-29 09:30:00 | 916.10 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-05-29 10:00:00 | 916.30 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-18 12:15:00 | 1025.20 | 2025-06-23 10:15:00 | 1022.95 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-06-18 13:15:00 | 1023.55 | 2025-06-23 10:15:00 | 1022.95 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-06-18 13:45:00 | 1023.05 | 2025-06-23 10:15:00 | 1022.95 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-06-18 15:00:00 | 1021.00 | 2025-06-23 10:15:00 | 1022.95 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1034.00 | 2025-06-26 09:15:00 | 991.80 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest1 | 2025-06-30 10:30:00 | 968.00 | 2025-07-08 12:15:00 | 922.92 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2025-07-01 11:30:00 | 971.50 | 2025-07-08 12:15:00 | 921.74 | PARTIAL | 0.50 | 5.12% |
| SELL | retest1 | 2025-06-30 10:30:00 | 968.00 | 2025-07-09 09:15:00 | 931.05 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-07-01 11:30:00 | 971.50 | 2025-07-09 09:15:00 | 931.05 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2025-07-01 13:15:00 | 970.25 | 2025-07-09 10:15:00 | 938.95 | STOP_HIT | 1.00 | 3.23% |
| SELL | retest2 | 2025-07-01 15:00:00 | 968.50 | 2025-07-11 10:15:00 | 920.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 15:00:00 | 968.50 | 2025-07-15 09:15:00 | 978.50 | STOP_HIT | 0.50 | -1.03% |
| SELL | retest2 | 2025-08-01 12:15:00 | 939.00 | 2025-08-05 09:15:00 | 968.15 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-08-01 12:45:00 | 936.50 | 2025-08-05 09:15:00 | 968.15 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-08-04 10:45:00 | 939.65 | 2025-08-05 09:15:00 | 968.15 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-08-04 11:30:00 | 938.80 | 2025-08-05 09:15:00 | 968.15 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-08-08 15:00:00 | 925.50 | 2025-08-12 14:15:00 | 944.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-08-18 13:30:00 | 963.40 | 2025-08-26 09:15:00 | 982.70 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2025-08-18 14:00:00 | 964.95 | 2025-08-26 09:15:00 | 982.70 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2025-09-10 12:00:00 | 971.45 | 2025-09-11 09:15:00 | 986.85 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-09-10 13:00:00 | 972.10 | 2025-09-11 09:15:00 | 986.85 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-11 13:15:00 | 971.80 | 2025-09-11 14:15:00 | 981.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-11 14:45:00 | 973.70 | 2025-09-11 15:15:00 | 985.90 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-24 11:15:00 | 972.15 | 2025-09-24 12:15:00 | 991.10 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-24 14:45:00 | 971.35 | 2025-09-29 14:15:00 | 922.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 15:15:00 | 970.90 | 2025-09-29 14:15:00 | 922.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:30:00 | 970.35 | 2025-09-29 14:15:00 | 921.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 11:15:00 | 975.70 | 2025-09-29 14:15:00 | 926.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:00:00 | 972.60 | 2025-09-29 14:15:00 | 923.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:45:00 | 971.35 | 2025-10-01 10:15:00 | 878.13 | TARGET_HIT | 0.50 | 9.60% |
| SELL | retest2 | 2025-09-24 15:15:00 | 970.90 | 2025-10-01 10:15:00 | 875.34 | TARGET_HIT | 0.50 | 9.84% |
| SELL | retest2 | 2025-09-25 09:30:00 | 970.35 | 2025-10-03 09:15:00 | 892.45 | STOP_HIT | 0.50 | 8.03% |
| SELL | retest2 | 2025-09-25 11:15:00 | 975.70 | 2025-10-03 09:15:00 | 892.45 | STOP_HIT | 0.50 | 8.53% |
| SELL | retest2 | 2025-09-25 12:00:00 | 972.60 | 2025-10-03 09:15:00 | 892.45 | STOP_HIT | 0.50 | 8.24% |
| BUY | retest2 | 2025-10-29 09:15:00 | 910.40 | 2025-11-04 11:15:00 | 902.15 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-30 14:15:00 | 900.90 | 2025-11-04 11:15:00 | 902.15 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-10-30 14:45:00 | 900.95 | 2025-11-04 11:15:00 | 902.15 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-11-03 09:15:00 | 911.25 | 2025-11-04 11:15:00 | 902.15 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-11-11 09:15:00 | 879.00 | 2025-11-12 09:15:00 | 892.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-01 15:15:00 | 871.55 | 2025-12-05 09:15:00 | 827.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 15:15:00 | 871.55 | 2025-12-05 14:15:00 | 845.00 | STOP_HIT | 0.50 | 3.05% |
| BUY | retest2 | 2025-12-26 10:15:00 | 903.00 | 2026-01-01 09:15:00 | 904.10 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-12-29 09:15:00 | 914.05 | 2026-01-01 14:15:00 | 904.45 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-29 14:15:00 | 900.60 | 2026-01-01 14:15:00 | 904.45 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-12-30 09:30:00 | 906.20 | 2026-01-01 14:15:00 | 904.45 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2026-01-01 09:15:00 | 915.30 | 2026-01-01 14:15:00 | 904.45 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-01-09 09:15:00 | 873.55 | 2026-01-16 15:15:00 | 833.29 | PARTIAL | 0.50 | 4.61% |
| SELL | retest2 | 2026-01-09 10:45:00 | 877.15 | 2026-01-19 09:15:00 | 829.87 | PARTIAL | 0.50 | 5.39% |
| SELL | retest2 | 2026-01-09 09:15:00 | 873.55 | 2026-01-20 15:15:00 | 789.43 | TARGET_HIT | 0.50 | 9.63% |
| SELL | retest2 | 2026-01-09 10:45:00 | 877.15 | 2026-01-21 09:15:00 | 786.19 | TARGET_HIT | 0.50 | 10.37% |
| BUY | retest2 | 2026-02-11 15:15:00 | 811.80 | 2026-02-12 09:15:00 | 800.05 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-02-25 11:30:00 | 758.60 | 2026-03-02 09:15:00 | 720.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:45:00 | 757.65 | 2026-03-02 09:15:00 | 719.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:30:00 | 758.60 | 2026-03-04 09:15:00 | 682.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 10:45:00 | 757.65 | 2026-03-04 09:15:00 | 681.88 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-03-13 09:15:00 | 912.10 | 2026-03-16 09:15:00 | 816.90 | STOP_HIT | 1.00 | -10.44% |
| SELL | retest2 | 2026-03-19 09:30:00 | 802.50 | 2026-03-23 09:15:00 | 762.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 11:00:00 | 802.50 | 2026-03-23 09:15:00 | 762.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 13:00:00 | 802.20 | 2026-03-23 09:15:00 | 762.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 09:30:00 | 803.55 | 2026-03-23 09:15:00 | 763.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 11:30:00 | 798.90 | 2026-03-23 09:15:00 | 758.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 798.00 | 2026-03-23 10:15:00 | 758.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:30:00 | 802.50 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2026-03-19 11:00:00 | 802.50 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2026-03-19 13:00:00 | 802.20 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2026-03-20 09:30:00 | 803.55 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2026-03-20 11:30:00 | 798.90 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2026-03-20 12:15:00 | 798.00 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.26% |
| BUY | retest1 | 2026-04-08 09:15:00 | 804.85 | 2026-04-09 09:15:00 | 885.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 846.35 | 2026-04-17 09:15:00 | 930.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:45:00 | 849.45 | 2026-04-17 09:15:00 | 934.40 | TARGET_HIT | 1.00 | 10.00% |
